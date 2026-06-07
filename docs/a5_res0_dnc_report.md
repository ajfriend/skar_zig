# A5 resolution-0 cells DNC at default settings (outer-iteration count scales with boundary point count)

**Status:** **FIXED.** Shipped mechanism (v0.4.0): **size-gated sparse
farthest-point FW init** (`algo.SEED_SPARSE_*` in `src/config.zig`,
`farthestPointSeed` in `src/skar.zig`; tests in `tests/a5_res0_test.zig`). This
*superseded* the v0.3.0 inner-FW boost — same a5_res0 fix but ~56× faster and
~3–6× faster on genuine medium/large inputs (which the boost had slowed). See
**Implemented (v0.4.0)** below for the shipped fix and the boost-vs-sparse data;
the earlier sections record how we got here (boost → survey → sparse), and the
original bug report is retained from **Evidence** onward.
**Found:** 2026-06-07, sweeping S2/A5 across all resolutions from skar_py
(`scripts/dggs/dnc_sweep.py`).
**Summary:** All **12 A5 resolution-0 base cells** (the largest A5 cells, ~70°
across) return `.did_not_converge` at skar's **default** settings
(`gap_tol = 1e-6`, `max_outer = 100`), with the gap stuck at ~0.04. This is
**not** a precision floor and **not** geometric hardness — it's that the
solver's **outer-iteration count scales with the number of boundary points**.
A5's default `cell_to_boundary` emits **320 points** for these cells (5 edges ×
64 segments), which needs ~145 outer iterations — just over the 100 cap. The
*same cells* reduced to their 5 corners converge in ~7 iterations. With
`max_outer` raised, all 12 converge cleanly to gap ~1e-7.

The solver should handle these — they are valid, well-conditioned, large cells.
The concern is that convergence cost blows up on redundant, near-collinear
boundary samples.

## Context

This is a *different* phenomenon from the H3 r7–r10 stall fixed in v0.2.0
(`a5_res0` here is **not** floored — more iterations *do* help), and *also*
distinct from the finest-resolution S2/A5 f64 gap floor in
`tests/dggs_dnc_test.zig`. The survey below confirms that finest-resolution band
**is** a genuine floor: the inner-FW fix does not rescue it (it slightly worsens
S2), so the `did_not_converge` there is correct and must stay. In the full S2/A5
resolution sweep, S2 was clean+monotonic and A5 was clean at every resolution
**except res 0** (the lone non-monotonic spike) and the finest-resolution band
(r28–r30).

---

## Investigation (2026-06-06): root cause found

### Root cause — the inner MVEE solve gets only one Frank-Wolfe step per cycle

The outer loop solves the inner 2D MVEE (minimum-area enclosing ellipse of the
gnomonic-projected points) by taking **exactly one Frank-Wolfe step per cycle**:
`mveeFw(wb.Ps, 1, 0.0, wb.Ql, wb.w)` in `src/skar.zig`. Newton polish then
refines the weights, but **only within the current active set**, and its
fraction-to-boundary step *never zeros a weight* — so it cannot remove a point
from the active set. The only thing that removes points is FW's pairwise
drop-step: **~2 points per outer iteration** (`FW_PER_NEWTON = 2`).

The duality gap cannot close until the active set
(`w > ACTIVE_THRESH = 1e-12`) collapses to the true MVEE support (~5 corners).
Instrumenting one 320-point cell shows the active set draining ~2/iter in
lockstep with the gap:

```
outer=  0 gap=2.816e-1 n_active=318 ...
outer= 10 gap=2.423e-1 n_active=297 ...
outer= 50 gap=1.228e-1 n_active=198 ...
outer= 99 gap=4.580e-2 n_active= 82 ...   (still draining at the cap)
```

So **outer iters ≈ (n_points − support) / 2 → linear in point count**. 320
points needs ~145 > the 100 cap → DNC. This matches the report's headline
exactly; it's the active-set drainage rate, not the geometry.

### Two ruled-out fix directions

- **Hull preprocessing cannot help** (explains Evidence 2). The 320 boundary
  points are all genuine convex-hull vertices even under the solver's *own
  gnomonic projection*: measured `orthographic_hull = 320` **and**
  `gnomonic_hull = 320`. A5 cell edges are **not** great circles — each edge
  sample bows outward from the corner-to-corner chord, so every sample is a true
  hull vertex. They are MVEE-redundant (the enclosing ellipse touches ~5), not
  hull-redundant. Reducing toward the 5 corners via the hull is therefore
  impossible.
- **Gradient-based pruning is ill-conditioned.** The cell is so near-circular
  that the most-interior active point still has MVEE gradient `g_lo ≈ 2.68` vs
  `3.0` for a binding point — interior and binding points are not cleanly
  separable. Pruning by gradient risks dropping near-binding constraints (the
  H3 r7–r10 lesson). Not viable.

### The fix: give the inner FW a real iteration budget

Replacing the 1-step inner solve with `mveeFw(wb.Ps, K, inner_tol, ...)` (K FW
steps, with `inner_tol` so easy cases still break after one step). All 12 a5_res0
cells, default `max_outer = 100`:

| K (inner FW/cycle) | a5_res0 converged | outer iters (avg) | time/solve   |
|--------------------|-------------------|-------------------|--------------|
| 1 (today)          | **0 / 12 (DNC)**  | capped at 100     | 2,155,727 µs |
| 2                  | 12 / 12           | 82                | —            |
| 10                 | 12 / 12           | 22                | 246,990 µs   |
| 20                 | 12 / 12           | 14                | 107,702 µs   |
| 50                 | 12 / 12           | 6                 | 26,144 µs    |
| 100                | 12 / 12           | 6                 | **4,142 µs** |
| 200 (tol 1e-10)    | 12 / 12           | 5                 | —            |

A larger budget collapses the active set in the **first** outer iteration, so
every later Newton polish (the O(k³) cost) runs on ~5 points instead of 320.
The fix is therefore **~520× faster** as well as correct, and at K ≳ 100 the
outer count is essentially **n-independent** (~5–6, pure cone-axis convergence)
— directly satisfying the report's goal of "convergence cost that doesn't
explode on redundant boundary samples." `inner_tol` keeps small/easy cases at
one FW step, so the common case is not slowed.

### Surprise: two finest-resolution "f64 floor" cells converge with the fix

> **⚠️ SUPERSEDED by the survey below.** This subsection over-generalized from
> two cells. The broad survey shows the finest-resolution *population* is
> genuinely floor-limited and a blanket fix makes S2 **worse**; `A5_CELL` /
> `S2_CELL` converting is not representative. The finest-cell DNC guard is
> correct and stays. Kept here for the record.

`tests/dggs_dnc_test.zig` asserts the finest S2/A5 cells (`A5_CELL`, `S2_CELL`)
*correctly* DNC at 1e-6, framed as a genuine f64 gap floor. With the fix *these
two* converge to **valid** cones (default `gap_tol = 1e-6`):

| cell    | baseline (1 FW) | with fix (K ≥ 50)                                   |
|---------|-----------------|-----------------------------------------------------|
| A5_CELL | DNC, gap 2.9e-5 | CONVERGED gap 3.8e-7, AR 2.21164596, maxViol −4e-8  |
| S2_CELL | DNC, gap 2.4e-6 | CONVERGED gap 1.2e-7, AR 1.21362116, maxViol 1.2e-8 |

The recovered aspect ratios match the documented references
(`2.21164606`, `1.21362116`), `checkFeasibility` ≤ roundoff. But the survey
below shows this does *not* generalize — most finest cells stay DNC, and S2 gets
worse. So these two are quirks, not evidence the guard is wrong.

## Survey validation (2026-06-06): the blanket fix regresses — gate it

Ran the in-repo DGGS survey (`zig build dggs-aspect`, 10k cells each of H3 r15 /
S2 L30 / A5 r30 — the finest-resolution, corner-only cells) baseline-vs-fix at
both `gap_tol = 1e-3` (where all converge) and the strict `1e-6`. The survey does
**not** include a5_res0 / densified boundaries, so it tests the *finest cells*
and the *already-converging population*, not a5_res0 itself (validated above).

**Blanket `K = 100` (always boost) — NOT a clean win:**

| metric (baseline → blanket fix) | h3        | s2                | a5          |
|---------------------------------|-----------|-------------------|-------------|
| DNC @ 1e-6                      | 0 → 0     | 2173 → **3318** ✗ | 4739 → 4712 |
| solve time @ 1e-3 (ms)          | 173 → 266 | 133 → 131         | 175 → 173   |
| solve time @ 1e-6 (ms)          | 175 → 271 | 216 → 240         | 356 → 376   |
| per-cell max rel ΔAR @ 1e-3     | 3.5e-8    | 9.1e-8            | 2.4e-7      |

So the blanket fix: (a) **does not** rescue the finest population — S2 DNC
*rises* by 1145, A5 ~flat; the finest cells are genuinely floor-limited and the
two converting fixtures were not representative. (b) **Slows near-circular H3
hexagons ~1.5×** (every small cell now grinds extra inner-FW steps). (c) Leaves
ARs unchanged (correctness fine). This **refutes claims #2 and #3** for a blanket
change, and confirms the finest-cell DNC guard should **stay**.

**Gated fix — boost only when it pays off:**

```zig
const inner_fw_iters: u32 = if (nw > 16) 100 else 1;   // nw = working-set size
const inner_fw_tol:  f64 = if (nw > 16) 1e-9 else 0.0;
```

Small cells (finest DGGS 4–6 pts, H3 hexagons 6 pts: `nw ≤ 16`) keep the
**bit-identical** 1-step path; only large/dense inputs (a5_res0: `nw = 320`) get
the boost. Measured:

- **a5_res0:** still fixed (routes through exactly the validated `K=100, 1e-9`).
- **Finest cells @ 1e-6:** DNC counts **bit-identical to baseline** (h3 0, s2
  2173, a5 4739); times within noise. Floor behavior unchanged.
- **H3 speed restored:** 173 → 179 ms @ 1e-3 (vs blanket 266 ms).
- **Full `zig build test` is GREEN** — no canary shifts, no finest-cell guard
  failure (every test cell is small → baseline path).

The gate is the recommended shape: it fixes a5_res0 with **zero** change to every
already-working case.

## Implemented (v0.4.0, 2026-06-07): size-gated sparse FW init

This is the **shipped** fix; it replaced the v0.3.0 boost (below) after the
follow-up experiment showed sparse init strictly dominates it.

The MVEE inner solve lets Frank–Wolfe move weight onto the support. FW *grows*
the support well but *prunes* it poorly (only a drop-step removes a point;
Newton can't zero a weight). So the uniform start `w_i = 1/nw` is the worst case
when the support is a small subset — a slow drain. **Fix:** for `nw >
SEED_SPARSE_MIN_POINTS` (16), seed only `SEED_SPARSE_K` (5) well-spread extreme
points (`farthestPointSeed`, greedy farthest-point) so FW grows *into* the
support. Small inputs keep the uniform start (optimal for near-circular cells).
The per-cycle inner FW returns to the plain 1-step schedule for everyone — the
boost machinery is gone.

**Measured (boost C0 vs sparse C3; a5_res0 at 50-rep µs, medium/large at 100-rep
bench):**

| metric                        | v0.3.0 boost | v0.4.0 sparse        |
|-------------------------------|--------------|----------------------|
| a5_res0 µs/solve              | 4055         | **74** (~56×)        |
| bench TOTAL µs (medium/large) | 1121         | **332** (~3.4×)      |
| a5_res0 converged             | 12/12        | 12/12                |
| finest f64 floor (s2/a5 DNC)  | 2173 / 4739  | 2173 / 4739 (held)   |
| h3 mean outer iters           | 1.0          | 1.0 (gate → uniform) |
| aspect ratios                 | —            | unchanged            |

So the v0.3.0 boost had a hidden cost: its `nw>16` gate also caught *genuine*
large polygons (countries, np400, ha_*) and ran 100 wasteful inner-FW steps on
them — sparse init undoes that while fixing a5_res0 better. The size gate is
load-bearing for the same reason as before: ungated sparse *slows* small
near-circular cells (h3 1 → ~11 iters) because uniform is already their optimum.

- **`src/config.zig`** — `algo.SEED_SPARSE_MIN_POINTS = 16`, `SEED_SPARSE_K = 5`
  (replaced `INNER_FW_BOOST_*`), full doc-comment.
- **`src/skar.zig`** — `farthestPointSeed` + gated init before the outer loop;
  `mveeFw` back to `(…, 1, 0.0, …)`.
- **`tests/a5_res0_test.zig`** — unchanged structure; iter ceilings hold
  (dense ≤ 20, sparse ≤ 4).
- **Validation:** full `zig build test -Dslow=true` green (no CANARY shifts —
  small cells bit-identical); DGGS @1e-6 floor unchanged; states/countries ARs
  unchanged. Full C0–C5 study lives on the experiment branch
  `exp/fw-sparse-init` (its copy of this report).

---

## (superseded) Implemented (2026-06-06): point-count-gated inner-FW boost

> Shipped in v0.3.0, replaced by the sparse FW init above in v0.4.0. Retained as
> history. The boost fixed a5_res0 but ran a per-cycle inner-FW budget that
> slowed genuine medium/large inputs ~3–6×.

The fix is the gated form of option 1 below.

- **`src/config.zig`** — new `algo` constants with a full doc-comment (the
  two-regime rationale + a `FUTURE:` note pointing at a fully-corrective /
  away-step inner FW that could unify the regimes and remove the branch):
  `INNER_FW_BOOST_MIN_POINTS = 16`, `INNER_FW_BOOST_ITERS = 100`,
  `INNER_FW_BOOST_TOL = 1e-9`.
- **`src/skar.zig`** — before the outer loop:
  ```zig
  const inner_fw_boost = nw > algo.INNER_FW_BOOST_MIN_POINTS;
  const inner_fw_iters: u32 = if (inner_fw_boost) algo.INNER_FW_BOOST_ITERS else 1;
  const inner_fw_tol:  f64 = if (inner_fw_boost) algo.INNER_FW_BOOST_TOL  else 0.0;
  ```
  used in the per-cycle `mveeFw` call. Inputs with `nw ≤ 16` take the
  **bit-identical** 1-step path; larger/denser inputs get the draining budget.
- **`tests/a5_res0_test.zig`** (+ fixture `tests/a5_res0_cells_dense.zig`,
  registered in `tests/all.zig`) — two regression tests at the strict default:
  all 12 dense 320-point cells converge (`gap ≤ 1e-6`); the 5-corner version of
  the same cell converges and yields the **same** aspect ratio (cross-checks the
  two regimes against each other).

**Validation:**

- Full `zig build test` **green** — no CANARY shifts, finest-cell DNC guard
  still passes (those are genuine floors; the gate leaves them on the 1-step
  path, bit-identical).
- a5_res0: 0/12 → **12/12** at default, ~145 → ~6 outer iters, ≈500× faster.
- Mid/large genuine polygons (the boost *does* engage: countries have 100s of
  vertices) show no regression — re-running the states/countries surveys vs the
  pre-fix solver: max relative ΔAR **6.8e-9** (states) / **1.7e-7** (countries)
  — input-precision noise; outer-iteration totals essentially flat (countries
  slightly *fewer*). So the boost is neutral-to-beneficial wherever there's an
  active set to drain, and only overhead-free 1-step where there isn't.

The remaining cost is intrinsic and accepted: this is a workaround that
*documents* the two regimes, not the final algorithm — see the `FUTURE:` note in
the `INNER_FW_BOOST_*` doc-comment.

## Fix options (ranked)

1. **Implemented — gate the inner-FW budget on working-set size.** Boost
   (`mveeFw(wb.Ps, K, inner_tol, …)`, K≈100, inner_tol≈1e-9) only when `nw`
   exceeds a threshold (~16); else keep today's `1, 0.0`. Validated end-to-end:
   a5_res0 fixed (~520× faster), every finest/small case bit-identical, full
   suite green. Expose `K`, `inner_tol`, and the threshold as named `algo`
   constants. (Threshold just needs to sit above max small-cell size ~6 and below
   a5_res0's 320; tie it to `n_hull` or pick ~16/32.)
2. **Rejected — blanket `K = 100`.** Refuted by the survey: worsens S2 finest
   convergence and slows H3 ~1.5×.
3. **Rejected — explicit support pruning** (drop step in Newton, or gradient
   threshold). Ill-conditioned on near-circular cells; risks the H3 r7–r10
   failure mode.
4. **Rejected — raise default `max_outer`.** Masks the scaling; leaves the ~2
   s/solve cost in place.

## Done / not-doing

- ✅ **Shipped: size-gated sparse FW init** (`algo.SEED_SPARSE_*`, v0.4.0) —
  see **Implemented (v0.4.0)**. Replaced the v0.3.0 inner-FW boost.
- ✅ **Regression tests** — dense 320-pt (all 12) + sparse 5-corner, each
  asserting convergence at the strict default **and** an outer-iteration ceiling
  (dense ≤ 20, sparse ≤ 4) as a slow-grind guard.
- ✅ **Finest-cell DNC guard kept** — those are genuine floors; the `σ_max·ε`
  narrative in `src/api.zig` stands. (An earlier draft proposed flipping that
  test to assert convergence — retracted.)

## Possible future work

- **Smarter-than-size gate.** The real discriminator is "redundant /
  non-symmetric," not size. The size gate skips *small irregular* polygons where
  sparse init would also help (ungated sparse beat the gate on states/countries
  iteration counts). A cheap proxy (hull ≫ lifted dim, or near-cocircularity)
  could capture those without hurting symmetric small cells.
- **Unify via away-step FW.** A fully-corrective / away-step inner FW that drives
  weights to exactly zero could drain the active set well enough that even the
  uniform start is fine, removing the gate. Bigger change; own validation. Design
  note: `docs/away-step-fw.md`.
- **(Optional) broaden the survey** to multi-resolution / densified boundaries
  for full coverage — deferred.

> **Reproducing the investigation:** the numbers in **Investigation** /
> **Survey validation** came from temporary harnesses (a `mveeFw`
> iteration-count knob, active-set/hull tracing, and `a5dbg` / survey-tolerance
> toggles) since reverted. The tables capture their full output.

## Evidence

### 1. Iteration count scales with boundary point count

Same 12 res-0 cells, varying only how finely `cell_to_boundary` segments each
edge (`max_outer=5000` so everything converges):

| segments/edge               | n points | DNC @ default (max_outer=100) | outer iters to converge (min/med/max) |
|-----------------------------|----------|-------------------------------|---------------------------------------|
| 1 (just the 5 corners)      | 5        | 0 / 12                        | 1 / 7 / 7                             |
| 2                           | 10       | 0 / 12                        | 8 / 9 / 11                            |
| 4                           | 20       | 0 / 12                        | 11 / 14 / 15                          |
| 8                           | 40       | 0 / 12                        | 21 / 24 / 28                          |
| **auto = 64** (the default) | **320**  | **12 / 12**                   | **142 / 144 / 149**                   |

Iterations grow roughly linearly with point count; the default `max_outer=100`
is exceeded around a few hundred points. The points are highly redundant — 64
near-collinear samples along each of 5 edges — so the extreme/active set is
still essentially just the 5 corners.

### 2. `n_hull` does not mitigate it

On the 320-point cells at default `max_outer=100`, varying the hull knob:

| n_hull | 10    | 20    | 50    | 100   | 320   |
|--------|-------|-------|-------|-------|-------|
| DNC    | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |

The hull-reduction parameter doesn't prevent the iteration blow-up.

### 3. All 12 cells, default vs raised iteration cap

Every cell DNCs at default (gap ~0.035–0.047) and converges at `max_outer=5000`
to gap ~1e-7 in ~142–149 iters:

```
id 200000000000000  default=did_not_converge(gap 4.58e-02)  maxouter5000=converged(gap 7.23e-07, it 144)
id 600000000000000  default=did_not_converge(gap 4.73e-02)  maxouter5000=converged(gap 1.91e-07, it 143)
id a00000000000000  default=did_not_converge(gap 4.53e-02)  maxouter5000=converged(gap 1.99e-07, it 149)
id e00000000000000  default=did_not_converge(gap 3.67e-02)  maxouter5000=converged(gap 3.24e-07, it 148)
id 1200000000000000 default=did_not_converge(gap 4.00e-02)  maxouter5000=converged(gap 6.76e-07, it 142)
id 1600000000000000 default=did_not_converge(gap 4.11e-02)  maxouter5000=converged(gap 6.07e-07, it 146)
id 1a00000000000000 default=did_not_converge(gap 4.07e-02)  maxouter5000=converged(gap 1.28e-07, it 144)
id 1e00000000000000 default=did_not_converge(gap 4.46e-02)  maxouter5000=converged(gap 5.90e-08, it 145)
id 2200000000000000 default=did_not_converge(gap 3.55e-02)  maxouter5000=converged(gap 5.38e-07, it 142)
id 2600000000000000 default=did_not_converge(gap 3.57e-02)  maxouter5000=converged(gap 5.36e-07, it 144)
id 2a00000000000000 default=did_not_converge(gap 3.72e-02)  maxouter5000=converged(gap 1.59e-07, it 145)
id 2e00000000000000 default=did_not_converge(gap 3.70e-02)  maxouter5000=converged(gap 5.59e-08, it 144)
```

## Geometry note

An A5 res-0 cell is a pentagon centred on a dodecahedron-face axis; its **5
corners are coplanar** (they lie on a small circle — note the identical z in the
fixture below). So the corners-only point set is essentially a circle → a
near-circular enclosing cone, trivial (~7 iters). The *densified* boundary
points bow off that small circle along the curved edges; with 320 of them the
solver takes ~145 iters.

## Reproducers

### Python (needs pya5)

```python
import a5, skar
for cid in a5.get_res0_cells():            # 12 base cells
    v = skar.to_vec3(
        [(la, lo) for lo, la in a5.cell_to_boundary(cid)],  # default: 320 pts
        geo='latlng_deg')
    print(skar.solve(v, geo='vec3').status)               # did_not_converge
    print(skar.solve(v, geo='vec3', max_outer=5000).status)  # converged, ~145 it
```

### Inline 5-corner fixture (cell `200000000000000`)

This **converges** (~7 iters) — included to show the underlying geometry is
trivial (note the shared z = coplanar corners). To reproduce the DNC, use the
densified 320-point boundary instead (companion file below).

```zig
const A5_RES0_CORNERS = [_][3]f64{
    .{ -0.3809559340728538, -0.47044139975172183, 0.7959632313708467 },
    .{  0.3296945010323943, -0.5076850109021148,  0.7959632313708467 },
    .{  0.5847183416148108,  0.1566748074353539,  0.7959632313708467 },
    .{  0.03168130793103096, 0.6045153670780082,  0.7959632313708467 },
    .{ -0.5651382165053821,  0.2169362361404745,  0.7959632313708467 },
};
```

### Full failing fixtures

All 12 cells at the default 320-point boundary (the actual `.did_not_converge`
inputs before the fix) are the committed test fixture
**`tests/a5_res0_cells_dense.zig`** (`pub const A5_RES0_CELLS =
[_][]const [3]f64{ ... }`), exercised by `tests/a5_res0_test.zig`. Feed any one
to `skar.solve` with default options to reproduce the pre-fix DNC (or confirm the
post-fix convergence).
