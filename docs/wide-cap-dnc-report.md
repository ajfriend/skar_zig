# Wide-angle inputs DNC: the outer axis iteration limit-cycles past ~81° cap radius

**Status:** root cause characterized; fix direction #1 (joint barrier-Newton
solver) **prototyped on this branch** — see **Prototype results** below.
Probe harnesses live on this branch (`investigate/wide-cap-dnc`):
`probe_dnc.zig` … `probe8.zig` at the repo root plus temporary `probe_*`
knobs in `src/skar.zig` (all marked TEMPORARY; revert before merging anything
to `main`).
**Found:** 2026-07-07, sweeping randomized spherical caps of increasing angular
radius (the `ha_*` construction) past the bundled `ha_14` (80°) case.
**Summary:** For dense random inputs, the solver hits a **hard wall at cap
radius ≈ 81–83°**: beyond it, `solve` returns `.did_not_converge` at *any*
iteration budget — the outer axis iteration falls into a **limit cycle**, not a
slow crawl. The failure is **not intrinsic to the problem**: the primal SDP is
jointly convex in `(A, b)` (paper, eq. primal), and a generic interior-point
conic solver (Clarabel) solves every failing probe case to optimality with
modest aspect ratios (1.16–1.54). The wall is an artifact of the fast path's
nonconvex parametrization (axis fixed-point iteration + damping heuristic).

This is the missing piece of the long-tail robustness picture. The other
long-tail families are in better shape:

- **Redundant dense boundaries** (a5_res0, 320 pts): fixed in v0.4.0 via
  size-gated sparse FW init; the away-step FW note
  (`docs/away-step-fw.md`) covers retiring the size-gate proxy.
- **Finest-resolution f64 gap floor** (S2 L30 / A5 r30): genuine κ-driven
  floor, documented on `SolveOptions.gap_tol`; correctly DNCs at 1e-6.
- **Extreme aspect ratio: NOT hard.** Symmetric stretched caps converge in
  **1 outer iteration at AR 1000** (gap 6e-7); a 174°-span arc at AR 328
  converges in 2. The `extreme_aspect_test.zig` "AR ≈ 80–100 plateau" on
  near-antipodal arcs is the *width* mechanism below, not an AR limit.

## Evidence

### 1. The wall (200 random points per cap, `max_outer = 500`, 10 seeds)

| cap radius | DNC /10 | median outer iters (converged) |
|-----------|---------|-------------------------------|
| 79°       | 0       | 19                            |
| 80°       | 0       | 30                            |
| 80.5°     | 0       | 35                            |
| 81°       | 0       | 53                            |
| 81.5°     | 4       | 102                           |
| 82°       | 7       | 92                            |
| 83°       | 10      | —                             |

Iteration counts climb steeply approaching the wall — the classic signature of
a first-order fixed-point iteration losing its contraction factor. `ha_14`'s
"widest angular extent we converge on" description is this wall, measured.

Raising `max_outer` to 2000 rescues nothing at ≥ 83°: the gap oscillates
between ~1e0 and 1e30 (1e30 = the dual-certificate Cholesky fails outright)
with the uncertified AR bouncing 1.1–44. One 82° seed crawled to gap 2e-4 at
2000 iters; the rest cycle forever.

### 2. It is width × density, not width alone (85° cap, 10 seeds)

| n points | DNC /10 |
|----------|---------|
| 5        | 0       |
| 10       | 0       |
| 20       | 1       |
| 50       | 4       |
| 200      | 10      |

Sparse wide inputs still converge; dense wide inputs never do.

### 3. Trace of a failing case (85°, seed 1)

`alpha` pins at `DAMP_MIN = 0.05` while the axis-gradient norm `c_norm`
oscillates 1 → 1.6e3. Since the b-step rotates the axis by ≈ atan(α·‖u‖) with
‖u‖ = c_norm, a pinned α of 0.05 against c_norm ~ 1e3 is a ~89° axis swing:
the iterate repeatedly overshoots the optimum, walks to within 2e-4 of the
feasible-cone boundary (points at 89.99° from the axis; `s_scale` = tan θ_max
hits 4.8e3), rebounds, and never settles. The gap is frequently 1e30 because
the constructed dual certificate's Cholesky fails far from optimality.

### 4. Ruled-out cheap fixes (all measured, `max_outer = 500`)

- **Better initial axis** — centroid warm-start produces a bit-identical
  trajectory (the Farkas b is already centroid-like). Not the init.
- **Trust-region cap on the axis step** (cap α·‖u‖ at 1.0 / 0.5 / 0.25 / 0.1):
  no change in any DNC cell; converged cases unaffected. Even small feasible
  steps fail ⇒ the *direction* is unreliable, not just the magnitude.
- **Inner-FW budget boost** (K=100, tol 1e-9, i.e. solve the fixed-axis MVEE
  well before each b-step), alone and combined with step caps: no change
  (one 82° seed flipped to converged; everything ≥ 84° still 100% DNC).

Mechanism, in short: the outer loop is a damped first-order fixed-point
iteration on the axis with **no merit function** — steps are accepted on
feasibility alone, and the damping controller only shrinks α *after* a bad
step. The sensitivity of the projected centroid to axis motion grows like
sec²(θ_max), so past ~81° the contraction condition would need α far below
`DAMP_MIN` (and n-dependent), while the heuristic has no way to find it. No
amount of step-size tuning fixes a non-descent iteration on this landscape.

### 5. The problem itself is easy: generic conic solver cross-check

The primal is jointly convex (paper, eq. primal): minimize −log det A subject
to ‖A·xᵢ‖₂ ≤ b·xᵢ and ‖b‖₂ ≤ 1, over A ∈ S³₊₊ and b ∈ R³. Solving the three
dumped failing cases with cvxpy + Clarabel (scratchpad `solve_sdp.py`):

| case      | status  | AR       | max angle from axis | axis eigval        | max constraint viol |
|-----------|---------|----------|---------------------|--------------------|---------------------|
| cap82_s1  | optimal | 1.159634 | 81.98°              | 0.577356 (≈ 1/√3)  | 7e-10               |
| cap85_s1  | optimal | 1.269181 | 84.99°              | 0.577369            | 1.4e-9              |
| cap89_s3  | optimal | 1.542028 | 88.87°              | 0.577355            | 5e-9                |

`b` comes out an eigenvector of `A` with eigenvalue 1/√3 to 1e-5, exactly as
the paper's optimality property predicts. These are well-conditioned,
modest-AR optima — nothing about the answers is extreme; only our path to them
is.

## Fix directions (ranked)

1. **Joint convex fallback solve.** Implement a small primal-dual /
   barrier-Newton method on the joint convex formulation — 9 unknowns
   (6 for A, 3 for b), n SOC constraints, log-det objective. Per Newton step:
   O(n) to assemble a 9×9 KKT system; expect a few tens of Newton steps
   regardless of geometry (self-concordant barrier ⇒ polynomial, globally
   convergent for every feasible input including 89.9° caps). Run it **only
   when the fast path returns `.did_not_converge`** (or trips an oscillation
   detector): zero cost on the hot DGGS path, and the library becomes
   robust-by-construction on the long tail. A native implementation should
   land in the tens-of-µs range for n ≤ a few hundred (hull-reduced) —
   comparable to today's medium cases.
2. **Oscillation detection → early DNC.** Independent of (1): today a
   wide-cap input burns all `max_outer` iterations producing a garbage last
   iterate (the DNC payload's uncertified AR can read 44 when the true
   optimum is 1.27). Detect the limit cycle (e.g. gap not improving over a
   window + `alpha` pinned at `DAMP_MIN`) and exit early — cheaper, and the
   DNC diagnostics stop being misleading. With (1) in place this becomes the
   fallback trigger.
3. **Away-step FW** (`docs/away-step-fw.md`, already planned): orthogonal to
   this — it addresses the redundant-boundary tail and the size-gate proxy,
   not the wide-angle wall (probe 4 above shows inner-solve quality is not
   the binding constraint here).
4. **Interim documentation.** Until (1) lands, `SolveOptions` / readme should
   state the supported regime: dense inputs are reliable up to ~80° angular
   radius from the optimal axis; beyond that expect DNC (and the fixture
   `ha_14` marks the edge). The infeasibility boundary at 90° is handled
   correctly (Farkas cert) — the gap is only 81–90°.

## Prototype results (2026-07-07): joint barrier-Newton path

Fix direction #1 is now prototyped on this branch: `src/joint.zig`, a
barrier-Newton interior-point method on the joint convex `(A, b)` formulation
(9 unknowns, damped Newton + log-barrier path-following), selected via the
experimental `SolveOptions.method` enum (`.fast` default = bit-identical old
behavior; `.joint`; `.auto` = fast, then joint on DNC). Preprocessing
(validation → Farkas → hull → coplanarity) is shared by both paths, and the
joint path certifies through the *same* constructed-dual machinery
(`dualityGapConstructed`), so both return identical `Outcome` semantics.
Measurements from `zig build ex-compare` (ReleaseFast, this machine):

**Robustness — the wide-angle hole is closed.** On the random-cap grid
(widths 60–89.5° × 10 seeds × n ∈ {20, 200}):

| method | DNC on the grid | worst cell |
|--------|-----------------|------------|
| fast   | up to 10/10 for n=200 at ≥ 84° (the wall) | — |
| joint  | **0 anywhere**, incl. 89.5°/200 pts | ~240 µs |
| auto   | **0 anywhere** | ~440 µs (fast burn + joint solve) |

Joint's ARs match the Clarabel cross-check to ≤ ~1e-4 relative on the three
committed fixtures (`tests/wide_cap_cells.zig`), with `checkFeasibility` at
machine epsilon. The regression tests (`tests/joint_test.zig`) pin: joint and
auto converge on all three fixtures; fast still DNCs on them; joint agrees
with fast (AR rel ≤ 1e-4) across easy/medium manifest cases; auto is
bit-identical to fast whenever fast converges.

**Runtime — joint is a fallback, not a replacement.** Across the 54
mutually-converged manifest cases, joint costs a mean ~12× the fast path's
median wall time (hex: 1 outer iter/≲1 µs vs 33 Newton steps/18 µs;
np400: 14 → 85 µs; ha_14: 41 → 85 µs). Newton-step counts sit at 33–70 for
nearly everything — geometry-independent, as IPM theory predicts — which is
exactly why it wins in the wide regime and loses on the hot path. `.auto`
keeps the hot path untouched by construction.

**Known limitation — extreme-κ certification floor (pure `.joint` only).**
4 of 61 manifest cases DNC under pure `.joint`: `h3_r12_ring10`,
`h3_r15_midLat`, `h3_r15_pent`, `h3_r15_ring10` — finest-resolution cells
with tangent eigenvalues σ ~ 1e6–1e7. The joint iterate *finds* the optimum
(ARs match fast to 6–7 digits) but the certified gap floors at ~1e-5: the
barrier multipliers λᵢ = 2sᵢ/(t·rᵢ) need rᵢ ~ 1/t, and in raw 3D coordinates
rᵢ = sᵢ² − ‖A·xᵢ‖² drowns in κ·ε cancellation noise (~σ_max·ε ≈ 1e-9) once
t ≳ 1e9. The fast path certifies these same cells fine because its inner
machinery works in the rescaled gnomonic chart (`rescaleP`). Consequences:

- `.auto` is unaffected — the fast path owns exactly these cells, so the
  fallback never fires there (measured: auto = fast on all four).
- Fix directions if pure joint ever needs them: certify in the scaled
  gnomonic chart (mirror `rescaleP`), or hand the joint axis b̂ to one
  fast-path finalization cycle (inner MVEE + polish at fixed near-optimal
  axis) — "joint for global convergence, fast for the finish."

**Verdict.** `.auto` delivers the robustness goal — zero DNCs across the
entire wide-cap grid and no change whatsoever to already-working inputs — at
zero hot-path cost. Pure `.joint` is not competitive as a default (12× on
easy cases) and shouldn't replace the fast path. Before flipping `.auto` on
as the default: tune the barrier schedule (µ, warm-start t across stages) to
cut the 33–70 Newton steps, consider an oscillation detector so the fast
path bails before burning all `max_outer` iterations pre-fallback (~halves
auto's worst-case), and decide whether the wide-cap fixtures should join the
cases manifest with `.auto` expectations.

**Superseding note:** the reduced path below dominates `.joint` on every
axis measured; if it holds up, the `.auto` fallback (and possibly the fast
path's role as the primary) should be re-pointed at `.reduced`.

## Prototype results (2026-07-07, later): reduced trust-region path

A follow-up question — "why does a 9-variable convex problem need 33–70
Newton steps?" — led to the barrier-schedule sweep (`probe9.zig`: µ-tuning
trims ~25–40% then floors at ~40–55 steps; even hex costs 22, and µ = 1000
destabilizes) and from there to the observation the joint IPM ignores: the
problem has a *reduced* convex structure.

**The reduction** (`src/reduced.zig`, `method = .reduced`). Define
h(b) = min_A { −log det A : ‖A·xᵢ‖ ≤ bᵀxᵢ }. Partial minimization of the
jointly convex primal makes h convex on the unit ball and radially
non-increasing, so its sphere minimum is the joint optimum and no spurious
strict local minima exist for a descent method. Three exact identities make
it cheap:

- the inner problem at fixed b **is** the fast path's 2D lifted MVEE: the
  lifted points [pᵢ; 1] are the coordinates of zᵢ = xᵢ/(bᵀxᵢ) in the
  [Q̂ | b] basis (`initWeights` + `mveeFw` + `newtonPolish` reused verbatim);
- h(b) = ½(log det S + 3·ln 3) + 2·ln s_scale from the design moment S in
  the rescaled chart;
- the envelope theorem gives ∇h(b) = −3·Σwᵢzᵢ, whose tangent component is
  −3·c — the weighted centroid the fast path already computes. **The fast
  path is gradient descent on h minus the merit function**; the reduced path
  adds the merit function and a 2D trust-region BFGS model (dogleg steps,
  model reset on a non-positive prediction), certifying each accepted
  iterate with the fast path's own `recoverAPerp` + `dualityGapConstructed`.

**Measured** (`ex-compare`, same protocol):

| axis | fast | joint | reduced |
|------|------|-------|---------|
| wide-cap grid DNC | wall at ~82° (10/10 at ≥84°, n=200) | 0 | **0** |
| manifest DNC | 0 (by construction) | 4 (extreme-κ floor) | **0** |
| mean slowdown vs fast (manifest) | 1× | 12.5× | **0.9×** |
| iterations | 1–30 outer | 33–119 Newton | **0–34 TR** |

Highlights: hex converges in **0 iterations** (the initial certificate
already passes), h3_res09 in 1 (vs fast's 4), ha_05 in 4 (vs 10), ha_14 in
21 (vs 30); the wide caps take 9–34 TR iterations where fast limit-cycles
forever; and the extreme-κ finest-res cells converge in 0 iterations — the
joint path's certification floor doesn't exist here because certification
runs in the scaled chart. At the widest densest cells (89°+, 200 pts) the
reduced path is the fastest method in *absolute* terms (~82–84 µs vs joint's
~180–240 µs, vs fast burning ~210 µs failing). ARs agree with fast to ≤1e-7
relative on mutually-converged cases and with Clarabel on the fixtures.

Where it's slower: dense mid-width caps (~60–80°, 200 pts) cost ~3–7× fast
(cold inner oracle runs the full FW budget on the first evaluation) — mean
across the manifest is still 0.9× because it wins elsewhere. Obvious tuning:
adaptive inner tolerance (loose early, tighten with the gap), and skipping
certification on early iterates.

One TR robustness fix is already in: near active-set kinks the BFGS
curvature pairs can leave the model ill-conditioned enough that the dogleg's
own prediction goes negative; the loop resets to the B₀·I model and retries
instead of declaring stationarity (found on the n=20/88°/seed-6 grid cell,
which now converges in 19 iterations to the same AR joint finds).

**Verdict (updated).** `.reduced` matches or beats the fast path on
iterations everywhere measured, closes the wide-angle hole with *better*
wall-time than the joint IPM, has no extreme-κ blind spot, and reuses the
fast path's inner machinery and certification wholesale. It is the natural
candidate to *replace* the two-path split: promote it to the `.auto`
fallback first (strictly better than `.joint` there), and — after broader
validation (full DGGS surveys at strict tol, states/countries, randomized
stress with rotations) — consider it as the default solver. The fast path
would remain as the specialized hot-path shortcut, or be retired if
`.reduced`'s 0–1-iteration behavior on small cells holds up under the
CANARY-style scrutiny.

The joint IPM keeps independent value as the *reference implementation*:
its convergence is schedule-driven and geometry-blind, which is exactly
what you want in a cross-check oracle (and it validated the reduced path's
ARs here, alongside Clarabel).

## Reproducing

All numbers from the probe harnesses on this branch:

- `probe_dnc.zig` — width sweep (wall discovery), AR sweeps (probes 2–4).
- `probe2.zig` — failing-case trace + centroid-init A/B (uses
  `skar.probe_trace` / `skar.probe_centroid_init`).
- `probe3.zig` — trust-region step-cap sweep (`skar.probe_step_cap`).
- `probe4.zig` — inner-FW budget × step cap grid (`skar.probe_inner_iters`).
- `probe5.zig` — dumps failing cases as JSON for the SDP cross-check.
- `probe6.zig` — wall localization + n-dependence.
- `probe_sdp.py` (cvxpy/Clarabel, PEP 723; `uv run probe_sdp.py`) — the
  convex cross-check; reads probe5's JSON dumps.

`zig build test -Dslow=true` is green on this branch (probe knobs default
off; the default path is bit-identical).
