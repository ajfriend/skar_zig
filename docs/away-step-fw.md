# TODO: away-step Frank–Wolfe to retire the sparse-init size gate

**Status:** not started — design note / future work.

The A5 res-0 fix has gone through two mechanisms: a size-gated inner-FW *boost*
(v0.3.0) and a size-gated *sparse farthest-point init* (v0.4.0, current — see
`a5_res0_dnc_report.md`). Both are gated on working-set size (`nw > 16`,
`algo.SEED_SPARSE_MIN_POINTS`). This note records the deeper change that would
remove the gate entirely.

## Why the gate exists (and why it's a proxy)

The MVEE inner solve moves Frank–Wolfe weight onto the support. FW **grows** the
support well (each step pulls weight onto the most-violated point) but **prunes**
it poorly: only a pairwise drop-step removes a point (~2 per outer iteration),
and Newton's fraction-to-boundary step can never zero a weight.

So the *quality of the weight init* is "how close it starts to the optimal weight
vector," and the optimum's shape is input-dependent:

- **Near-circular small cells** (e.g. H3 hexagons): the enclosing ellipse touches
  *all* vertices, so the optimal weights are near-**uniform**. `w = 1/n` is
  already the answer → converges in ~1 outer iteration. A sparse seed *breaks the
  symmetry* (skews the moment `M = Σ wᵢ·Pᵢ·Pᵢᵀ`, hence the recovered shape and the
  cone-axis gradient), costing iterations to undo (~1 → ~11).
- **Redundant/dense inputs** (a5_res0: 320 points, ellipse touches ~5): the
  optimal support is **sparse**. Uniform init is far (must drain ~315 weights to
  zero, ~2/iter → ~145 outer iters → DNC); a sparse seed is close.

The real discriminator is **support sparsity** (`support/n`), not size. The size
gate is a cheap proxy and is imperfect: small *irregular* polygons have sparse
support and would benefit from sparse init, but the gate skips them (measured:
ungated sparse beat the gate on states/countries iteration counts).

No *pure init* escapes this. Fixing a5_res0 requires a hard-sparse active set
(hard zeros); not regressing symmetric cells forbids zeroing an arbitrary subset
of a symmetric point set. A soft weighting (e.g. `wᵢ ∝ dist-from-centroid`)
preserves symmetry but leaves every point above the active threshold, so it
doesn't drain a5_res0. The conflict is structural.

## The fix: make the solver prune fast, so the init stops mattering

Replace the plain/pairwise inner FW with an **away-step (a.k.a. fully-corrective)
Frank–Wolfe** that can drive weights to **exactly zero** quickly — the standard
Wolfe away-step MVEE algorithm (Todd & Yıldırım; Kumar & Yıldırım). Then:

- Keep **uniform init everywhere** → symmetric small cells stay on their optimum
  (no regression, no gate, no seed routine).
- The away-step solver **drains the redundant points fast** from that uniform
  start → a5_res0 converges in a handful of outer iterations too.

This removes `algo.SEED_SPARSE_*`, the `nw > 16` branch in `solve()`, and
`farthestPointSeed` — the init becomes irrelevant because the solver is robust to
it.

## Sketch

`mveeFw` (`src/skar.zig`) currently does, per inner iteration: build
`S = Σ wᵢ·qᵢ·qᵢᵀ`, Cholesky, find `j_max` (max `gᵢ = qᵢᵀ S⁻¹ qᵢ`) and the
min-gradient *active* point `j_min`, then a pairwise swap (move mass `j_min →
j_max`) or a vanilla FW fallback.

Away-step FW makes the **away direction first-class**:

- **Toward step** (standard FW): add mass to `j_max`. Progress ∝ `g_max − 3`.
- **Away step**: *remove* mass from `j_min` (the worst point still carrying
  weight). Progress ∝ `3 − g_min`. The max away step is bounded by
  `w[j_min]/(1 − w[j_min])`; taking the full max step **zeros** `w[j_min]` — the
  drop that the current pairwise step only manages ~2/iter.

Each iteration pick whichever direction (toward / away) gives more progress, with
the standard away-step line-search/step-size. This both grows and *aggressively
prunes* the support, so a uniform start collapses to the true support in O(log)
rather than O(n) steps. (The current pairwise swap is a restricted hybrid; the
generalization is letting the away step run to the drop boundary on its own.)

## Acceptance criteria (same gauntlet as the v0.4.0 change)

Prototype behind a toggle on an experiment branch and measure vs current main:

- **a5_res0** (`tests/a5_res0_cells_dense.zig`): 12/12 converge at strict
  default, few outer iters, fast (target ≈ the v0.4.0 sparse numbers, ~74
  µs/solve).
- **Symmetric small cells** (h3 hexagons): with **uniform init**, must stay at ~1
  outer iter (this is the whole point — no symmetric-cell regression).
- **Medium/large** (`zig build ex-bench`, states/countries): wall-time ≤ v0.4.0.
- **Finest f64 floor** (`zig build dggs-aspect` @ `gap_tol = 1e-6`): DNC counts
  unchanged (s2 2173 / a5 4739).
- **Full suite** `zig build test -Dslow=true` green. Expect **CANARY
  iteration-count shifts** (away-step changes trajectories) — flag for human
  confirmation per repo policy (`tests/dggs_dnc_test.zig`), don't silently bump.

If all hold: delete `algo.SEED_SPARSE_*` + `farthestPointSeed` + the gate, revert
init to plain uniform, and update `a5_res0_dnc_report.md`.

## Risks / notes

- **Numerical care** on the away step-size and the drop boundary (clamp like the
  existing pairwise `step > w[jm]` guard; keep `tol.WEIGHT_ACTIVE`).
- **Outer-loop coupling:** weights warm-start across outer iterations and feed
  the b-axis; confirm the away step doesn't destabilize the damped axis update.
- **Cholesky-break path** (`S.cholesky() orelse break`) must still behave when
  the support is tiny early on.
- Bigger change than the gated seed (~tens of lines of real algorithm + step-size
  logic), hence its own branch + validation rather than a drive-by.

## Pointers

- `src/skar.zig` — `mveeFw` (inner solver), the gated init in `solve()`,
  `farthestPointSeed`.
- `src/config.zig` — `algo.SEED_SPARSE_{MIN_POINTS,K}` (to be removed).
- `docs/a5_res0_dnc_report.md` — full history (boost → survey → sparse) and the
  measured boost-vs-sparse comparison.
- Literature: Todd & Yıldırım, "On Khachiyan's algorithm for the computation of
  minimum-volume enclosing ellipsoids"; Kumar & Yıldırım, away-step / WAFW MVEE.
