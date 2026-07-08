# Proposal: away-step Frank–Wolfe for the inner MVEE solver

**Status:** proposed, staged — ready to execute in its own session(s) after
the `investigate/wide-cap-dnc` branch lands. Supersedes the earlier
future-work note at this path (written before the reduced solver existed;
its goal stands, its context and plan are updated here).
**Estimated size:** ~1 focused day of algorithm work + 1–2 days of
validation/reconciliation, using the measurement infrastructure already on
the branch. Stage 1 carries no fast-path risk; stage 2 is the
CANARY-shifting step and needs explicit human sign-off.

## Why (three reasons now, was one)

1. **Retire the sparse-init size gate** (the original motivation). The MVEE
   inner FW grows support well but prunes poorly, so init quality matters
   and is input-dependent: uniform is optimal for symmetric small cells,
   sparse seeding for redundant dense boundaries (a5_res0). The
   `algo.SEED_SPARSE_*` size gate is a proxy for "support sparsity" and
   misses small irregular polygons. An inner solver that prunes fast makes
   the init irrelevant: uniform everywhere, delete `farthestPointSeed`,
   `SEED_SPARSE_MIN_POINTS`, `SEED_SPARSE_K`, and the gate branch.

2. **Eliminate the near-singular drop-step hazard at the source.** The
   pairwise step's fallback (`step = w[jm]` when `det_G ≤ NEAR_SING`,
   `src/skar.zig` `mveeFw`) fires on noise-level descent signals at
   converged designs and zeroes a genuine support point that `newtonPolish`
   cannot resurrect (its active set is w-thresholded). This bit the reduced
   path four times in one validation day (h3_res09 + cap82 during the
   rounds-oracle detour; New York during burst tuning — see
   docs/reduced-solver.md "What it took" and the tuning ledger entry). The
   fast path is only accidentally shielded: it runs 2 FW steps per outer
   and stops iterating the moment its certificate passes, so it rarely
   executes FW at a converged design. Away-step FW replaces the ad-hoc
   drop with a line-search-justified away step whose drop boundary is a
   deliberate event.

3. **Shrink the reduced oracle's burst defense.** `config.reduced.
   INNER_BURST = 64` exists specifically so a mid-burst noise-drop can
   self-heal before the stall-exit h-sample. That minimum evaluation cost
   is the main residual of the reduced path's 2–3× gap on mid-size
   synthetic caps (np400, ha_14). With a drop-safe inner solver the burst
   can shrink toward "a few", directly attacking that residual.

## The algorithm

Current `mveeFw` per iteration: build S = Σ wᵢqᵢqᵢᵀ, Cholesky, gradients
gᵢ = qᵢᵀS⁻¹qᵢ, toward-vertex `j_max` (max g), away-vertex `j_min` (min g
among active), then either a pairwise swap with step `a/(2·det_G)` capped
at `w[jm]` (with the hazardous near-singular fallback) or a vanilla FW
step.

Away-step FW keeps the same per-iteration quantities and changes the
decision:

- **Toward step** (mass onto `j_max`): first-order progress ∝ g_max − 3.
- **Away step** (mass off `j_min`): progress ∝ 3 − g_min; direction
  w + γ·(w − e_jmin); max step γ_max = w[jm]/(1 − w[jm]), and taking
  γ_max zeroes `w[jm]` — the drop, now reached only when the 1-D line
  search sends it there.
- Pick the direction with more first-order progress; step size by the
  closed-form 1-D minimizer of the log-det objective along the chosen
  direction (the existing `det_G` machinery is exactly this quadratic for
  the pairwise direction; derive the toward/away analogues — same
  ingredients: g_max, g_min, g_cross), clamped to the boundary. Keep
  `tol.WEIGHT_ACTIVE` semantics for the active set.

Literature anchors: Todd & Yıldırım (Khachiyan/MVEE); Kumar & Yıldırım
(away-step / WAFW for MVEE). Linear convergence with away steps is the
standard result; the practical draw here is O(log) support collapse from a
uniform start instead of the current ~2-drops-per-outer drain.

~60–100 lines replacing the current loop body, plus config constants.
Implement as a sibling function (or comptime flag) so both the old and new
inner solvers exist during evaluation — stage 1 depends on that.

## Staged plan

### Stage 1 — reduced oracle only (no fast-path exposure)

Wire the away-step solver into `reduced.evalH` and the RECERT phase only;
`solveFast` keeps today's `mveeFw` verbatim, so the default path stays
bit-identical and NO fast CANARY can shift.

Do in stage 1:
- Implement + unit-test the line search (agreement with the pairwise step
  on non-degenerate pairs; drop boundary exactness; no negative weights).
- Try shrinking `INNER_BURST` (64 → 8-ish) and, if the hazard is truly
  gone, simplifying the stall exit.
- Full battery, all of which exists on the branch and runs in minutes:
  - `zig build test -Dslow=true` — the reduced CANARY pins
    (dggs_dnc_test "CANARY(reduced)" 0/3/0/3/3, a5_res0 reduced ceilings,
    wide-cap ceilings in joint_test) are the tripwires; shifts here are
    *expected* and each needs a story.
  - probe14 (DGGS 30k × 2 tols: parity + floor counts), probe19
    (states 50/50, countries 177/177 at default budget), probe20
    (rotations 160/160), probe13 (a5_res0), probe22 (canary cells),
    `zig build ex-compare` (manifest mean + wide-cap grid).
- Success criteria: all convergence counts hold or improve; np400/ha_14
  ratio vs fast improves measurably (target ≤ 1.5×); no new failure
  shapes. The catalogued shapes to watch: oracle (value, gradient)
  inconsistency read as a wrong TR slope; corrupted-state stall exits;
  cert-edge thrash.

### Stage 2 — fast-path adoption + gate deletion (CANARY sign-off)

Swap `solveFast`'s inner solver to away-step, revert init to plain uniform
everywhere, delete `algo.SEED_SPARSE_*` + `farthestPointSeed` + the
`initWeights` branch.

- **Every fast CANARY pin will likely shift** — per repo policy each shift
  is flagged and explained to a human, never silently bumped. This is the
  step that requires explicit sign-off before starting.
- Acceptance (the original note's gauntlet, all harnesses now in-repo):
  a5_res0 12/12 at strict default **with uniform init**, few iters, wall
  time ≤ current (~74 µs-class); symmetric small cells stay ~1 fast outer
  iteration with uniform init (the whole point — no symmetric-cell
  regression); `ex-bench` per-case small cells not slower (ignore TOTAL);
  DGGS floor DNC counts unchanged at 1e-6; states/countries unchanged;
  full slow suite with reconciled canaries.
- Only after stage 2 lands does the doc cleanup happen: update
  a5_res0_dnc_report.md (the gate it documents is gone) and
  reduced-solver.md (drop-hazard notes become history).

Stage 1 is worth doing even if stage 2 stalls: the reduced path is the
consumer that actually steps on the hazard, and months of stage-1 burn-in
is exactly the evidence stage 2's sign-off wants.

## Risks (updated from the original note)

- **Drop-boundary numerics**: clamp like the existing `step > w[jm]`
  guard; keep `tol.WEIGHT_ACTIVE`; property-test that weights stay in the
  simplex.
- **Warm-start coupling** (bigger deal now than when first written): the
  reduced path warm-starts weights across axis moves and re-runs the
  solver at converged designs — the exact regime where the old solver
  misbehaved. Stage 1's battery covers it; watch the oracle-consistency
  failure shape specifically.
- **Aggressive draining destabilizing the fast path's damping** (stage 2
  only): away steps change the weight trajectory feeding the axis update;
  the damped controller was co-tuned with the old cadence. If ha_*-band
  cells wobble, that's the first place to look.
- **Cholesky-break path**: `S.cholesky() orelse break` must still behave
  when the support is tiny early on (sparse starts are gone in stage 2 —
  uniform start keeps S full-rank from the outset, which actually
  *removes* a risk).

## Pointers

- `src/skar.zig` — `mveeFw` (the loop to replace), `initWeights` /
  `farthestPointSeed` (stage-2 deletions), `newtonPolish` interaction.
- `src/reduced.zig` — `evalH` + RECERT phase (stage-1 call sites),
  `config.reduced.INNER_*` (burst defense to retire).
- `docs/reduced-solver.md` — the four drop-step incidents, the
  oracle-consistency lesson, and the validation ledger this proposal's
  battery mirrors.
- `docs/a5_res0_dnc_report.md` — the drain story and boost-vs-sparse
  history (stage 2 updates it).
- Todd & Yıldırım; Kumar & Yıldırım (away-step MVEE).
