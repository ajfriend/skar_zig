# The reduced solver: trust-region descent on h(b)

**Status:** prototype on branch `investigate/wide-cap-dnc`
(`src/reduced.zig`, `SolveOptions.method = .reduced`). This document is
the tracking doc for the method: the writeup, the measurements so far,
and the validation ledger toward promoting it (first as the `.auto`
fallback, possibly as the default solver). History of how we got here:
`docs/wide-cap-dnc-report.md` (the wide-angle DNC investigation and the
joint-IPM prototype it superseded).

## The idea in one paragraph

The solve is really a 2D problem wearing a 9-dimensional coat. Define

    h(b) = min over A of { −log det A : ‖A·xᵢ‖₂ ≤ bᵀxᵢ for all i }

— "the best cone whose axis-of-projection is b." Partial minimization
of a jointly convex problem is convex, so **h is convex on the unit
ball**, and it only improves as ‖b‖ grows, so its minimum over the
sphere is the global optimum of the full problem. Minimizing h over the
sphere is a *two-dimensional* smooth-ish convex-structured optimization,
and every ingredient a second-order method needs is already in the
codebase: the inner minimization at fixed b is exactly the fast path's
2D MVEE, the gradient of h falls out of the inner solution for free,
and h itself is computable — so descent can be *verified*, not hoped
for. The reduced solver is a textbook trust-region BFGS on h over the
sphere, built entirely from parts the fast path already had.

## Everything, seen through h

The value of the reduction isn't just the new solver — it's that every
solver and every historical failure in this codebase becomes a
one-line statement about h:

- **The fast path is gradient descent on h minus the merit function.**
  Its "weighted centroid" axis-update direction c is exactly −∇h(b)/3
  in the tangent plane (envelope theorem, see below). What it lacks is
  any evaluation of h: steps are accepted on feasibility alone, and the
  `DampState` controller adjusts the step size only *after* a possibly
  bad step is already taken. The reduced path adds the merit function
  and a second-order model; nothing else about the per-iteration
  machinery changes.
- **The wide-cap limit cycle is what unguarded gradient descent does on
  a stiff landscape.** Past ~82° cap radius the gradient field of h
  stiffens like sec²(θ_max); a fixed damping heuristic can't find the
  shrinking stable step size, and with no merit function there is
  nothing to reject the overshoots. The trust region rejects them by
  construction — same gradient, same oracle, no limit cycle.
- **The fast path's quasi-Newton preconditioner is a Hessian model
  without a safety net.** `quasiNewtonAxisDirection` (M⁻¹·c) is already
  groping toward a second-order model of h; the reduced path's BFGS is
  the same instinct with a merit function to keep it honest and a trust
  region to bound it.
- **The joint IPM is what you pay for refusing the reduction.** It
  solves the full 9-variable problem with a barrier schedule that is
  geometry-blind by design: ~33–119 Newton steps on everything,
  including 22 on hex (whose answer is nearly the identity). That
  blindness makes it a poor product solver and an excellent *reference
  oracle* — it cross-validates the reduced path's answers precisely
  because it shares none of its structure.
- **`DampState` vs the trust region:** both adapt a step size, but the
  trust-region ratio ρ compares predicted against *actual* decrease in
  h and restores state on rejection; `DampState` shrinks α after
  accepting a step it never evaluated. The trust region is the damping
  controller with hindsight replaced by foresight.
- **"hex converges in 1 outer iteration" and "reduced converges in 0"
  are the same fact.** The initial axis from the halfspace check is
  already optimal; the fast path spends its iteration discovering that
  via one FW cycle, the reduced path's iteration-0 certificate says it
  outright.
- **The a5_res0 sparse-init fix (v0.4.0) was an oracle improvement, not
  an outer-method improvement** — it made one evaluation of h cheaper
  on redundant inputs. It therefore transfers to the reduced path
  unchanged (measured below: 12/12 dense cells in ≤ 2 TR iterations).
  Likewise the proposed away-step FW (`docs/away-step-fw.md`) is a
  better h-oracle; it would speed up *every* path equally and is
  orthogonal to the choice of outer method.
- **The extreme-κ certification floor that sank the pure joint path
  doesn't exist here** because h is evaluated and certified in the
  rescaled gnomonic chart — the same well-conditioned coordinates the
  fast path certifies in. The joint IPM certifies in raw 3D, where
  σ_max·ε cancellation noise floors the gap at ~1e-5 on finest-res
  cells.

## The three identities that make it cheap

1. **The oracle is the existing inner MVEE.** At fixed b, substituting
   zᵢ = xᵢ/(bᵀxᵢ) turns the inner problem into the centered 3D minimum
   volume enclosing ellipsoid of the zᵢ — the D-optimal design problem.
   The lifted chart points qᵢ = [pᵢ; 1] that `mveeFw` already iterates
   on are exactly the coordinates of zᵢ in the [Q̂ | b] basis, so the
   chart MVEE and the inner minimization coincide. One h-evaluation =
   `projectGnomonic` + `rescaleP` + `initWeights`(first call) +
   `mveeFw` + `newtonPolish` — the fast path's inner cycle with a real
   iteration budget instead of one step.
2. **The value.** With S = Σ wᵢ·qᵢ·qᵢᵀ the design moment in the scaled
   chart: h(b) = ½·(log det S + 3·ln 3) + 2·ln s_scale. (D-optimal
   design values transform by a constant under the chart rescaling; the
   2·ln s_scale term puts evaluations at different b on a common
   scale.)
3. **The gradient is free.** By the envelope theorem,
   ∇h(b) = −3·Σ wᵢ·zᵢ, whose tangent-plane component is −3·c where
   c = Σ wᵢ·pᵢ is the weighted centroid `computeMoments` already
   returns. No differentiation of the inner solution is needed.

Certification reuses `recoverAPerp` + `dualityGapConstructed`
wholesale, so `.reduced` returns the same certified `Outcome` as the
fast path, with the same gap semantics.

## The algorithm

Standard 2D trust-region BFGS over the sphere (retraction:
b′ = normalize(b + Q̂·u)):

1. Evaluate h, g = −3c, and the certificate at the current b; stop when
   the certified |gap| ≤ `gap_tol`.
2. Dogleg step u against the 2×2 BFGS model B within radius Δ. If the
   model's own predicted decrease is non-positive (BFGS can go
   ill-conditioned near active-set kinks), reset B = B₀·I and re-solve
   — with a fresh model the prediction is positive whenever g ≠ 0.
3. Trial-evaluate h(b′) (weights snapshot/restored on rejection — the
   warm start is part of the state). ρ = actual/predicted decrease;
   reject and shrink Δ if ρ < 0.05; accept, certify, BFGS-update
   (curvature-guarded, with the 2×2 model parallel-transported between
   tangent bases), and possibly grow Δ otherwise.

Knobs live in `config.reduced` (inner oracle budget/tolerance,
trust-region constants, B₀ = 3·I — the exact Hessian of h at a circular
optimum). All prototype values, untuned.

## Performance so far (2026-07-07, `zig build ex-compare`, ReleaseFast)

All measurements in this section (ex-compare, the probes, and the
`joint_test.zig` assertions) solve to the library default
**`gap_tol = 1e-6`** — a *certified* duality gap: convergence means the
constructed dual certificate proves the iterate is within 1e-6 of
optimal in −log det units, not that iteration merely stalled. "DNC"
likewise means "could not certify 1e-6," under the default
`max_outer = 100` except where a probe says otherwise. Converged gaps
in the tables often land well below the tolerance (e.g. 3e-11 on
cap82_s1) because the last trust-region step overshoots it; AR
agreement between converged solvers is correspondingly tighter than the
gap bound (≤ 1e-7 relative vs fast; the looser ~1e-4 checks against the
Clarabel references reflect the stored references' 7-significant-digit
precision, not solver disagreement).

Headline table:

| axis | fast | joint IPM | reduced |
|------|------|-----------|---------|
| wide-cap grid DNC (60–89.5° × 10 seeds × n ∈ {20, 200}) | wall at ~82°; 10/10 DNC ≥ 84° at n=200 | 0 | **0** |
| bundled manifest DNC (61 cases) | 0 | 4 (extreme-κ floor) | **0** |
| a5_res0 dense (12 × 320 pts) | 12/12, ~6 iters | not measured | **12/12, ≤ 2 iters** |
| mean median-time vs fast (manifest, mutually converged) | 1× | 12.5× | **0.9×** (1.7× after the re-cert phase landed — floor cells now buy their certificates; tuning item below) |
| iterations | 1–30 outer | 33–119 Newton | **0–34 TR** |

Selected per-case rows (iterations / min µs):

| case | fast | reduced | note |
|------|------|---------|------|
| hex | 1 / ≲1 | **0** / ≲1 | initial certificate passes outright |
| h3_res09 | 4 / 4 | **1** / 3 | |
| h3_res15 | 4 / 3 | **1** / 3 | |
| h3_r15_midLat (κ ~ 1e7) | 1 / 2 | **0** / 5 | pure joint floors at gap ~3e-5 here |
| np400 | 5 / 13 | 3 / 69 | oracle cost, see below |
| ha_05 | 10 / 21 | **4** / 37 | |
| ha_14 (80° cap) | 30 / 40 | **21** / 133 | |
| dnc_small_wide | 11 / 72 | **6** / 53 | |
| cap82_s1 … cap89_s3 (fixtures) | DNC forever | **14–34** / 84–367 | ARs match Clarabel to ≤ ~1e-4 rel |

Reading the split: **iterations are ≤ fast everywhere measured** —
h-descent with a real model needs fewer steps than h-descent with a
damping heuristic, exactly as the framing predicts. Wall-time is ahead
of fast wherever iterations dominate and behind (~3–7×) on dense
mid-width inputs where the *first* cold oracle call runs a long FW
budget (np400, ha_*, 200-pt caps at 60–80°). At the widest dense cells
(≥ 89°, 200 pts) reduced is the fastest method in absolute terms
(~82–84 µs vs joint ~180–240 µs, vs fast burning ~210 µs failing).
AR agreement with fast on mutually-converged cases: ≤ 1e-7 relative
(most ≤ 1e-13).

Robustness fix found during the grid sweep: n=20 / 88° / seed 6
initially stalled at gap 0.15 — near an active-set kink the BFGS
curvature pairs left the model ill-conditioned enough that the dogleg's
own prediction went negative, and the loop mistook that for
stationarity. The model-reset-and-retry in step 2 fixed it (now 19
iterations to the AR the joint oracle confirms). Worth keeping in mind
as the failure shape to watch for in stress tests: *model corruption
masquerading as convergence*.

## DGGS survey validation (2026-07-07, probe14: 10k cells × {h3 r15, s2 L30, a5 r30})

Ledger item done. Matrix {fast, reduced} × {1e-3, 1e-6} (+ joint at 1e-6):

| system, tol | fast DNC | reduced DNC | maxRelΔAR (both converged) |
|-------------|----------|-------------|-----------------------------|
| h3 @1e-3 and @1e-6 | 0 / 0 | **0 / 0** | 3.5e-8 |
| s2 @1e-3 | 0 | **0** | 9.4e-8 |
| s2 @1e-6 (floor) | 2173 | **1434** | 8.3e-8 |
| a5 @1e-3 | 0 | **0** | 2.4e-7 |
| a5 @1e-6 (floor) | 4739 | **4237** | 1.4e-7 |

Full parity at the survey tolerance, and at the strict 1e-6 the reduced
path certifies **more** of the floor-marginal population than fast on
both S2 (+739 cells) and A5 (+502) — its axis sits at the h-minimum, so
its certificate attempts are better centered than fast's wandering
iterates. (The joint IPM certifies almost nothing here — 473 / 0 / 0 —
its raw-3D certification floor, as documented.) Survey wall-times within
~2× of fast.

### What it took: the re-certification phase (and two reverted detours)

Getting here surfaced a mechanism worth its own framing. On extreme-κ
cells the constructed certificate is sensitive to the incidental
numerical state at noise amplitude — the first cert's M-Cholesky fails
*for the fast path too* (measured on A5 res-30); fast passes on its
second outer iteration purely because iterating re-samples the state.
The reduced path's TR loop, having *correctly* found h stationary,
would compute one certificate and stop. And everything it can do at a
bit-frozen axis is **idempotent**: an oracle re-run reproduces its
state, a raw FW step is a no-op once g_max < 3 numerically, polish is
at its fixed point. The sampling lever fast enjoys is axis motion. So
the fix is honest about that: **the re-cert phase is a few fast-path
outer iterations warm-started at the TR optimum** — FW step → polish →
certify → damped axis micro-step (`config.reduced.RECERT_MAX` bounds
it). TR for the global descent, fast iteration for the terminal
certification.

Fixed for real along the way: a **NEG_GAP ordering bug** — a
converged-at-noise gap can be slightly negative (−5e-9 on H3 r15
cells), so the acceptance check must run before the hard
NegativeDualityGap guard, mirroring the fast path's break-before-guard
ordering.

Tried and reverted (history note on `config.reduced.INNER_*`): a
rounds/burst/patience oracle that alternated short FW bursts with
polish, with best-w tracking and a round-0 baseline. Every variant
could return an *under-refined* inner state — and then the envelope
gradient −3·c is not the gradient of the h being reported (that
identity needs inner optimality). The trust region reads the mismatch
as a systematically wrong slope (measured ρ → −7.95 as Δ → 0 on cap82)
and stalls. Lesson, in the running framing: **the trust region is only
as honest as its oracle's (value, gradient) consistency** — an oracle
allowed to return non-optimal states must return the matching fixed-w
gradient, and it's simpler to keep the oracle inner-optimal.

Also isolated en route: `mveeFw`'s near-singular pairwise fallback
(`step = w[jm]`, a full drop) fires at converged designs and
obliterates a needed support point that polish cannot resurrect (its
active set is w-thresholded). The fast path co-evolved with this sharp
edge — it stops running FW the moment its cert passes. Any future
oracle change (and the away-step FW work) should treat that fallback as
the known hazard.

## Validation ledger

Done (this branch):

- [x] `zig build test -Dslow=true` green with `.reduced` tests included;
      no CANARY shifts (default path untouched).
- [x] Wide-cap fixtures (`tests/wide_cap_cells.zig`): converge, AR vs
      Clarabel ≤ ~1e-4 rel, feasibility ≤ 1e-12 (`tests/joint_test.zig`).
- [x] Manifest agreement vs fast incl. the extreme-κ cells pure joint
      fails on (`tests/joint_test.zig`).
- [x] Wide-cap grid 0 DNC (ex-compare part 2).
- [x] a5_res0 dense 12/12 in ≤ 2 iterations (probe13).
- [x] Cross-validated against two independent oracles: Clarabel (SDP)
      and the joint IPM.

- [x] **DGGS surveys with `.reduced`** (h3/s2/a5, 10k cells each) at
      1e-3 and 1e-6 — full parity at 1e-3; *better* than fast at the
      1e-6 floor (see the DGGS survey validation section; probe14).

Open, roughly in order:

- [ ] **States/countries surveys with `.reduced`**: AR agreement +
      iteration counts (Tennessee at 41 fast outer iters is the
      interesting row).
- [ ] **Randomized rotation stress** (extreme_aspect-style SO(3) sweeps)
      on the wide-cap and high-AR geometries; watch for the
      model-corruption failure shape.
- [ ] **Oracle cost tuning**: adaptive inner tolerance (loose early,
      tighten with the certified gap), skip certification while the
      gradient is large, revisit `INNER_ITERS`. Target: close the 3–7×
      gap on dense mid-width inputs without disturbing the 0–1-iteration
      behavior on small cells.
- [ ] **Repoint `.auto`'s fallback** from `.joint` to `.reduced`
      (strictly better on every measured axis) once the surveys pass.
- [ ] **CANARY-style iteration pins for `.reduced`** on the small-cell
      hot path (mirror `tests/dggs_dnc_test.zig`) before any default
      flip.
- [ ] **Decide the endgame**: `.reduced` as default with fast retired,
      or fast kept as the small-cell shortcut. Requires the tuning item
      and a bench story on the 4–10-point hot path.
- [ ] Revert the TEMPORARY probe knobs (`skar.probe_*`,
      `reduced.probe_trace`, `config.joint.probe_mu`) and decide the
      probes' fate before anything merges to `main`.

## Reproducing

- `zig build ex-compare` — manifest × {fast, joint, reduced} + the
  wide-cap grid × {fast, joint, reduced, auto}.
- `zig build test -Dslow=true` — includes `tests/joint_test.zig`
  (fixtures + agreement for both experimental paths).
- Branch probes: `probe10.zig` (reduced smoke + timing), `probe11.zig` /
  `probe12.zig` (the seed-6 stall + trace), `probe13.zig` (a5_res0
  dense), `probe9.zig` (the barrier-schedule sweep that motivated the
  reduction), `probe_sdp.py` (Clarabel cross-check).
