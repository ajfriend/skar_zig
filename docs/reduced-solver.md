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

Standard 2D trust region over the sphere (retraction:
b′ = normalize(b + Q̂·u)), with an *analytic per-evaluation model
Hessian* — the majorant Hessian — instead of BFGS (see the
majorant-model section below for the history):

1. Evaluate h, g = −3c, the certificate, and the model Hessian B̃ at
   the current b; stop when the certified |gap| ≤ `gap_tol`. B̃ is the
   fixed-w (majorant) Hessian of h on the sphere, computed in chart
   quantities from the design Cholesky already in hand:
   B̃ = 3·Σwᵢgᵢ·pᵢpᵢᵀ − 2·Σᵢⱼwᵢwⱼc²ᵢⱼ·pᵢpⱼᵀ + (Σwᵢgᵢ)·I₂. Since
   h̃_w ≥ h with equality at the current point, its curvature
   over-estimates the envelope's — steps are conservative and nearly
   always accepted. At a circular optimum B̃ → 3·I, which is where the
   old BFGS seed B₀ = 3 came from; it survives only as the non-PD
   fallback, now derived rather than fitted.
2. Dogleg step u against B̃ within radius Δ (isotropic fallback if B̃
   is non-PD from roundoff or far-field states).
3. Trial-evaluate h(b′) (weights snapshot/restored on rejection).
   ρ = actual/predicted decrease: reject and shrink if ρ < 0.05;
   accept otherwise, shrinking gently if ρ < ¼ (the quadratic model
   over-promised — third-order terms of h dominate over this radius;
   without the textbook ρ < ¼ rule the loop creeps at ρ ≈ 0.15) and
   growing on ρ ≥ 0.7 radius-limited steps. No model state crosses
   iterations — nothing to transport, nothing to corrupt.

Knobs live in `config.reduced` (inner oracle budget/tolerance,
trust-region constants). All prototype values.

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

## The majorant-Hessian model (option A, branch `exp/majorant-hessian`)

The question "are we exploiting all the problem's properties?" had one
big yes-we-aren't: the model curvature was generic BFGS, when the
problem *offers* its Hessian. For frozen weights, h̃_w(b) is a smooth
closed-form **global majorant of h touching at the current point**
(h = min over inner states), so its Hessian — O(active²) from
forward-solves against the already-factored design Cholesky — is a
derived, per-evaluation model with a built-in safety property: model
curvature ≥ envelope curvature, so steps are conservative and nearly
always accepted.

**What replacing BFGS with it bought, measured** (full battery vs the
BFGS baseline at the `investigate/wide-cap-dnc` tip):

- Convergence byte-parity everywhere it matters: DGGS counts identical
  (incl. the 1e-6 floor), states 50/50 (max iters 44→5 for fast vs
  reduced, reduced mean 2.4), countries 177/177 (mean 3.0), rotations
  160/160, a5_res0 12/12, all CANARY pins unchanged (0/3/0/3/3).
- Wide-cap fixtures 17/23/22 iterations vs BFGS's 20/34/14 — slightly
  fewer total and much more uniform.
- Wall-time a wash (manifest mean 1.5× vs 1.4×, within run noise).
- **The real win is deletion**: the BFGS update, curvature guard,
  tangent-basis transport, and the model-reset-on-corruption machinery
  are gone. The model is recomputed fresh each evaluation from analytic
  structure — the "model corruption masquerading as convergence"
  failure shape (the seed-6 bug) is now structurally impossible.

**Two trust-region lessons collected on the way** (both now in
`config.reduced` doc-comments): without the textbook ρ < ¼
shrink-on-accepted-step rule, the loop can creep at ρ ≈ 0.15 forever
when h's third-order terms dominate the quadratic over the current
radius (cap89: 83 iterations; 28 with the rule); and the shrink for
accepted-but-poor steps must be gentler than the rejection shrink or Δ
oscillates across the model-fidelity boundary with GROW (cap89: 28 →
22 with SHRINK_POOR = 0.5). Notably, BFGS never showed the creep — its
secant pairs *measure* the far-field stiffness the analytic quadratic
can't see. The ρ-based radius rules recover that adaptivity while
keeping the stateless model.

**Option B (exact envelope Hessian via KKT sensitivity): assessed, not
worth it now.** The remaining wide-cap iterations are spent in the
far-field creep where the binding error is *third-order*, and the
exact Hessian (≤ the majorant's) would over-promise *more* there, not
less; its quadratic-convergence benefit only trims the ~2–4 tail
iterations near the optimum. It would add KKT sensitivity solves and
code for a marginal gain — revisit only if a workload emerges whose
cost concentrates in the endgame iterations.

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

- [x] **States/countries surveys with `.reduced`** (probe19, strict
      1e-6): states 50/50 both paths, reduced mean 2.7 outer iters vs
      fast's 13.4, maxRelΔAR 1.7e-8. Countries: reduced **177/177 at
      the default `max_outer = 100`** with a max of 5 iterations —
      including France, which the fast path can only solve because the
      survey exec quietly raises `max_outer` to 1000 for it (France
      needs 140 fast iterations; Chile 97; both extreme-AR elongated
      shapes, AR 7.8 / 6.2 — the fast path's degrading-contraction
      regime, invisible to the second-order model). maxRelΔAR 4.7e-8.
- [x] **Randomized rotation stress** (probe20: 8 geometries × 20 SO(3)
      rotations — wide-cap fixtures, arcs to AR 143, patch AR 17320,
      stretched caps): 160/160 converged, all gaps certified ≤ 1e-6,
      worst AR drift 1e-7 (mostly ≤ 1e-11); symmetric caps converge in
      0 iterations under arbitrary rotation. Neither catalogued failure
      shape appeared.
- [x] **Repoint `.auto`'s fallback** from `.joint` to `.reduced` — done;
      grid re-validated (auto 0 DNC everywhere, and its worst-case cost
      dropped ~25% since the reduced fallback beats the joint solve on
      the widest cells).

- [x] **Oracle cost tuning** (2026-07-07): two changes — (a) FW runs in
      bursts of 64 with a stall exit on the design value (κ-limited
      cells stop grinding the full budget at noise amplitude); (b)
      certification of accepted TR iterates is gated on the accepted
      step's predicted decrease (`pred ≤ 100·gap_tol` — pred is in gap
      units, so the gate is scale-aware; a ‖g‖-based gate was tried
      first and mis-fired on elongated regions whose Hessian scale
      ≫ B₀). A third change fell out of putting the fast path's CANARY
      cells through `.reduced`: (c) a **pred-noise exit** — when the
      step's predicted decrease falls below the merit function's own
      resolution (pred ≤ 1e-14·(1+|h|)), the ratio test can never
      verify a step, so the loop hands off to the RECERT phase instead
      of rejecting the same unresolvable interior Newton step ~26 times
      while Δ marches to its floor (measured on the H3 r9 canary:
      |g| = 3e-10, pred = 2e-20, cert at 6.6e-6 vs tol 1e-6 — 27
      iterations before the fix, 3 after; the A5 common canary went
      30 → 3 and the a5 survey's mean iterations 14.0 → 3.0).

      Results, survey aggregates: at the strict 1e-6, whole-config
      wall time showed s2 0.70× / a5 0.64× "faster than fast" — but
      see the CORRECTION below: that was a DNC-burn artifact. All
      convergence behavior held across the battery (fixtures, DGGS
      parity, states 50/50, countries 177/177, a5_res0, rotations,
      slow suite green).

      **CORRECTION (probe27, fair metric).** Whole-config survey times
      conflate success cost with failure cost: fast honestly burns its
      full `max_outer` (~50 µs/cell) on floor cells before reporting
      DNC, so any config containing DNCs overstates fast's time.
      Measured per-cell (min of 3) on the MUTUALLY-CONVERGED subsets
      only:

      | system, tol | both-converged | fast | reduced | ratio |
      |---|---|---|---|---|
      | h3, either tol | 10000 | 147 ms | 222 ms | **1.50×** |
      | s2 @1e-3 | 10000 | 121 ms | 124 ms | 1.02× |
      | s2 @1e-6 | 7250 | 91 ms | 91 ms | **1.00×** |
      | a5 @1e-3 | 10000 | 163 ms | 175 ms | 1.07× |
      | a5 @1e-6 | 3879 | 67 ms | 70 ms | 1.04× |

      So on successes: parity on s2/a5, and fast genuinely 1.5× faster
      on the h3 family — the real hot-path gap the fusion work must
      close. Two facts survive the correction in reduced's favor: it
      certifies more floor cells at 1e-6 (net +723 s2 / +466 a5,
      though not a per-cell superset), and its cost of FAILURE is
      ~2.5× cheaper (20–25 vs ~50 µs/DNC cell — it stops at
      stationarity instead of burning the budget), which bulk
      pipelines at strict tolerance do pay for. Residual on mid-size
      synthetic caps (np400 ~2.3×, ha_14 ~2.7×) unchanged.

- [x] **Eager first certificate — the fusion work's cadence half —
      CLOSED the h3 gap** (2026-07-08). Iteration 0 now runs the fast
      path's exact opening cadence (two FW steps, one polish,
      certify) before any full-precision oracle work; the full oracle
      + trust region engage only when that certificate fails. Safe
      w.r.t. the oracle-consistency lesson: the eager certificate is a
      pure upper-bound check that never feeds the TR model. Fair
      metric (probe27, mutually-converged only) after:

      | system, tol | before | after |
      |---|---|---|
      | h3 @1e-3 | 1.50× | **1.02×** |
      | h3 @1e-6 | 1.50× | **0.89×** |
      | s2 / a5 (either tol) | 1.00–1.07× | 0.97–1.02× |

      Reduced is now at parity or faster with fast on every DGGS
      system at both tolerances on successes, with the floor-coverage
      and cheap-failure advantages intact, CANARY pins unchanged
      (0/3/0/3/3), and the full battery green (geographies, fixtures,
      rotations, a5_res0, slow suite). In the running framing: the
      reduced method's iteration 0 is now *literally* the fast path's
      first outer iteration, and the trust region is what happens
      instead of the damped wander when that first certificate doesn't
      pass — the two designs have converged into one. The dedup half
      of the fusion work (shared design state) was measured
      unnecessary after this and skipped.

      CANARY-cell comparison (fast pins vs reduced, post-fix): H3 r15
      1 → **0**, S2 L30 1 → **0**, A5 hard tail 4 → **3**, H3 r9
      2 → 3, A5 common 2 → 3. Reduced meets or beats the pins on cells
      whose initial certificate passes outright and pays +1 on the two
      cert-edge cells that route through the RECERT phase.

      Two failed variants documented for the record, both re-runs of
      the oracle-consistency lesson: a burst of 16 let mveeFw's
      destructive near-singular drop step corrupt-and-partially-recover
      *inside* the stall window, so the exit landed on a corrupted
      state (New York DNC'd — trial h stuck 2.9e-3 high at Δ = 1e-12);
      and an incoming-baseline restore (round-0 redux) re-broke the
      wide caps by returning unrefined states. The burst must be big
      enough for the drop to self-heal before the h-sample; no
      snapshots, no restores.

Open, roughly in order:

- [x] **Guard `mveeFw`'s drop step at the source** — done on
      `exp/majorant-hessian`, threshold-free: the full-mass drop is
      taken only when the exact log-det change of the rank-2 update,
      (1 + γ·g_max)(1 − γ·g_min) + γ²·g_cross² (all quantities already
      in hand), exceeds 1. Ran the CANARY gauntlet on the shared
      solver: **every fast-path pin held** — the blocked-drop scenario
      never occurs on fast trajectories, so the guard is
      behavior-invisible there while making oracle-state corruption
      impossible. Two follow-on negative results recorded in
      docs/away-step-fw.md "Stage 1 findings": away-step FW as the
      oracle is slower than pairwise on large near-circular supports
      (reverted; `mveeFwAway` kept in-tree for the record), and
      shrinking the oracle burst below 64 harms robustness AND speed
      even with the guard — the burst floor was never the residual
      cost.
- [x] **CANARY-style iteration pins for `.reduced`** — landed in three
      places, same flag-don't-bump policy as the fast pins:
      `tests/dggs_dnc_test.zig` "CANARY(reduced)" section (the same
      five cells, pinned at 0 / 3 / 0 / 3 / 3 — the 0s are
      initial-cert-passes, the 3s are cert-edge cells routed through
      RECERT; notable: the fast path's "hard tail" A5 cell costs the
      same as the common one under reduced);
      `tests/a5_res0_test.zig` (dense ≤ 8, sparse ≤ 2);
      `tests/joint_test.zig` (wide-cap fixture ceilings 30 / 50 / 25).
      These pins already paid for themselves once — asking "how does
      reduced do on the canaries?" is what surfaced the 26-rejection
      Δ-collapse thrash the survey means had hidden.
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
