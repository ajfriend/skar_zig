# Roadmap: remaining algorithmic improvements (speed / convergence / stability)

**Status:** proposal, not started. Written 2026-07-10, from a survey of the
post-0.6.0 state (trust default, drop guard landed, exact envelope Hessian,
eager cert). Supersedes nothing; items here are additive to the record in
docs/trust-solver.md and docs/away-step-fw.md.

## Where the remaining costs actually are

The outer-method problem is essentially solved: the trust path has 0 DNC on
the wide-cap grid, ≤ 5 iterations on geographies, and success-parity with the
alternating path on every DGGS system (probe27 fair metric). What's left
splits into three residuals, and only the first is an MVEE-subproblem
question:

1. **Wall-clock on the mid-size dense band** (np400 ~2.3×, ha_14 ~2.7× vs
   alternating) — pure oracle cost. MVEE work pays here.
2. **The 1e-6 certificate floor on finest-res cells** (~1400–4200 survey DNCs
   at strict tol) — lives in the *certificate arithmetic*, not the MVEE. A
   better inner solver does nothing for it.
3. **The small-cell hot path** — already at parity (0.89–1.02×). Protect
   (CANARY pins, ex-bench per-case), don't optimize.

In the h framing: the outer methods now buy (value, gradient) pairs only when
needed; what's left is making each pair cheaper (items 1–3, 5) and making the
certificate's noise floor lower than the tolerances we certify (item 4).

## Ranked items

### 1. Range-space Newton polish — best single item (speed + latent stability)

`newtonPolish` (src/newton.zig) solves a dense bordered (k+1)×(k+1) LU per
Newton iteration, O(k³), up to 20 iterations, inside every oracle evaluation.
Its Hessian H_ij = (qᵢᵀW⁻¹qⱼ)² is exactly the C∘C matrix whose rank ≤ 6
structure the exact envelope Hessian already exploits (src/trust.zig, evalH
part 2): Schur square of a rank-3 Gram.

- **Speed:** on dense near-circular supports (ha_*, geographies — exactly the
  residual band, k up to ~60) the same 6-dim range-space substitution turns
  each Newton iteration from O(k³) into O(k·36) + a 7×7 bordered solve. The
  biggest asymptotic win available anywhere in the code.
- **Stability:** for k > 6 the current KKT system is *exactly singular* in
  exact arithmetic — the LU succeeds only because roundoff regularizes it,
  and it can amplify the null-space component of the gradient (active-set
  noise) into the step: the same 1/ε amplification measured and fixed for the
  Hessian correction (~1e5 blowup on a 60-point ring, see
  config.trust.EXACT_HESSIAN). The pseudo-inverse step (flat directions of
  the degenerate optimal face projected out) is the principled direction.
  This is a latent cousin of the drop-step hazard: quiet today, but the shape
  that produces "corrupted state at a converged design."

Touches the shared polish ⇒ full gauntlet (slow suite with CANARY discipline,
ex-bench per-case, a5_res0, geographies, rotations). But it's derived
structure, not a heuristic — the enabling math is written and validated in
trust.zig.

### 2. Harman–Pronzato elimination inside `evalH`

Within one oracle evaluation b is fixed, so the fixed-b MVEE admits the
standard elimination test: points whose gradient gᵢ falls below a bound
derived from the current duality gap provably cannot be in the support and
can be dropped for the rest of the evaluation. Every FW burst currently pays
an O(n) scan over the full working set for 64–320 iterations; on geographies
(hundreds of hull vertices, support ~5–10) and elongated dense inputs,
elimination collapses that scan after the first burst.

- Conservative (provable exclusion, optimum unchanged), literature-standard
  for D-optimal design (Harman & Pronzato 2007).
- Per-evaluation only: the support depends on b, so eliminated points return
  as candidates at the next axis.
- Caveat: helps least exactly where ha_* hurts (near-circular supports where
  most hull points genuinely touch the ellipse). Measure on geographies and
  np400 first; don't judge it on ha_*.

### 3. Harvest the deferred dedup/fusion 20–30%

Already scoped in docs/trust-solver.md ("the dedup half of the fusion work"):
`mveeFw`, `newtonPolish`, and `certifyAt` each rebuild and refactor the design
state S. The eager cert made this irrelevant to the DGGS path, but it applies
verbatim to every full-oracle evaluation — the np400/ha band that is the
remaining loss. Known size, known cost (threading polish internals across the
newton/trust boundary), zero algorithmic risk.

### 4. The certificate floor: extended-precision probe (the convergence lever)

The only convergence lever left is the f64 gap floor, not the solver. The
tell is already in the record (CLAUDE.md, dggs_dnc_test): WHICH finest-res
cells sit above vs below 1e-6 is path-dependent at noise level — so a large
fraction of the floor population sits within a small factor of the tolerance.

`dualityGapConstructed` runs on k ≤ ~10 active points; evaluating its
noise-critical pieces (λᵢ = 3wᵢ/(b·xᵢ), the LᵀZL triple product, `logDet`)
in compensated/double-double arithmetic costs essentially nothing at that k.

The experiment is cheap and cleanly falsifiable: re-certify a batch of
floor-DNC cells with a high-precision gap **at the same iterate**.

- Gaps drop below 1e-6 ⇒ thousands of survey cells convert; the "honest DNC"
  boundary moves; ship the compensated cert.
- Gaps don't drop ⇒ the floor is iterate quantization (w, b themselves at
  f64) and the question closes permanently.

Either outcome is worth having. Note the per-path DNC facts pinned in
tests/dggs_dnc_test.zig will shift if this ships — that's the expected
signature, flag-and-reconcile per CANARY policy.

### 5. SIMD the gradient scan

The `L.solve(qᵢ)` + dot loop in `mveeFw` is the hot kernel of every burst.
Embarrassingly parallel across points; an SoA layout with `@Vector` batching
should give 2–4× on large n. Implementation-level rather than algorithmic,
multiplies with items 2–3, touches no math. Guard: bit-drift on small cells
is acceptable per repo policy (precision drift ≠ regression), but the
small-cell CANARY iteration counts must hold.

## Smaller / conditional

- **Smarter-than-size sparse-seed gate** (the FUTURE note on
  `algo.SEED_SPARSE_MIN_POINTS`): a cheap redundancy proxy (e.g. hull-vertex
  count vs seed-spread ratio) would extend the sparse-seed win to small
  *irregular* polygons the size gate skips. Low risk, modest win.
- **Fully-corrective FW while the support is small** (grow-one-point, exact
  polish over the support each step): attractive on sparse-support inputs,
  but shares away-step FW's measured failure profile — degrades exactly on
  large near-circular supports, where its per-step polish is O(k³) (unless
  item 1 lands first, which changes that math). If tried at all: gate on
  support size, treat as an experiment.

## What not to retry (measured dead ends)

- Away-step FW as the oracle — slower on large near-circular supports
  (ha_05 56 → 261 µs; docs/away-step-fw.md "Stage 1 findings").
- Oracle bursts below 64 — worse robustness AND speed (New York DNC,
  np400 69 → 95 µs).
- Budget/inexact-oracle schemes — three reverts, all oracle-inconsistency
  (ρ → −7.95 on cap82); the (value, gradient) identity needs inner
  optimality.
- BFGS-style model state — deleted for cause (model corruption masquerading
  as convergence).
- Tikhonov on the rank-deficient k×k system — 1/ε null-space amplification.
- Touching `solveAlternating` — it's the bit-stable reference; everything
  above lands in the shared inner machinery or the trust path.

## Suggested order

Start with item 1 (range-space polish): the only item that improves speed on
the residual band *and* removes a latent numerical hazard. Item 4 is the
cheapest experiment with the largest potential convergence payoff and can run
in parallel as a probe. Items 2–3 follow if the np400/ha band still matters
after 1.
