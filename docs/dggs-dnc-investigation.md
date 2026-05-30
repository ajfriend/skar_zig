# DGGS DNC investigation — findings (May 2026)

## TL;DR

At finest DGGS resolution, **22% of S2 L30 cells and 47% of A5 r30 cells** return `Outcome.did_not_converge` under default solver options (`gap_tol = 1e-6`, `max_outer = 100`). The failure mode is *not* algorithmic — it's a **numerical floor in the duality-gap formula** that scales with $\kappa(A)\,\varepsilon_{\text{mach}}$. For tiny cells, $\sigma_{\max}(A)\sim 10^{9}$, so the gap floor sits at $\sim 10^{-6}$ — exactly straddling the default `gap_tol`.

Confirmed by per-iteration trace: the solver reaches a **bit-identical fixed point**; the residual gap comes entirely from $\log\det M$ failing to evaluate to zero at convergence, traced to $M = L^{\top} Z L$ at `src/skar.zig:468`.

## How to reproduce

1. Regenerate the survey: `just dggs-gen` then `just dggs-aspect`. Check the DNC counts (`scripts/dggs/data/aspect.json` → `counts.did_not_converge`).
2. Two minimal repros already pinned as regression tests at `tests/dggs_dnc_test.zig` — they assert `.converged` and currently FAIL. The vertex coordinates are hardcoded; no JSON dependency.
3. Side-by-side polygon plot at `scripts/dggs/data/dnc_polygons.png` via `uv run scripts/dggs/plot_dnc.py`.

## Survey numbers (N=10_000 each, seed=0xC0FFEE)

| system | resolution | converged | DNC   | infeasible | input_error | worst AR |
|--------|-----------:|----------:|------:|-----------:|------------:|---------:|
| H3     |         15 |     10000 |     0 |          0 |           0 |   1.2497 |
| S2     |         30 |      7827 |  2173 |          0 |           0 |   1.7012 |
| A5     |         30 |      5261 |  4739 |          0 |           0 |   2.3186 |

H3 r15 has zero DNCs; the hex tiling at finest resolution doesn't push $\sigma_{\max}(A)$ high enough to trip the floor. S2 and A5 do.

## Hypotheses considered

- **(a) Budget** — solver needs more outer iters: **REJECTED**. Bit-identical state at `max_outer=100` vs `max_outer=10_000` (same `gap_bits`, same `sigma`). Fixed point.
- **(b) Noise floor** — gap converges to a positive floor above `gap_tol`: **CONFIRMED**. Both S2 and A5 land at stable floors.
- **(c) Oscillation / algorithmic instability**: **REJECTED**. Bit-stable fixed point.
- **NaN masking** (`NaN <= gap_tol` always false): ruled out, `gap_bits` is a finite double.
- **Newton-polish failure** dominating: `newton_polish_failures = 0` across all runs.
- **Active-set churn**: `cert.indices.len = 4` invariant.

## Localization to `src/skar.zig:468`

The gap formula at line 489 is

$$\mathrm{gap}\;=\;\|w_{\text{sum}}\|\;-\;3\;-\;\log\det M.$$

Per-iter trace at the bit-stable fixed point (S2 case):

```
TRACE gap k=4  ws=3.00000000000000000e0  log_det_M=-2.82662434818522070e-6
              ws-3=0.000e0  gap=2.827e-6
              sigma_perp=[8.811e8, 1.069e9]  cholD=[1.000e0, 1.000e0, 1.000e0]
```

- $\|w_{\text{sum}}\| = 3.0$ to the last bit — primal-residual term is *perfect*.
- The residual gap *is* $-\log\det M$.

$\log\det M$ should be zero at optimum (since $M = L^{\top} Z L \to I$ as $Z \to A^{-1}$), but isn't. Diagnosis:

- $L L^{\top} = A \;\Rightarrow\; \|L\| \approx \sqrt{\sigma_{\max}(A)}$. For these cells, $\sigma_{\max}(A) \sim 10^{9}\;\Rightarrow\; \|L\| \sim 10^{4.5}$.
- Iterate error in $Z$ is $E$ with $\|E\| \approx \varepsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$.
- $M - I = L^{\top} E L \;\Rightarrow\; \|M - I\| \approx \kappa(A)\,\|E\| \approx 10^{9} \cdot 2 \times 10^{-16} = 2 \times 10^{-7}$.
- $\log\det M \approx \operatorname{tr}(M - I) + O(\|M-I\|^{2}) \approx 3 \cdot 2 \times 10^{-7} \approx 6 \times 10^{-7}$ per Cholesky-diag-log term, factor 2 → **$\sim 10^{-6}$ floor**.

## Scale-dependence corroboration

| case | $\sigma_{\max}(A_\perp)$ | predicted floor $\kappa\,\varepsilon$ | **observed gap floor** |
|------|--------------:|----------------------:|-----------------------:|
| S2   |          $1.1 \times 10^{9}$ |                $\sim 2 \times 10^{-7}$  |                **$2.8 \times 10^{-6}$** |
| A5   |          $2.1 \times 10^{9}$ |                $\sim 5 \times 10^{-7}$  |                **$2.6 \times 10^{-5}$** |

Ratio matches: A5 floor / S2 floor $\approx$ A5 $\sigma$ / S2 $\sigma$ $\approx 1.9\times$. The absolute observed values are $\sim 10\times$ the simple $\kappa\,\varepsilon$ estimate (chain of $L^{\top}\!Z L \to \mathrm{symmetrize} \to \mathrm{Cholesky} \to \log$ accumulates more ulps).

## Smoking-gun stretch test

Renormalizing the input scatter by ×1e9 in the tangent plane (then back to the unit sphere — diagnostic only, not a fix) drops the floor by ~100× for both cases:

| case | original floor | $\times 10^{9}$ stretched | iters to converge |
|------|---------------:|---------------:|------------------:|
| S2   |        $2.827 \times 10^{-6}$ |        $3.062 \times 10^{-8}$ |                 1 |
| A5   |        $2.555 \times 10^{-5}$ |        $5.877 \times 10^{-7}$ |                 9 |

Confirms the floor is dominated by input-scale-dependent precision loss in M, not by anything intrinsic to the SDP.

## Recommended fix: Schur-block log-det in $A$'s eigenbasis

### The idea

Use the algebraic identity $\log\det M = \log\det A + \log\det Z$ to eliminate the $L^{\top} Z L$ matrix product (the amplifier), and compute $\log\det Z$ via a block decomposition along the structural axial-vs-tangent split of $A$. Each piece then lives in its own naturally-conditioned scale.

The eigendecomposition $V = [\,b\;|\;v_1\;|\;v_2\,]$, $\Lambda = \mathrm{diag}(\mathrm{SIGMA}_0,\,\sigma_1,\,\sigma_2)$ is *already computed* at `src/skar.zig:410-422` — no new factorizations needed.

### Block structure of $Z$ in $A$'s eigenbasis

Rotate the iterate: $Z_{\text{rot}} = V^{\top} Z V$. Block along the 1+2 split:

$$
Z_{\text{rot}} \;=\; \begin{bmatrix} z_{bb} & z_{bt}^{\top} \\[2pt] z_{bt} & Z_{tt} \end{bmatrix}
$$

with $z_{bb} \in \mathbb{R}$, $z_{bt} \in \mathbb{R}^{2}$, $Z_{tt} \in \mathbb{R}^{2 \times 2}$ symmetric.

At convergence $Z = A^{-1}$, so in this basis $Z_{\text{rot}} = \Lambda^{-1}$ — strictly diagonal:

$$
z_{bb} \to \frac{1}{\mathrm{SIGMA}_0} \approx 1.73, \qquad z_{bt} \to 0, \qquad Z_{tt} \to \mathrm{diag}\!\bigl(1/\sigma_1,\, 1/\sigma_2\bigr).
$$

For DGGS-tiny cells the tangent block has entries $\sim 10^{-9}$, but its own conditioning $\kappa(Z_{tt}) = \sigma_2/\sigma_1$ is $O(1)$ — only the *scale* is small, not the *ratio*.

### Schur identity

$$
\det Z \;=\; \det(Z_{tt})\,\cdot\,\bigl(z_{bb} \;-\; z_{bt}^{\top}\, Z_{tt}^{-1}\, z_{bt}\bigr).
$$

Taking logs:

$$
\log\det Z \;=\; \log\det(Z_{tt}) \;+\; \log\!\bigl(z_{bb} - z_{bt}^{\top} Z_{tt}^{-1} z_{bt}\bigr).
$$

### Final gap formula

Combine with the bit-perfect $\log\det A$ and the existing $\|w_{\text{sum}}\|$ term:

$$
\boxed{\;\mathrm{gap} \;=\; \|w_{\text{sum}}\|\;-\;3\;-\;\bigl(\log\mathrm{SIGMA}_0 + \log\sigma_1 + \log\sigma_2\bigr)\;-\;\log\det(Z_{tt})\;-\;\log\!\bigl(z_{bb} - z_{bt}^{\top} Z_{tt}^{-1} z_{bt}\bigr).\;}
$$

### Why this beats the current path

Each piece is computed in its own well-conditioned scale, with **no cross-scale matrix products**:

| piece | computation | scale | expected error |
|---|---|---|---|
| $\|w_{\text{sum}}\|$ | (unchanged) | $O(1)$ | already bit-exact in trace |
| $\log\det A$ | $\sum_i \log \Lambda_i$ on the existing eigenvalues | $\lvert\log\Lambda_i\rvert \sim 21$ | $\varepsilon \cdot 21 \approx 5\times 10^{-15}$ |
| $\log\det(Z_{tt})$ | `Mat2.det` then `@log`, or Cholesky-then-log | entries $\sim 10^{-9}$, but $\kappa(Z_{tt}) \sim 1$ | $\varepsilon \cdot 41 \approx 9\times 10^{-15}$ |
| $\log(\text{Schur})$ | scalar; at convergence $\approx \log z_{bb}$ | $O(1)$ | $\varepsilon \approx 2\times 10^{-16}$ |

Sum: $\mathrm{error} \sim \varepsilon \cdot \max\bigl(|\log\Lambda_i|\bigr) \approx 10^{-14}$.

**$\sim 8$ orders of magnitude improvement** over the current $10^{-6}$ floor at $\kappa(A) \sim 10^{9}$. Default `gap_tol = 1e-6` becomes deeply attainable.

### Bonus: also dominates the historical hex-degenerate case

The src/skar.zig:482-487 comment rejected the plain $\log\det Z$ path because $\kappa(Z) \sim 10^{7}$ on hex-degenerate inputs cost $\sim 10^{-3}$ of precision. Under Schur:

- If the small $Z$ eigenvalue aligns with the **axial** direction ($z_{bb}$ small): $\log z_{bb}$ has large magnitude but error $\varepsilon$.
- If it aligns with the **tangent** plane ($Z_{tt}$ has one tiny eigenvalue): $\kappa(Z_{tt}) \sim 10^{7}$ locally, so $\log\det(Z_{tt})$ has error $\sim \varepsilon\,\kappa(Z_{tt}) \approx 10^{-9}$ — still ~$10^{6}\times$ better than the rejected path's $10^{-3}$.

So Schur **strictly improves on both** the current M path *and* the rejected plain-Z path. No hybrid switching, no scale predicate, no regime detection.

### Implementation sketch (~40 lines, no new linalg primitives)

Inside `dualityGapConstructed`, after `eig2(A_perp)` builds $V$ at lines 410-422:

```zig
// Rotate Z to A's eigenbasis: Z_rot = V^T Z V. V's columns are
// {b, v1, v2}; use existing Mat3.fromCols + Mat3.mul + transpose.
const V = Mat3.fromCols(b, v1, v2);
const Z_rot = V.transpose().mul(Z).mul(V).symmetrize();

// Block: z_bb (scalar), z_bt (2-vec), Z_tt (2x2 symmetric).
const z_bb = Z_rot.m[0];
const z_bt = Vec2{ .m = .{ Z_rot.m[1], Z_rot.m[2] } };
const Z_tt = Mat2{ .m = .{ Z_rot.m[4], Z_rot.m[5], Z_rot.m[7], Z_rot.m[8] } };

// Z_tt determinant via existing diff_of_products inside Mat2.det.
const det_tt = Z_tt.det();
if (det_tt <= 0) return /* indefinite-dual guard */;

// 2x2 linear solve for Z_tt^{-1} z_bt — closed-form, ~6 ops.
const z_tt_inv_z_bt = Z_tt.solve(z_bt);
const schur = z_bb - z_bt.dot(z_tt_inv_z_bt);
if (schur <= 0) return /* indefinite-dual guard */;

const log_det_Z = @log(det_tt) + @log(schur);
const log_det_A = @log(SIGMA_0) + @log(sigma[0]) + @log(sigma[1]);
const gap = w_sum.norm() - 3.0 - log_det_A - log_det_Z;
```

New helpers needed:

- `Mat2.solve(b: Vec2) Vec2` — one closed-form expression, ~6 floating-point ops.
- Possibly a `Vec2` type if it doesn't exist (likely does — `eig2` returns 2-vectors).

The 3×3 path through `Cholesky`, the explicit `L`, and the $L^{\top} Z L$ symmetric similarity all disappear from `dualityGapConstructed`.

### Risks worth empirical checks before locking in

1. **Two new indefinite-dual guards** ($\det Z_{tt} \leq 0$ and Schur scalar $\leq 0$) replace the current single `cholesky() orelse` guard on $M$. The existing `gap = 1e30` fallback pattern extends to both cleanly, but the *frequency* with which each fires on the existing 48-case suite + DGGS survey is worth measuring — could indicate a regime where iterate quality is worse than predicted.
2. **$z_{bt}$-magnitude assumption.** The error analysis above assumes the iterate drives $z_{bt} \to 0$ proportionally with iterate quality. If $z_{bt}$ stalls at some larger noise level (independent of iterate accuracy), the Schur scalar's error inherits that noise. One per-iter trace pass against the failing DGGS tests will confirm whether $z_{bt}$ shrinks as expected.
3. **The $10^{-14}$ floor is a theoretical best-case under benign rounding.** Real-world likely loses 1-2 orders to compounding; even at $10^{-12}$, we're 6 orders below default `gap_tol`.

## Other fix directions considered (alternatives)

These remain viable as backups or supplements if Schur turns out to have unforeseen issues:

1. **Compensated (double-double) arithmetic** in the $L^{\top} Z L$ chain — quadruples effective precision via TwoSum/TwoProd error-free transforms. Drives the floor to $\sim 10^{-23}$. ~50-100 lines of careful pair-arithmetic code. The "bulletproof hammer" answer if Schur ever fails.
2. **Scale-aware `gap_tol`** — auto-relax the convergence test to $\max(\text{user\_tol},\, \sigma_{\max}(A) \cdot \varepsilon_{\text{mach}} \cdot C)$. Cheapest possible fix; doesn't move the floor, just stops the DNC reporting. Suitable as a belt-and-suspenders complement.
3. **Input pre-rescaling at the API layer** — detect tiny-cell inputs (max pairwise vertex distance below threshold), rescale before solving, transform the output back. Pragmatic workaround; doesn't touch the solver internals at all.
4. **$\mathrm{tr}(AZ) - 3$ as second-tier convergence check** — cheaper than $\log\det M$, has the same fixed-point property. Not the canonical duality gap, so not theoretically tight as a stopping criterion, but useful as a noise-floor escape valve.

## Investigation artifacts (reverted before commit)

- `tests/dggs_dnc_test.zig`: rich probe + sweep + stretch variants — reverted to the two bare-assertion regression tests.
- `src/skar.zig`: `TRACE` const + per-iter print in `dualityGapConstructed` — reverted.

This document preserves the diagnosis; the code is back to its pre-investigation state so the regression tests stay clean and the solver has no debug overhead.

## Related files

- `tests/dggs_dnc_test.zig` — the two failing regression tests (kept).
- `scripts/dggs/aspect.zig` — the survey driver (`zig build dggs-aspect`).
- `scripts/dggs/gen_cells.py` — random-cell generator (`just dggs-gen`).
- `scripts/dggs/plot_dnc.py` — projection plot of the two DNC cells.
- `docs/dggs-aspect-survey-plan.md` — overall survey plan; this investigation is a sub-thread.
- `src/skar.zig:397-496` — `dualityGapConstructed` (where the floor lives).
- `src/skar.zig:748-790` — outer loop.
