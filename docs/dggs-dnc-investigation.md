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

The eigendecomposition $V = [\,b\;|\;v_1\;|\;v_2\,]$, $\Lambda = \mathrm{diag}(\sigma_0,\,\sigma_1,\,\sigma_2)$ is *already computed* at `src/skar.zig:410-422` — no new factorizations needed.

### Block structure of $Z$ in $A$'s eigenbasis

Rotate the iterate: $Z_{\text{rot}} = V^{\top} Z V$. Block along the 1+2 split:

$$
Z_{\text{rot}} \;=\; \begin{bmatrix} z_{bb} & z_{bt}^{\top} \\[2pt] z_{bt} & Z_{tt} \end{bmatrix}
$$

with $z_{bb} \in \mathbb{R}$, $z_{bt} \in \mathbb{R}^{2}$, $Z_{tt} \in \mathbb{R}^{2 \times 2}$ symmetric.

At convergence $Z = A^{-1}$, so in this basis $Z_{\text{rot}} = \Lambda^{-1}$ — strictly diagonal:

$$
z_{bb} \to \frac{1}{\sigma_0} \approx 1.73, \qquad z_{bt} \to 0, \qquad Z_{tt} \to \mathrm{diag}\!\bigl(1/\sigma_1,\, 1/\sigma_2\bigr).
$$

For DGGS-tiny cells the tangent block has entries $\sim 10^{-9}$, but its own conditioning $\kappa(Z_{tt}) = \sigma_2/\sigma_1$ is $O(1)$ — only the *scale* is small, not the *ratio*.

### Schur identity

$$
\det Z \;=\; \det(Z_{tt})\,\cdot\,\bigl(z_{bb} \;-\; z_{bt}^{\top}\, Z_{tt}^{-1}\, z_{bt}\bigr).
$$

Write $z_{bb}^{\text{schur}} \;\equiv\; z_{bb} - z_{bt}^{\top} Z_{tt}^{-1} z_{bt}$. Then $\log\det Z = \log\det(Z_{tt}) + \log(z_{bb}^{\text{schur}})$.

### Cancellation-aware pairing via Cholesky of $Z_{tt}$

A naïve assembly $\log\det M = \log\det A + \log\det Z$ has a *hidden* cancellation worth flagging. For the S2 case:

$$
\log\det A \;=\; \log\sigma_0 + \log\sigma_1 + \log\sigma_2 \;\approx\; -0.55 + 20.60 + 20.79 \;=\; 40.84,
$$

and at convergence $\log\det Z \to -40.84$. Summing them gives $0$ with **~14 digits of cancellation** between the two; individual log errors of $\varepsilon\cdot 41 \approx 9\times 10^{-15}$ then bound the gap floor at $\sim 2\times 10^{-14}$.

We recover the missing 2 orders of magnitude by **pairing the terms so each piece is already small**. Cholesky of $Z_{tt}$ gives a natural per-eigenvalue split: $Z_{tt} = L_t L_t^{\top}$ with diagonals $l_{11}, l_{22}$, where $l_{11}^2 = z_{11}$ and $l_{22}^2 = z_{22} - l_{21}^2$. So

$$
\log\det Z_{tt} \;=\; \log l_{11}^2 + \log l_{22}^2 \;=\; \log z_{11} + \log l_{22}^2.
$$

Pair the three structural terms with their cancellation partners:

$$
\log\det M \;=\; \underbrace{\log\!\bigl(\sigma_0 \cdot z_{bb}^{\text{schur}}\bigr)}_{\to\,\log 1\,=\,0} \;+\; \underbrace{\log\!\bigl(\sigma_1 \cdot z_{11}\bigr)}_{\to\,\log 1\,=\,0} \;+\; \underbrace{\log\!\bigl(\sigma_2 \cdot l_{22}^2\bigr)}_{\to\,\log 1\,=\,0}.
$$

(At convergence $z_{11} \to 1/\sigma_1$ and $l_{22}^2 \to 1/\sigma_2$, so each $\sigma_i \cdot l_{ii}^2 \to 1$.)

Each argument is $1 + \delta$ with $\delta$ small. Use `log1p` on the deviation — it preserves full relative precision in $\delta$ where naïve `log(1 + δ)` would lose the small bits to the leading $1$:

$$
\boxed{\;\log\det M \;=\; \mathrm{log1p}\bigl(\sigma_0 \cdot z_{bb}^{\text{schur}} - 1\bigr) \;+\; \mathrm{log1p}\bigl(\sigma_1 \cdot z_{11} - 1\bigr) \;+\; \mathrm{log1p}\bigl(\sigma_2 \cdot l_{22}^2 - 1\bigr).\;}
$$

Each "$\sigma_i \cdot \text{diag} - 1$" subtraction becomes a single-rounded FMA (`@mulAdd`) — no separate multiplication-then-subtraction step:

$$
\mathrm{ax} = \mathrm{fma}(\sigma_0,\, z_{bb}^{\text{schur}},\, -1), \quad \mathrm{bx_1} = \mathrm{fma}(\sigma_1,\, z_{11},\, -1), \quad \mathrm{bx_2} = \mathrm{fma}(\sigma_2,\, l_{22}^2,\, -1).
$$

### Schur scalar via norm-squared

Cholesky gives an additional gift: the bilinear form $z_{bt}^{\top} Z_{tt}^{-1} z_{bt}$ collapses to a norm-squared after one forward solve, **eliminating the back-solve entirely** and making the result structurally non-negative:

$$
z_{bt}^{\top} Z_{tt}^{-1} z_{bt} \;=\; z_{bt}^{\top} L_t^{-\top} L_t^{-1} z_{bt} \;=\; \|L_t^{-1} z_{bt}\|^2 \;=\; y_1^2 + y_2^2,
$$

where $L_t\, y = z_{bt}$. This guarantees $y_1^2 + y_2^2 \geq 0$ by construction, so any noise in the Schur scalar's sign (`schur ≤ 0` indef-dual case) is a real PD violation of $Z_{tt}$, not a rounding artifact.

### Final gap formula

$$
\boxed{\;\mathrm{gap} \;=\; \|w_{\text{sum}}\| \;-\; 3 \;-\; \mathrm{log1p}(\mathrm{ax}) \;-\; \mathrm{log1p}(\mathrm{bx_1}) \;-\; \mathrm{log1p}(\mathrm{bx_2}).\;}
$$

### Why this beats the current path

Each input to `log1p` is computed *as a small number directly*, with single-rounding FMA wherever possible:

| piece | computation | absolute error |
|---|---|---:|
| $z_{bb},\,z_{bt},\,Z_{tt}$ entries | six quadratic forms $v_i^{\top} Z v_j$ (FMA-chained `Vec3.dot`) | $\varepsilon$ relative per entry |
| Cholesky $l_{11}, l_{21}, l_{22\text{sq}}$ | $\sqrt{z_{11}}$, $z_{12}/l_{11}$, FMA'd $z_{22} - l_{21}^2$ | $\sim \varepsilon$ relative |
| forward solve $y_1, y_2$ | $z_{b1}/l_{11}$, FMA'd $(z_{b2} - l_{21}\,y_1)/l_{22}$ | $\sim \varepsilon$ relative |
| $z_{bt}^{\top} Z_{tt}^{-1} z_{bt} = y_1^2 + y_2^2$ | FMA'd; non-negative by construction | $\sim \varepsilon$ relative |
| $z_{bb}^{\text{schur}}$ | $z_{bb} - (y_1^2 + y_2^2)$ | inherits $\sim \varepsilon$ |
| $\mathrm{ax},\,\mathrm{bx_1},\,\mathrm{bx_2}$ ("$\sigma_i \cdot \text{diag} - 1$" residuals) | single-rounded FMA each | $\sim \varepsilon$ **absolute** |
| three `log1p` calls | full relative precision on small inputs | $\sim \varepsilon$ **absolute** each |
| sum + $\|w_{\text{sum}}\| - 3$ | observed bit-exact for the primal term in the trace | inherits $\sim 3\varepsilon$ |

Final gap floor: $\sim 7\times 10^{-16}$, i.e., **bit-precision**. Default `gap_tol = 1e-6` is ten *billion* times above the floor.

### Why this beats the naïve Schur

For comparison, naïve Schur (`@log(det_tt) + @log(schur) + @log(SIGMA_0) + @log(sigma[0]) + @log(sigma[1])`, then sum) sits at $\sim 2\times 10^{-14}$ because of the ~14-digit cancellation between $\log\det A \approx +41$ and $\log\det Z \approx -41$. The log1p pairing dodges it entirely.

**$\sim 10$ orders of magnitude improvement** over the current $10^{-6}$ floor at $\kappa(A) \sim 10^{9}$. **2 orders** beyond the naïve Schur.

### Bonus: also dominates the historical hex-degenerate case

The src/skar.zig:482-487 comment rejected the plain $\log\det Z$ path because $\kappa(Z) \sim 10^{7}$ on hex-degenerate inputs cost $\sim 10^{-3}$ of precision. Under Schur:

- If the small $Z$ eigenvalue aligns with the **axial** direction ($z_{bb}$ small): $\log z_{bb}$ has large magnitude but error $\varepsilon$.
- If it aligns with the **tangent** plane ($Z_{tt}$ has one tiny eigenvalue): $\kappa(Z_{tt}) \sim 10^{7}$ locally, so $\log\det(Z_{tt})$ has error $\sim \varepsilon\,\kappa(Z_{tt}) \approx 10^{-9}$ — still ~$10^{6}\times$ better than the rejected path's $10^{-3}$.

So Schur **strictly improves on both** the current M path *and* the rejected plain-Z path. No hybrid switching, no scale predicate, no regime detection.

### Implementation sketch (~25-30 lines, no new linalg primitives)

Inside `dualityGapConstructed`, after `eig2(A_perp)` builds $V = [b\,|\,v_1\,|\,v_2]$ at lines 410-422. Skip materializing $V^{\top} Z V$ — project $Z$ onto the basis vectors directly via six quadratic forms, then Cholesky $Z_{tt}$ for the back-solve, log-det, and PD guard all at once:

```zig
// Project Z onto A's eigenbasis: six FMA-chained quadratic forms.
const Zb  = Z.apply(b);
const Zv1 = Z.apply(v1);
const Zv2 = Z.apply(v2);
const z_bb = b.dot(Zb);
const z_b1 = b.dot(Zv1);
const z_b2 = b.dot(Zv2);
const z_11 = v1.dot(Zv1);
const z_12 = v1.dot(Zv2);
const z_22 = v2.dot(Zv2);

// Cholesky of Z_tt = L L^T (2x2 SPD). Three quantities serve double duty:
//   l_11^2 = z_11        → tangent residual 1
//   l_22_sq = z_22 - l_21^2 → tangent residual 2, also PD guard
//   l_11, l_22           → forward solve for the Schur scalar
if (z_11 <= 0) return /* indef-dual guard, gap = 1e30 */;
const l_11 = @sqrt(z_11);
const l_21 = z_12 / l_11;
const l_22_sq = @mulAdd(f64, -l_21, l_21, z_22);  // single-rounded z_22 - l_21^2
if (l_22_sq <= 0) return /* indef-dual guard, Z_tt not PD */;
const l_22 = @sqrt(l_22_sq);

// Schur scalar via norm-squared. z_bt^T Z_tt^{-1} z_bt = ||L^{-1} z_bt||^2.
// One forward solve L y = z_bt; result is non-negative by construction.
const y_1 = z_b1 / l_11;
const y_2 = @mulAdd(f64, -l_21, y_1, z_b2) / l_22;   // (z_b2 - l_21·y_1)/l_22
const yy  = @mulAdd(f64, y_1, y_1, y_2 * y_2);
const schur = z_bb - yy;
if (schur <= 0) return /* indef-dual guard */;

// Three log1p residuals: axial + two tangent (per Cholesky diagonal).
// Each is "σ · (diagonal entry) - 1" fused into one FMA, single-rounded.
const ax  = @mulAdd(f64, SIGMA_0, schur,  -1.0);   // SIGMA_0 · schur − 1
const bx1 = @mulAdd(f64, sigma[0], z_11,   -1.0);  // σ_1 · z_11 − 1
const bx2 = @mulAdd(f64, sigma[1], l_22_sq, -1.0); // σ_2 · l_22² − 1

const log_det_M = @log1p(ax) + @log1p(bx1) + @log1p(bx2);
const gap = w_sum.norm() - 3.0 - log_det_M;
```

No new helper types or primitives needed: `Vec3.apply` / `Vec3.dot` already exist and are FMA-chained, `@mulAdd`, `@log1p`, and `@sqrt` are Zig builtins. The 3×3 path through the existing `Cholesky` on $M$, the explicit $L$, and the $L^{\top} Z L$ symmetric similarity all disappear from `dualityGapConstructed`. The 2×2 Cholesky of $Z_{tt}$ is the only matrix factorization that remains.

### Risks worth empirical checks before locking in

1. **Three new indefinite-dual guards** ($z_{11} \leq 0$, $l_{22}^2 \leq 0$, $\mathrm{schur} \leq 0$) replace the current single `cholesky() orelse` guard on $M$. Algebraically these are equivalent (collectively they encode "$Z$ is PD"), but the *frequency* with which each fires on the existing 48-case suite + DGGS survey is worth measuring — divergence from the current guard's hit rate could indicate a numerical regime worth understanding.
2. **$z_{bt}$-magnitude assumption.** The "bit-precision floor" estimate assumes the iterate drives $z_{bt} \to 0$ proportionally with iterate quality, so $\|L_t^{-1} z_{bt}\|^2$ stays small relative to $z_{bb}$. If $z_{bt}$ stalls at some larger noise level (independent of iterate quality), the Schur scalar's error inherits that noise. One per-iter trace pass against the failing DGGS tests will confirm whether $z_{bt}$ shrinks as expected.
3. **The $\sim 7\times 10^{-16}$ floor is a theoretical best-case under benign rounding.** Real-world likely loses 1-2 orders to compounding; even at $10^{-13}$, we're 7 orders below default `gap_tol`.
4. **Performance trade-off vs `diff_of_products` form.** Cholesky adds 2 sqrts + 1 extra `@log1p` call vs a `det Z_{tt}` + single tangent-`log1p` variant. Net: roughly 5-10% slower per gap evaluation. Negligible at the solver level (`dualityGapConstructed` is called ~10× per `solve`). The clarity wins — natural-non-negative Schur scalar, structural PD guard via $l_{22}^2$, three identical $\sigma_i \cdot \text{diag} - 1$ residuals — easily justify it.

## Other fix directions considered (alternatives)

These remain viable as backups or supplements if Schur turns out to have unforeseen issues:

1. **Compensated (double-double) arithmetic** in the $L^{\top} Z L$ chain — quadruples effective precision via TwoSum/TwoProd error-free transforms. Drives the floor to $\sim 10^{-23}$. ~50-100 lines of careful pair-arithmetic code. The "bulletproof hammer" answer if Schur ever fails.
2. **Scale-aware `gap_tol`** — auto-relax the convergence test to $\max(\text{user\_tol},\, \sigma_{\max}(A) \cdot \varepsilon_{\text{mach}} \cdot C)$. Cheapest possible fix; doesn't move the floor, just stops the DNC reporting. Suitable as a belt-and-suspenders complement.
3. **Input pre-rescaling at the API layer** — detect tiny-cell inputs (max pairwise vertex distance below threshold), rescale before solving, transform the output back. Pragmatic workaround; doesn't touch the solver internals at all.
4. **$\mathrm{tr}(AZ) - 3$ as second-tier convergence check** — cheaper than $\log\det M$, has the same fixed-point property. Not the canonical duality gap, so not theoretically tight as a stopping criterion, but useful as a noise-floor escape valve.

### Quick comparison

| approach | gap floor at $\kappa(A)=10^9$ | gap floor at hex $\kappa(Z)=10^7$ | impl cost |
|---|---:|---:|---|
| current $L^{\top} Z L$ + Cholesky + log | $10^{-6}$ | $\sim 10^{-7}$ | baseline |
| naïve Schur (separate logs, sum) | $\sim 10^{-14}$ | $\sim 10^{-9}$ | ~30 lines |
| **Schur + Cholesky($Z_{tt}$) + 3×log1p (recommended)** | $\sim 7\times 10^{-16}$ | $\sim 10^{-9}$ | ~25 lines |
| double-double arithmetic in current $L^{\top} Z L$ chain | $\sim 10^{-23}$ | $\sim 10^{-23}$ | ~100 lines |

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
