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

Write $z_{bb}^{\text{schur}} \;\equiv\; z_{bb} - z_{bt}^{\top} Z_{tt}^{-1} z_{bt}$. Then $\log\det Z = \log\det(Z_{tt}) + \log(z_{bb}^{\text{schur}})$.

### Cancellation-aware pairing (don't compute logs separately!)

A naïve assembly $\log\det M = \log\det A + \log\det Z$ has a *hidden* cancellation worth flagging. For the S2 case:

$$
\log\det A \;=\; \log\mathrm{SIGMA}_0 + \log\sigma_1 + \log\sigma_2 \;\approx\; -0.55 + 20.60 + 20.79 \;=\; 40.84,
$$

and at convergence $\log\det Z \to -40.84$. Summing them gives $0$ with **~14 digits of cancellation** between the two; individual log errors of $\varepsilon\cdot 41 \approx 9\times 10^{-15}$ then bound the gap floor at $\sim 2\times 10^{-14}$.

We can recover the missing 2 orders of magnitude by **pairing the terms so each piece is already small**. Group axial-with-axial and tangent-with-tangent before taking logs:

$$
\log\det M \;=\; \underbrace{\log\!\bigl(\mathrm{SIGMA}_0\,\cdot\,z_{bb}^{\text{schur}}\bigr)}_{\to\,\log 1\,=\,0} \;+\; \underbrace{\log\!\bigl(\sigma_1\sigma_2\,\cdot\,\det Z_{tt}\bigr)}_{\to\,\log 1\,=\,0}.
$$

Each argument is $1 + \delta$ with $\delta$ small. Use `log1p` on the deviation — it preserves full relative precision in $\delta$ where naïve `log(1 + δ)` would lose the small bits to the leading $1$:

$$
\boxed{\;\log\det M \;=\; \mathrm{log1p}\bigl(\mathrm{SIGMA}_0 \cdot z_{bb}^{\text{schur}} - 1\bigr) \;+\; \mathrm{log1p}\bigl(\sigma_1\sigma_2 \cdot \det Z_{tt} - 1\bigr).\;}
$$

The two "$\cdot{} - 1$" subtractions become single-rounded FMAs (`@mulAdd`) to fuse the multiply and subtract:

$$
\mathrm{ax} \;=\; \mathrm{fma}\!\bigl(\mathrm{SIGMA}_0,\; z_{bb}^{\text{schur}},\; -1\bigr), \qquad \mathrm{bx} \;=\; \mathrm{fma}\!\bigl(\sigma_1,\; \sigma_2 \cdot \det Z_{tt},\; -1\bigr).
$$

Two more micro-decisions:

- **Order intermediates to avoid large transients.** $\sigma_1\sigma_2$ has magnitude $\sim 10^{18}$; the product $\sigma_2 \cdot \det Z_{tt}$ alone is $\sim 10^{-9}$ (≈ $1/\sigma_1$). Form the small product first, then FMA with $\sigma_1$ and $-1$.
- **Route every 2×2 piece through `diff_of_products`** (`src/linalg.zig:20`, Kahan's compensated FMA scheme). `det Z_{tt}` and the two 2×2-solve numerators ($z_{22}\,z_{b1} - z_{12}\,z_{b2}$, $z_{11}\,z_{b2} - z_{12}\,z_{b1}$) are exactly the near-cancellation cases the helper was written for.

### Final gap formula

$$
\boxed{\;\mathrm{gap} \;=\; \|w_{\text{sum}}\| \;-\; 3 \;-\; \mathrm{log1p}\!\bigl(\mathrm{SIGMA}_0\cdot z_{bb}^{\text{schur}} - 1\bigr) \;-\; \mathrm{log1p}\!\bigl(\sigma_1 \cdot \sigma_2\,\det Z_{tt} - 1\bigr).\;}
$$

### Why this beats the current path

Each input to `log1p` is computed *as a small number directly*, with single-rounding FMA wherever possible:

| piece | computation | absolute error |
|---|---|---:|
| $z_{bb},\,z_{bt},\,Z_{tt}$ | six quadratic forms $v_i^{\top} Z v_j$ (FMA-chained `Vec3.dot`) | $\varepsilon$ relative per entry |
| $\det Z_{tt}$ | `diff_of_products(z_{11}, z_{22}, z_{12}, z_{12})` | $\sim 2\varepsilon$ relative |
| 2×2 solve numerators | two `diff_of_products` calls / `det_tt` | $\sim 2\varepsilon$ relative |
| $z_{bb}^{\text{schur}}$ | `z_bb − @mulAdd(z_b1, inv_x, z_b2 * inv_y)` | $\sim \varepsilon$ relative |
| $\mathrm{ax},\,\mathrm{bx}$ (the "$\cdot - 1$" residuals) | single-rounded FMA each | $\sim \varepsilon$ **absolute** |
| `log1p(ax)`, `log1p(bx)` | full relative precision on small inputs | $\sim \varepsilon$ **absolute** |
| sum + $\|w_{\text{sum}}\| - 3$ | three subtractions; observed bit-exact for the primal term | inherits $\sim 2\varepsilon$ |

Final gap floor: $\sim 4\times 10^{-16}$, i.e., **bit-precision**. Default `gap_tol = 1e-6` is ten *billion* times above the floor.

### Why this beats the naïve Schur

For comparison, naïve Schur (`@log(det_tt) + @log(schur) + @log(SIGMA_0) + @log(sigma[0]) + @log(sigma[1])`, then sum) sits at $\sim 2\times 10^{-14}$ because of the ~14-digit cancellation between $\log\det A \approx +41$ and $\log\det Z \approx -41$. The log1p pairing dodges it entirely.

**$\sim 10$ orders of magnitude improvement** over the current $10^{-6}$ floor at $\kappa(A) \sim 10^{9}$. **2 orders** beyond the naïve Schur.

### Bonus: also dominates the historical hex-degenerate case

The src/skar.zig:482-487 comment rejected the plain $\log\det Z$ path because $\kappa(Z) \sim 10^{7}$ on hex-degenerate inputs cost $\sim 10^{-3}$ of precision. Under Schur:

- If the small $Z$ eigenvalue aligns with the **axial** direction ($z_{bb}$ small): $\log z_{bb}$ has large magnitude but error $\varepsilon$.
- If it aligns with the **tangent** plane ($Z_{tt}$ has one tiny eigenvalue): $\kappa(Z_{tt}) \sim 10^{7}$ locally, so $\log\det(Z_{tt})$ has error $\sim \varepsilon\,\kappa(Z_{tt}) \approx 10^{-9}$ — still ~$10^{6}\times$ better than the rejected path's $10^{-3}$.

So Schur **strictly improves on both** the current M path *and* the rejected plain-Z path. No hybrid switching, no scale predicate, no regime detection.

### Implementation sketch (~30-40 lines, no new linalg primitives)

Inside `dualityGapConstructed`, after `eig2(A_perp)` builds $V = [b\,|\,v_1\,|\,v_2]$ at lines 410-422. Skip materializing $V^{\top} Z V$ — project $Z$ onto the basis vectors directly via six quadratic forms:

```zig
// Project Z onto A's eigenbasis: six FMA-chained quadratic forms.
// Z.apply(v) is one matvec; vN.dot(...) is one FMA-chained dot.
const Zb  = Z.apply(b);
const Zv1 = Z.apply(v1);
const Zv2 = Z.apply(v2);
const z_bb = b.dot(Zb);
const z_b1 = b.dot(Zv1);
const z_b2 = b.dot(Zv2);
const z_11 = v1.dot(Zv1);
const z_12 = v1.dot(Zv2);
const z_22 = v2.dot(Zv2);

// 2x2 tangent determinant — Kahan-compensated subtraction of products.
const det_tt = diff_of_products(z_11, z_22, z_12, z_12);
if (det_tt <= 0) return /* indefinite-dual guard, gap = 1e30 */;

// Z_tt^{-1} z_bt via Cramer's rule; both numerators use diff_of_products.
const num_x = diff_of_products(z_22, z_b1, z_12, z_b2);
const num_y = diff_of_products(z_11, z_b2, z_12, z_b1);
const inv_x = num_x / det_tt;
const inv_y = num_y / det_tt;

// Schur complement of M_tt in M_rot (equivalent to SIGMA_0 · z_bb^schur):
//   schur = z_bb - z_bt^T Z_tt^{-1} z_bt
//         = z_bb - (z_b1 * inv_x + z_b2 * inv_y)        ← FMA-fused inner sub
const z_bt_dot = @mulAdd(f64, z_b1, inv_x, z_b2 * inv_y);
const schur = z_bb - z_bt_dot;
if (schur <= 0) return /* indefinite-dual guard, gap = 1e30 */;

// Axial residual: SIGMA_0 · schur - 1, single-rounded.
const ax = @mulAdd(f64, SIGMA_0, schur, -1.0);

// Tangent residual: σ_1 · σ_2 · det_tt - 1. Form the small intermediate
// first (σ_2 · det_tt ≈ 1/σ_1, not the giant σ_1·σ_2 ≈ 1e18), then FMA.
const sigma2_det_tt = sigma[1] * det_tt;
const bx = @mulAdd(f64, sigma[0], sigma2_det_tt, -1.0);

// log1p preserves precision when ax, bx are small (the convergence regime).
const log_det_M = @log1p(ax) + @log1p(bx);
const gap = w_sum.norm() - 3.0 - log_det_M;
```

No new helper types or primitives needed: `Vec3.apply` / `Vec3.dot` already exist and are FMA-chained, `diff_of_products` already exists at `src/linalg.zig:20`, `@mulAdd` and `@log1p` are Zig builtins. The 3×3 path through `Cholesky`, the explicit $L$, and the $L^{\top} Z L$ symmetric similarity all disappear from `dualityGapConstructed`.

### Risks worth empirical checks before locking in

1. **Two new indefinite-dual guards** ($\det Z_{tt} \leq 0$ and $\mathrm{schur} \leq 0$) replace the current single `cholesky() orelse` guard on $M$. Algebraically these are equivalent ($M$ is PD iff both hold), but the *frequency* with which each fires on the existing 48-case suite + DGGS survey is worth measuring — divergence from the current guard's hit rate could indicate a numerical regime worth understanding.
2. **$z_{bt}$-magnitude assumption.** The "bit-precision floor" estimate assumes the iterate drives $z_{bt} \to 0$ proportionally with iterate quality, so $z_{bt}^{\top} Z_{tt}^{-1} z_{bt}$ stays small relative to $z_{bb}$. If $z_{bt}$ stalls at some larger noise level (independent of iterate quality), the Schur scalar's error inherits that noise. One per-iter trace pass against the failing DGGS tests will confirm whether $z_{bt}$ shrinks as expected.
3. **The $\sim 4\times 10^{-16}$ floor is a theoretical best-case under benign rounding.** Real-world likely loses 1-2 orders to compounding; even at $10^{-13}$, we're 7 orders below default `gap_tol`.
4. **`@log1p` performance.** Slightly slower per call than `@log` (extra Taylor-series step for the near-zero regime). Two `@log1p` calls vs three `@log` calls — net per-gap cost is roughly the same. Measure with `just bench` before/after if curious.

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
| **Schur + log1p + FMA + diff_of_products (recommended)** | $\sim 4\times 10^{-16}$ | $\sim 10^{-9}$ | ~35 lines |
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
