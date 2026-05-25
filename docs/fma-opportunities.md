# Remaining FMA / cancellation-hygiene opportunities

The linalg pass (`c901dc7`, `5ef5f19`, `fed7637`) covered the
primitives in `src/linalg.zig`. This doc lists the remaining
hand-rolled arithmetic in `src/skar.zig` and `src/newton.zig` that
follows the same patterns — places where switching to `@mulAdd`
chains or `diff_of_products` should be a net win on precision (and
often a small perf win as a side effect).

We work through these **one at a time**. Each item gets its own
commit with a before/after bench comparison (3 runs to filter
system noise), and the `just test-slow` gate must stay green
(21 tests, 100% coverage).

The general principle stays the same as the linalg pass: if a test
shifts because the math is more accurate, **adapt the test**,
don't revert the math. See `memory/feedback_precision_drift.md` for
the standing user preference.

---

## A. Hot inner loops with FMA wins

These are the highest-payoff sites — tight loops where the cost
shows up directly in `ex-bench` and the precision improvement
propagates through many iterations.

### A1. `mveeFw` rank-1 accumulation (`src/skar.zig:207-212`)

Hand-rolled sym-rank-1 update inlined to skip mirror writes inside
the inner-inner loop:

```zig
S.m[0] += wi * qi.m[0] * qi.m[0];
S.m[1] += wi * qi.m[0] * qi.m[1];
S.m[2] += wi * qi.m[0] * qi.m[2];
S.m[4] += wi * qi.m[1] * qi.m[1];
S.m[5] += wi * qi.m[1] * qi.m[2];
S.m[8] += wi * qi.m[2] * qi.m[2];
```

**Fix:** precompute `wq_i = wi * qi.m[i]` for i=0,1,2 and use
`@mulAdd` for each `+=`. Same logic as `Mat3.addSymRank1` minus the
mirror writes (which are correctly deferred to a single pass at
lines 214-216 after the loop). Multiplication order `(wi·qi_r)·qi_c`
is preserved.

**Expected impact:** measurable. Loop runs ~N · max_iter times per
`mveeFw` call, and `mveeFw` is called every outer iteration.

### A2. LU elimination + tri-solve (`src/newton.zig:86, 107, 113`)

Three canonical `a -= b * c` sites in the bordered-KKT LU solver:

```zig
data[i*n + j] -= data[i*n + kk] * data[kk*n + j];   // line 86, elimination
b[i] -= data[i*n + j] * b[j];                        // line 107, forward
b[i] -= data[i*n + j] * b[j];                        // line 113, back
```

**Fix:** `@mulAdd(f64, -data[...], data[...], data[...])` etc. Same
shape we used in `Chol3.forwardSolve` / `backSolve`.

**Expected impact:** runs every Newton polish step. Smaller payoff
than A1 (LU is more rare than FW) but consistent with the rest of
the codebase.

---

## B. `diff_of_products` candidates

Quick drop-ins; cancellation-resistant `a*b − c*d` form.

### B3. `det_G` in `mveeFw` (`src/skar.zig:245`)

```zig
const det_G = g_max * g_min - g_cross * g_cross;
```

**Fix:** `diff_of_products(g_max, g_min, g_cross, g_cross)`. Inside
the 2-point step decision; can cancellate near the FW-vanilla
fallback boundary.

### B4. Coplanarity discriminant (`src/skar.zig:631`)

```zig
const det = c00 * c11 - c01 * c01;
```

**Fix:** `diff_of_products(c00, c11, c01, c01)`. This is THE
discriminant that drives the coplanarity gate (`4·det/trace²`).
Improving its precision could matter for inputs sitting near the
cutoff (the existing tests probe this band).

### B5. Convex hull `cross2` (`src/halfspace.zig:95`)

```zig
return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
```

**Fix:** precompute the four differences (or fold via `diff_of_products`
on the products). Cancellation possible when the three points are
nearly collinear — and that's exactly the case the predicate
exists to disambiguate.

---

## C. Minor / stylistic

Optional, marginal benefit. Worth touching only if we're already
in the function for another reason.

### C6. `rescaleP` sum-of-squares (`src/skar.zig:74`)

```zig
const sq = p[0] * p[0] + p[1] * p[1];
```

**Fix:** `@mulAdd(f64, p[1], p[1], p[0] * p[0])`. One fewer rounding.

### C7. `denom` in `recoverAPerp`-adjacent code (`src/skar.zig:292`)

```zig
const denom = @sqrt(tr + 2.0 * s_det);
```

**Fix:** `@sqrt(@mulAdd(f64, 2.0, s_det, tr))`. One fewer rounding.

---

## Suggested order

1. **B5** (`cross2`) — quick, low risk, isolated to halfspace.zig
2. **B3** (`det_G` in mveeFw) — quick, isolated change
3. **B4** (coplanarity discriminant) — quick, but worth running the
   coplanarity-band tests carefully since this drives the gate
4. **A1** (`mveeFw` rank-1) — larger change, biggest payoff
5. **A2** (LU solver) — consistent with Chol3 fix, easy to verify

C-tier swept up at the end if we want a clean grep'able result.

## Per-item workflow

For each item:

1. Capture 3 baseline bench runs into a scratch
2. Apply the change
3. `just test-slow` — must stay 21/21 + 100%
4. 3 after-bench runs; eyeball the diff
5. Commit with: what changed, expected vs observed precision win,
   measured perf delta (or "noise" if no signal)

If a test shifts: update the test, don't revert. AR values are the
ground-truth contract; sub-ulp drift in iter counts or exact gap
values is acceptable.

If perf clearly regresses (more than the run-to-run noise band of
~3-5%): investigate before committing. The expected outcome on
every item is "flat or slightly faster" — anything else means we
got something wrong.
