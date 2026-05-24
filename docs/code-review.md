# Code review: project-wide

Findings from a high-recall multi-angle review of the project at the
current HEAD state (review run after the coplanarity-check work
landed). Scope was the whole solver — `src/skar.zig` and supporting
files in `tests/`, `bench/`, `cli/`, `src/root.zig`.

Severity is rough; ordering puts silent-wrong-output bugs first,
constrained-impact real bugs next, then doc/contract gaps that mislead
callers without crashing them. None of these break the current test
suite — they describe failure modes that aren't exercised yet or
contract drifts that would manifest under specific caller use.

---

## 1. `dualityGapConstructed` takes `@sqrt(sigma[0])` without a non-negativity guard — *RESOLVED*

- **Location:** `src/skar.zig` — the L = V·√Λ construction inside
  `dualityGapConstructed` (around the `@sqrt(sigma[0])`,
  `@sqrt(sigma[1])` calls).
- **Severity:** silent NaN propagation. Critical.
- **Status:** Resolved with a tolerance-banded guard after `eig2`:
  ulp-scale negatives are clipped to 0 (sqrt is well-defined; the
  existing Cholesky-null guard then routes the iteration through
  gap=1e30 as "no progress"), but meaningfully-negative eigenvalues
  raise the new `SolveError.NegativeEigenvalue`. Threshold is
  `tol.PSD_NEG_REL · max(sigma)`, mirroring `tol.NEG_GAP`'s shape.
  `dualityGapConstructed` now returns `SolveError!GapResult`; the
  single caller `try`s it.

`eig2(A_perp)` can return a slightly-negative smaller eigenvalue when
A_perp is near-rank-1 — for example, when Newton polish has
concentrated weight on a near-collinear active set after a chain of
b-updates. `@sqrt(−1e-18) = NaN`, which then poisons L, the M = LᵀZL
product, the Cholesky guard (`s <= 0` is silently false on NaN
comparisons), and the returned gap.

**Failure scenario.** Input whose hull active set is near-coplanar
but slipped past the coplanarity check (ratio just above threshold,
or the check is disabled). Newton polish drives A_perp's smaller
eigenvalue to roundoff-negative. The gap path returns 1e30 forever
(via the Cholesky-null branch) and the solver exits `did_not_converge`
after MAX_OUTER iterations — no diagnostic distinguishes "didn't
converge" from "NaN poisoned the iteration."

**Fix direction.** Clamp `sigma[i]` to `@max(sigma[i], 0)` before
taking sqrt. Optionally also raise `SolveError.NegativeDualityGap` if
the eigenvalue is below `-tol.NEG_GAP` — that's a "theorem violation"
in the sense that A_perp should be PSD by construction.

---

## 2. `recoverAPerp` takes `@sqrt(Minv.det())` without a PSD guard — *RESOLVED*

- **Location:** `src/skar.zig` — `recoverAPerp`'s `s_det` and `denom`
  computations.
- **Severity:** silent NaN propagation. Critical.
- **Status:** Resolved with the same tolerance-banded guard pattern as
  #1. det(Minv) noise below `tol.PSD_NEG_REL · trace²` is clipped to
  0; meaningful negatives raise the new `SolveError.SingularMoment`.
  `recoverAPerp` now returns `SolveError!Mat2`; the caller `try`s it.

`M` (the weighted moment matrix of the 2D projected points) is PSD in
exact arithmetic, but its inverse `Minv` can have det numerically
negative via roundoff when M is near-singular. `@sqrt(negative) =
NaN`, which propagates through `Minv_half` into A_perp, then into the
next b-step. NaN comparisons are silently false in both the
convergence check (`@abs(gap) <= gap_tol`) and the bug guard (`gap <
-NEG_GAP`), so no error fires.

**Failure scenario.** An outer iteration where Newton polish
concentrates weight on three nearly-collinear active points (common on
H3 hex children with one dominant corner cluster). `M.det() ~ 1e-30`
with FP noise; `Minv` has huge entries; `det(Minv)` rounds negative.
s_det = NaN, denom = NaN, A_perp = NaN. The solver finishes with
NaN-filled `Info.sigma` and `Info.Q`, and the caller sees a
`did_not_converge` (or, less commonly, a converged-with-NaN result if
the NaN comparison happens to flip the right way).

**Fix direction.** Either clamp `Minv.det()` to `@max(_, 0)` before
sqrt (correct for PSD matrices to ulp), or surface as
`SolveError.NegativeDualityGap`-style failure when the det is below
a tolerance. The clamp is cheaper and matches the same fix pattern as
finding #1.

---

## 3. `defer if (last_info) |*li| li.deinit();` registered after the loop — *RESOLVED*

- **Location:** `cli/main.zig` (the warmup + timed loops) and
  `bench/main.zig` (the timed loop).
- **Severity:** real leak under mid-loop error paths.
- **Status:** Resolved by moving the `defer` to immediately after the
  `var last_info` declaration in both files. Warmup loops use
  `catch continue` + a local `info.deinit()` and were already
  leak-free. Mechanical one-line move per file.

In both files the pattern is:

```zig
var last_info: ?sphar.Info = null;
for (0..N_RUNS) |r| {
    const info = try sphar.solve(...);  // may error
    if (last_info) |*li| li.deinit();
    last_info = info;
}
defer if (last_info) |*li| li.deinit();
```

The `defer` is registered *after* the loop, so if `solve` errors on
iteration `r > 0`, the previous iteration's `last_info` is on the
heap (cert allocated on the parent allocator) but its deinit never
runs.

**Failure scenario.** Run CLI with `--n-runs=2` on an input that
triggers `SolveError.NegativeDualityGap` on the second call but not
the first. r=0 succeeds and stores last_info; r=1's `try` propagates
the error and unwinds. Leak of `cert.indices` + `cert.lambdas`.
GeneralPurposeAllocator's leak check fires; with smp_allocator (cli's
default) the leak is silent.

**Fix direction.** Move the `defer if (last_info) ...` to immediately
after the `var last_info = null;` declaration. One-line move per file.

---

## 4. `halfspaceCheck` on empty input divides by zero, returns `.infeasible` with NaN cert

- **Location:** `src/skar.zig` — `halfspaceCheck`'s opening
  `z = z.scale(1.0 / n)` line.
- **Severity:** wrong status on a precondition-violating input.

With `n=0`, `z.scale(1.0/0.0)` makes `z = (NaN, NaN, NaN)`. The FW
loop body is empty, `all_positive` stays vacuously true, `nz = NaN
> NEAR_SING` is false, `b_out = null`. `solve` takes the infeasible
branch and returns `.infeasible` with an empty Farkas cert and
`claimed_gap = NaN`. A caller can't distinguish "I passed an empty
list" from "this is a real infeasibility result."

**Failure scenario.** Caller does `sphar.solve(allocator,
&[_][3]f64{}, ...)` — perhaps because upstream deduplication
collapsed all inputs, or because they're testing edge cases. Gets
`.infeasible` with NaN gap, looks at the Farkas cert (empty), is
confused.

**Fix direction.** Validate `X.len >= 1` at the entry to `solve` and
return either a new status (`insufficient_input`) or an error
(`SolveError.InsufficientInput` — fits the "caller's input is
malformed" category alongside potential future variants). Combine with
finding #7 below.

---

## 5. Back-to-back `try allocator.alloc(...)` without `errdefer` on the first — *RESOLVED*

- **Location:** `src/skar.zig` — three sites:
  - `buildFarkasCert` (the indices + lambdas pair)
  - the `.coplanar_input` early-return path
  - the final convergence path's cert assembly
- **Severity:** real leak under OOM only; constrained.
- **Status:** Resolved by adding `errdefer allocator.free(indices);`
  between the two allocations at all three sites. The coplanar_input
  path was refactored from a struct-literal initializer to two
  statements to accommodate the errdefer.

Each site does `const indices = try allocator.alloc(u32, k); const
lambdas = try allocator.alloc(f64, k);`. If the second alloc fails
(budget-limited allocator, fault injector, true OOM right at the
boundary), the first slice leaks because no `errdefer` was registered
for it.

**Failure scenario.** FixedBufferAllocator that fits the first
allocation but not the second, or a test fault-injector that fails
the kth call. solve returns `OutOfMemory` to the caller. Caller never
receives an Info and so never calls `deinit`. u32 slice is dropped.

**Fix direction.** Add `errdefer allocator.free(indices);` between the
two allocations at all three sites. Mechanical change.

---

## 6. `Info.b()` and `Info.A()` silently return zero on non-converged statuses

- **Location:** `src/skar.zig` — `Info.b()` and `Info.A()` methods;
  their doc comments do not document a non-converged contract.
- **Severity:** caller gets garbage with no warning.

`Info.b()` says "Cone axis: first column of Q." On `.infeasible`,
`.coplanar_input`, and (depending on path) `.did_not_converge`,
`Info.Q` stays at `Mat3.zero`, so `b()` silently returns `(0,0,0)`.
`A()` similarly returns the zero matrix.

Unlike `aspectRatio()` (documented to return NaN) and
`checkFeasibility()` (documented to return +inf), these methods have
no sentinel and no caveat.

**Failure scenario.** Caller writes `if (info.status == .infeasible)
renderFarkasDirection(info.b());` expecting a best-effort axis. Gets
`(0,0,0)`. Normalizes it → NaN. Downstream rendering / numerics fail
silently.

**Fix direction.** Update the doc comments on `b()` and `A()` to spell
out the precondition (only meaningful when `status == .converged`).
Cheap. A more invasive fix would be to populate `Q[:,0]` with the
halfspaceCheck axis even on `.infeasible` / `.coplanar_input` so the
caller has *something* sensible to inspect, but that's a behavioral
change requiring a careful think about the cone-axis vs.
halfspace-axis distinction.

---

## 7. `isCoplanarInput` flags n=1 and n=2 as `coplanar_input`, conflating two failure modes

- **Location:** `src/skar.zig` — `isCoplanarInput`'s `tr <= 0` guard.
- **Severity:** misleading status; caller can't distinguish.

For `n=1`, the centered scatter is identically zero (the single point
equals its own mean), `tr = 0`, the guard returns true → `solve`
returns `.coplanar_input`. For `n=2`, two points always lie on a
line, scatter is rank-1 (one nonzero, one zero eigenvalue), again
triggers the check.

These are inherent low-dimensional degeneracies, not the "all points
lie on a great circle" failure mode the variant name and docstring
suggest. The caller can't distinguish "I gave too few points" from
"I gave a degenerate spatial configuration" — they need different
upstream fixes.

**Failure scenario.** Caller passes 1 or 2 points (perhaps from
upstream deduplication that collapsed an input set). Reads the docs,
thinks "my input has rank-deficient geometry," but the actual issue
is "I need more points."

**Fix direction.** Either add an explicit `insufficient_points`
status variant (paired with finding #4's empty-input handling), or
broaden the `coplanar_input` docstring to mention that `n < 3`
inherently trips it. The status-rename option is cleanest but a
breaking API change.

---

## 8. `halfspaceCheck` conflates true Farkas infeasibility with FW numerical stall

- **Location:** `src/skar.zig` — the `nz <= NEAR_SING` branch in
  `halfspaceCheck` (which sets `b_out = null` but takes the
  `all_positive` path), and the unconditional `.infeasible` label in
  `solve`'s caller code.
- **Severity:** wrong status on borderline-feasible inputs.

`halfspaceCheck` returns `b_out = null` in two distinct cases: (a)
true Farkas infeasibility (FW converged with `all_positive = false`),
and (b) FW numerically stalled with `nz <= NEAR_SING` even though
`all_positive` was true. `solve` labels both as `.infeasible` and
runs `buildFarkasCert` — producing a Cert-shaped object that
satisfies the documented Farkas invariants (λ ≥ 0, ∑λ=1, small
residual) but in case (b) corresponds to a numerical stall rather
than a witness of infeasibility.

**Failure scenario.** Input whose true feasible direction is
near-orthogonal to its centroid (nearly-antipodal pairs with a small
tie-breaker that keeps them barely feasible). FW finds `all_positive`
true at an iterate where `‖z‖ < NEAR_SING`. `solve` returns
`.infeasible` with a cert that passes the integration test's Farkas
invariants. Caller dispatches on `.infeasible` and rejects valid data.

**Fix direction.** Distinguish the two cases in `HalfspaceResult` (a
separate enum variant for "stall" vs. "infeasibility witness") and
have `solve` either retry with a perturbed seed or return a
distinct status.

---

## 9. `coplanarity_tol = NaN` silently disables the check; `+inf` rejects everything

- **Location:** `src/skar.zig` — the `if (coplanarity_tol > 0 ...)`
  gate, plus the `4·det < tol·tr²` trigger.
- **Severity:** input validation gap; silent misbehavior.

`NaN > 0` is false, so passing NaN silently disables the check —
undocumented behavior. The doc says "Pass ≤ 0 to disable" but doesn't
mention NaN. Conversely, `tol = +inf` causes every input to flag
(since `4·det/tr² ∈ [0, 1] < inf` always), silently turning the
solver into a no-op rejector.

**Failure scenario.** Caller's config loader produces NaN when an env
var is unset or malformed (common when porting from Python). solve()
runs with the check disabled. A genuinely coplanar input then reaches
`recoverAPerp` → NaN → silent wrong output (see finding #2). Or +inf:
every input returns `.coplanar_input` regardless of geometry — looks
like the library is broken.

**Fix direction.** Validate `coplanarity_tol` at the boundary: assert
finite, or treat NaN as user error (panic / return an error). Cheap.

---

## 10. `Cert.claimed_gap` undocumented for `did_not_converge` and `coplanar_input`

- **Location:** `src/skar.zig` — `Cert` struct doc-comment for
  `claimed_gap` mentions only `.converged` and `.infeasible`.
- **Severity:** caller misuse risk under a uniform "quality metric"
  pattern.

`claimed_gap` on `.coplanar_input` is silently 0 (initialized in the
Info literal, never overwritten). On `.did_not_converge` it's the
last computed gap — could be anything from near-zero to `1e30`. The
docstring doesn't say either, so a caller treating it as a uniform
quality metric will misclassify these cases.

**Failure scenario.** Caller pipeline: `if (info.cert.claimed_gap <
user_tol) accept(info);`. On `.coplanar_input`, `claimed_gap = 0 <
user_tol` → caller accepts an Info with zeroed Q/sigma → downstream
code uses zero matrices, wrong geometry. On `.did_not_converge` with
gap just below user_tol, caller accepts a non-converged result.

**Fix direction.** Document `claimed_gap` semantics for all four
statuses. The fix is a doc-comment update; behavior stays the same.
Optionally also set `claimed_gap` to a sentinel (`+inf`?) on the
`.coplanar_input` early-return so the "uniform metric" pattern still
works.

---

## Items considered and refuted

- **Hull collapse + coplanarity check interaction.** Addressed by the
  earlier #1 from the coplanarity-check review (the check now runs on
  Xw rather than Xv; when hull falls back to Xv, the check still
  fires correctly on whatever the solver iterates on).
- **`n_hull` accepting any negative as "disable" while docs say -1.**
  Pure doc style. The behavior — any negative disables — is
  reasonable; doc says `-1` because it's the canonical sentinel, not
  because other negatives are forbidden.
- **`convexHull2d` u32 underflow on n=0 or n=1.** Currently
  unreachable because `hullPreprocess` early-returns before the call
  when `Xv.len <= n_hull` (and `n_hull >= 0` is also gated). Worth
  noting if the function ever becomes more broadly used.
- **`mveeFw`'s inner-tol early-exit on FW_PER_NEWTON > 1 cycles.** A
  subtle convergence-stall concern that hasn't been observed in
  practice; the FW step is structurally needed when geometry has
  changed, but in the steady-state regime its absence is
  observationally harmless.
- **Negative or NaN `gap_tol` causing the solver to silently run
  MAX_OUTER iterations.** Similar to finding #9 (NaN-tol disables
  check). The fix is the same shape — validate tolerances at the
  boundary.
