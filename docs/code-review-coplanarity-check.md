# Code review: coplanarity preprocessing check

Findings from a high-recall multi-angle review of the diff that introduced
`Status.coplanar_input` and the 2D-scatter coplanarity check in `solve`
(working tree state at the time of review; see `src/skar.zig`,
`tests/extreme_aspect.zig`, `bench/main.zig`, `cli/main.zig`,
`tests/integration.zig`).

Severity is rough; ordering is most-actionable first. None of these break
the current test suite — they describe failure modes that aren't exercised
yet or fragilities that would manifest under future change.

---

## 1. Coplanarity check runs on full input, but solver iterates on the hull subset — *RESOLVED*

- **Location:** `src/skar.zig` — the new `if (coplanarity_tol >= 0)` block
  sits between `halfspaceCheck` and the hull preprocessing.
- **Severity:** correctness gap.
- **Status:** Resolved by moving the check to step 2.5 (after hull
  preprocessing), running it on `Xw` rather than `Xv`. For small-n
  inputs that bypass hull preprocessing, `Xw == Xv` so behavior is
  unchanged; for large-n inputs with hull enabled, the check now sees
  what the solver will actually iterate on. Same move addresses #9.

The check measures the 2D scatter of the **full** input `Xv`, but if
`n_hull` is enabled (default 10) the solver then iterates on the hulled
subset `Xw_storage` only. An adversarial input whose full-cloud scatter is
full-rank can still have a near-collinear *hull*, which is exactly the
shape the check was supposed to guard against.

**Failure scenario.** `n = 1000`: 997 jittered points in a tight cluster
around vertex A (contributing positive 2D scatter), plus 3 hull-defining
outliers at B, C, D that happen to be near-collinear in tangent space.
Full-cloud scatter passes the check. Hull preprocessing reduces input to
`{B, C, D}`. Solver runs on near-collinear input → great-circle failure
mode.

**Fix direction.** Run the check on `Xw_storage` after hull reduction, or
duplicate it (cheap on the hulled subset).

---

## 2. `checkFeasibility(info, X)` returns 0 on `coplanar_input` — *RESOLVED*

- **Location:** `src/skar.zig` — `checkFeasibility` reads `info.A()` and
  `info.b()`, both of which are zero on the new early-return path.
- **Severity:** silent downstream pitfall.
- **Status:** Resolved by adding a guard at the top of
  `checkFeasibility`: any non-converged status returns `+inf`. The
  `inf` sentinel composes cleanly with the typical
  `checkFeasibility(...) <= tol` gate — it always rejects, so a caller
  using the function as a "is this Info usable" check no longer sees
  apparent feasibility on a rejected input. Function doc updated to
  document the precondition explicitly. New test assertion exercises
  the guard on a `.coplanar_input` result.

On `Status.coplanar_input`, `Info.Q` is the zero matrix and
`Info.sigma = .{0, 0, 0}`. So `info.A() = 0`, `info.b() = (0,0,0)`. The
feasibility inequality `‖A·x‖ ≤ b·x` collapses to `0 ≤ 0` for every input
point, and `checkFeasibility` returns max violation 0 — apparent
feasibility on a rejected input.

**Failure scenario.** A pipeline calls `solve` then uses
`checkFeasibility(info, X) <= tol` as the gate for "usable output."
Great-circle input is flagged as `coplanar_input`. The feasibility gate
trivially passes (0 ≤ tol). Pipeline proceeds with garbage `Info`.

**Fix direction.** `checkFeasibility` should either gate on
`status == .converged` first, or return a sentinel (NaN, ∞) on any
non-converged status. Document the contract.

---

## 3. `aspectRatio()` returns NaN on `coplanar_input`, but docs and callers don't anticipate it — *RESOLVED*

- **Location:** `src/skar.zig` (Info struct doc), `bench/main.zig:80`
  (unconditional print of `info.aspectRatio()`).
- **Severity:** silent NaN propagation; cosmetic in bench but real in
  callers.
- **Status:** Resolved by updating the `aspectRatio()` field doc to
  list all NaN-producing statuses (not just `.infeasible`) and to
  recommend callers gate on `.converged` before reading. Bench's
  unconditional print still emits "nan" for any non-converged row —
  consistent with how `.infeasible` rows already printed.

`aspectRatio()` computes `sigma[2] / sigma[1]` and the field-level doc
comment notes it returns NaN via 0/0 specifically on `INFEASIBLE`. With
`coplanar_input` also yielding `sigma = .{0, 0, 0}`, that comment is now
incomplete. `bench/main.zig:80` prints `aspectRatio()` unconditionally —
a `coplanar` row would emit `nan` in the AR column, mirroring how
`infeas` rows print.

**Failure scenario.** A caller writes
`if (info.status == .infeasible) skip_ar else use(info.aspectRatio())`.
The solver returns `.coplanar_input`, the caller takes the `use` branch,
and NaN propagates silently.

**Fix direction.** Update the field-level doc to mention all NaN-producing
statuses. Either change `aspectRatio()` to assert / return Option on
non-converged status, or document that callers must guard explicitly.

---

## 4. One-pass centered-scatter formula is cancellation-prone — *RESOLVED*

- **Location:** `src/skar.zig` — `c00 = s00 - ps0 * ps0 * inv_n` and
  analogues.
- **Severity:** numerical edge case; can misclassify on legitimate input.
- **Status:** Resolved by switching `isCoplanarInput` to a two-pass
  accumulator: pass 1 computes the mean, pass 2 accumulates squared
  deviations from the mean. Each deviation term is small and
  non-negative, so the subtraction-induced cancellation is gone and
  `tr ≥ 0` is structural rather than a roundoff coincidence. All 7
  existing tests pass without modification. Bench impact lost in
  measurement noise (sub-µs per case, runs on the hull subset which
  is typically ≤ 10 points).

The centered scatter uses the textbook one-pass form
`Var = Σx² − (Σx)²/n`. When the tangent-plane projections have a large
mean relative to their spread — e.g., the FW iterate `b` from
`halfspaceCheck` ends up far from the data centroid — `s00` and
`ps0·ps0/n` are nearly equal, and the subtraction can drop most digits.
For small-but-positive spreads, `c00` can become roundoff-negative, and
the `tr ≤ 0` guard fires on legitimate full-rank input.

**Failure scenario.** Input where FW terminates at a feasible `b` far
from the natural centroid (e.g., points crowded near one edge of the
feasible cone). Tangent-plane projections then have
`ps0/n ≈ p ≈ |projection|`, so `s00 − ps0²/n` cancels by ~14 digits. For
a real spread at ~1e-15, `tr` can flip sign and a valid input is
flagged.

**Fix direction.** Switch to the two-pass form (compute mean, then
accumulate squared deviations) — same number of passes overall (~n adds
× 2), no cancellation. Welford's online algorithm is another option but
overkill for n ≥ ~50.

---

## 5. Empty-literal slice passed to `allocator.free` on the early-return path — *RESOLVED*

- **Location:** `src/skar.zig` — the initial `Info` literal sets
  `cert.indices = &[_]u32{}` and `cert.lambdas = &[_]f64{}`. These are
  returned as-is on `Status.coplanar_input`.
- **Severity:** fragile; works on standard allocators, breaks on strict
  custom ones.
- **Status:** Resolved by allocating zero-length slices on the
  `.coplanar_input` early-return (mirrors the `.infeasible` branch).
  `Info.deinit` now always frees allocator-owned pointers across all
  statuses.

Other status paths (`infeasible`, `converged`, `did_not_converge`) allocate
`cert.indices` / `cert.lambdas` via the allocator even when length is
zero, so `Info.deinit` is uniform there. The `coplanar_input` early-return
leaves the static empty-literal pointers in place. Standard Zig
allocators (GPA, ArenaAllocator, page_allocator, c_allocator) no-op on
zero-length free, so this is benign today.

**Failure scenario.** Caller uses a tracking / bounds-checking custom
allocator that asserts every freed pointer was previously allocated by
this allocator. Sends a great-circle input. Solver returns
`Status.coplanar_input`. Caller's `defer info.deinit()` calls
`allocator.free` on a `.rodata` pointer → assertion failure / crash.

**Fix direction.** Mirror the `.infeasible` branch: allocate zero-length
slices through `allocator` even on `coplanar_input`. Or document that
`Info.deinit` requires an allocator that no-ops zero-length free.

---

## 6. `coplanarity_tol = 0` is a partial-disable, not "maximally strict" — *RESOLVED*

- **Location:** `src/skar.zig` — gate condition
  `if (coplanarity_tol >= 0)` plus trigger `4·det < tol · tr²`.
- **Severity:** docs/behavior mismatch around the boundary value.
- **Status:** Resolved by changing the gate from `>= 0` to `> 0` so
  exactly-zero is treated as disabled (matching the silent behavior it
  had before). The parameter doc now spells out *why* 0 is disabled
  rather than being a usable strictness setting.

The gate accepts `0` as "enabled," but the trigger inequality becomes
`4·det < 0`, which is unreachable for a PSD 2×2 matrix (centered
scatter is PSD by construction). So `coplanarity_tol = 0` only retains
the `tr ≤ 0` companion branch — almost the same as `-1` (the documented
disable knob). A user reading "tighter catches only essentially-exact
coplanarity" and choosing `0` for maximum strictness gets the opposite.

**Failure scenario.** User picks `tol = 0` expecting maximum strictness.
Sends a near-degenerate input with `4·det/tr² ~ 1e-30`. The check does
NOT fire. Solver proceeds and either NaNs out or surfaces
`NegativeDualityGap` — the precise failure modes the user thought they
were guarding against.

**Fix direction.** Either reject `tol == 0` at the call boundary
(meaningless value), or change the gate to `tol > 0`. Update the
docstring either way to spell out the disable/enable semantics
unambiguously.

---

## 7. Tests rely on `try` + the NaN-not-error path with check disabled

- **Location:** `tests/extreme_aspect.zig` — the "negative control" arm
  of `coplanarity check flags great-circle inputs` (passes
  `coplanarity_tol = -1` on great-circle input through `try`) and the
  loose-tol arm of `coplanarity check cutoff is near the parameter's
  value`.
- **Severity:** test fragility for future changes.

Both tests pass `-1` to disable the check and use bare `try` on inputs
that historically produced NaN-filled `Info` rather than throwing. If a
future tightening of the dual-gap guard or Cholesky logic shifts those
inputs from NaN to `SolveError.NegativeDualityGap`, `try` propagates
and the tests fail with an error that has nothing obvious to do with
the coplanarity check.

**Failure scenario.** Someone tightens `tol.NEG_GAP` or Cholesky's
positivity guard. Great-circle inputs with the check disabled now
trigger `NegativeDualityGap`. The negative-control assertions never
run; the test failure points at coplanarity when the cause is
elsewhere.

**Fix direction.** Wrap the disabled-check call with
`catch |err| ...` and explicitly accept both `NegativeDualityGap` and
NaN-Info outcomes (or just stop asserting anything beyond "doesn't
get flagged as coplanar_input" — i.e., use `if (sphar.solve(...))
|info| { ... } else |_| {}`).

---

## 8. Name vs behavior — `coplanar_input` actually detects "near-collinear in tangent plane" — *RESOLVED (doc only)*

- **Location:** `src/skar.zig` — the geometric meaning of the status.
- **Severity:** semantic / naming.
- **Status:** Resolved with a docstring tweak rather than a rename
  (renaming churns the public API). The `Status.coplanar_input` doc
  now spells out that the detection is slightly broader than the
  literal name — short arcs on non-equatorial latitude circles can
  also trip it via near-collinear tangent-plane projections.

The status name and docstring frame this as "points coplanar with the
origin." Operationally the check measures "2D scatter rank after
projecting to the tangent plane at `b`." These differ on small-circle
latitude inputs compressed into a short arc: such inputs are NOT
coplanar with the origin (they sit at `z = const ≠ 0`), but their
tangent-plane projections are near-collinear and get flagged.

Two ways to read this:

- The check is "correctly" stricter than the name suggests, and the name
  is misleading.
- The check is over-conservative — short-arc inputs at non-equatorial
  latitudes might in principle be solvable for an extremely thin cone.

Either is defensible; the inconsistency between the name and the
operational meaning is what's worth flagging.

**Fix direction.** Either rename (e.g., `tangent_collinear`,
`projected_collinear`) or document in the status' doc-comment that the
detection is broader than the name strictly suggests.

---

## 9. O(n) coplanarity check runs before hull preprocessing — *RESOLVED*

- **Location:** `src/skar.zig` — check sits between `halfspaceCheck` and
  hull reduction.
- **Severity:** performance regression at large n; invisible in current
  bench.
- **Status:** Resolved alongside #1 by moving the check to after hull
  preprocessing. For large-n with hull enabled, the check now runs on
  the hulled subset (~hull_size instead of n).

For very large inputs (n ≫ n_hull), the check adds an unavoidable full
O(n) pass even though the solver only operates on the small hull. Bench
caps at n=400 so this is invisible there. Related to finding #1
(running the check on the hulled subset would address both performance
and correctness).

**Failure scenario.** Caller passes n = 1e6, n_hull = 10. Pre-change:
halfspaceCheck (small FW iters) + hull preprocessing (O(n log n)) +
solver over hull (tiny). Post-change: same + extra O(n) pass for the
coplanarity check. For workloads where solver iter count is small, the
check can be a meaningful fraction of total time.

**Fix direction.** Move the check to after hull reduction (also
addresses #1) — cheap on the hull, equally informative since the hull
of a coplanar input is coplanar.

---

## 10. Comment math error in test rationale — *RESOLVED*

- **Location:** `tests/extreme_aspect.zig` — the third arm of
  `coplanarity check cutoff is near the parameter's value`.
- **Severity:** minor; future-maintenance footgun.
- **Status:** Resolved — "3700×" replaced with "2.7e6×".

Comment says "ratio (2.7e-14) is now ~3700× above the tighter threshold
(1e-20)" but the actual ratio is `2.7e-14 / 1e-20 = 2.7e6` (millions,
not thousands). The test still passes — the headroom is enormous either
way — but a maintainer reading "~3700× cushion" and tightening the
threshold based on that number would miscalibrate.

**Fix direction.** Replace "3700×" with "2.7e6×" (or equivalent).

---

## Items considered and refuted

- **Rotational variance via `pickRefAxis` tie-break:** The check
  computes `det(C)` and `trace(C)` on the 2D scatter, both of which are
  invariant under orthonormal change of basis (rotating the tangent
  basis just rotates the scatter as `R·C·Rᵀ`, preserving det and trace).
  So the trigger is independent of which orthonormal basis
  `b.orthoBasis()` happens to pick.
- **Non-unit-norm inputs skewing the check:** The solver's API
  precondition is unit vectors; non-unit inputs are out-of-scope.
- **`Info.Q` zeroed on `coplanar_input` discards `b` from
  halfspaceCheck:** Design choice consistent with "no meaningful
  payload." Could be a future enhancement to expose `b` for diagnostic
  purposes, but it's not a bug.
