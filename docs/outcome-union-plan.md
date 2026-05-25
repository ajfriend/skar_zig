# Plan: replace `Info` + `Status` with an `Outcome` tagged union

**Branch**: `outcome-union` (prototype; merge decision held until the
implementation lands and we eyeball the migrated call sites).

## Why

The current public API has the **cvxpy footgun**: `solve` returns
`Info` regardless of outcome, with a `status: Status` field the
caller is supposed to check before reading `info.aspectRatio()`,
`info.b()`, `info.A()`, or `info.cert`. We document this ("callers
that care should gate on `status == .converged`"), but documentation
is the weakest enforcement we can offer — the type system isn't
helping. A user who writes

```zig
const info = try solve(allocator, X, .{});
const ar = info.aspectRatio();  // NaN on .infeasible / .did_not_converge / .coplanar_input
useDownstream(ar);
```

gets garbage silently. That's exactly the failure mode cvxpy is
infamous for (`problem.value` is set regardless of `problem.status`,
users grab it without checking, results contaminate downstream code).

The fix: replace the flat `Info` + `Status` with a **tagged union**
where each variant carries only the data that's meaningful for it,
and Zig's exhaustive-switch check forces every caller to acknowledge
every outcome. The footgun becomes *unrepresentable* — there's no
top-level `aspectRatio` to accidentally call; you have to switch
first and reach it through the `Converged` variant.

## The shape

### `Outcome` union

```zig
pub const Outcome = union(enum) {
    /// A valid cone was found; full eigendecomposition + primal cert.
    converged: Converged,

    /// Proven infeasible — no hemisphere contains all input points.
    /// Carries the Farkas certificate as proof.
    infeasible: Infeasible,

    /// Solver hit `max_outer` without closing the gap. Last iterate
    /// is available for warm-start / inspection; no certified cone.
    did_not_converge: PartialInfo,

    /// Input is rank-deficient (all points on a single great circle).
    /// No cone is definable; nothing to inspect.
    coplanar_input: void,

    pub fn deinit(self: *Outcome) void {
        switch (self.*) {
            .converged => |*c| c.deinit(),
            .infeasible => |*i| i.deinit(),
            .did_not_converge, .coplanar_input => {},
        }
    }
};
```

### Per-variant structs

```zig
pub const Converged = struct {
    Q: Mat3,
    sigma: [3]f64,
    cert: Cert,                  // primal certificate
    outer_iters: u32,
    newton_polish_failures: u32,
    allocator: Allocator,

    pub fn aspectRatio(self: Converged) f64 { return self.sigma[2] / self.sigma[1]; }
    pub fn b(self: Converged) Vec3 { return self.Q.col(0); }
    pub fn A(self: Converged) Mat3 { /* materialize from eigendecomp */ }
    pub fn deinit(self: *Converged) void { /* free cert slices */ }
};

pub const Infeasible = struct {
    cert: Cert,                  // Farkas: λ ≥ 0, ∑λ = 1, ‖∑λᵢxᵢ‖ small
    residual: f64,               // the witness magnitude
    allocator: Allocator,

    pub fn deinit(self: *Infeasible) void { /* free cert slices */ }
};

pub const PartialInfo = struct {
    Q: Mat3,                     // last iterate — not certified
    sigma: [3]f64,
    last_gap: f64,
    outer_iters: u32,
    newton_polish_failures: u32,
    // no cert: we didn't converge
};
```

Each variant only exposes the methods that make sense for it.
`Converged.aspectRatio` is a plain `f64`-returning method (no
fallible accessor — the type system already guarantees we're in the
converged branch). `Infeasible` doesn't have an `aspectRatio` method
at all — calling it is a compile error.

### Error set (essentially unchanged)

```zig
pub fn solve(
    allocator: Allocator,
    X: []const [3]f64,
    opts: SolveOptions,
) (SolveError || InputError || Allocator.Error)!Outcome
```

The errors keep their current semantics: each one means the call
**couldn't produce a structured outcome at all**.

- `Allocator.Error` — host couldn't allocate.
- `InputError.InsufficientPoints` / `InvalidTolerance` — caller's
  arguments were malformed; we bailed before the algorithm ran.
- `SolveError.NegativeDualityGap` / `NegativeEigenvalue` /
  `SingularMoment` — internal-correctness violations; the
  algorithm ran but produced a result that disqualifies itself.

`Status` as a public enum **goes away**. The union tag replaces it.

## Migration

### Caller pattern, before → after

```zig
// Before:
const info = try sphar.solve(allocator, X, .{});
defer info.deinit();
switch (info.status) {
    .converged => |_| use(info.aspectRatio()),
    .infeasible => useFarkas(info.cert),
    .did_not_converge => report(info.cert.claimed_gap),
    .coplanar_input => noteRankDeficient(),
}

// After:
var outcome = try sphar.solve(allocator, X, .{});
defer outcome.deinit();
switch (outcome) {
    .converged => |c| use(c.aspectRatio()),
    .infeasible => |i| useFarkas(i.cert),
    .did_not_converge => |p| report(p.last_gap),
    .coplanar_input => noteRankDeficient(),
}
```

### Sites to migrate

Internal:
- `src/skar.zig` — `solve` body: assemble per-variant outcomes
  instead of populating a single `Info`. Drop the `var info = Info{
  .status = .did_not_converge, ... }` initializer. Each early-return
  path constructs its variant directly.
- `src/skar.zig` — `checkFeasibility(info, X)` — currently takes
  `Info` and returns `+inf` if `info.status != .converged`. Switch
  to taking `Converged` (or `Outcome` and switching internally).
- `src/skar.zig` — `Info` struct + methods deleted; replaced by the
  three per-variant structs.
- `src/root.zig` — re-exports updated: `Outcome`, `Converged`,
  `Infeasible`, `PartialInfo`; drop `Info`, `Status`, `Cert` (Cert
  may stay if multiple variants share its shape).

Tests (4 in-tree files):
- `src/tests/integration_test.zig` — the convergence-baseline test
  uses `info.status`, `info.aspectRatio()`, `info.cert`, `info.Q`.
  Migrate to switch on outcome, asserting on the right variant.
- `src/tests/integration_test.zig` — the infeasible/DNC/shape-invariants
  tests similarly migrate.
- `src/tests/extreme_aspect_test.zig` — the rotation-invariance,
  coplanarity, OOM, and acceptBUpdate-fallback tests all reference
  `info.status` and accessors; migrate. The 3 negative tests using
  fake-Info construction need fake-Outcome construction instead.
- The `checkRotationInvariance` and `checkArEq` helpers update for
  the new return shape.

External demos:
- `examples/basic.zig` — was 30 lines; becomes ~25 lines with the
  switch (or stays ~30 with a `case .converged => |c| { ... }`
  inline pattern).
- `examples/status.zig` — already a switch-on-status; near drop-in
  rename to switch-on-outcome.
- `cli/main.zig` — prints status string; migrate.
- `bench/main.zig` — prints per-case status; migrate.

### Order of operations

1. Define `Outcome` + per-variant structs in `src/skar.zig`. Keep
   `Info` + `Status` temporarily so the build stays green.
2. Add an `Info → Outcome` adapter or rewrite `solve` directly.
3. Migrate `src/skar.zig`'s `solve` body to construct `Outcome`
   variants directly.
4. Migrate tests one file at a time. Run `just test` after each.
5. Migrate cli / bench / examples.
6. Delete `Info`, `Status`, the adapter — old types fully retired.
7. Update `dev.md` and `README.md`.

Each step a separate commit; the branch should be bisectable.

## Verification

### Behavior preservation

The 17 existing tests must all pass after migration, with no logic
change beyond the type translation. Coverage gate (100% line) must
remain green at every commit.

### Type-level proof the footgun is closed

A test that *tries* to use `outcome.aspectRatio()` directly (without
switching) should fail to compile. We can capture this in a comment
on the `Converged` struct's `aspectRatio` method:

```zig
/// Method on `Converged` only — calling it on `Outcome` directly
/// is a compile error, by design.
pub fn aspectRatio(self: Converged) f64 { ... }
```

The compile-error-by-construction is the win; no runtime test needed.

### Performance

Tagged unions in Zig are a tag + the largest variant. `Outcome`'s
largest variant is `Converged` which is currently ~200 bytes
(Mat3 + sigma + cert pointers + counters). The bench should be
indistinguishable from current within noise.

## Risk and rollback

- **Risk**: bigger surface change than anything else we've done.
  ~10 caller sites + the solve body. Mitigation: branch + per-step
  commits, easy to bisect or roll back.
- **Risk**: a downstream user already imports `Status` or `Info`
  from our package. Mitigation: this is an unreleased solo project
  (per the README), so no external users.
- **Rollback**: don't merge the branch. Branch lives as a reference;
  main stays on the current shape.

## Out of scope (this round)

- Renaming `Cert` or moving it inside variants. It's used by both
  `Converged` and `Infeasible` with the same shape but different
  semantics for `claimed_gap`. Could later split into `PrimalCert`
  and `FarkasCert` but not necessary for the union refactor.
- Adding new error variants (e.g., `DidNotConverge` as an error
  rather than a variant). We've decided variant; the doc above
  spells out why.
- Public re-export of internal modules. Tests reach internals via
  filesystem `@import` (unchanged from current).

## Decision point after implementation

After the branch is implemented and passing tests + coverage, eyeball
the migrated examples and caller sites. The merge decision rests on:

- **Does the call-site code actually read better?** (subjective but
  the most important signal)
- **Did we lose any information the current API exposed?** (Cert
  semantics, claimed_gap meaning, partial-iterate fields)
- **Did `solve`'s body get simpler or messier?** (it currently
  threads `info` through every phase; the union version builds the
  outcome at the end)

If the answers are good: merge. If not: discard the branch; the
plan and discussion live on as documentation of the design space
we considered.
