# skar

Spherical aspect-ratio solver. Given a point set on the unit sphere,
finds the tightest ellipsoidal cone enclosing it (parameterized by a
PSD matrix `A` and unit axis `b`) and returns the cone's axis ratio.

A standalone, std-only Zig package — no third-party dependencies.

## Quick start

```sh
zig build ex-basic     # runs examples/basic.zig — happy-path only
zig build ex-status    # runs examples/status.zig — every Outcome branch
```

[`examples/basic.zig`](examples/basic.zig) is the minimum call:
define points, call `solve`, switch on the outcome, print the aspect
ratio and axis. [`examples/status.zig`](examples/status.zig) shows
the canonical switch over every variant of the `Outcome` union.

In a Zig package, depend on `skar` and call into the public API:

```zig
const skar = @import("skar");

var outcome = try skar.solve(allocator, points, .{});
defer outcome.deinit();

switch (outcome) {
    .converged => |c| {
        const axis = c.b();
        const aspect = c.aspectRatio();
        // ...
    },
    .infeasible, .did_not_converge => { /* ... */ },
}
```

`solve` returns a tagged union, so accessors like `aspectRatio()`,
`b()`, `A()` only exist on the `Converged` variant — there's no
top-level method to accidentally call on a non-converged result.

Solver selection: `SolveOptions.method` defaults to `.auto`, an alias
for the library's recommended method (currently `.trust`, a
trust-region solver — the default since 0.6.0). Pin `.trust` or
`.alternating` (the original solver, bit-stable with pre-0.6.0
defaults) if you need version-stable behavior.

## Layout

- `src/root.zig` — public API re-exports
- `src/api.zig` — public API surface (types + methods + `checkFeasibility`)
- `src/skar.zig` — solver core: preprocessing, the alternating path, dispatch (std-only)
- `src/trust.zig` — the trust-region solver path (what `.auto` resolves to)
- `src/linalg.zig`, `src/halfspace.zig`, `src/newton.zig`, `src/config.zig` — internal modules
- `tests/*_test.zig` — top-level tests (run via `zig build test`)
- `tests/cases/cases.zig` — comptime manifest over the .zon files; exposed as the `cases` build module
- `tests/cases/cases_test.zig` — tests driven by the case manifest
- `tests/cases/zon/*.zon` — fixture point sets + expected outcomes (data only)
- `test_root.zig` — test-target root at repo level
- `examples/basic.zig`, `examples/status.zig`, `examples/cases.zig` — end-user usage demos
- `examples/bench.zig` — per-case timing (release-built; run via `zig build ex-bench`)
- `examples/compare.zig` — alternating-vs-trust comparison (release-built; `zig build ex-compare`)
- `dev.md` — developer-workflow guide (coverage, layout, conventions)

## Build

```sh
zig build              # builds the library
zig build ex-basic     # runs examples/basic.zig
zig build ex-status    # runs examples/status.zig
zig build ex-cases -- hex      # runs one bundled case
zig build ex-cases -- --all    # runs every bundled case
zig build ex-bench     # runs the per-case timing bench (release-built)
zig build test         # fast unit suite (no coverage)
```

Equivalent `just` targets are in `justfile`. The full suite +
100% line-coverage gate under kcov is `just test-slow` (the
pre-commit / CI check); see `dev.md` for the full workflow.

## License

MIT — see [LICENSE](LICENSE).
