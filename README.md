# skar

Spherical aspect-ratio solver. Given a point set on the unit sphere,
finds the tightest ellipsoidal cone enclosing it (parameterized by a
PSD matrix `A` and unit axis `b`) and returns the cone's axis ratio.

Extracted from the Zig solver in
[`2025-09-29_conic_aspect_ratio`](../2025-09-29_conic_aspect_ratio) as a
standalone, std-only Zig package.

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

## Layout

- `src/root.zig` — public API re-exports
- `src/skar.zig` — solver core (~1300 lines, std-only)
- `src/linalg.zig`, `src/halfspace.zig`, `src/newton.zig`, `src/config.zig` — internal modules
- `src/tests/*_test.zig` — in-tree tests (run via `zig build test`)
- `tests/cases.zig` — fixture loader (shared by tests, cli, bench)
- `cases/*.txt` — fixture point sets
- `cli/main.zig` — `skar-cli`: solve a single case file, emit one JSONL line
- `bench/main.zig` — `skar-bench`: per-case timing across the bundled set
- `examples/basic.zig`, `examples/status.zig` — end-user usage demos
- `dev.md` — developer-workflow guide (coverage, layout, conventions)

## Build

```sh
zig build              # builds lib + skar-cli + skar-bench
zig build ex-basic     # runs examples/basic.zig
zig build test         # runs tests under kcov via `just test`; see dev.md
zig build bench        # runs the timing bench
```

Equivalent `just` targets are in `justfile`. The `test` recipe
enforces 100% line coverage via kcov (see `dev.md` for the full
workflow).
