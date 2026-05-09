# skar

Spherical aspect-ratio solver. Given a point set on the unit sphere,
finds the tightest ellipsoidal cone enclosing it (parameterized by a PSD
matrix `A` and unit axis `b`) and returns the cone's axis ratio.

Extracted from the Zig solver in
[`2025-09-29_conic_aspect_ratio`](../2025-09-29_conic_aspect_ratio) as a
standalone, std-only Zig package.

## Layout

- `src/skar.zig` — solver core (~1500 lines, std-lib only)
- `src/root.zig` — public API re-exports
- `src/cases.zig` — case-file loader
- `cli/main.zig` — `skar-cli`: solve a single case, emit one JSONL line
- `bench/main.zig` — `skar-bench`: per-case timing across the bundled set
- `tests/integration.zig` — convergence + certificate tests against bundled cases
- `cases/*.txt` — fixture point sets

## Build

```sh
zig build              # builds lib + skar-cli + skar-bench
zig build test         # runs integration tests
zig build bench        # runs the timing bench
```

Equivalent `just` targets are in `justfile`.
