# Development notes

## Dependencies

- **[zig](https://ziglang.org/)** 0.15.2+ — the language. `brew install zig`
  on macOS; see ziglang.org for other platforms.
- **[just](https://github.com/casey/just)** — task runner.
  `brew install just`.
- **[kcov](https://github.com/SimonKagstrom/kcov)** — line-coverage tool
  used by `just test`. `brew install kcov` on macOS,
  `apt-get install kcov` on Debian/Ubuntu.
- **[jq](https://stedolan.github.io/jq/)** — used by `just test` to check
  the coverage threshold. `brew install jq` / `apt-get install jq`.

## Common commands

| Command | What it does |
| --- | --- |
| `just test` | Build the test binary, run tests under `kcov`, print line coverage, fail if not exactly 100%. The inner-loop iteration command. |
| `just build` | Build the library + CLI + bench (release-optimized). |
| `just coverage` | Same as `just test`, then prints the path to the HTML report. |
| `just bench` | Run the benchmark suite (uses the release binary). |
| `just clean` | Remove `zig-out/`, `.zig-cache/`, `coverage/`. |

## Coverage

`kcov` runs the test binary as a black box: it instruments the binary
with traps at each source line and records which lines execute. It
writes two directories under `coverage/`:

- `coverage/skar-test/` — the **merged** HTML report; open
  `coverage/skar-test/index.html` to browse covered lines.
- `coverage/skar-test.<hash>/` — the **per-binary** report containing
  the `coverage.json` summary that `just test` parses with `jq` to
  enforce the threshold.

Both contain the same aggregate percentages today (one binary, one
run). If you're debugging a gate failure, the JSON is in the
hash-suffixed sibling, not the merged dir.

The gate enforces **100% line coverage** across both production code
(`src/*.zig`, `src/tests/*.zig`) and the case-loader helper
(`cases/cases.zig`). Test code isn't exempt — dead test helpers are
dead code too.

What "100% line coverage" buys you:

- Every line in every shipped function is reached by some test.
- `comptime` branches that aren't realized at runtime don't appear in
  the binary, so they don't show as uncovered — Zig's comptime is
  naturally well-suited to line coverage.

What 100% line coverage **doesn't** buy you:

- **Branch coverage.** A one-line `if (a) x() else y()` counts as one
  line. Both sides being executed isn't measured. Discipline plus code
  review fill the gap; explicit tests for both branches are the norm.

### Why kcov and not LLVM source-based coverage?

Zig 0.15.x / 0.16.x doesn't expose the LLVM coverage flags
(`-fprofile-instr-generate`, `-fcoverage-mapping`) that would feed
`llvm-cov` and unlock branch coverage. The CLI only has `-ffuzz` for
instrumentation. Track
[ziglang/zig#352](https://github.com/ziglang/zig/issues/352) — when
LLVM-style coverage lands upstream, swap the kcov pipeline in
`justfile` for an `llvm-profdata merge` + `llvm-cov show` flow and
pick up branch coverage automatically.

## Source layout

The `src/` directory contains library code only — nothing in there
should be test- or diagnostic-specific.

| File | Role |
| --- | --- |
| `src/root.zig` | Module entry point — thin re-export shim over `api.zig` + `solve` from `skar.zig`. |
| `src/api.zig` | Public API surface: `Outcome` (`Converged` / `Infeasible` / `DidNotConverge`), `Cert`, `SolveError` / `InputError` / `SolveOptions`, `checkFeasibility`. Read this file end-to-end to learn the API. |
| `src/skar.zig` | Algorithm orchestration: mvee/gap inner code, outer-loop driver, `solve`. |
| `src/linalg.zig` | Linear algebra primitives: Vec2/3, Mat2/3/3x2, Chol3, `eig2`. |
| `src/config.zig` | Internal tuning: `SIGMA_0`, `algo` (algorithm tuning), `tol` (numerical tolerances). |
| `src/halfspace.zig` | Geometric preprocessing: `halfspaceCheck`, `convexHull2d`, `projectGnomonic`. |
| `src/newton.zig` | Newton polish on the D-optimal dual + bordered KKT/LU. |

## Test layout

Test files live **inside** `src/tests/` so they're in the same Zig
module path as the library sources — internals are reached via
filesystem `@import("../halfspace.zig")` etc., without needing a
"test-only" namespace in the public API. `zig build test` uses
`src/root.zig` as the test root; the `test {}` block at the bottom
of `root.zig` pulls in `src/tests/all.zig`, which `comptime`-imports
each `*_test.zig` file.

| File | Role |
| --- | --- |
| `src/tests/all.zig` | Aggregator: `comptime { _ = @import(...); }` for each `*_test.zig`. Pulled in by `root.zig`'s `test {}` block. |
| `src/tests/integration_test.zig` | Iterates the comptime case manifest from `cases/cases.zig`; validates convergence + certificates per-case. |
| `src/tests/extreme_aspect_test.zig` | Rotation-invariance, coplanarity, near-degenerate edge-case tests on synthesized inputs. Also hits internal helpers (`acceptBUpdate`, `convexHull2d`) via filesystem imports for branches not reachable through `solve` for all inputs. |
| `cases/cases.zig` | Comptime manifest over `cases/*.zon` — defines the `Case` schema and the `all` list. Exposed as the `cases` build module; imported by tests / bench / the `ex-cases` example. |

To add a new test file: create `src/tests/<name>_test.zig`, then add
`_ = @import("<name>_test.zig");` to `src/tests/all.zig`. The test
binary picks it up automatically.

## Bench

`bench/main.zig` produces the `skar-bench` binary: min/median
timings over a hand-picked subset of the case manifest. Not part of
the library; links against `src/root.zig` through the `skar` build
module.

For ad-hoc shell-driven runs of a single case, use the `ex-cases`
example: `zig build ex-cases -- <name>` or `-- --all`.
