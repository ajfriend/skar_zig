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
| `just test` | Fast test loop — skips long-running randomized stress tests, no coverage gate. Sub-second; the inner-loop iteration command. |
| `just test-slow` | Full suite + 100% line coverage gate under `kcov`. Builds with `-Dslow=true` so randomized stress tests run. ~10s; the pre-commit / CI check. |
| `just build` | Build the library (release-optimized). |
| `just coverage` | Same as `just test-slow`, then prints the path to the HTML report. |
| `just bench` | Run the benchmark suite (release-built `ex-bench`). |
| `just clean` | Remove `zig-out/`, `.zig-cache/`, `coverage/`. |

### Two test tiers

`just test` is the dev fast loop — runs every test except those gated
on `-Dslow`. `just test-slow` adds the slow ones and enforces the
coverage gate. The slow flag is plumbed via `build.zig`'s
`b.addOptions("test_options", ...)` into a `test_options` module
that gated tests import:

```zig
const test_options = @import("test_options");

test "my slow test" {
    if (!test_options.slow) return error.SkipZigTest;
    // ...
}
```

Slow tests show up as `SKIP` in the fast tier and `OK` in the slow
tier. Coverage only makes sense on the slow tier — fast-tier
coverage would be incomplete by design.

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
(`src/*.zig`, `tests/*.zig`) and the case manifest (`tests/cases/cases.zig`).
Test code isn't exempt — dead test helpers are dead code too. The
gate runs under `just test-slow`, not `just test` — slow-tier tests
(currently cap_test) exercise lines that the fast tier doesn't reach.

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

Tests live at top-level `tests/`, not inside `src/`. The test target
roots at `test_root.zig` at the repo root; its module's
filesystem-import scope therefore covers both `src/` (the library
under test, reached via `@import("../src/foo.zig")` from test files)
and `tests/` (the test files themselves). This lets tests reach
internals like `acceptBUpdate` or `convexHull2d` directly, without
re-exporting them through the public API.

| File | Role |
| --- | --- |
| `test_root.zig` | Test-target root at the repo level. One `test {}` block that pulls in `tests/all.zig`. |
| `tests/all.zig` | Aggregator: `comptime { _ = @import(...); }` for each test file. |
| `tests/solver_test.zig` | Synthetic property/contract tests of `solve` (e.g. the `max_outer` DNC contract). No fixture dependency. |
| `tests/extreme_aspect_test.zig` | Rotation-invariance, coplanarity, near-degenerate edge-case tests on synthesized inputs. Also hits internal helpers (`acceptBUpdate`, `convexHull2d`) via filesystem imports for branches not reachable through `solve` for all inputs. |
| `tests/cases/cases.zig` | Comptime manifest over `tests/cases/zon/*.zon` — defines the `Case` schema and the `all` list. Exposed as the `cases` build module; imported by tests / bench / the `ex-cases` example. |
| `tests/cases/cases_test.zig` | Tests driven by the case manifest: cases.byName lookup, per-case outcome dispatch, Q/sigma shape invariants on np100. Lives next to `cases.zig` but is not part of the cases module compilation. |
| `tests/cases/zon/*.zon` | Per-case fixture: description + tags + points + expected outcome. |
| `tests/dggs_dnc_test.zig` | Regression tests pinning DGGS cells at finest resolution that DNC at default options. Currently FAIL; see `docs/dggs-dnc-investigation.md` for the diagnosis. |

To add a new test file: create `tests/<name>_test.zig`, then add
`_ = @import("<name>_test.zig");` to `tests/all.zig`. The test
binary picks it up automatically.

## Examples

Four single-file programs under `examples/`, each wired into
`build.zig` via `addExample`:

| Step | Source | Role |
| --- | --- | --- |
| `ex-basic` | `examples/basic.zig` | Minimum API call — solve + read AR + axis. |
| `ex-status` | `examples/status.zig` | Full `Outcome` switch with per-variant inspection. |
| `ex-cases` | `examples/cases.zig` | Runs a bundled case by name (`-- hex`) or iterates the whole manifest (`-- --all`). |
| `ex-bench` | `examples/bench.zig` | Per-case timing across a hand-picked subset. Forced to `.ReleaseFast` in `build.zig` regardless of the top-level optimize flag — Debug timings are noise. |

`addExample` accepts an optional optimize override (`null` inherits
the project-wide flag); only `ex-bench` uses it today. Examples
also receive pass-through args after `--`; only `ex-cases` reads
them today.
