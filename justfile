_:
    just --list

# Fast test loop. Skips long-running randomized stress tests
# (e.g. cap_test). No coverage gate. Sub-second; use this while
# iterating. Run `just test-slow` before committing.
test:
    zig build install-test
    ./zig-out/bin/skar-test

# Full test suite + 100% line coverage gate. Builds with -Dslow=true
# so the randomized stress tests run, then measures coverage under
# kcov. Slower (~10s) — the pre-commit / CI check.
test-slow:
    zig build install-test -Dslow=true
    rm -rf coverage
    kcov --include-pattern=src/,tests/ coverage zig-out/bin/skar-test
    @n=$(ls -1d coverage/skar-test.*/ 2>/dev/null | wc -l | tr -d ' '); \
        if [ "$n" != "1" ]; then echo "expected exactly 1 coverage/skar-test.*/ dir, got $n"; exit 1; fi
    @jq -r '"skar coverage: \(.percent_covered)%"' coverage/skar-test.*/coverage.json
    @jq -e '(.percent_covered | tonumber) >= 100' coverage/skar-test.*/coverage.json > /dev/null

# Build the library (optimized).
build:
    zig build -Doptimize=ReleaseFast

# Run `just test-slow` and print where the HTML coverage report landed.
coverage: test-slow
    @echo "open coverage/skar-test/index.html"

# Run the minimal usage example (examples/basic.zig).
ex-basic:
    zig build ex-basic

# Run the full status-handling example (examples/status.zig).
ex-status:
    zig build ex-status

# Run the per-case timing bench (examples/bench.zig, forced ReleaseFast by build.zig).
bench:
    zig build ex-bench

# Generate DGGS cell boundary data (H3/S2/A5) under scripts/dggs/data/.
# Output is gitignored; regenerate any time via this target.
# Edit constants (N, SEED, resolutions) at the top of gen_cells.py.
dggs-gen:
    uv run scripts/dggs/gen_cells.py

# Run skar over every generated DGGS cell; writes scripts/dggs/data/aspect.json.
# Depends on `just dggs-gen` having run first.
dggs-aspect:
    zig build dggs-aspect

# Remove build artifacts, coverage output, and generated DGGS data.
clean:
    rm -rf zig-out .zig-cache coverage scripts/dggs/data
