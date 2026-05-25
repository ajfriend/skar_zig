_:
    just --list

# Build, run tests under kcov, fail if line coverage isn't 100%.
test:
    zig build install-test
    rm -rf coverage
    kcov --include-pattern=src/,tests/ coverage zig-out/bin/skar-test
    @n=$(ls -1d coverage/skar-test.*/ 2>/dev/null | wc -l | tr -d ' '); \
        if [ "$n" != "1" ]; then echo "expected exactly 1 coverage/skar-test.*/ dir, got $n"; exit 1; fi
    @jq -r '"skar coverage: \(.percent_covered)%"' coverage/skar-test.*/coverage.json
    @jq -e '(.percent_covered | tonumber) >= 100' coverage/skar-test.*/coverage.json > /dev/null

# Build the library and CLI / bench binaries (optimized).
build:
    zig build -Doptimize=ReleaseFast

# Run `just test` and print where the HTML coverage report landed.
coverage: test
    @echo "open coverage/skar-test/index.html"

# Run the minimal usage example (examples/basic.zig).
example:
    zig build example

# Run the full status-handling example (examples/status.zig).
example-status:
    zig build example-status

# Run the benchmark suite (uses the release-built binary from `just build`).
bench: build
    ./zig-out/bin/skar-bench

# Remove build artifacts and coverage output.
clean:
    rm -rf zig-out .zig-cache coverage
