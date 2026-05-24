_:
    just --list

build:
    zig build -Doptimize=ReleaseFast

test:
    zig build test --summary all

bench: build
    ./zig-out/bin/skar-bench

clean:
    rm -rf zig-out .zig-cache
