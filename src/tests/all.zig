//! Aggregator: pulls every `*_test.zig` in this directory into a
//! single compilation unit. Imported by `root.zig`'s `test {}` block
//! so `zig build test` (and therefore the kcov-based coverage gate)
//! discovers them all.

comptime {
    _ = @import("integration_test.zig");
    _ = @import("extreme_aspect_test.zig");
}
