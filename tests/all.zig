//! Aggregator: pulls every test file into a single compilation
//! unit. Imported by `test_root.zig` so `zig build test` (and
//! therefore the kcov-based coverage gate) discovers them all.

comptime {
    _ = @import("cases/cases_test.zig");
    _ = @import("solver_test.zig");
    _ = @import("extreme_aspect_test.zig");
    _ = @import("cap_test.zig");
    _ = @import("stretched_cap_test.zig");
    _ = @import("linalg_test.zig");
    _ = @import("dggs_dnc_test.zig");
}
