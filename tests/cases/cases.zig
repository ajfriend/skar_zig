//! Static manifest of solver test cases.
//!
//! Each case lives in its own `cases/*.zon` file with shape:
//!   .{
//!     .description = "...",
//!     .tags = .{ "...", ... },
//!     .points = .{ .{x, y, z}, ... },
//!     .expected = .{ .converged = .{ .ar = 1.234 } }  // or .infeasible
//!   }
//!
//! Everything below is compiled into the binary at build time — no
//! filesystem reads at runtime. Adding a case = drop a `.zon` file +
//! append one line to `all`. The schema enforces shape at compile time;
//! the test loop in `src/tests/integration_test.zig` runs every entry,
//! so an unlisted case is the only way to escape coverage.

const std = @import("std");

pub const Expected = union(enum) {
    /// Solver should converge with this aspect ratio (matched within
    /// the integration test's tolerance, currently 1e-6).
    converged: struct { ar: f64 },
    /// Solver should detect infeasibility. Universal sanity checks
    /// (λ ≥ 0, ∑λ ≈ 1, ‖∑ λᵢ xᵢ‖ ≈ residual) live in the test loop;
    /// no per-case residual value is stored.
    infeasible,
};

pub const Case = struct {
    description: []const u8,
    tags: []const []const u8,
    points: []const [3]f64,
    expected: Expected,
};

pub const Entry = struct {
    name: []const u8,
    case: Case,
};

pub const all: []const Entry = &.{
    .{ .name = "dnc_small_wide", .case = @import("zon/dnc_small_wide.zon") },
    .{ .name = "h3_r12_equator", .case = @import("zon/h3_r12_equator.zon") },
    .{ .name = "h3_r12_midLat", .case = @import("zon/h3_r12_midLat.zon") },
    .{ .name = "h3_r12_pent", .case = @import("zon/h3_r12_pent.zon") },
    .{ .name = "h3_r12_ring10", .case = @import("zon/h3_r12_ring10.zon") },
    .{ .name = "h3_r15_equator", .case = @import("zon/h3_r15_equator.zon") },
    .{ .name = "h3_r15_midLat", .case = @import("zon/h3_r15_midLat.zon") },
    .{ .name = "h3_r15_pent", .case = @import("zon/h3_r15_pent.zon") },
    .{ .name = "h3_r15_ring10", .case = @import("zon/h3_r15_ring10.zon") },
    .{ .name = "h3_r5_equator", .case = @import("zon/h3_r5_equator.zon") },
    .{ .name = "h3_r5_midLat", .case = @import("zon/h3_r5_midLat.zon") },
    .{ .name = "h3_r5_pent", .case = @import("zon/h3_r5_pent.zon") },
    .{ .name = "h3_r5_ring10", .case = @import("zon/h3_r5_ring10.zon") },
    .{ .name = "h3_r9_equator", .case = @import("zon/h3_r9_equator.zon") },
    .{ .name = "h3_r9_midLat", .case = @import("zon/h3_r9_midLat.zon") },
    .{ .name = "h3_r9_pent", .case = @import("zon/h3_r9_pent.zon") },
    .{ .name = "h3_r9_ring10", .case = @import("zon/h3_r9_ring10.zon") },
    .{ .name = "h3_res05", .case = @import("zon/h3_res05.zon") },
    .{ .name = "h3_res09", .case = @import("zon/h3_res09.zon") },
    .{ .name = "h3_res12", .case = @import("zon/h3_res12.zon") },
    .{ .name = "h3_res15", .case = @import("zon/h3_res15.zon") },
    .{ .name = "ha_05", .case = @import("zon/ha_05.zon") },
    .{ .name = "ha_08", .case = @import("zon/ha_08.zon") },
    .{ .name = "ha_10", .case = @import("zon/ha_10.zon") },
    .{ .name = "ha_12", .case = @import("zon/ha_12.zon") },
    .{ .name = "ha_14", .case = @import("zon/ha_14.zon") },
    .{ .name = "hex", .case = @import("zon/hex.zon") },
    .{ .name = "ico_00", .case = @import("zon/ico_00.zon") },
    .{ .name = "ico_01", .case = @import("zon/ico_01.zon") },
    .{ .name = "ico_02", .case = @import("zon/ico_02.zon") },
    .{ .name = "ico_03", .case = @import("zon/ico_03.zon") },
    .{ .name = "ico_04", .case = @import("zon/ico_04.zon") },
    .{ .name = "ico_05", .case = @import("zon/ico_05.zon") },
    .{ .name = "ico_06", .case = @import("zon/ico_06.zon") },
    .{ .name = "ico_07", .case = @import("zon/ico_07.zon") },
    .{ .name = "ico_08", .case = @import("zon/ico_08.zon") },
    .{ .name = "ico_09", .case = @import("zon/ico_09.zon") },
    .{ .name = "ico_10", .case = @import("zon/ico_10.zon") },
    .{ .name = "ico_11", .case = @import("zon/ico_11.zon") },
    .{ .name = "ico_12", .case = @import("zon/ico_12.zon") },
    .{ .name = "ico_13", .case = @import("zon/ico_13.zon") },
    .{ .name = "ico_14", .case = @import("zon/ico_14.zon") },
    .{ .name = "ico_15", .case = @import("zon/ico_15.zon") },
    .{ .name = "ico_16", .case = @import("zon/ico_16.zon") },
    .{ .name = "ico_17", .case = @import("zon/ico_17.zon") },
    .{ .name = "ico_18", .case = @import("zon/ico_18.zon") },
    .{ .name = "ico_19", .case = @import("zon/ico_19.zon") },
    .{ .name = "infeas_antipodal", .case = @import("zon/infeas_antipodal.zon") },
    .{ .name = "near_collinear", .case = @import("zon/near_collinear.zon") },
    .{ .name = "np100", .case = @import("zon/np100.zon") },
    .{ .name = "np20", .case = @import("zon/np20.zon") },
    .{ .name = "np400", .case = @import("zon/np400.zon") },
    .{ .name = "oct_n0", .case = @import("zon/oct_n0.zon") },
    .{ .name = "oct_n1", .case = @import("zon/oct_n1.zon") },
    .{ .name = "oct_n2", .case = @import("zon/oct_n2.zon") },
    .{ .name = "oct_n3", .case = @import("zon/oct_n3.zon") },
    .{ .name = "oct_s0", .case = @import("zon/oct_s0.zon") },
    .{ .name = "oct_s1", .case = @import("zon/oct_s1.zon") },
    .{ .name = "oct_s2", .case = @import("zon/oct_s2.zon") },
    .{ .name = "oct_s3", .case = @import("zon/oct_s3.zon") },
    .{ .name = "wide_cap82", .case = @import("zon/wide_cap82.zon") },
    .{ .name = "wide_cap85", .case = @import("zon/wide_cap85.zon") },
    .{ .name = "wide_cap89", .case = @import("zon/wide_cap89.zon") },
};

/// Look up a case by name. Linear scan; the manifest is tiny.
pub fn byName(name: []const u8) ?Case {
    for (all) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.case;
    }
    return null;
}
