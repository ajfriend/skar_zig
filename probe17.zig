//! Temporary probe 17: trace h3_res09 + ha_08 reduced regressions.
const std = @import("std");
const sphar = @import("src/root.zig");
const skar = @import("src/skar.zig");
const reduced = @import("src/reduced.zig");
const cases = @import("tests/cases/cases.zig");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    for ([_][]const u8{ "h3_res09", "ha_08" }) |name| {
        const case = cases.byName(name) orelse continue;
        std.debug.print("=== {s} ===\n", .{name});
        reduced.probe_trace = true;
        skar.probe_gap_trace = true;
        var o = try sphar.solve(allocator, case.points, .{ .method = .reduced, .max_outer = 60 });
        reduced.probe_trace = false;
        skar.probe_gap_trace = false;
        switch (o) {
            .converged => |c| std.debug.print("  => converged it={d} gap={e:.3}\n", .{ c.outer_iters, c.gap }),
            .did_not_converge => |p| std.debug.print("  => DNC it={d} gap={e:.3}\n", .{ p.outer_iters, p.gap }),
            .infeasible => {},
        }
        o.deinit();
    }
}
