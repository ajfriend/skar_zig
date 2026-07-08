//! Temporary probe 18: trace cap82 regression after round-0 oracle change.
const std = @import("std");
const sphar = @import("src/root.zig");
const reduced = @import("src/reduced.zig");
const wide = @import("tests/wide_cap_cells.zig");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    reduced.probe_trace = true;
    var o = try sphar.solve(allocator, &wide.CAP82_S1, .{ .method = .reduced, .max_outer = 25 });
    reduced.probe_trace = false;
    switch (o) {
        .converged => |c| std.debug.print("=> converged it={d} gap={e:.3}\n", .{ c.outer_iters, c.gap }),
        .did_not_converge => |p| std.debug.print("=> DNC it={d} gap={e:.3}\n", .{ p.outer_iters, p.gap }),
        .infeasible => {},
    }
    o.deinit();
}
