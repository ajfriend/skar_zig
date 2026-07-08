//! Temporary probe 23: trace the H3 r9 CANARY cell under reduced.
const std = @import("std");
const sphar = @import("src/root.zig");
const reduced = @import("src/reduced.zig");

const H3_R9_CELL = [_][3]f64{
    .{ -0.8586175701975843, 0.28761239723198995, -0.42432885490673883 },
    .{ -0.8586271933201559, 0.28762660191847433, -0.42429975342908594 },
    .{ -0.8586197375801148, 0.2876590246563569, -0.42429286085392487 },
    .{ -0.8586026585975493, 0.2876772430738286, -0.42431506980858175 },
    .{ -0.8585930353179254, 0.2876630384891841, -0.42434417162336724 },
    .{ -0.8586004911779209, 0.28763061538522544, -0.42435106414636176 },
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    reduced.probe_trace = true;
    var o = try sphar.solve(allocator, &H3_R9_CELL, .{ .method = .reduced });
    reduced.probe_trace = false;
    switch (o) {
        .converged => |c| std.debug.print("=> converged it={d} gap={e:.3}\n", .{ c.outer_iters, c.gap }),
        else => std.debug.print("=> not converged\n", .{}),
    }
    o.deinit();
}
