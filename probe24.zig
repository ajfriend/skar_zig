//! Temporary probe 24: pin values for reduced canaries (a5_res0 sparse).
const std = @import("std");
const sphar = @import("src/root.zig");

const A5_RES0_CORNERS = [_][3]f64{
    .{ -0.3809559340728538, -0.47044139975172183, 0.7959632313708467 },
    .{ 0.3296945010323943, -0.5076850109021148, 0.7959632313708467 },
    .{ 0.5847183416148108, 0.1566748074353539, 0.7959632313708467 },
    .{ 0.03168130793103096, 0.6045153670780082, 0.7959632313708467 },
    .{ -0.5651382165053821, 0.2169362361404745, 0.7959632313708467 },
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var o = try sphar.solve(allocator, &A5_RES0_CORNERS, .{ .method = .reduced });
    defer o.deinit();
    switch (o) {
        .converged => |c| std.debug.print("a5_res0 sparse reduced: it={d} gap={e:.3} AR={d:.8}\n", .{ c.outer_iters, c.gap, c.aspectRatio() }),
        else => std.debug.print("not converged\n", .{}),
    }
}
