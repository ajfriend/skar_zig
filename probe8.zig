//! Temporary probe 8: joint-path stall diagnosis on finest-res cells.
//! Run: zig run -O ReleaseFast probe8.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const cases = @import("tests/cases/cases.zig");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const names = [_][]const u8{ "h3_r12_ring10", "h3_r15_midLat", "h3_r15_pent", "h3_r15_ring10", "h3_res15" };
    for (names) |name| {
        const case = cases.byName(name) orelse continue;
        var o = try sphar.solve(allocator, case.points, .{ .method = .joint });
        defer o.deinit();
        switch (o) {
            .converged => |c| std.debug.print("{s:16} converged it={d:4} gap={e:10.3} AR={d:.6} sigma=({e:.2},{e:.2},{e:.2})\n", .{ name, c.outer_iters, c.gap, c.aspectRatio(), c.sigma[0], c.sigma[1], c.sigma[2] }),
            .did_not_converge => |p| std.debug.print("{s:16} DNC       it={d:4} gap={e:10.3} AR~{d:.6} sigma=({e:.2},{e:.2},{e:.2})\n", .{ name, p.outer_iters, p.gap, p.sigma[2] / p.sigma[1], p.sigma[0], p.sigma[1], p.sigma[2] }),
            .infeasible => {},
        }
    }
}
