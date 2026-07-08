//! Temporary probe 13: reduced on the a5_res0 dense 320-pt cells.
const std = @import("std");
const sphar = @import("src/root.zig");
const a5 = @import("tests/a5_res0_cells_dense.zig");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var max_it: u32 = 0;
    var n_conv: u32 = 0;
    var t_total: f64 = 0;
    for (a5.A5_RES0_CELLS, 0..) |pts, i| {
        const t0 = std.time.nanoTimestamp();
        var o = try sphar.solve(allocator, pts, .{ .method = .reduced });
        const t1 = std.time.nanoTimestamp();
        defer o.deinit();
        t_total += @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
        switch (o) {
            .converged => |c| {
                n_conv += 1;
                if (c.outer_iters > max_it) max_it = c.outer_iters;
                var of = try sphar.solve(allocator, pts, .{});
                defer of.deinit();
                const ar_f = of.converged.aspectRatio();
                if (i < 3) std.debug.print("cell {d}: it={d} gap={e:.2} AR={d:.8} (fast AR={d:.8})\n", .{ i, c.outer_iters, c.gap, c.aspectRatio(), ar_f });
            },
            else => std.debug.print("cell {d}: NOT converged\n", .{i}),
        }
    }
    std.debug.print("reduced: {d}/12 converged, max iters {d}, total {d:.0}us ({d:.0}us/cell)\n", .{ n_conv, max_it, t_total, t_total / 12.0 });
}
