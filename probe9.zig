//! Temporary probe 9: barrier schedule (mu) vs Newton-step count.
//! Run: zig run -O ReleaseFast probe9.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const config = @import("src/config.zig");
const cases = @import("tests/cases/cases.zig");
const wide = @import("tests/wide_cap_cells.zig");

const Item = struct { name: []const u8, pts: []const [3]f64 };

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var items = std.ArrayList(Item){};
    defer items.deinit(allocator);
    try items.append(allocator, .{ .name = "cap85_s1", .pts = &wide.CAP85_S1 });
    try items.append(allocator, .{ .name = "cap89_s3", .pts = &wide.CAP89_S3 });
    for ([_][]const u8{ "hex", "h3_res09", "np400", "ha_14" }) |n| {
        try items.append(allocator, .{ .name = n, .pts = (cases.byName(n) orelse unreachable).points });
    }

    const mus = [_]f64{ 10, 30, 100, 300, 1000 };
    std.debug.print("{s:10}", .{"case"});
    for (mus) |m| std.debug.print("  mu={d:5.0}   ", .{m});
    std.debug.print("\n", .{});
    for (items.items) |item| {
        std.debug.print("{s:10}", .{item.name});
        for (mus) |m| {
            config.joint.probe_mu = m;
            var o = try sphar.solve(allocator, item.pts, .{ .method = .joint });
            defer o.deinit();
            switch (o) {
                .converged => |c| std.debug.print("  {d:3} {d:6.0}us", .{ c.outer_iters, 0.0 }),
                .did_not_converge => |p| std.debug.print("  DNC({d:3})  ", .{p.outer_iters}),
                .infeasible => {},
            }
        }
        config.joint.probe_mu = 0;
        std.debug.print("\n", .{});
    }

    // Timing at the best-looking mu on two representatives.
    std.debug.print("\ntiming (min of 50 reps):\n", .{});
    for ([_]f64{ 10, 100, 300 }) |m| {
        config.joint.probe_mu = m;
        for (items.items) |item| {
            var tmin: f64 = 1e30;
            for (0..50) |_| {
                const t0 = std.time.nanoTimestamp();
                var o = try sphar.solve(allocator, item.pts, .{ .method = .joint });
                const t1 = std.time.nanoTimestamp();
                o.deinit();
                const us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
                if (us < tmin) tmin = us;
            }
            std.debug.print("  mu={d:4.0} {s:10} {d:7.1}us\n", .{ m, item.name, tmin });
        }
        config.joint.probe_mu = 0;
    }
}
