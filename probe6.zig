//! Temporary probe 6: locate the wide-cap wall; n-dependence.
//! Run: zig run -O ReleaseFast probe6.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const Vec3 = sphar.Vec3;

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

fn capPoints(allocator: std.mem.Allocator, rng: std.Random, n: usize, cap_deg: f64) ![][3]f64 {
    const pts = try allocator.alloc([3]f64, n);
    const cos_max = @cos(deg(cap_deg));
    var R = sphar.Mat3.randomNormal(rng);
    R.orthonormalize();
    for (pts) |*p| {
        const z = cos_max + rng.float(f64) * (1.0 - cos_max);
        const phi = 2.0 * std.math.pi * rng.float(f64);
        const s = @sqrt(1.0 - z * z);
        const v = Vec3{ .m = .{ s * @cos(phi), s * @sin(phi), z } };
        p.* = R.apply(v).m;
    }
    return pts;
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Wall location: 10 seeds per width, n=200, count DNC.
    std.debug.print("n=200, max_outer=500: width -> #DNC/10 (median iters of converged)\n", .{});
    const widths = [_]f64{ 79, 80, 80.5, 81, 81.5, 82, 83 };
    for (widths) |wdeg| {
        var dnc: u32 = 0;
        var iters_buf: [10]u32 = undefined;
        var nconv: usize = 0;
        var seed: u64 = 1;
        while (seed <= 10) : (seed += 1) {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const pts = try capPoints(allocator, rng, 200, wdeg);
            defer allocator.free(pts);
            var o = try sphar.solve(allocator, pts, .{ .max_outer = 500 });
            defer o.deinit();
            switch (o) {
                .converged => |c| {
                    iters_buf[nconv] = c.outer_iters;
                    nconv += 1;
                },
                else => dnc += 1,
            }
        }
        std.mem.sort(u32, iters_buf[0..nconv], {}, std.sort.asc(u32));
        const med: u32 = if (nconv > 0) iters_buf[nconv / 2] else 0;
        std.debug.print("  w={d:5.1}  DNC {d:2}/10  median iters {d}\n", .{ wdeg, dnc, med });
    }

    // n-dependence at fixed width 85.
    std.debug.print("\nwidth=85, max_outer=500: n -> #DNC/10\n", .{});
    const ns = [_]usize{ 5, 10, 20, 50, 200 };
    for (ns) |n| {
        var dnc: u32 = 0;
        var seed: u64 = 1;
        while (seed <= 10) : (seed += 1) {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const pts = try capPoints(allocator, rng, n, 85);
            defer allocator.free(pts);
            var o = try sphar.solve(allocator, pts, .{ .max_outer = 500 });
            defer o.deinit();
            switch (o) {
                .converged => {},
                else => dnc += 1,
            }
        }
        std.debug.print("  n={d:4}  DNC {d:2}/10\n", .{ n, dnc });
    }
}
