//! Temporary probe 11: find + diagnose the one reduced DNC (n=20, 88deg).
//! Run: zig run -O ReleaseFast probe11.zig
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

    var seed: u64 = 1;
    while (seed <= 10) : (seed += 1) {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const pts = try capPoints(allocator, rng, 20, 88);
        defer allocator.free(pts);
        var o = try sphar.solve(allocator, pts, .{ .method = .reduced });
        defer o.deinit();
        switch (o) {
            .converged => |c| std.debug.print("seed={d} converged it={d:3} gap={e:10.3} AR={d:.6}\n", .{ seed, c.outer_iters, c.gap, c.aspectRatio() }),
            .did_not_converge => |p| {
                std.debug.print("seed={d} DNC       it={d:3} gap={e:10.3} AR~{d:.6}\n", .{ seed, p.outer_iters, p.gap, p.sigma[2] / p.sigma[1] });
                // What do the other methods say?
                var oj = try sphar.solve(allocator, pts, .{ .method = .joint });
                defer oj.deinit();
                switch (oj) {
                    .converged => |c| std.debug.print("        joint: converged gap={e:10.3} AR={d:.6}\n", .{ c.gap, c.aspectRatio() }),
                    else => std.debug.print("        joint: not converged\n", .{}),
                }
                var of = try sphar.solve(allocator, pts, .{ .max_outer = 2000 });
                defer of.deinit();
                switch (of) {
                    .converged => |c| std.debug.print("        fast(2000): converged it={d} AR={d:.6}\n", .{ c.outer_iters, c.aspectRatio() }),
                    else => std.debug.print("        fast(2000): not converged\n", .{}),
                }
                // reduced with a bigger outer budget?
                var o2 = try sphar.solve(allocator, pts, .{ .method = .reduced, .max_outer = 2000 });
                defer o2.deinit();
                switch (o2) {
                    .converged => |c| std.debug.print("        reduced(2000): converged it={d} gap={e:10.3}\n", .{ c.outer_iters, c.gap }),
                    .did_not_converge => |p2| std.debug.print("        reduced(2000): DNC it={d} gap={e:10.3}\n", .{ p2.outer_iters, p2.gap }),
                    else => {},
                }
            },
            .infeasible => {},
        }
    }
}
