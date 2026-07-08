//! Temporary probe 12: trace reduced on the n=20 w=88 seed=6 stall.
const std = @import("std");
const sphar = @import("src/root.zig");
const reduced = @import("src/reduced.zig");
const Vec3 = sphar.Vec3;

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(6);
    const rng = prng.random();
    const pts = try allocator.alloc([3]f64, 20);
    defer allocator.free(pts);
    const cos_max = @cos(deg(88));
    var R = sphar.Mat3.randomNormal(rng);
    R.orthonormalize();
    for (pts) |*p| {
        const z = cos_max + rng.float(f64) * (1.0 - cos_max);
        const phi = 2.0 * std.math.pi * rng.float(f64);
        const s = @sqrt(1.0 - z * z);
        const v = Vec3{ .m = .{ s * @cos(phi), s * @sin(phi), z } };
        p.* = R.apply(v).m;
    }
    reduced.probe_trace = true;
    var o = try sphar.solve(allocator, pts, .{ .method = .reduced, .max_outer = 40 });
    defer o.deinit();
    reduced.probe_trace = false;
}
