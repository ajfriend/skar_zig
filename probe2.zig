//! Temporary probe 2: wide-cap failure diagnosis.
//! Run: zig run -O ReleaseFast probe2.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const skar = @import("src/skar.zig");
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

fn report(label: []const u8, outcome: sphar.Outcome) void {
    switch (outcome) {
        .converged => |c| std.debug.print("{s}  converged  iters={d:4}  gap={e:10.3}  AR={d:.4}\n", .{ label, c.outer_iters, c.gap, c.aspectRatio() }),
        .did_not_converge => |p| std.debug.print("{s}  DNC        iters={d:4}  gap={e:10.3}  AR~{d:.4}\n", .{ label, p.outer_iters, p.gap, p.sigma[2] / p.sigma[1] }),
        .infeasible => |i| std.debug.print("{s}  infeasible residual={e:.3}\n", .{ label, i.residual }),
    }
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var buf: [64]u8 = undefined;

    // A) Trace one failing case: r=85 seed=1, first 30 iters.
    std.debug.print("== trace: cap r=85 seed=1, default init ==\n", .{});
    {
        var prng = std.Random.DefaultPrng.init(1);
        const rng = prng.random();
        const pts = try capPoints(allocator, rng, 200, 85);
        defer allocator.free(pts);
        skar.probe_trace = true;
        var o = try sphar.solve(allocator, pts, .{ .max_outer = 30 });
        skar.probe_trace = false;
        o.deinit();

        std.debug.print("\n== trace: same case, centroid init ==\n", .{});
        skar.probe_centroid_init = true;
        skar.probe_trace = true;
        var o2 = try sphar.solve(allocator, pts, .{ .max_outer = 30 });
        skar.probe_trace = false;
        skar.probe_centroid_init = false;
        o2.deinit();
    }

    // B) Sweep with centroid init: does the wall move?
    std.debug.print("\n== sweep widths 80-89, centroid init vs default, max_outer=500 ==\n", .{});
    const widths = [_]f64{ 80, 82, 84, 85, 86, 87, 88, 89 };
    for (widths) |wdeg| {
        var seed: u64 = 1;
        while (seed <= 5) : (seed += 1) {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const pts = try capPoints(allocator, rng, 200, wdeg);
            defer allocator.free(pts);
            var o1 = try sphar.solve(allocator, pts, .{ .max_outer = 500 });
            defer o1.deinit();
            skar.probe_centroid_init = true;
            var o2 = try sphar.solve(allocator, pts, .{ .max_outer = 500 });
            skar.probe_centroid_init = false;
            defer o2.deinit();
            const l1 = try std.fmt.bufPrint(&buf, "r={d:2.0} s={d} default ", .{ wdeg, seed });
            report(l1, o1);
            const l2 = try std.fmt.bufPrint(&buf, "         centroid", .{});
            report(l2, o2);
        }
    }
}
