//! Temporary probe: characterize slow-convergence regimes.
//! Run: zig run -O ReleaseFast probe_dnc.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const Vec3 = sphar.Vec3;

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

/// n random points in a spherical cap of angular radius `cap_deg`,
/// randomly rotated. Same construction style as the ha_* cases.
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

fn arcPoints(span_deg: f64, eps: f64) [3][3]f64 {
    const half = deg(span_deg / 2.0);
    var pts: [3][3]f64 = .{
        .{ @cos(-half), @sin(-half), eps },
        .{ 1.0, 0.0, -eps },
        .{ @cos(half), @sin(half), eps },
    };
    for (&pts) |*p| {
        const nn = @sqrt(p.*[0] * p.*[0] + p.*[1] * p.*[1] + p.*[2] * p.*[2]);
        p.*[0] /= nn;
        p.*[1] /= nn;
        p.*[2] /= nn;
    }
    return pts;
}

fn stretchedCap(n: usize, half_angle: f64, stretch: f64, out: []Vec3) void {
    const s = @sin(half_angle);
    const c = @cos(half_angle);
    const n_f = @as(f64, @floatFromInt(n));
    for (out, 0..) |*p, i| {
        const phi = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / n_f;
        const v = Vec3{ .m = .{ stretch * s * @cos(phi), s * @sin(phi), c } };
        p.* = v.normalize();
    }
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

    // ---- Probe 1: wide random caps, 200 pts, sweep width. 5 seeds each.
    // max_outer=100 (default) and 2000 (does it eventually converge?)
    std.debug.print("== probe 1: 200 random pts in cap of angular radius r (deg), default max_outer=100 vs 2000 ==\n", .{});
    const widths = [_]f64{ 60, 70, 75, 80, 82, 84, 85, 86, 87, 88, 89 };
    for (widths) |wdeg| {
        var seed: u64 = 1;
        while (seed <= 5) : (seed += 1) {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const pts = try capPoints(allocator, rng, 200, wdeg);
            defer allocator.free(pts);
            var o1 = try sphar.solve(allocator, pts, .{});
            defer o1.deinit();
            var o2 = try sphar.solve(allocator, pts, .{ .max_outer = 2000 });
            defer o2.deinit();
            const label = try std.fmt.bufPrint(&buf, "cap r={d:2.0} seed={d}", .{ wdeg, seed });
            report(label, o1);
            const label2 = try std.fmt.bufPrint(&buf, "           mo=2000", .{});
            report(label2, o2);
        }
    }

    // ---- Probe 2: near-antipodal arc plateau, sweep eps (AR) at span 174.
    std.debug.print("\n== probe 2: arc span=174deg, 3 pts, sweep eps; max_outer=5000 ==\n", .{});
    const epss = [_]f64{ 0.05, 0.025, 0.018, 0.015, 0.012, 0.010, 0.008, 0.005 };
    for (epss) |eps| {
        var pts = arcPoints(174.0, eps);
        var o = try sphar.solve(allocator, pts[0..], .{ .max_outer = 5000, .coplanarity_tol = 1e-12 });
        defer o.deinit();
        const label = try std.fmt.bufPrint(&buf, "eps={d:.3}", .{eps});
        report(label, o);
    }

    // ---- Probe 3: stretched cap, iterations vs stretch (AR) at half-angle 15deg.
    std.debug.print("\n== probe 3: stretched cap (16 pts, half-angle 15deg), iters vs stretch; max_outer=5000 ==\n", .{});
    const stretches = [_]f64{ 2, 5, 10, 20, 50, 100, 200, 500, 1000 };
    for (stretches) |st| {
        var out: [16]Vec3 = undefined;
        stretchedCap(16, deg(15.0), st, &out);
        const pts: [][3]f64 = @ptrCast(out[0..]);
        var o = try sphar.solve(allocator, pts, .{ .max_outer = 5000 });
        defer o.deinit();
        const label = try std.fmt.bufPrint(&buf, "stretch={d:6.0}", .{st});
        report(label, o);
    }

    // ---- Probe 4: same stretched cap but wider (half-angle 45deg) — width x AR interaction.
    std.debug.print("\n== probe 4: stretched cap (16 pts, half-angle 45deg), iters vs stretch; max_outer=5000 ==\n", .{});
    const stretches2 = [_]f64{ 2, 5, 10, 20, 50 };
    for (stretches2) |st| {
        var out: [16]Vec3 = undefined;
        stretchedCap(16, deg(45.0), st, &out);
        const pts: [][3]f64 = @ptrCast(out[0..]);
        var o = try sphar.solve(allocator, pts, .{ .max_outer = 5000 });
        defer o.deinit();
        const label = try std.fmt.bufPrint(&buf, "stretch={d:6.0}", .{st});
        report(label, o);
    }
}
