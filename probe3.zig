//! Temporary probe 3: does a trust-region cap on the axis step fix wide caps?
//! Run: zig run -O ReleaseFast probe3.zig
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

fn statusChar(o: sphar.Outcome) u8 {
    return switch (o) {
        .converged => 'C',
        .did_not_converge => 'D',
        .infeasible => 'I',
    };
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Sweep width x step_cap. Report status + iters per (width, seed, cap).
    const widths = [_]f64{ 80, 82, 84, 85, 86, 87, 88, 89, 89.5 };
    const caps = [_]f64{ 0, 1.0, 0.5, 0.25, 0.1 };
    std.debug.print("status/iters at max_outer=500, 200 pts. cap=0 is current behavior.\n", .{});
    std.debug.print("width seed |", .{});
    for (caps) |c| std.debug.print("  cap={d:4.2}    |", .{c});
    std.debug.print("\n", .{});
    for (widths) |wdeg| {
        var seed: u64 = 1;
        while (seed <= 5) : (seed += 1) {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const pts = try capPoints(allocator, rng, 200, wdeg);
            defer allocator.free(pts);
            std.debug.print("{d:5.1} {d:4} |", .{ wdeg, seed });
            for (caps) |c| {
                skar.probe_step_cap = c;
                var o = try sphar.solve(allocator, pts, .{ .max_outer = 500 });
                defer o.deinit();
                const iters: u32 = switch (o) {
                    .converged => |cc| cc.outer_iters,
                    .did_not_converge => |p| p.outer_iters,
                    .infeasible => 0,
                };
                std.debug.print("  {c} it={d:4}   |", .{ statusChar(o), iters });
            }
            skar.probe_step_cap = 0;
            std.debug.print("\n", .{});
        }
    }

    // Sanity on easy cases with the cap enabled: hex-like + ha_05-like should not slow down.
    std.debug.print("\nsanity: 200pts r=30 cap sweep\n", .{});
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const pts = try capPoints(allocator, rng, 200, 30);
    defer allocator.free(pts);
    for (caps) |c| {
        skar.probe_step_cap = c;
        var o = try sphar.solve(allocator, pts, .{ .max_outer = 500 });
        defer o.deinit();
        const iters: u32 = switch (o) {
            .converged => |cc| cc.outer_iters,
            .did_not_converge => |p| p.outer_iters,
            .infeasible => 0,
        };
        std.debug.print("cap={d:4.2}  {c} it={d}\n", .{ c, statusChar(o), iters });
    }
    skar.probe_step_cap = 0;
}
