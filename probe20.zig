//! Temporary probe 20: SO(3) rotation stress for the reduced path
//! (validation-ledger item). Each geometry is solved canonically, then
//! under N random rotations; assert convergence, certified gap ≤ tol,
//! and AR rotation-invariance. Watch for the two catalogued failure
//! shapes: model corruption masquerading as convergence, and
//! oracle-consistency stalls.
//!   zig run -O ReleaseFast probe20.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const wide = @import("tests/wide_cap_cells.zig");
const Vec3 = sphar.Vec3;

const N_ROT: u32 = 20;
const GAP_TOL: f64 = 1e-6;
const AR_REL_TOL: f64 = 1e-4;

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

fn normalizeAll(pts: [][3]f64) void {
    for (pts) |*p| {
        const n = @sqrt(p.*[0] * p.*[0] + p.*[1] * p.*[1] + p.*[2] * p.*[2]);
        p.*[0] /= n;
        p.*[1] /= n;
        p.*[2] /= n;
    }
}

fn arcPoints(span_deg: f64, eps: f64, out: *[3][3]f64) void {
    const half = deg(span_deg / 2.0);
    out.* = .{
        .{ @cos(-half), @sin(-half), eps },
        .{ 1.0, 0.0, -eps },
        .{ @cos(half), @sin(half), eps },
    };
    normalizeAll(out);
}

fn patchPoints(L: f64, eps: f64, out: *[3][3]f64) void {
    out.* = .{
        .{ L, 0.0, 1.0 },
        .{ -L, 0.0, 1.0 },
        .{ 0.0, eps, 1.0 },
    };
    normalizeAll(out);
}

fn stretchedCap(n: usize, half_angle: f64, stretch: f64, out: [][3]f64) void {
    const s = @sin(half_angle);
    const c = @cos(half_angle);
    const n_f = @as(f64, @floatFromInt(n));
    for (out, 0..) |*p, i| {
        const phi = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / n_f;
        p.* = .{ stretch * s * @cos(phi), s * @sin(phi), c };
    }
    normalizeAll(out);
}

const Case = struct { name: []const u8, pts: []const [3]f64 };

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arc174: [3][3]f64 = undefined;
    arcPoints(174.0, 0.025, &arc174);
    var arc90: [3][3]f64 = undefined;
    arcPoints(90.0, 0.005, &arc90);
    var patch: [3][3]f64 = undefined;
    patchPoints(0.1, 1e-5, &patch);
    var scap15: [16][3]f64 = undefined;
    stretchedCap(16, deg(15.0), 5.0, &scap15);
    var scap45: [16][3]f64 = undefined;
    stretchedCap(16, deg(45.0), 20.0, &scap45);

    const cases = [_]Case{
        .{ .name = "cap82_s1 (wide)", .pts = &wide.CAP82_S1 },
        .{ .name = "cap85_s1 (wide)", .pts = &wide.CAP85_S1 },
        .{ .name = "cap89_s3 (wide)", .pts = &wide.CAP89_S3 },
        .{ .name = "arc174 AR~63", .pts = &arc174 },
        .{ .name = "arc90 AR~143", .pts = &arc90 },
        .{ .name = "patch AR~17320", .pts = &patch },
        .{ .name = "scap15 AR=5", .pts = &scap15 },
        .{ .name = "scap45 AR=20", .pts = &scap45 },
    };

    var total_fail: u32 = 0;
    for (cases) |case| {
        // Canonical solve for the reference AR.
        var canon = try sphar.solve(allocator, case.pts, .{ .method = .reduced, .coplanarity_tol = 1e-12 });
        defer canon.deinit();
        if (canon != .converged) {
            std.debug.print("{s:18}  CANONICAL NOT CONVERGED\n", .{case.name});
            total_fail += 1;
            continue;
        }
        const canon_ar = canon.converged.aspectRatio();

        var prng = std.Random.DefaultPrng.init(0xCA7);
        const rng = prng.random();
        var dnc: u32 = 0;
        var bad_gap: u32 = 0;
        var worst_drift: f64 = 0;
        var max_iters: u32 = 0;
        const rot_pts = try allocator.alloc([3]f64, case.pts.len);
        defer allocator.free(rot_pts);
        var k: u32 = 0;
        while (k < N_ROT) : (k += 1) {
            var R = sphar.Mat3.randomNormal(rng);
            R.orthonormalize();
            for (case.pts, 0..) |p, i| rot_pts[i] = R.apply(.{ .m = p }).m;
            var o = try sphar.solve(allocator, rot_pts, .{ .method = .reduced, .coplanarity_tol = 1e-12 });
            defer o.deinit();
            switch (o) {
                .converged => |c| {
                    if (@abs(c.gap) > GAP_TOL) bad_gap += 1;
                    const drift = @abs(c.aspectRatio() - canon_ar) / canon_ar;
                    if (drift > worst_drift) worst_drift = drift;
                    if (c.outer_iters > max_iters) max_iters = c.outer_iters;
                },
                else => dnc += 1,
            }
        }
        const failed = dnc > 0 or bad_gap > 0 or worst_drift > AR_REL_TOL;
        if (failed) total_fail += 1;
        std.debug.print("{s:18}  AR={d:10.4}  DNC {d:2}/{d}  badgap {d}  maxIters {d:3}  worstDrift {e:.2}  {s}\n", .{
            case.name, canon_ar, dnc, N_ROT, bad_gap, max_iters, worst_drift,
            if (failed) "FAIL" else "ok",
        });
    }
    std.debug.print("\n{s}: {d} case(s) failed\n", .{ if (total_fail == 0) "PASS" else "FAIL", total_fail });
}
