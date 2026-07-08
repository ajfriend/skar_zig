//! Temporary probe 7: smoke-test the joint barrier solver.
//! Run: zig run -O ReleaseFast probe7.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const cases = @import("tests/cases/cases.zig");
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

fn report(label: []const u8, outcome: sphar.Outcome, pts: []const [3]f64) void {
    switch (outcome) {
        .converged => |c| {
            const viol = sphar.checkFeasibility(c, pts);
            std.debug.print("{s}  converged  it={d:4}  gap={e:10.3}  AR={d:.6}  viol={e:9.2}\n", .{ label, c.outer_iters, c.gap, c.aspectRatio(), viol });
        },
        .did_not_converge => |p| std.debug.print("{s}  DNC        it={d:4}  gap={e:10.3}  AR~{d:.4}\n", .{ label, p.outer_iters, p.gap, p.sigma[2] / p.sigma[1] }),
        .infeasible => |i| std.debug.print("{s}  infeasible residual={e:.3}\n", .{ label, i.residual }),
    }
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var buf: [96]u8 = undefined;

    // Wide caps that DNC on the fast path. Clarabel refs:
    // cap82_s1 AR=1.159634, cap85_s1 AR=1.269181, cap89_s3 AR=1.542028.
    const wide = [_]struct { w: f64, seed: u64, ref: f64 }{
        .{ .w = 82, .seed = 1, .ref = 1.159634 },
        .{ .w = 85, .seed = 1, .ref = 1.269181 },
        .{ .w = 89, .seed = 3, .ref = 1.542028 },
        .{ .w = 89.5, .seed = 5, .ref = 0 },
    };
    std.debug.print("== wide caps (200 pts), method=joint ==\n", .{});
    for (wide) |c| {
        var prng = std.Random.DefaultPrng.init(c.seed);
        const rng = prng.random();
        const pts = try capPoints(allocator, rng, 200, c.w);
        defer allocator.free(pts);
        var o = try sphar.solve(allocator, pts, .{ .method = .joint });
        defer o.deinit();
        const label = try std.fmt.bufPrint(&buf, "cap{d:4.1}_s{d} (ref AR {d:.6})", .{ c.w, c.seed, c.ref });
        report(label, o, pts);
        var oa = try sphar.solve(allocator, pts, .{ .method = .auto });
        defer oa.deinit();
        const label2 = try std.fmt.bufPrint(&buf, "            auto           ", .{});
        report(label2, oa, pts);
    }

    // Easy / bundled cases, joint vs fast AR agreement.
    std.debug.print("\n== bundled cases, method=joint vs fast ==\n", .{});
    const names = [_][]const u8{ "hex", "h3_res09", "np20", "np400", "ha_05", "ha_14", "dnc_small_wide" };
    for (names) |name| {
        const case = cases.byName(name) orelse continue;
        var of = try sphar.solve(allocator, case.points, .{});
        defer of.deinit();
        var oj = try sphar.solve(allocator, case.points, .{ .method = .joint });
        defer oj.deinit();
        const lf = try std.fmt.bufPrint(&buf, "{s:16} fast ", .{name});
        report(lf, of, case.points);
        const lj = try std.fmt.bufPrint(&buf, "{s:16} joint", .{name});
        report(lj, oj, case.points);
    }
}
