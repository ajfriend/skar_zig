//! Temporary probe 10: smoke-test the reduced TR-BFGS solver.
//! Run: zig run -O ReleaseFast probe10.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const cases = @import("tests/cases/cases.zig");
const wide = @import("tests/wide_cap_cells.zig");

fn report(label: []const u8, outcome: sphar.Outcome, pts: []const [3]f64) void {
    switch (outcome) {
        .converged => |c| {
            const viol = sphar.checkFeasibility(c, pts);
            std.debug.print("{s}  converged  it={d:4}  gap={e:10.3}  AR={d:.6}  viol={e:9.2}\n", .{ label, c.outer_iters, c.gap, c.aspectRatio(), viol });
        },
        .did_not_converge => |p| std.debug.print("{s}  DNC        it={d:4}  gap={e:10.3}  AR~{d:.6}\n", .{ label, p.outer_iters, p.gap, p.sigma[2] / p.sigma[1] }),
        .infeasible => |i| std.debug.print("{s}  infeasible residual={e:.3}\n", .{ label, i.residual }),
    }
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var buf: [96]u8 = undefined;

    std.debug.print("== wide-cap fixtures, method=reduced (Clarabel refs: 1.159634 / 1.269181 / 1.542028) ==\n", .{});
    const fixtures = [_]struct { name: []const u8, pts: []const [3]f64 }{
        .{ .name = "cap82_s1", .pts = &wide.CAP82_S1 },
        .{ .name = "cap85_s1", .pts = &wide.CAP85_S1 },
        .{ .name = "cap89_s3", .pts = &wide.CAP89_S3 },
    };
    for (fixtures) |f| {
        var o = try sphar.solve(allocator, f.pts, .{ .method = .reduced });
        defer o.deinit();
        const label = try std.fmt.bufPrint(&buf, "{s:16}", .{f.name});
        report(label, o, f.pts);
    }

    std.debug.print("\n== bundled cases incl. extreme-kappa cells, fast vs reduced ==\n", .{});
    const names = [_][]const u8{
        "hex",           "h3_res09",      "h3_res15",    "np20",   "np400",
        "ha_05",         "ha_14",         "dnc_small_wide",
        "h3_r12_ring10", "h3_r15_midLat", "h3_r15_pent", "h3_r15_ring10",
    };
    for (names) |name| {
        const case = cases.byName(name) orelse continue;
        var of = try sphar.solve(allocator, case.points, .{});
        defer of.deinit();
        var orr = try sphar.solve(allocator, case.points, .{ .method = .reduced });
        defer orr.deinit();
        const lf = try std.fmt.bufPrint(&buf, "{s:16} fast   ", .{name});
        report(lf, of, case.points);
        const lr = try std.fmt.bufPrint(&buf, "{s:16} reduced", .{name});
        report(lr, orr, case.points);
    }

    std.debug.print("\n== timing (min of 100 reps) fast vs reduced ==\n", .{});
    const tnames = [_][]const u8{ "hex", "h3_res09", "np400", "ha_14" };
    for (tnames) |name| {
        const case = cases.byName(name) orelse continue;
        for ([_]sphar.Method{ .fast, .reduced }) |m| {
            var tmin: f64 = 1e30;
            for (0..100) |_| {
                const t0 = std.time.nanoTimestamp();
                var o = try sphar.solve(allocator, case.points, .{ .method = m });
                const t1 = std.time.nanoTimestamp();
                o.deinit();
                const us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
                if (us < tmin) tmin = us;
            }
            std.debug.print("  {s:10} {s:8} {d:7.1}us\n", .{ name, @tagName(m), tmin });
        }
    }
    for (fixtures) |f| {
        for ([_]sphar.Method{ .joint, .reduced }) |m| {
            var tmin: f64 = 1e30;
            for (0..50) |_| {
                const t0 = std.time.nanoTimestamp();
                var o = try sphar.solve(allocator, f.pts, .{ .method = m });
                const t1 = std.time.nanoTimestamp();
                o.deinit();
                const us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
                if (us < tmin) tmin = us;
            }
            std.debug.print("  {s:10} {s:8} {d:7.1}us\n", .{ f.name, @tagName(m), tmin });
        }
    }
}
