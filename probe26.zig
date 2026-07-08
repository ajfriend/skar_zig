//! Temporary probe 26: where does hot-path solve time go?
//! Times halfspaceCheck alone vs full solve (fast, reduced).
const std = @import("std");
const sphar = @import("src/root.zig");
const halfspace = @import("src/halfspace.zig");
const cases = @import("tests/cases/cases.zig");
const Vec3 = sphar.Vec3;

fn bench(comptime f: anytype, args: anytype, reps: usize) f64 {
    var tmin: f64 = 1e30;
    for (0..reps) |_| {
        const t0 = std.time.nanoTimestamp();
        @call(.auto, f, args) catch unreachable;
        const t1 = std.time.nanoTimestamp();
        const us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
        if (us < tmin) tmin = us;
    }
    return tmin;
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const names = [_][]const u8{ "hex", "h3_res09", "h3_res15", "np400", "ha_05" };
    std.debug.print("{s:10} {s:>12} {s:>10} {s:>10}\n", .{ "case", "halfspace_us", "fast_us", "reduced_us" });
    for (names) |name| {
        const case = cases.byName(name) orelse continue;
        const Xv: []const Vec3 = @ptrCast(case.points);

        const hs_fn = struct {
            fn run(alloc: std.mem.Allocator, X: []const Vec3) !void {
                var arena = std.heap.ArenaAllocator.init(alloc);
                defer arena.deinit();
                _ = try halfspace.halfspaceCheck(arena.allocator(), X);
            }
        };
        const solve_fn = struct {
            fn run(alloc: std.mem.Allocator, pts: []const [3]f64, m: sphar.Method) !void {
                var o = try sphar.solve(alloc, pts, .{ .method = m });
                o.deinit();
            }
        };
        const t_hs = bench(hs_fn.run, .{ allocator, Xv }, 300);
        const t_fast = bench(solve_fn.run, .{ allocator, case.points, .fast }, 300);
        const t_red = bench(solve_fn.run, .{ allocator, case.points, .reduced }, 300);
        std.debug.print("{s:10} {d:>12.2} {d:>10.2} {d:>10.2}\n", .{ name, t_hs, t_fast, t_red });
    }
}
