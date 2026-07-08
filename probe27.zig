//! Temporary probe 27: fair DGGS timing — DNC counted as occurrences,
//! wall time compared ONLY on mutually-converged cells (per-cell min of
//! N reps). Also reports each method's average cost of failure for
//! information. Run from repo root: zig run -O ReleaseFast probe27.zig
const std = @import("std");
const sphar = @import("src/root.zig");

const INPUT_DIR = "scripts/dggs/data";
const SYSTEMS = [_][]const u8{ "h3", "s2", "a5" };
const TOLS = [_]f64{ 1e-3, 1e-6 };
const REPS = 3;

const InputJson = struct {
    system: []const u8,
    resolution: i64,
    n_requested: u64,
    n_unique: u64,
    cells: []InputCell,
};
const InputCell = struct { id: []const u8, vertices: [][3]f64 };

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    for (SYSTEMS) |sys| {
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}.json", .{ INPUT_DIR, sys });
        defer allocator.free(path);
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 1 << 30);
        defer allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(InputJson, allocator, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const cells = parsed.value.cells;
        const n = cells.len;

        for (TOLS) |tol_| {
            const conv = try allocator.alloc([2]bool, n);
            defer allocator.free(conv);
            const tus = try allocator.alloc([2]f64, n);
            defer allocator.free(tus);

            const methods = [_]sphar.Method{ .fast, .reduced };
            for (methods, 0..) |method, mi| {
                for (cells, 0..) |cell, k| {
                    var tmin: f64 = 1e30;
                    var ok = false;
                    for (0..REPS) |_| {
                        const t0 = std.time.nanoTimestamp();
                        var o = sphar.solve(allocator, cell.vertices, .{ .gap_tol = tol_, .method = method }) catch {
                            ok = false;
                            break;
                        };
                        const t1 = std.time.nanoTimestamp();
                        ok = (o == .converged);
                        o.deinit();
                        const us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
                        if (us < tmin) tmin = us;
                    }
                    conv[k][mi] = ok;
                    tus[k][mi] = tmin;
                }
            }

            var n_both: u32 = 0;
            var n_fast_only: u32 = 0;
            var n_red_only: u32 = 0;
            var n_neither: u32 = 0;
            var t_both: [2]f64 = .{ 0, 0 };
            var t_fail: [2]f64 = .{ 0, 0 };
            var n_fail: [2]u32 = .{ 0, 0 };
            for (0..n) |k| {
                const cf = conv[k][0];
                const cr = conv[k][1];
                if (cf and cr) {
                    n_both += 1;
                    t_both[0] += tus[k][0];
                    t_both[1] += tus[k][1];
                } else if (cf) {
                    n_fast_only += 1;
                } else if (cr) {
                    n_red_only += 1;
                } else {
                    n_neither += 1;
                }
                for (0..2) |mi| {
                    if (!conv[k][mi]) {
                        t_fail[mi] += tus[k][mi];
                        n_fail[mi] += 1;
                    }
                }
            }
            std.debug.print("{s} tol={e:8.0}: both={d} fast-only={d} reduced-only={d} neither={d}\n", .{ sys, tol_, n_both, n_fast_only, n_red_only, n_neither });
            std.debug.print("   time on BOTH-converged: fast {d:7.1}ms  reduced {d:7.1}ms  ratio {d:.2}x\n", .{ t_both[0] / 1000.0, t_both[1] / 1000.0, t_both[1] / t_both[0] });
            if (n_fail[0] > 0 or n_fail[1] > 0) {
                const mf: f64 = if (n_fail[0] > 0) t_fail[0] / @as(f64, @floatFromInt(n_fail[0])) else 0;
                const mr: f64 = if (n_fail[1] > 0) t_fail[1] / @as(f64, @floatFromInt(n_fail[1])) else 0;
                std.debug.print("   cost of failure (mean us/DNC cell): fast {d:.1} ({d} cells)  reduced {d:.1} ({d} cells)\n", .{ mf, n_fail[0], mr, n_fail[1] });
            }
        }
        std.debug.print("\n", .{});
    }
}
