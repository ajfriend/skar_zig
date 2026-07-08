//! Temporary probe 14: DGGS survey (h3/s2/a5 finest-res, 10k cells each)
//! through the reduced path — the validation-ledger item in
//! docs/reduced-solver.md. Matrix: {fast, reduced} × {1e-3, 1e-6} plus
//! joint × 1e-6 (extreme-κ floor comparison). Reads the same JSON inputs
//! as scripts/dggs/aspect.zig; run from the repo root:
//!   zig run -O ReleaseFast probe14.zig
const std = @import("std");
const sphar = @import("src/root.zig");

const INPUT_DIR = "scripts/dggs/data";
const SYSTEMS = [_][]const u8{ "h3", "s2", "a5" };

const InputJson = struct {
    system: []const u8,
    resolution: i64,
    n_requested: u64,
    n_unique: u64,
    cells: []InputCell,
};
const InputCell = struct {
    id: []const u8,
    vertices: [][3]f64,
};

const Config = struct { method: sphar.Method, tol: f64, label: []const u8 };
const CONFIGS = [_]Config{
    .{ .method = .fast, .tol = 1e-3, .label = "fast/1e-3   " },
    .{ .method = .reduced, .tol = 1e-3, .label = "reduced/1e-3" },
    .{ .method = .fast, .tol = 1e-6, .label = "fast/1e-6   " },
    .{ .method = .reduced, .tol = 1e-6, .label = "reduced/1e-6" },
    .{ .method = .joint, .tol = 1e-6, .label = "joint/1e-6  " },
};

const Run = struct {
    converged: u32 = 0,
    dnc: u32 = 0,
    infeasible: u32 = 0,
    input_error: u32 = 0,
    iters_sum: u64 = 0,
    iters_max: u32 = 0,
    ms: i64 = 0,
    /// Per-cell AR; NaN when not converged. Indexed by cell order.
    ars: []f64,
};

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

        std.debug.print("== {s} (res={d}, {d} cells) ==\n", .{ sys, parsed.value.resolution, cells.len });
        std.debug.print("{s:14} {s:>9} {s:>7} {s:>4} {s:>4} {s:>10} {s:>9} {s:>7}\n", .{ "config", "converged", "DNC", "inf", "ipe", "iters_mean", "iters_max", "ms" });

        var runs: [CONFIGS.len]Run = undefined;
        for (CONFIGS, 0..) |cfg, ci| {
            var run = Run{ .ars = try allocator.alloc(f64, cells.len) };
            @memset(run.ars, std.math.nan(f64));
            const t0 = std.time.milliTimestamp();
            for (cells, 0..) |cell, k| {
                var outcome = sphar.solve(allocator, cell.vertices, .{ .gap_tol = cfg.tol, .method = cfg.method }) catch |err| switch (err) {
                    error.InsufficientPoints, error.CoplanarInput, error.InvalidTolerance => {
                        run.input_error += 1;
                        continue;
                    },
                    else => return err,
                };
                defer outcome.deinit();
                switch (outcome) {
                    .converged => |c| {
                        run.converged += 1;
                        run.ars[k] = c.aspectRatio();
                        run.iters_sum += c.outer_iters;
                        if (c.outer_iters > run.iters_max) run.iters_max = c.outer_iters;
                    },
                    .did_not_converge => run.dnc += 1,
                    .infeasible => run.infeasible += 1,
                }
            }
            run.ms = std.time.milliTimestamp() - t0;
            runs[ci] = run;
            const mean: f64 = if (run.converged > 0)
                @as(f64, @floatFromInt(run.iters_sum)) / @as(f64, @floatFromInt(run.converged))
            else
                0;
            std.debug.print("{s:14} {d:>9} {d:>7} {d:>4} {d:>4} {d:>10.2} {d:>9} {d:>7}\n", .{
                cfg.label, run.converged, run.dnc, run.infeasible, run.input_error, mean, run.iters_max, run.ms,
            });
        }

        // Parity: reduced vs fast at each tolerance — status disagreements
        // and max relative AR diff over cells converged in both.
        const pairs = [_][2]usize{ .{ 0, 1 }, .{ 2, 3 } };
        for (pairs) |pr| {
            const f = runs[pr[0]];
            const r = runs[pr[1]];
            var only_f: u32 = 0;
            var only_r: u32 = 0;
            var max_rel: f64 = 0;
            for (f.ars, r.ars) |af, ar| {
                const cf = !std.math.isNan(af);
                const cr = !std.math.isNan(ar);
                if (cf and !cr) only_f += 1;
                if (cr and !cf) only_r += 1;
                if (cf and cr) {
                    const rel = @abs(af - ar) / af;
                    if (rel > max_rel) max_rel = rel;
                }
            }
            std.debug.print("  parity {s} vs {s}: fast-only-converged={d} reduced-only-converged={d} maxRelΔAR={e:.2}\n", .{
                CONFIGS[pr[0]].label, CONFIGS[pr[1]].label, only_f, only_r, max_rel,
            });
        }
        std.debug.print("\n", .{});
        for (runs) |run| allocator.free(run.ars);
    }
}
