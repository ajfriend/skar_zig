//! Per-case timing bench. Iterates a hand-picked subset of the
//! comptime case manifest, runs the solver N times per case, prints
//! per-case min/median μs.
//!
//! Uses std.heap.smp_allocator (Zig's fast thread-safe production allocator)
//! — see the allocator note in skar.zig.

const std = @import("std");
const sphar = @import("skar");
const cases = @import("cases");

const N_WARMUP: u32 = 5;
const N_RUNS: u32 = 100;
const TOL: f64 = 1e-6;

fn cmpF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    // Representative subset of the full manifest. Intentionally fewer
    // cases than `cases.all` — bench is for cross-config timing, not
    // completeness; the full case-coverage gate is the test suite.
    const CASE_NAMES: []const []const u8 = &.{
        "hex",      "np20",     "np100",    "np400",
        "h3_res05", "h3_res09", "h3_res12", "h3_res15",
        "ha_05",    "ha_08",    "ha_10",    "ha_12",   "ha_14",
        "infeas_antipodal", "near_collinear",
    };

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("case                    status    n   iters  time_min_us  time_median_us  aspect_ratio  np_fail\n", .{});
    try stdout.print("----------------------  --------  --  -----  -----------  --------------  ------------  -------\n", .{});

    var total_converged_min: f64 = 0;
    var total_converged_median: f64 = 0;
    var n_converged: u32 = 0;

    for (CASE_NAMES) |name| {
        const case = cases.byName(name) orelse {
            try stdout.print("{s:22}  unknown case (not in manifest)\n", .{name});
            continue;
        };
        const X = case.points;

        // Warm up.
        for (0..N_WARMUP) |_| {
            var outcome = sphar.solve(allocator, X, .{ .gap_tol = TOL, .n_hull = 10, .coplanarity_tol = 1e-12 }) catch continue;
            outcome.deinit();
        }

        var times = try allocator.alloc(f64, N_RUNS);
        defer allocator.free(times);

        var last_outcome: ?sphar.Outcome = null;
        defer if (last_outcome) |*lo| lo.deinit();
        for (0..N_RUNS) |r| {
            const t0 = std.time.nanoTimestamp();
            const outcome = try sphar.solve(allocator, X, .{ .gap_tol = TOL, .n_hull = 10, .coplanarity_tol = 1e-12 });
            const t1 = std.time.nanoTimestamp();
            times[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
            if (last_outcome) |*lo| lo.deinit();
            last_outcome = outcome;
        }

        std.mem.sort(f64, times, {}, cmpF64);
        const t_min = times[0];
        const t_median = times[N_RUNS / 2];

        if (last_outcome) |lo| {
            const status_str = switch (lo) {
                .converged => "ok",
                .infeasible => "infeas",
                .did_not_converge => "DNC",
            };
            // Per-variant: only Converged/DidNotConverge carry iteration
            // counters; Infeasible bails in halfspaceCheck before iterating.
            // Aspect ratio is only meaningful on Converged.
            var outer_iters: u32 = 0;
            var newton_polish_failures: u32 = 0;
            var aspect_ratio: f64 = 0;
            switch (lo) {
                .converged => |c| {
                    outer_iters = c.outer_iters;
                    newton_polish_failures = c.newton_polish_failures;
                    aspect_ratio = c.aspectRatio();
                },
                .did_not_converge => |p| {
                    outer_iters = p.outer_iters;
                    newton_polish_failures = p.newton_polish_failures;
                    // Uncertified ratio from the last iterate — useful when
                    // chasing a DNC regression. `DidNotConverge` intentionally
                    // omits an `aspectRatio()` method since the value isn't
                    // certified; compute it inline here.
                    aspect_ratio = p.sigma[2] / p.sigma[1];
                },
                .infeasible => {},
            }
            try stdout.print("{s:22}  {s:8}  {d:2}  {d:5}  {d:11.2}  {d:14.2}  {d:12.6}  {d:7}\n", .{
                name,        status_str, X.len,
                outer_iters, t_min,      t_median,
                aspect_ratio, newton_polish_failures,
            });
            if (lo == .converged) {
                total_converged_min += t_min;
                total_converged_median += t_median;
                n_converged += 1;
            }
        }
    }

    // Only converged-case timings are meaningful for cross-config comparison.
    // DNC cases always hit MAX_OUTER and infeasible cases bail in halfspace
    // check — neither reflects solver inner-loop performance.
    try stdout.print("----------------------  --------  --  -----  -----------  --------------\n", .{});
    try stdout.print("{s:22}  {s:8}  {d:2}  {s:5}  {d:11.2}  {d:14.2}\n", .{
        "TOTAL (converged only)", "ok", n_converged, "—",
        total_converged_min, total_converged_median,
    });
}
