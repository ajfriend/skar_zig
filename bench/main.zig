//! Per-case timing bench. Reads cases/*.txt, runs the solver N times,
//! prints per-case min/median μs.
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
        const path = try std.fmt.allocPrint(allocator, "cases/{s}.txt", .{name});
        defer allocator.free(path);
        const X = cases.loadCase(allocator, path) catch |err| {
            try stdout.print("{s:22}  load error: {s}\n", .{ name, @errorName(err) });
            continue;
        };
        defer allocator.free(X);

        // Warm up.
        for (0..N_WARMUP) |_| {
            var info = sphar.solve(allocator, X, TOL, 10, 1e-12) catch continue;
            info.deinit();
        }

        var times = try allocator.alloc(f64, N_RUNS);
        defer allocator.free(times);

        var last_info: ?sphar.Info = null;
        for (0..N_RUNS) |r| {
            const t0 = std.time.nanoTimestamp();
            const info = try sphar.solve(allocator, X, TOL, 10, 1e-12);
            const t1 = std.time.nanoTimestamp();
            times[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
            if (last_info) |*li| li.deinit();
            last_info = info;
        }
        defer if (last_info) |*li| li.deinit();

        std.mem.sort(f64, times, {}, cmpF64);
        const t_min = times[0];
        const t_median = times[N_RUNS / 2];

        if (last_info) |li| {
            const status_str = switch (li.status) {
                .converged => "ok",
                .infeasible => "infeas",
                .did_not_converge => "DNC",
                .coplanar_input => "coplanar",
            };
            try stdout.print("{s:22}  {s:8}  {d:2}  {d:5}  {d:11.2}  {d:14.2}  {d:12.6}  {d:7}\n", .{
                name,           status_str,      X.len,
                li.outer_iters, t_min, t_median,
                li.aspectRatio(), li.newton_polish_failures,
            });
            if (li.status == .converged) {
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
