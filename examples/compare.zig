//! Fast-vs-joint solver path comparison (EXPERIMENTAL joint prototype;
//! see docs/wide-cap-dnc-report.md).
//!
//! Part 1: every bundled manifest case × {fast, joint} — status,
//! iterations (outer loops for fast, Newton steps for joint), min /
//! median wall µs, aspect ratio, gap. Answers "what does joint cost on
//! the cases fast already handles?"
//!
//! Part 2: the wide-cap robustness grid (random caps by width × seed
//! × density) × {fast, joint, auto} — DNC counts and median times.
//! Answers "does joint/auto close the wide-angle hole, and at what
//! price?"
//!
//! Force-built ReleaseFast (timing meaningless in Debug).

const std = @import("std");
const sphar = @import("skar");
const cases = @import("cases");
const Vec3 = sphar.Vec3;

const N_WARMUP: u32 = 3;
const N_RUNS: u32 = 30;
const GRID_RUNS: u32 = 5;

fn cmpF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

const RunStats = struct {
    status: []const u8,
    iters: u32,
    t_min_us: f64,
    t_median_us: f64,
    ar: f64,
    gap: f64,
};

/// Run `method` on `pts` `runs` times; report the last outcome + timing.
fn measure(
    allocator: std.mem.Allocator,
    pts: []const [3]f64,
    method: sphar.Method,
    runs: u32,
    times_buf: []f64,
) !RunStats {
    for (0..N_WARMUP) |_| {
        var o = sphar.solve(allocator, pts, .{ .method = method }) catch continue;
        o.deinit();
    }
    var last: ?sphar.Outcome = null;
    defer if (last) |*lo| lo.deinit();
    for (0..runs) |r| {
        const t0 = std.time.nanoTimestamp();
        const o = try sphar.solve(allocator, pts, .{ .method = method });
        const t1 = std.time.nanoTimestamp();
        times_buf[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
        if (last) |*lo| lo.deinit();
        last = o;
    }
    std.mem.sort(f64, times_buf[0..runs], {}, cmpF64);

    var stats = RunStats{
        .status = "?",
        .iters = 0,
        .t_min_us = times_buf[0],
        .t_median_us = times_buf[runs / 2],
        .ar = 0,
        .gap = 0,
    };
    switch (last.?) {
        .converged => |c| {
            stats.status = "ok";
            stats.iters = c.outer_iters;
            stats.ar = c.aspectRatio();
            stats.gap = c.gap;
        },
        .did_not_converge => |p| {
            stats.status = "DNC";
            stats.iters = p.outer_iters;
            stats.ar = p.sigma[2] / p.sigma[1]; // uncertified, diagnostic only
            stats.gap = p.gap;
        },
        .infeasible => {
            stats.status = "infeas";
        },
    }
    return stats;
}

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

/// Same construction as the wide-cap investigation probes: n random
/// points in a spherical cap of angular radius cap_deg, random rotation.
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

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const stdout = std.fs.File.stdout().deprecatedWriter();

    var times: [N_RUNS]f64 = undefined;

    // ---------- Part 1: bundled manifest, fast vs joint vs reduced ----------
    try stdout.print("== part 1: bundled cases (n={d} runs; iters = outer loops for fast/reduced, Newton steps for joint) ==\n\n", .{N_RUNS});
    try stdout.print("{s:22} {s:3} | {s:6} {s:5} {s:9} {s:11} | {s:6} {s:5} {s:9} | {s:7} {s:5} {s:9} {s:11}\n", .{
        "case",    "n",
        "fast",    "iters", "min_us", "ar",
        "joint",   "iters", "min_us",
        "reduced", "iters", "min_us", "ar_rel_diff",
    });
    var n_joint: u32 = 0;
    var joint_slowdown_sum: f64 = 0;
    var n_red: u32 = 0;
    var red_slowdown_sum: f64 = 0;
    for (cases.all) |entry| {
        const pts = entry.case.points;
        const f = try measure(allocator, pts, .fast, N_RUNS, &times);
        const j = try measure(allocator, pts, .joint, N_RUNS, &times);
        const r = try measure(allocator, pts, .reduced, N_RUNS, &times);
        var ar_rel: f64 = 0;
        if (std.mem.eql(u8, f.status, "ok") and std.mem.eql(u8, r.status, "ok")) {
            ar_rel = @abs(r.ar - f.ar) / f.ar;
        }
        // Sub-µs fast solves are below the clock resolution; skip them
        // in the ratios rather than dividing by ~0.
        if (f.t_median_us >= 1.0 and std.mem.eql(u8, f.status, "ok")) {
            if (std.mem.eql(u8, j.status, "ok")) {
                joint_slowdown_sum += j.t_median_us / f.t_median_us;
                n_joint += 1;
            }
            if (std.mem.eql(u8, r.status, "ok")) {
                red_slowdown_sum += r.t_median_us / f.t_median_us;
                n_red += 1;
            }
        }
        try stdout.print("{s:22} {d:3} | {s:6} {d:5} {d:9.2} {d:11.6} | {s:6} {d:5} {d:9.2} | {s:7} {d:5} {d:9.2} {e:11.2}\n", .{
            entry.name, pts.len,
            f.status,   f.iters, f.t_min_us, f.ar,
            j.status,   j.iters, j.t_min_us,
            r.status,   r.iters, r.t_min_us, ar_rel,
        });
    }
    try stdout.print("\nmean median-time slowdown vs fast (mutually-converged, fast ≥ 1µs): joint {d:.1}x ({d}), reduced {d:.1}x ({d})\n", .{
        joint_slowdown_sum / @as(f64, @floatFromInt(n_joint)), n_joint,
        red_slowdown_sum / @as(f64, @floatFromInt(n_red)),     n_red,
    });

    // ---------- Part 2: wide-cap robustness grid ----------
    const widths = [_]f64{ 60, 75, 80, 81, 82, 84, 86, 88, 89, 89.5 };
    const ns = [_]usize{ 20, 200 };
    const n_seeds: u64 = 10;
    const methods = [_]sphar.Method{ .fast, .joint, .reduced, .auto };

    try stdout.print("\n== part 2: wide-cap grid, DNC counts /{d} seeds and median µs (n runs = {d}) ==\n\n", .{ n_seeds, GRID_RUNS });
    try stdout.print("{s:5} {s:5} | {s:14} | {s:14} | {s:14} | {s:14}\n", .{ "n", "width", "fast", "joint", "reduced", "auto" });
    try stdout.print("{s:5} {s:5} | {s:6} {s:7} | {s:6} {s:7} | {s:6} {s:7} | {s:6} {s:7}\n", .{ "", "", "DNC", "med_us", "DNC", "med_us", "DNC", "med_us", "DNC", "med_us" });

    var grid_times: [GRID_RUNS]f64 = undefined;
    for (ns) |n| {
        for (widths) |wdeg| {
            var dnc = [_]u32{ 0, 0, 0, 0 };
            var med = [_]f64{ 0, 0, 0, 0 };
            var seed: u64 = 1;
            while (seed <= n_seeds) : (seed += 1) {
                var prng = std.Random.DefaultPrng.init(seed);
                const rng = prng.random();
                const pts = try capPoints(allocator, rng, n, wdeg);
                defer allocator.free(pts);
                for (methods, 0..) |method, mi| {
                    const st = try measure(allocator, pts, method, GRID_RUNS, &grid_times);
                    if (!std.mem.eql(u8, st.status, "ok")) dnc[mi] += 1;
                    med[mi] += st.t_median_us;
                }
            }
            const denom: f64 = @floatFromInt(n_seeds);
            try stdout.print("{d:5} {d:5.1} | {d:2}/10 {d:7.0} | {d:2}/10 {d:7.0} | {d:2}/10 {d:7.0} | {d:2}/10 {d:7.0}\n", .{
                n,            wdeg,
                dnc[0],       med[0] / denom,
                dnc[1],       med[1] / denom,
                dnc[2],       med[2] / denom,
                dnc[3],       med[3] / denom,
            });
        }
    }
    try stdout.print("\n(med_us for a DNC-heavy fast cell is the cost of burning max_outer; auto pays that plus the joint solve.)\n", .{});
}
