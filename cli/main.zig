//! Uniform CLI shim for the skar solver. Reads a single case file,
//! runs the solver, emits one JSONL line on stdout.
//!
//! Usage:
//!   skar-cli <case-file> [--tol FLOAT] [--max-iter INT]
//!                         [--warmup INT] [--n-runs INT]

const std = @import("std");
const sphar = @import("skar");
const cases = @import("cases");

const SOLVER_NAME = "skar";
const N_HULL: i32 = 10;
const DEFAULT_TOL: f64 = 1e-6;

const Args = struct {
    case_path: []const u8,
    tol: f64 = DEFAULT_TOL,
    max_iter: ?u32 = null, // sphar.solve currently has no max_iter knob; reserved
    warmup: u32 = 0,
    n_runs: u32 = 1,
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    // Don't free here; caller's arena owns argv.

    if (argv.len < 2) return error.MissingCasePath;

    var args: Args = .{ .case_path = "" };
    var positional_seen = false;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--tol")) {
            i += 1;
            args.tol = try std.fmt.parseFloat(f64, argv[i]);
        } else if (std.mem.eql(u8, a, "--max-iter")) {
            i += 1;
            args.max_iter = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--warmup")) {
            i += 1;
            args.warmup = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--n-runs")) {
            i += 1;
            args.n_runs = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (!positional_seen) {
            args.case_path = a;
            positional_seen = true;
        } else {
            return error.UnexpectedArg;
        }
    }
    if (!positional_seen) return error.MissingCasePath;
    return args;
}

fn outcomeTag(outcome: sphar.Outcome) []const u8 {
    return switch (outcome) {
        .converged => "converged",
        .infeasible => "infeasible",
        .did_not_converge => "did_not_converge",
        .coplanar_input => "coplanar_input",
    };
}

fn writeVec3(w: anytype, v: [3]f64) !void {
    try w.print("[{d},{d},{d}]", .{ v[0], v[1], v[2] });
}

fn writeLambdas(w: anytype, indices: []const u32, lambdas: []const f64) !void {
    try w.writeAll("{");
    for (indices, lambdas, 0..) |idx, lam, k| {
        if (k > 0) try w.writeAll(",");
        try w.print("\"{d}\":{d}", .{ idx, lam });
    }
    try w.writeAll("}");
}

fn writeRecord(w: anytype, args: Args, outcome: sphar.Outcome, time_s: f64) !void {
    const case_stem = cases.caseStem(args.case_path);

    try w.writeAll("{");
    try w.print("\"solver\":\"{s}\",", .{SOLVER_NAME});
    try w.print("\"case\":\"{s}\",", .{case_stem});
    try w.print("\"status\":\"{s}\",", .{outcomeTag(outcome)});
    try w.print("\"tolerance\":{d},", .{args.tol});
    try w.print("\"time_s\":{d}", .{time_s});

    // Iteration counters are 0 on the variants that bail before iterating
    // (Infeasible bails in halfspaceCheck; coplanar_input bails in
    // preprocessing). Only Converged and PartialInfo carry them.
    var outer_iters: u32 = 0;
    var newton_polish_failures: u32 = 0;

    switch (outcome) {
        .converged => |c| {
            try w.print(",\"aspect_ratio\":{d},", .{c.aspectRatio()});
            try w.writeAll("\"Q\":[");
            try writeVec3(w, c.Q.col(0).m);
            try w.writeAll(",");
            try writeVec3(w, c.Q.col(1).m);
            try w.writeAll(",");
            try writeVec3(w, c.Q.col(2).m);
            try w.print("],\"sigma\":[{d},{d},{d}],", .{ c.sigma[0], c.sigma[1], c.sigma[2] });
            try w.writeAll("\"lambdas\":");
            try writeLambdas(w, c.cert.indices, c.cert.lambdas);
            try w.print(",\"claimed_gap\":{d}", .{c.cert.claimed_gap});
            outer_iters = c.outer_iters;
            newton_polish_failures = c.newton_polish_failures;
        },
        .infeasible => |i| {
            try w.writeAll(",\"lambdas\":");
            try writeLambdas(w, i.cert.indices, i.cert.lambdas);
        },
        .did_not_converge => |p| {
            outer_iters = p.outer_iters;
            newton_polish_failures = p.newton_polish_failures;
        },
        .coplanar_input => {},
    }

    try w.print(",\"instrumentation\":{{\"outer_iters\":{d},\"newton_polish_failures\":{d}}}", .{
        outer_iters,
        newton_polish_failures,
    });

    try w.writeAll("}\n");
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const stdout = std.fs.File.stdout().deprecatedWriter();

    const args = parseArgs(allocator) catch |err| {
        std.debug.print("error parsing args: {s}\n", .{@errorName(err)});
        std.debug.print("usage: skar-cli <case-file> [--tol F] [--max-iter N] [--warmup N] [--n-runs N]\n", .{});
        std.process.exit(2);
    };

    const X = try cases.loadCase(allocator, args.case_path);
    defer allocator.free(X);

    // Warmup — discard timing.
    for (0..args.warmup) |_| {
        var outcome = sphar.solve(allocator, X, .{ .gap_tol = args.tol, .n_hull = N_HULL, .coplanarity_tol = 1e-12 }) catch continue;
        outcome.deinit();
    }

    const n_runs = if (args.n_runs == 0) 1 else args.n_runs;
    var times = try allocator.alloc(f64, n_runs);
    defer allocator.free(times);

    var last_outcome: ?sphar.Outcome = null;
    defer if (last_outcome) |*lo| lo.deinit();
    for (0..n_runs) |r| {
        const t0 = std.time.nanoTimestamp();
        const outcome = try sphar.solve(allocator, X, .{ .gap_tol = args.tol, .n_hull = N_HULL, .coplanarity_tol = 1e-12 });
        const t1 = std.time.nanoTimestamp();
        times[r] = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
        if (last_outcome) |*lo| lo.deinit();
        last_outcome = outcome;
    }

    var sum: f64 = 0;
    for (times) |t| sum += t;
    const time_s = sum / @as(f64, @floatFromInt(n_runs));

    try writeRecord(stdout, args, last_outcome.?, time_s);
}
