//! EXPLORATION PROBE (perf/trust-losing-cases; delete before merge):
//! bulk fair comparison of .alternating vs .trust over the real survey
//! datasets, per the repo's measurement policy: DNC counts reported per
//! method, timing compared ONLY on the mutually-converged subset,
//! per-cell min over REPS runs.
//!
//! Mid-resolution H3 inputs (the r9-class table-stakes check) are
//! generated OUTSIDE the repo — see the session scratchpad's
//! gen_midres_h3.py (same schema as scripts/dggs/gen_cells.py, but
//! H3 r5/r9) — and read from MIDRES_DIR; missing files are skipped
//! with a note. Run from the repo root, ReleaseFast (`zig build
//! ex-bulkcmp`).

const std = @import("std");
const skar = @import("skar");

const REPS = 5;

/// Where gen_midres_h3.py wrote h3_r5.json / h3_r9.json. Edit in place
/// if reproducing on another machine.
const MIDRES_DIR = "/private/tmp/claude-501/-Users-aj-work-skar-zig/e24ef9a7-eb9e-4c91-9f3f-66da8c82bbc3/scratchpad";

const Status = enum { ok, dnc, infeas, input_err };

const CellRes = struct {
    st: Status,
    ar: f64,
    t_us: f64, // min over REPS
};

fn runMethod(
    allocator: std.mem.Allocator,
    verts: []const [3]f64,
    method: skar.Method,
    tol: f64,
) !CellRes {
    var best: f64 = std.math.inf(f64);
    var st: Status = .input_err;
    var ar: f64 = 0;
    var timer = try std.time.Timer.start();
    for (0..REPS) |_| {
        timer.reset();
        const o_or_err = skar.solve(allocator, verts, .{ .method = method, .gap_tol = tol });
        const ns = timer.read();
        var o = o_or_err catch |err| switch (err) {
            error.InsufficientPoints, error.CoplanarInput, error.InvalidTolerance => return .{ .st = .input_err, .ar = 0, .t_us = 0 },
            else => return err,
        };
        defer o.deinit();
        const dt = @as(f64, @floatFromInt(ns)) / 1000.0;
        if (dt < best) best = dt;
        switch (o) {
            .converged => |c| {
                st = .ok;
                ar = c.aspectRatio();
            },
            .did_not_converge => st = .dnc,
            .infeasible => st = .infeas,
        }
    }
    return .{ .st = st, .ar = ar, .t_us = best };
}

fn cmpF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn runDataset(
    allocator: std.mem.Allocator,
    name: []const u8,
    cells: []const []const [3]f64,
    tol: f64,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const n = cells.len;

    const alt = try allocator.alloc(CellRes, n);
    defer allocator.free(alt);
    const tru = try allocator.alloc(CellRes, n);
    defer allocator.free(tru);

    for (cells, 0..) |verts, i| {
        alt[i] = try runMethod(allocator, verts, .alternating, tol);
        tru[i] = try runMethod(allocator, verts, .trust, tol);
    }

    var counts_alt = [_]u32{ 0, 0, 0, 0 };
    var counts_tru = [_]u32{ 0, 0, 0, 0 };
    var dnc_t_alt: f64 = 0;
    var dnc_t_tru: f64 = 0;
    var mut_alt_times = std.ArrayListUnmanaged(f64){};
    defer mut_alt_times.deinit(allocator);
    var mut_tru_times = std.ArrayListUnmanaged(f64){};
    defer mut_tru_times.deinit(allocator);
    var max_ar_rel: f64 = 0;
    var only_alt: u32 = 0; // converged for alt but not trust
    var only_tru: u32 = 0;

    for (alt, tru) |a, t| {
        counts_alt[@intFromEnum(a.st)] += 1;
        counts_tru[@intFromEnum(t.st)] += 1;
        if (a.st == .dnc) dnc_t_alt += a.t_us;
        if (t.st == .dnc) dnc_t_tru += t.t_us;
        if (a.st == .ok and t.st == .ok) {
            try mut_alt_times.append(allocator, a.t_us);
            try mut_tru_times.append(allocator, t.t_us);
            const rel = @abs(t.ar - a.ar) / a.ar;
            if (rel > max_ar_rel) max_ar_rel = rel;
        } else if (a.st == .ok) {
            only_alt += 1;
        } else if (t.st == .ok) {
            only_tru += 1;
        }
    }

    try stdout.print("\n== {s} (n={d}) @ gap_tol {e:.0} ==\n", .{ name, n, tol });
    try stdout.print("  alt:   conv {d:6}  DNC {d:5}  infeas {d}  input_err {d}", .{ counts_alt[0], counts_alt[1], counts_alt[2], counts_alt[3] });
    if (counts_alt[1] > 0) try stdout.print("  (mean DNC cost {d:.0} µs)", .{dnc_t_alt / @as(f64, @floatFromInt(counts_alt[1]))});
    try stdout.print("\n", .{});
    try stdout.print("  trust: conv {d:6}  DNC {d:5}  infeas {d}  input_err {d}", .{ counts_tru[0], counts_tru[1], counts_tru[2], counts_tru[3] });
    if (counts_tru[1] > 0) try stdout.print("  (mean DNC cost {d:.0} µs)", .{dnc_t_tru / @as(f64, @floatFromInt(counts_tru[1]))});
    try stdout.print("\n", .{});
    try stdout.print("  converged-only-by: alt {d}, trust {d}\n", .{ only_alt, only_tru });

    const m = mut_alt_times.items.len;
    if (m == 0) {
        try stdout.print("  mutual: none\n", .{});
        return;
    }
    var sum_a: f64 = 0;
    var sum_t: f64 = 0;
    for (mut_alt_times.items) |v| sum_a += v;
    for (mut_tru_times.items) |v| sum_t += v;
    std.mem.sort(f64, mut_alt_times.items, {}, cmpF64);
    std.mem.sort(f64, mut_tru_times.items, {}, cmpF64);
    const fm: f64 = @floatFromInt(m);
    try stdout.print("  mutual {d}: alt total {d:.1} ms, mean {d:.2} µs, median {d:.2} µs\n", .{ m, sum_a / 1000.0, sum_a / fm, mut_alt_times.items[m / 2] });
    try stdout.print("             trust total {d:.1} ms, mean {d:.2} µs, median {d:.2} µs\n", .{ sum_t / 1000.0, sum_t / fm, mut_tru_times.items[m / 2] });
    try stdout.print("             trust/alt total {d:.2}x | max AR reldiff {e:.2}\n", .{ sum_t / sum_a, max_ar_rel });
}

const DggsJson = struct {
    cells: []struct { id: []const u8, vertices: [][3]f64 },
};
const StatesJson = struct {
    states: []struct { name: []const u8, vertices: [][3]f64 },
};
const CountriesJson = struct {
    countries: []struct { name: []const u8, vertices: [][3]f64 },
};

fn loadVertexLists(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    comptime T: type,
    comptime field: []const u8,
    path: []const u8,
) !?[]const []const [3]f64 {
    var file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 30);
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(T, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const entries = @field(parsed.value, field);
    const out = try arena.alloc([]const [3]f64, entries.len);
    for (entries, 0..) |e, i| {
        out[i] = try arena.dupe([3]f64, e.vertices);
    }
    return out;
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const stdout = std.fs.File.stdout().deprecatedWriter();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const h3 = (try loadVertexLists(allocator, arena, DggsJson, "cells", "scripts/dggs/data/h3.json")).?;
    const s2 = (try loadVertexLists(allocator, arena, DggsJson, "cells", "scripts/dggs/data/s2.json")).?;
    const a5 = (try loadVertexLists(allocator, arena, DggsJson, "cells", "scripts/dggs/data/a5.json")).?;
    const states = (try loadVertexLists(allocator, arena, StatesJson, "states", "scripts/states/data/states.json")).?;
    const countries = (try loadVertexLists(allocator, arena, CountriesJson, "countries", "scripts/countries/data/countries.json")).?;
    const h3_r5 = try loadVertexLists(allocator, arena, DggsJson, "cells", MIDRES_DIR ++ "/h3_r5.json");
    const h3_r9 = try loadVertexLists(allocator, arena, DggsJson, "cells", MIDRES_DIR ++ "/h3_r9.json");

    // The table-stakes check: bulk DGGS surveys across resolutions, at
    // the strict default (floor cells DNC on both — count them) and at
    // the survey tolerance (everything converges — clean timing).
    for ([_]f64{ 1e-6, 1e-3 }) |tol| {
        if (h3_r5) |cells| try runDataset(allocator, "h3 r5", cells, tol);
        if (h3_r9) |cells| try runDataset(allocator, "h3 r9", cells, tol);
        try runDataset(allocator, "h3 r15", h3, tol);
        try runDataset(allocator, "s2 r30", s2, tol);
        try runDataset(allocator, "a5 r29", a5, tol);
    }
    if (h3_r5 == null or h3_r9 == null) {
        try stdout.print("\n(mid-res H3 file(s) missing under {s} — generate with gen_midres_h3.py)\n", .{MIDRES_DIR});
    }
    // Hard/irregular regions at the strict default.
    try runDataset(allocator, "states", states, 1e-6);
    try runDataset(allocator, "countries", countries, 1e-6);
}
