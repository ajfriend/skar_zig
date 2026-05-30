//! Step 2 of the DGGS aspect-ratio survey
//! (see docs/dggs-aspect-survey-plan.md).
//!
//! Reads scripts/dggs/data/{h3,s2,a5}.json (produced by step 1), calls
//! `skar.solve` on every cell, and writes scripts/dggs/data/aspect.json
//! with per-system outcome counts, the AR distribution for converged
//! cells, and the full details of the single worst-AR cell per system
//! (needed by step 4's ellipse plot).
//!
//! Wired as `zig build dggs-aspect` (ReleaseFast). Run from the repo
//! root; the input/output paths are relative to CWD.

const std = @import("std");
const skar = @import("skar");

const INPUT_DIR = "scripts/dggs/data";
const OUTPUT_PATH = "scripts/dggs/data/aspect.json";
const SYSTEMS = [_][]const u8{ "h3", "s2", "a5" };

/// Shape of one input file (scripts/dggs/data/<sys>.json).
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

/// Tally of `skar.solve` outcomes for one system.
const Counts = struct {
    converged: u32 = 0,
    infeasible: u32 = 0,
    did_not_converge: u32 = 0,
    /// `InputError.CoplanarInput` / `InsufficientPoints` — caller-side
    /// rejections, not solver failures. Counted separately so step 3
    /// can surface degeneracy rates without conflating them with DNC.
    input_error: u32 = 0,
};

/// Full details for one cell — only kept for the worst-AR converged
/// cell per system, since step 4 needs `b`, `A`, and the vertices to
/// project + draw the enclosing ellipse.
const WorstCase = struct {
    id: []const u8,
    ar: f64,
    b: [3]f64,
    /// Row-major 3×3, materialized from `Converged.A()`.
    A: [3][3]f64,
    gap: f64,
    outer_iters: u32,
    vertices: [][3]f64,
};

const SystemResult = struct {
    n_total: u32,
    counts: Counts,
    /// AR values for converged cells, in input order. Length ==
    /// `counts.converged`. No ids — the histogram doesn't need them.
    ars: []f64,
    worst: ?WorstCase,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // One arena holds the output-side allocations (per-system AR lists
    // and the deep-copied WorstCase payloads). Per-input arenas are
    // scoped to processSystem and freed before moving on.
    var out_arena = std.heap.ArenaAllocator.init(allocator);
    defer out_arena.deinit();
    const out_alloc = out_arena.allocator();

    var results: [SYSTEMS.len]SystemResult = undefined;
    for (SYSTEMS, 0..) |sys, i| {
        results[i] = try processSystem(allocator, out_alloc, sys);
    }

    try writeOutput(allocator, &results);
    printSummary(&results);
}

fn processSystem(
    gpa: std.mem.Allocator,
    out_alloc: std.mem.Allocator,
    sys: []const u8,
) !SystemResult {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.json", .{ INPUT_DIR, sys });
    defer gpa.free(path);

    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(gpa, 1 << 30);
    defer gpa.free(bytes);

    // The parsed payload owns its own arena (per std.json convention);
    // everything we want to keep past `parsed.deinit()` is copied into
    // `out_alloc` before this function returns.
    var parsed = try std.json.parseFromSlice(InputJson, gpa, bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const input = parsed.value;

    std.debug.print("[{s}] solving {d} cells (res={d})...\n", .{ sys, input.cells.len, input.resolution });

    var counts: Counts = .{};
    var ars = try std.ArrayListUnmanaged(f64).initCapacity(out_alloc, input.cells.len);
    var worst: ?WorstCase = null;
    var first_dnc_printed = false;

    const t0 = std.time.milliTimestamp();
    for (input.cells) |cell| {
        const outcome_or_err = skar.solve(gpa, cell.vertices, .{});
        var outcome = outcome_or_err catch |err| switch (err) {
            error.InsufficientPoints, error.CoplanarInput, error.InvalidTolerance => {
                counts.input_error += 1;
                continue;
            },
            else => return err, // OOM, SolveError — propagate
        };
        defer outcome.deinit();

        switch (outcome) {
            .converged => |c| {
                counts.converged += 1;
                const ar = c.aspectRatio();
                try ars.append(out_alloc, ar);
                if (worst == null or ar > worst.?.ar) {
                    worst = try captureWorst(out_alloc, cell, c);
                }
            },
            .infeasible => counts.infeasible += 1,
            .did_not_converge => {
                counts.did_not_converge += 1;
                if (!first_dnc_printed) {
                    first_dnc_printed = true;
                    dumpDnc(sys, cell);
                }
            },
        }
    }
    const dt_ms = std.time.milliTimestamp() - t0;
    std.debug.print("  done in {d}ms\n", .{dt_ms});

    return .{
        .n_total = @intCast(input.cells.len),
        .counts = counts,
        .ars = ars.items,
        .worst = worst,
    };
}

/// Print the first DNC cell per system to stderr in a copy-pasteable
/// Zig array literal — feeds straight into a regression test.
fn dumpDnc(sys: []const u8, cell: InputCell) void {
    std.debug.print("\n=== first DNC for {s} ===\n", .{sys});
    std.debug.print("id: {s}\n", .{cell.id});
    std.debug.print("const pts = [_][3]f64{{\n", .{});
    for (cell.vertices) |v| {
        std.debug.print("    .{{ {e:.17}, {e:.17}, {e:.17} }},\n", .{ v[0], v[1], v[2] });
    }
    std.debug.print("}};\n\n", .{});
}

fn captureWorst(
    out_alloc: std.mem.Allocator,
    cell: InputCell,
    c: skar.Converged,
) !WorstCase {
    const id_copy = try out_alloc.dupe(u8, cell.id);
    const verts_copy = try out_alloc.dupe([3]f64, cell.vertices);

    const bv = c.b();
    const A_mat = c.A();
    var A_rows: [3][3]f64 = undefined;
    for (0..3) |r| for (0..3) |k| {
        A_rows[r][k] = A_mat.m[r * 3 + k];
    };

    return .{
        .id = id_copy,
        .ar = c.aspectRatio(),
        .b = .{ bv.m[0], bv.m[1], bv.m[2] },
        .A = A_rows,
        .gap = c.gap,
        .outer_iters = c.outer_iters,
        .vertices = verts_copy,
    };
}

fn writeOutput(
    gpa: std.mem.Allocator,
    results: *const [SYSTEMS.len]SystemResult,
) !void {
    // Build a typed wrapper object so std.json can stringify the whole
    // payload with field names matching SYSTEMS.
    const Wrapped = struct {
        h3: SystemResult,
        s2: SystemResult,
        a5: SystemResult,
    };
    const payload: Wrapped = .{
        .h3 = results[0],
        .s2 = results[1],
        .a5 = results[2],
    };

    var file = try std.fs.cwd().createFile(OUTPUT_PATH, .{ .truncate = true });
    defer file.close();
    var write_buf: [4096]u8 = undefined;
    var writer = file.writer(&write_buf);
    try std.json.Stringify.value(payload, .{}, &writer.interface);
    try writer.interface.flush();
    _ = gpa;
}

fn printSummary(results: *const [SYSTEMS.len]SystemResult) void {
    std.debug.print("\n{s:5}  {s:>6}  {s:>10}  {s:>10}  {s:>3}  {s:>3}  {s:>8}\n", .{
        "sys", "total", "converged", "DNC", "inf", "ipe", "worstAR",
    });
    for (SYSTEMS, results) |sys, r| {
        const worst_ar: f64 = if (r.worst) |w| w.ar else std.math.nan(f64);
        std.debug.print("{s:5}  {d:>6}  {d:>10}  {d:>10}  {d:>3}  {d:>3}  {d:>8.4}\n", .{
            sys,
            r.n_total,
            r.counts.converged,
            r.counts.did_not_converge,
            r.counts.infeasible,
            r.counts.input_error,
            worst_ar,
        });
    }
    std.debug.print("\nwrote {s}\n", .{OUTPUT_PATH});
}
