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

// The survey solves at gap_tol = 1e-3, NOT skar's strict 1e-6 default. At
// finest resolution the S2/A5 cells (sub-meter scatters, κ(A) ~ σ_max ~ 1e9)
// hit an f64 duality-gap floor of ~1e-4–1e-3 and return `.did_not_converge`
// at 1e-6 — correctly, since f64 can't certify a tighter bound. But their
// aspect ratios are accurate regardless of the gap (input-precision-limited,
// ~7 digits). Running at 1e-3 lets every cell converge, so the AR
// distribution (step 3 histograms) is complete rather than silently dropping
// ~22% of S2 and ~47% of A5. See tests/dggs_dnc_test.zig and
// SolveOptions.gap_tol for the floor's derivation.
const SURVEY_GAP_TOL: f64 = 1e-3;

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

/// Full details for one converged cell — kept for the best- and worst-AR
/// cells per system, since step 4 needs `b`, `A`, and the vertices to
/// project + draw the enclosing-cone cross-section.
const CellDetail = struct {
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
    /// Smallest-AR (most circular) and largest-AR converged cell — the
    /// two columns of the step-4 extremes plot.
    best: ?CellDetail,
    worst: ?CellDetail,
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
    checkConvergence(&results);
}

/// Verify the survey got full convergence: at gap_tol = 1e-3 every
/// finest-resolution cell should converge to a usable AR. Any non-converged
/// outcome (DNC / infeasible / input_error) means the distribution is
/// incomplete — print a per-system breakdown and exit non-zero so the
/// pipeline (`just dggs-all`) fails loudly instead of silently dropping cells.
fn checkConvergence(results: *const [SYSTEMS.len]SystemResult) void {
    var total_bad: u64 = 0;
    for (results, 0..) |r, i| {
        const c = r.counts;
        const bad = c.did_not_converge + c.infeasible + c.input_error;
        if (bad > 0) {
            std.debug.print(
                "  non-converged [{s}]: {d}/{d} (DNC {d}, infeasible {d}, input_error {d})\n",
                .{ SYSTEMS[i], bad, r.n_total, c.did_not_converge, c.infeasible, c.input_error },
            );
        }
        total_bad += bad;
    }
    if (total_bad > 0) {
        std.debug.print("\nFAIL: {d} cell(s) did not converge — survey AR distribution is incomplete.\n", .{total_bad});
        std.process.exit(1);
    }
    std.debug.print("\nconvergence: OK — every cell converged in all systems.\n", .{});
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
    var best: ?CellDetail = null;
    var worst: ?CellDetail = null;
    var first_dnc_printed = false;

    const t0 = std.time.milliTimestamp();
    for (input.cells) |cell| {
        const outcome_or_err = skar.solve(gpa, cell.vertices, .{ .gap_tol = SURVEY_GAP_TOL });
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
                    worst = try captureDetail(out_alloc, cell, c);
                }
                if (best == null or ar < best.?.ar) {
                    best = try captureDetail(out_alloc, cell, c);
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
        .best = best,
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

fn captureDetail(
    out_alloc: std.mem.Allocator,
    cell: InputCell,
    c: skar.Converged,
) !CellDetail {
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
        .outer_iters = c.diag.totalIters(),
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
