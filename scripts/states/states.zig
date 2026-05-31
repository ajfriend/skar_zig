//! Step 2 of the US-states aspect-ratio example (mirrors scripts/dggs/aspect.zig).
//!
//! Reads scripts/states/data/states.json (produced by step 1), calls
//! `skar.solve` on every state's flattened vertex set, and writes
//! scripts/states/data/states_aspect.json with one detail record per state
//! (name, AR, cone axis `b`, shape matrix `A`, gap, outer iters) — everything
//! step 3's per-state ellipse plot needs.
//!
//! Wired as `zig build states-aspect` (ReleaseFast). Run from the repo root;
//! the input/output paths are relative to CWD.

const std = @import("std");
const skar = @import("skar");

const INPUT_PATH = "scripts/states/data/states.json";
const OUTPUT_PATH = "scripts/states/data/states_aspect.json";

// US states are large, well-conditioned spherical polygons — nowhere near the
// sub-meter κ·ε duality-gap floor that forced the DGGS survey down to 1e-3
// (see scripts/dggs/aspect.zig). The strict skar default converges cleanly, so
// keep it.
const STATES_GAP_TOL: f64 = 1e-6;

/// Shape of the input file. `rings` (the per-ring lon/lat for the plot) is
/// ignored here via `ignore_unknown_fields` — the solver only needs the
/// flattened `vertices`.
const InputJson = struct {
    source_url: []const u8,
    n_states: u64,
    states: []InputState,
};
const InputState = struct {
    name: []const u8,
    vertices: [][3]f64,
};

/// Tally of `skar.solve` outcomes across all states.
const Counts = struct {
    converged: u32 = 0,
    infeasible: u32 = 0,
    did_not_converge: u32 = 0,
    /// `InputError.CoplanarInput` / `InsufficientPoints` — caller-side
    /// rejections, not solver failures.
    input_error: u32 = 0,
};

/// Full result for one converged state — kept for every state (one PNG each),
/// since step 3 needs `b`, `A`, and the AR to draw the enclosing-cone ellipse.
const StateResult = struct {
    name: []const u8,
    ar: f64,
    b: [3]f64,
    /// Row-major 3×3, materialized from `Converged.A()`.
    A: [3][3]f64,
    gap: f64,
    outer_iters: u32,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // One arena holds the output-side allocations (the per-state result list
    // with its duped name strings). The parsed input owns its own arena and
    // is freed before we write.
    var out_arena = std.heap.ArenaAllocator.init(allocator);
    defer out_arena.deinit();
    const out_alloc = out_arena.allocator();

    var counts: Counts = .{};
    const results = try processStates(allocator, out_alloc, &counts);

    try writeOutput(results);
    printSummary(results);
    checkConvergence(counts);
}

fn processStates(
    gpa: std.mem.Allocator,
    out_alloc: std.mem.Allocator,
    counts: *Counts,
) ![]StateResult {
    var file = try std.fs.cwd().openFile(INPUT_PATH, .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(gpa, 1 << 30);
    defer gpa.free(bytes);

    var parsed = try std.json.parseFromSlice(InputJson, gpa, bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const input = parsed.value;

    std.debug.print("solving {d} states...\n", .{input.states.len});

    var results = try std.ArrayListUnmanaged(StateResult).initCapacity(out_alloc, input.states.len);

    const t0 = std.time.milliTimestamp();
    for (input.states) |state| {
        const outcome_or_err = skar.solve(gpa, state.vertices, .{ .gap_tol = STATES_GAP_TOL });
        var outcome = outcome_or_err catch |err| switch (err) {
            error.InsufficientPoints, error.CoplanarInput, error.InvalidTolerance => {
                counts.input_error += 1;
                std.debug.print("  input_error [{s}]: {s}\n", .{ state.name, @errorName(err) });
                continue;
            },
            else => return err, // OOM, SolveError — propagate
        };
        defer outcome.deinit();

        switch (outcome) {
            .converged => |c| {
                counts.converged += 1;
                try results.append(out_alloc, try captureResult(out_alloc, state, c));
            },
            .infeasible => {
                counts.infeasible += 1;
                std.debug.print("  infeasible [{s}]\n", .{state.name});
            },
            .did_not_converge => {
                counts.did_not_converge += 1;
                std.debug.print("  did_not_converge [{s}]\n", .{state.name});
            },
        }
    }
    const dt_ms = std.time.milliTimestamp() - t0;
    std.debug.print("  done in {d}ms\n", .{dt_ms});

    return results.items;
}

fn captureResult(
    out_alloc: std.mem.Allocator,
    state: InputState,
    c: skar.Converged,
) !StateResult {
    const name_copy = try out_alloc.dupe(u8, state.name);

    const bv = c.b();
    const A_mat = c.A();
    var A_rows: [3][3]f64 = undefined;
    for (0..3) |r| for (0..3) |k| {
        A_rows[r][k] = A_mat.m[r * 3 + k];
    };

    return .{
        .name = name_copy,
        .ar = c.aspectRatio(),
        .b = .{ bv.m[0], bv.m[1], bv.m[2] },
        .A = A_rows,
        .gap = c.gap,
        .outer_iters = c.outer_iters,
    };
}

fn writeOutput(results: []StateResult) !void {
    const Wrapped = struct { states: []StateResult };
    const payload: Wrapped = .{ .states = results };

    var file = try std.fs.cwd().createFile(OUTPUT_PATH, .{ .truncate = true });
    defer file.close();
    var write_buf: [4096]u8 = undefined;
    var writer = file.writer(&write_buf);
    try std.json.Stringify.value(payload, .{}, &writer.interface);
    try writer.interface.flush();
}

fn printSummary(results: []StateResult) void {
    std.debug.print("\n{s:<22}  {s:>8}  {s:>5}\n", .{ "state", "AR", "iters" });
    for (results) |r| {
        std.debug.print("{s:<22}  {d:>8.4}  {d:>5}\n", .{ r.name, r.ar, r.outer_iters });
    }
    std.debug.print("\nwrote {s}\n", .{OUTPUT_PATH});
}

/// Every US state fits well within a hemisphere and is well-conditioned, so at
/// the strict default tolerance all should converge. Any non-converged outcome
/// (DNC / infeasible / input_error) means corrupt input or a regression — print
/// the tally and exit non-zero so `just states-all` fails loudly.
fn checkConvergence(counts: Counts) void {
    const bad = counts.did_not_converge + counts.infeasible + counts.input_error;
    const n_total = counts.converged + bad;
    if (bad > 0) {
        std.debug.print(
            "\nFAIL: {d}/{d} state(s) did not converge (DNC {d}, infeasible {d}, input_error {d}).\n",
            .{ bad, n_total, counts.did_not_converge, counts.infeasible, counts.input_error },
        );
        std.process.exit(1);
    }
    std.debug.print("\nconvergence: OK — all {d} states converged.\n", .{counts.converged});
}
