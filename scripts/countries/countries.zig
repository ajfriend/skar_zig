//! Step 2 of the top-100-countries aspect-ratio example
//! (mirrors scripts/states/states.zig).
//!
//! Reads scripts/countries/data/countries.json (produced by step 1), calls
//! `skar.solve` on every country's flattened vertex set, and writes
//! scripts/countries/data/countries_aspect.json with one detail record per
//! *converged* country (name, AR, cone axis `b`, shape matrix `A`, gap, outer
//! iters) — everything step 3's per-country ellipse plot needs.
//!
//! Some countries are harder than US states. We raise `max_outer` (see
//! MAX_OUTER below) so the slow-to-certify ones still converge at the strict
//! tolerance, and we treat any remaining non-convergence as a non-fatal skip:
//! a country whose points don't fit in any hemisphere would be proven
//! `.infeasible` by skar's Farkas/halfspace test, and we report+skip rather
//! than fail. The run still succeeds; only real solver errors (OOM /
//! SolveError) propagate.
//!
//! Wired as `zig build countries-aspect` (ReleaseFast). Run from the repo root;
//! the input/output paths are relative to CWD.

const std = @import("std");
const skar = @import("skar");

const INPUT_PATH = "scripts/countries/data/countries.json";
const OUTPUT_PATH = "scripts/countries/data/countries_aspect.json";

// Countries are large, well-conditioned spherical polygons like US states —
// nowhere near the DGGS sub-meter duality-gap floor — so the strict skar
// default tolerance is fine.
const COUNTRIES_GAP_TOL: f64 = 1e-6;

// ...but a few feasible countries with very elongated, transoceanic point sets
// need more than skar's default 100 outer iterations to certify the gap at
// 1e-6: e.g. France (incl. French Guiana, AR≈7.8) converges in ~138, Chile
// (incl. Easter Island, AR≈6.2) in ~109. The aspect ratios are already stable
// well before then — only the gap certificate lags — so 1000 is ample headroom
// and stays cheap (ReleaseFast, 100 countries, tens of ms total). Genuinely
// infeasible inputs are caught up front by the feasibility test regardless of
// this cap, so raising it can't manufacture a bogus cone.
const COUNTRIES_MAX_OUTER: u32 = 1000;

/// Shape of the input file. `rings` (the per-ring lon/lat for the plot) is
/// ignored here via `ignore_unknown_fields` — the solver only needs the
/// flattened `vertices`.
const InputJson = struct {
    source_url: []const u8,
    n_countries: u64,
    countries: []InputCountry,
};
const InputCountry = struct {
    name: []const u8,
    vertices: [][3]f64,
};

/// Tally of `skar.solve` outcomes across all countries.
const Counts = struct {
    converged: u32 = 0,
    infeasible: u32 = 0,
    did_not_converge: u32 = 0,
    /// `InputError.CoplanarInput` / `InsufficientPoints` — caller-side
    /// rejections, not solver failures.
    input_error: u32 = 0,
};

/// Full result for one converged country — kept for every converged country
/// (one PNG each), since step 3 needs `b`, `A`, and the AR to draw the
/// enclosing-cone ellipse.
const CountryResult = struct {
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

    // One arena holds the output-side allocations (the per-country result list
    // with its duped name strings). The parsed input owns its own arena and
    // is freed before we write.
    var out_arena = std.heap.ArenaAllocator.init(allocator);
    defer out_arena.deinit();
    const out_alloc = out_arena.allocator();

    var counts: Counts = .{};
    const results = try processCountries(allocator, out_alloc, &counts);

    try writeOutput(results);
    printSummary(results);
    printSkips(counts);
}

fn processCountries(
    gpa: std.mem.Allocator,
    out_alloc: std.mem.Allocator,
    counts: *Counts,
) ![]CountryResult {
    var file = try std.fs.cwd().openFile(INPUT_PATH, .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(gpa, 1 << 30);
    defer gpa.free(bytes);

    var parsed = try std.json.parseFromSlice(InputJson, gpa, bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const input = parsed.value;

    std.debug.print("solving {d} countries...\n", .{input.countries.len});

    var results = try std.ArrayListUnmanaged(CountryResult).initCapacity(out_alloc, input.countries.len);

    const t0 = std.time.milliTimestamp();
    for (input.countries) |country| {
        const outcome_or_err = skar.solve(gpa, country.vertices, .{
            .gap_tol = COUNTRIES_GAP_TOL,
            .max_outer = COUNTRIES_MAX_OUTER,
        });
        var outcome = outcome_or_err catch |err| switch (err) {
            error.InsufficientPoints, error.CoplanarInput, error.InvalidTolerance => {
                counts.input_error += 1;
                std.debug.print("  skip [{s}]: input_error ({s})\n", .{ country.name, @errorName(err) });
                continue;
            },
            else => return err, // OOM, SolveError — propagate
        };
        defer outcome.deinit();

        switch (outcome) {
            .converged => |c| {
                counts.converged += 1;
                try results.append(out_alloc, try captureResult(out_alloc, country, c));
            },
            // Expected for globe-spanning countries: no hemisphere contains all
            // the points, so there's no enclosing cone. Skip, don't fail.
            .infeasible => {
                counts.infeasible += 1;
                std.debug.print("  skip [{s}]: infeasible (exceeds a hemisphere)\n", .{country.name});
            },
            .did_not_converge => {
                counts.did_not_converge += 1;
                std.debug.print("  skip [{s}]: did_not_converge\n", .{country.name});
            },
        }
    }
    const dt_ms = std.time.milliTimestamp() - t0;
    std.debug.print("  done in {d}ms\n", .{dt_ms});

    return results.items;
}

fn captureResult(
    out_alloc: std.mem.Allocator,
    country: InputCountry,
    c: skar.Converged,
) !CountryResult {
    const name_copy = try out_alloc.dupe(u8, country.name);

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

fn writeOutput(results: []CountryResult) !void {
    const Wrapped = struct { countries: []CountryResult };
    const payload: Wrapped = .{ .countries = results };

    var file = try std.fs.cwd().createFile(OUTPUT_PATH, .{ .truncate = true });
    defer file.close();
    var write_buf: [4096]u8 = undefined;
    var writer = file.writer(&write_buf);
    try std.json.Stringify.value(payload, .{}, &writer.interface);
    try writer.interface.flush();
}

fn printSummary(results: []CountryResult) void {
    std.debug.print("\n{s:<28}  {s:>8}  {s:>5}\n", .{ "country", "AR", "iters" });
    for (results) |r| {
        std.debug.print("{s:<28}  {d:>8.4}  {d:>5}\n", .{ r.name, r.ar, r.outer_iters });
    }
    std.debug.print("\nwrote {s}\n", .{OUTPUT_PATH});
}

/// Report the converged/skipped split. Skips (infeasible / DNC / input_error)
/// are expected for some countries, so this never exits non-zero — it just
/// summarizes what made it into the plot set.
fn printSkips(counts: Counts) void {
    const skipped = counts.infeasible + counts.did_not_converge + counts.input_error;
    const n_total = counts.converged + skipped;
    std.debug.print(
        "\n{d}/{d} converged, {d} skipped (infeasible {d}, DNC {d}, input_error {d}).\n",
        .{ counts.converged, n_total, skipped, counts.infeasible, counts.did_not_converge, counts.input_error },
    );
}
