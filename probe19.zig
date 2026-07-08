//! Temporary probe 19: states + countries surveys through .reduced
//! (validation-ledger item in docs/reduced-solver.md). Reads the same
//! JSON inputs as scripts/{states,countries}/*.zig; both surveys run at
//! the strict default gap_tol = 1e-6. Run from the repo root:
//!   zig run -O ReleaseFast probe19.zig
const std = @import("std");
const sphar = @import("src/root.zig");

const Region = struct { name: []const u8, vertices: [][3]f64 };

const StatesJson = struct {
    source_url: []const u8,
    n_states: u64,
    states: []Region,
};
const CountriesJson = struct {
    source_url: []const u8,
    n_countries: u64,
    countries: []Region,
};

const Tally = struct {
    converged: u32 = 0,
    dnc: u32 = 0,
    infeasible: u32 = 0,
    input_error: u32 = 0,
    iters_sum: u64 = 0,
    iters_max: u32 = 0,
    ms: i64 = 0,
};

fn runSet(allocator: std.mem.Allocator, label: []const u8, regions: []const Region) !void {
    const methods = [_]sphar.Method{ .fast, .reduced };
    var tallies: [2]Tally = .{ .{}, .{} };
    const ars = try allocator.alloc([2]f64, regions.len);
    defer allocator.free(ars);
    for (ars) |*a| a.* = .{ std.math.nan(f64), std.math.nan(f64) };

    for (methods, 0..) |method, mi| {
        const t0 = std.time.milliTimestamp();
        for (regions, 0..) |region, k| {
            var outcome = sphar.solve(allocator, region.vertices, .{ .method = method }) catch |err| switch (err) {
                error.InsufficientPoints, error.CoplanarInput, error.InvalidTolerance => {
                    tallies[mi].input_error += 1;
                    continue;
                },
                else => return err,
            };
            defer outcome.deinit();
            switch (outcome) {
                .converged => |c| {
                    tallies[mi].converged += 1;
                    ars[k][mi] = c.aspectRatio();
                    tallies[mi].iters_sum += c.outer_iters;
                    if (c.outer_iters > tallies[mi].iters_max) tallies[mi].iters_max = c.outer_iters;
                },
                .did_not_converge => {
                    tallies[mi].dnc += 1;
                    std.debug.print("  [{s}] DNC: {s} ({s})\n", .{ label, region.name, @tagName(method) });
                },
                .infeasible => tallies[mi].infeasible += 1,
            }
        }
        tallies[mi].ms = std.time.milliTimestamp() - t0;
    }

    var max_rel: f64 = 0;
    var max_rel_name: []const u8 = "";
    var only: [2]u32 = .{ 0, 0 };
    for (ars, 0..) |a, k| {
        const cf = !std.math.isNan(a[0]);
        const cr = !std.math.isNan(a[1]);
        if (cf and !cr) only[0] += 1;
        if (cr and !cf) only[1] += 1;
        if (cf and cr) {
            const rel = @abs(a[0] - a[1]) / a[0];
            if (rel > max_rel) {
                max_rel = rel;
                max_rel_name = regions[k].name;
            }
        }
    }

    std.debug.print("== {s} ({d} regions, gap_tol=1e-6) ==\n", .{ label, regions.len });
    for (methods, 0..) |method, mi| {
        const t = tallies[mi];
        const mean: f64 = if (t.converged > 0) @as(f64, @floatFromInt(t.iters_sum)) / @as(f64, @floatFromInt(t.converged)) else 0;
        std.debug.print("  {s:8} converged={d:3} DNC={d} inf={d} ipe={d} iters mean={d:.1} max={d} {d}ms\n", .{
            @tagName(method), t.converged, t.dnc, t.infeasible, t.input_error, mean, t.iters_max, t.ms,
        });
    }
    std.debug.print("  parity: fast-only={d} reduced-only={d} maxRelΔAR={e:.2} ({s})\n\n", .{ only[0], only[1], max_rel, max_rel_name });
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    {
        var file = try std.fs.cwd().openFile("scripts/states/data/states.json", .{});
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 1 << 30);
        defer allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(StatesJson, allocator, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        try runSet(allocator, "states", parsed.value.states);
    }
    {
        var file = try std.fs.cwd().openFile("scripts/countries/data/countries.json", .{});
        defer file.close();
        const bytes = try file.readToEndAlloc(allocator, 1 << 30);
        defer allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(CountriesJson, allocator, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        try runSet(allocator, "countries", parsed.value.countries);
    }
}
