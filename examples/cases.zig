//! Run one or all bundled cases through the solver.
//!
//! Pass arguments after `--`:
//!   zig build ex-cases -- hex      # run a single named case
//!   zig build ex-cases -- --all    # iterate the whole manifest
//!   zig build ex-cases             # no args: print usage + known cases
//!
//! Demonstrates how to reach into the bundled `cases` module from
//! user code: `cases.byName(...)` for a single lookup, `cases.all`
//! to iterate the full manifest.

const std = @import("std");
const skar = @import("skar");
const cases = @import("cases");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    if (argv.len != 2) {
        printUsage(argv[0]);
        return;
    }
    const arg = argv[1];
    if (std.mem.eql(u8, arg, "--all")) {
        try runAll(allocator);
    } else {
        try runOne(allocator, arg);
    }
}

fn printUsage(prog: []const u8) void {
    std.debug.print("usage: {s} <case-name> | --all\n", .{prog});
    std.debug.print("\nknown cases:\n", .{});
    for (cases.all) |entry| std.debug.print("  {s}\n", .{entry.name});
}

fn runOne(allocator: std.mem.Allocator, name: []const u8) !void {
    const case = cases.byName(name) orelse {
        std.debug.print("unknown case: {s}\n", .{name});
        std.debug.print("use --all to see what's available.\n", .{});
        return error.UnknownCase;
    };

    std.debug.print("{s}: {s}\n", .{ name, case.description });
    std.debug.print("  points: {d}\n", .{case.points.len});

    var outcome = try skar.solve(allocator, case.points, .{});
    defer outcome.deinit();

    switch (outcome) {
        .converged => |c| {
            std.debug.print("  converged: AR = {d:.6}, gap = {e:.3} after {d} iters\n", .{
                c.aspectRatio(), c.gap, c.outer_iters,
            });
        },
        .infeasible => |i| {
            std.debug.print("  infeasible: residual = {e:.3}, {d} active points\n", .{
                i.residual, i.cert.indices.len,
            });
        },
        .did_not_converge => |p| {
            std.debug.print("  did_not_converge: gap = {e:.3} after {d} iters\n", .{
                p.gap, p.outer_iters,
            });
        },
    }
}

fn runAll(allocator: std.mem.Allocator) !void {
    std.debug.print("{s:22}  {s:11}  {s}\n", .{ "case", "status", "metric" });
    std.debug.print("{s:22}  {s:11}  {s}\n", .{ "----------------------", "-----------", "------------" });
    for (cases.all) |entry| {
        var outcome = try skar.solve(allocator, entry.case.points, .{});
        defer outcome.deinit();
        switch (outcome) {
            .converged => |c| std.debug.print("{s:22}  {s:11}  AR={d:.6}\n", .{
                entry.name, "converged", c.aspectRatio(),
            }),
            .infeasible => |i| std.debug.print("{s:22}  {s:11}  residual={e:.3}\n", .{
                entry.name, "infeasible", i.residual,
            }),
            .did_not_converge => |p| std.debug.print("{s:22}  {s:11}  gap={e:.3}\n", .{
                entry.name, "DNC", p.gap,
            }),
        }
    }
}
