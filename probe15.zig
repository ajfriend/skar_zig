//! Temporary probe 15: diagnose reduced@1e-3 DNCs on a5 res-30 cells.
const std = @import("std");
const sphar = @import("src/root.zig");
const reduced = @import("src/reduced.zig");

const InputJson = struct {
    system: []const u8,
    resolution: i64,
    n_requested: u64,
    n_unique: u64,
    cells: []InputCell,
};
const InputCell = struct { id: []const u8, vertices: [][3]f64 };

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var file = try std.fs.cwd().openFile("scripts/dggs/data/a5.json", .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 30);
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(InputJson, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    var found: u32 = 0;
    for (parsed.value.cells) |cell| {
        var o = try sphar.solve(allocator, cell.vertices, .{ .gap_tol = 1e-3, .method = .reduced });
        defer o.deinit();
        if (o == .did_not_converge) {
            found += 1;
            std.debug.print("=== DNC cell {s} (n={d} pts): reduced it={d} gap={e:.3} ===\n", .{ cell.id, cell.vertices.len, o.did_not_converge.outer_iters, o.did_not_converge.gap });
            var of = try sphar.solve(allocator, cell.vertices, .{ .gap_tol = 1e-3 });
            defer of.deinit();
            if (of == .converged) {
                std.debug.print("  fast: converged it={d} gap={e:.3} AR={d:.8}\n", .{ of.converged.outer_iters, of.converged.gap, of.converged.aspectRatio() });
            }
            reduced.probe_trace = true;
            var ot = try sphar.solve(allocator, cell.vertices, .{ .gap_tol = 1e-3, .method = .reduced });
            reduced.probe_trace = false;
            ot.deinit();
            if (found >= 2) break;
        }
    }
}
