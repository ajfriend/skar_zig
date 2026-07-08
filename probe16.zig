//! Temporary probe 16: why does the initial reduced cert fail on a5 res-30?
const std = @import("std");
const sphar = @import("src/root.zig");
const skar = @import("src/skar.zig");
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
            std.debug.print("=== cell {s} (n={d}) ===\n", .{ cell.id, cell.vertices.len });
            skar.probe_gap_trace = true;
            std.debug.print("  reduced certs:\n", .{});
            var o2 = try sphar.solve(allocator, cell.vertices, .{ .gap_tol = 1e-3, .method = .reduced, .max_outer = 3 });
            o2.deinit();
            std.debug.print("  fast certs:\n", .{});
            var o3 = try sphar.solve(allocator, cell.vertices, .{ .gap_tol = 1e-3, .max_outer = 5 });
            switch (o3) {
                .converged => |c| std.debug.print("  fast converged it={d} gap={e:.3}\n", .{ c.outer_iters, c.gap }),
                else => {},
            }
            o3.deinit();
            skar.probe_gap_trace = false;
            if (found >= 2) break;
        }
    }
    _ = reduced.probe_trace;
}
