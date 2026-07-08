//! Temporary probe 21: trace the New York regression (stall exit).
const std = @import("std");
const sphar = @import("src/root.zig");
const reduced = @import("src/reduced.zig");

const StatesJson = struct {
    source_url: []const u8,
    n_states: u64,
    states: []Region,
};
const Region = struct { name: []const u8, vertices: [][3]f64 };

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var file = try std.fs.cwd().openFile("scripts/states/data/states.json", .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 30);
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(StatesJson, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    for (parsed.value.states) |st| {
        if (!std.mem.eql(u8, st.name, "New York")) continue;
        std.debug.print("New York: n={d} vertices\n", .{st.vertices.len});
        reduced.probe_trace = true;
        var o = try sphar.solve(allocator, st.vertices, .{ .method = .reduced, .max_outer = 40 });
        reduced.probe_trace = false;
        switch (o) {
            .converged => |c| std.debug.print("=> converged it={d} gap={e:.3} AR={d:.6}\n", .{ c.outer_iters, c.gap, c.aspectRatio() }),
            .did_not_converge => |p| std.debug.print("=> DNC it={d} gap={e:.3} AR~{d:.6}\n", .{ p.outer_iters, p.gap, p.sigma[2] / p.sigma[1] }),
            .infeasible => {},
        }
        o.deinit();
    }
}
