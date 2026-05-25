//! Minimal example: solve for the tightest enclosing cone of a
//! 3-point set on the unit sphere.
//!
//! Run with:
//!   zig build example
//!
//! Just the happy-path call: pass points, get the cone axis and
//! aspect ratio. See `examples/status.zig` for the full pattern
//! showing every Status outcome.

const std = @import("std");
const skar = @import("skar");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Three unit vectors at the standard basis directions — the
    // vertices of one octant of the unit sphere. By 3-fold symmetry
    // around (1,1,1)/√3 the optimal enclosing cone is circular
    // (aspect ratio = 1).
    const points = [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    };

    var info = try skar.solve(allocator, &points, .{});
    defer info.deinit();

    const b = info.b();
    std.debug.print("aspect ratio: {d:.6}\n", .{info.aspectRatio()});
    std.debug.print("cone axis:    ({d:.4}, {d:.4}, {d:.4})\n", .{ b.m[0], b.m[1], b.m[2] });
}
