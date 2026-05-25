//! Minimal example: solve for the tightest enclosing cone of a
//! 3-point set on the unit sphere.
//!
//! Run with:
//!   zig build ex-basic
//!
//! Just the happy-path call: pass points, switch on the outcome, and
//! print the cone axis + aspect ratio. See `examples/status.zig`
//! (`zig build ex-status`) for the full pattern showing every variant.

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

    var outcome = try skar.solve(allocator, &points, .{});
    defer outcome.deinit();

    // `solve` returns a tagged union. Switch on it before touching any
    // payload — the type system enforces that you handle every variant.
    // This example only cares about the happy path; see ex-status for
    // the full set.
    const c = switch (outcome) {
        .converged => |c| c,
        else => return error.UnexpectedOutcome,
    };
    const b = c.b();
    std.debug.print("aspect ratio: {d:.6}\n", .{c.aspectRatio()});
    std.debug.print("cone axis:    ({d:.4}, {d:.4}, {d:.4})\n", .{ b.m[0], b.m[1], b.m[2] });
}
