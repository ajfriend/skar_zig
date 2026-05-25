//! Full outcome-handling example: shows the canonical switch on the
//! `Outcome` tagged union and per-variant inspection.
//!
//! Run with:
//!   zig build ex-status
//!
//! `solve` returns three distinct outcomes; only `.converged` produces
//! a usable cone. `.infeasible` and `.did_not_converge` are still
//! valid library responses — the caller dispatches on the union tag
//! to decide what to do. Structural input problems (too few points,
//! rank-deficient X) propagate as `InputError` via `try`.

const std = @import("std");
const skar = @import("skar");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const points = [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    };

    // Solve with default options. Pass `.{}` for sensible defaults;
    // override individual fields with named-field syntax:
    //   .{ .gap_tol = 1e-9 }
    //   .{ .coplanarity_tol = -1 }   // disable the coplanarity check
    //   .{ .max_outer = 500 }
    //
    // `solve` can also return `InputError` (caller passed bad
    // arguments — too few points, bad tolerance), `SolveError`
    // (library internal-correctness violation), or `OutOfMemory`.
    // All three propagate via `try`.
    var outcome = try skar.solve(allocator, &points, .{});
    defer outcome.deinit();

    switch (outcome) {
        .converged => |c| {
            // `c` is a `Converged` — accessors like `aspectRatio()`,
            // `b()`, `A()` live here, not on `Outcome` directly. The
            // type system prevents calling them without first switching.
            const b = c.b(); // Vec3 — cone axis
            const aspect = c.aspectRatio();
            std.debug.print("converged: aspect ratio = {d:.6}\n", .{aspect});
            std.debug.print("  cone axis     b = ({d:.4}, {d:.4}, {d:.4})\n", .{ b.m[0], b.m[1], b.m[2] });
            std.debug.print("  duality gap     = {e:.3}\n", .{c.gap});
            std.debug.print("  outer iters     = {d}\n", .{c.outer_iters});
            std.debug.print("  active in cert  = {d} of {d} input points\n", .{ c.cert.indices.len, points.len });
        },
        .infeasible => |i| {
            // No hemisphere contains all input points. `i.cert` is a
            // Farkas certificate (λ ≥ 0, Σλ = 1, with ‖Σ λᵢ xᵢ‖ near
            // zero); the witness magnitude lives on `i.residual`.
            std.debug.print("infeasible: no hemisphere fits all points\n", .{});
            std.debug.print("  Farkas residual = {e:.3}\n", .{i.residual});
        },
        .did_not_converge => |p| {
            // Solver hit `max_outer` without closing the gap. The
            // last iterate is in p.Q / p.sigma but isn't a verified
            // certificate; p.last_gap holds the last computed gap.
            std.debug.print("did_not_converge: hit max iterations ({d})\n", .{p.outer_iters});
            std.debug.print("  last gap = {e:.3}\n", .{p.last_gap});
        },
    }
}
