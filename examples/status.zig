//! Full status-handling example: shows the canonical switch on
//! `Info.status` and the per-branch inspection of the result.
//!
//! Run with:
//!   zig build example-status
//!
//! `solve` returns four distinct outcomes; only `.converged`
//! produces a usable cone. The other three are still valid library
//! responses — the caller dispatches on `status` to decide what to
//! do.

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
    var info = try skar.solve(allocator, &points, .{});
    defer info.deinit();

    switch (info.status) {
        .converged => {
            const b = info.b(); // Vec3 — cone axis
            const aspect = info.aspectRatio();
            std.debug.print("converged: aspect ratio = {d:.6}\n", .{aspect});
            std.debug.print("  cone axis     b = ({d:.4}, {d:.4}, {d:.4})\n", .{ b.m[0], b.m[1], b.m[2] });
            std.debug.print("  duality gap     = {e:.3}\n", .{info.cert.claimed_gap});
            std.debug.print("  outer iters     = {d}\n", .{info.outer_iters});
            std.debug.print("  active in cert  = {d} of {d} input points\n", .{ info.cert.indices.len, points.len });
        },
        .infeasible => {
            // No hemisphere contains all input points. `info.cert`
            // holds the Farkas certificate (λ ≥ 0, Σλ = 1, with
            // ‖Σ λᵢ xᵢ‖ near zero — the residual is `claimed_gap`).
            std.debug.print("infeasible: no hemisphere fits all points\n", .{});
            std.debug.print("  Farkas residual = {e:.3}\n", .{info.cert.claimed_gap});
        },
        .did_not_converge => {
            // Solver hit `max_outer` without closing the gap. The
            // last iterate is in info.Q/info.sigma but isn't a
            // verified certificate.
            std.debug.print("did_not_converge: hit max iterations ({d})\n", .{info.outer_iters});
            std.debug.print("  last gap = {e:.3}\n", .{info.cert.claimed_gap});
        },
        .coplanar_input => {
            // Input is rank-deficient (all points on a single great
            // circle). The SDP is structurally degenerate; the
            // solver bails before iterating.
            std.debug.print("coplanar_input: rank-deficient input\n", .{});
        },
    }
}
