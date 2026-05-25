//! Minimal example: solve for the tightest enclosing cone of a small
//! point set on the unit sphere.
//!
//! Run with:
//!   zig build example
//!
//! Demonstrates the canonical happy-path call shape, how to read
//! `Info` for each status, and how to clean up via `defer
//! info.deinit()`. Error handling uses `try` — `solve` can also
//! return `InputError` (caller-bad-input) or `SolveError`
//! (internal-correctness violation) on top of `OutOfMemory`; both
//! propagate via `try` and `main` returns them.

const std = @import("std");
const skar = @import("skar");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 6 unit vectors arranged as a regular hexagon at half-angle
    // ~11.5° around the +z axis. The optimal enclosing cone is a
    // circular cone (aspect ratio = 1).
    const r = 0.19866933079506122; // sin(11.46°)
    const z = 0.98006657784124163; // cos(11.46°)
    const cos60 = 0.5;
    const sin60 = @sqrt(3.0) / 2.0;
    const points = [_][3]f64{
        .{ r,             0,             z },
        .{ r * cos60,     r * sin60,     z },
        .{ -r * cos60,    r * sin60,     z },
        .{ -r,            0,             z },
        .{ -r * cos60,    -r * sin60,    z },
        .{ r * cos60,     -r * sin60,    z },
    };

    // Solve with default options. Pass `.{}` for sensible defaults;
    // override individual fields with named-field syntax to taste:
    //   .{ .gap_tol = 1e-9 }
    //   .{ .coplanarity_tol = -1 }   // disable the coplanarity check
    //   .{ .max_outer = 500 }
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
