//! Regression for the A5 resolution-0 DNC (outer-iteration count scaling
//! with boundary-point count). See `docs/a5_res0_dnc_report.md` and the
//! `algo.INNER_FW_BOOST_MIN_POINTS` doc-comment in `src/config.zig`.
//!
//! Two cases, both at skar's strict default (`gap_tol = 1e-6`,
//! `max_outer = 100`):
//!   - the "many vertices" cell: the full 320-point `cell_to_boundary`
//!     polygon. Pre-fix this DNC'd (needed ~145 outer iterations); with the
//!     gated inner-FW boost it converges in ~6.
//!   - the "few vertices" cell: the same cell reduced to its 5 pentagon
//!     corners. Below the boost threshold, so it exercises the bit-identical
//!     1-step path — it always converged, and pins that both representations
//!     of the same cell certify the same enclosing cone.

const std = @import("std");
const skar = @import("../src/root.zig");
const dense = @import("a5_res0_cells_dense.zig");

// The 5 main pentagon corners of A5 res-0 cell 200000000000000 — i.e.
// dense.A5_RES0_CELLS[0] decimated to its corners (note the shared z: the
// corners are coplanar on a small circle). Same cell, ~5 points instead of 320.
const A5_RES0_CORNERS = [_][3]f64{
    .{ -0.3809559340728538, -0.47044139975172183, 0.7959632313708467 },
    .{ 0.3296945010323943, -0.5076850109021148, 0.7959632313708467 },
    .{ 0.5847183416148108, 0.1566748074353539, 0.7959632313708467 },
    .{ 0.03168130793103096, 0.6045153670780082, 0.7959632313708467 },
    .{ -0.5651382165053821, 0.2169362361404745, 0.7959632313708467 },
};

test "a5 res-0 dense (320-pt) boundary cells converge at the strict default" {
    const allocator = std.testing.allocator;
    // All 12 base cells, each the full 320-point boundary. Pre-fix every one
    // returned .did_not_converge at the default cap; all must now certify.
    var it_max: u32 = 0;
    for (dense.A5_RES0_CELLS) |cell| {
        var o = try skar.solve(allocator, cell, .{}); // default gap_tol = 1e-6, max_outer = 100
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .converged);
        try std.testing.expect(o.converged.gap <= 1e-6);
        if (o.converged.outer_iters > it_max) it_max = o.converged.outer_iters;
    }
    // Performance-regression guard on the BOOSTED path (this is what the gated
    // inner-FW fix buys). With the boost these converge in a handful of outer
    // iterations (observed max 7); pre-fix they needed ~145 (> the 100 cap →
    // DNC), and the rejected `max_outer = 200` alternative "converges" but
    // grinds ~145. The ceiling sits far below that failure band, so it catches
    // either regression while tolerating ±a-few-iter drift across platforms.
    try std.testing.expect(it_max <= 20);
}

test "a5 res-0 sparse (5-corner) cell converges and matches the dense AR" {
    const allocator = std.testing.allocator;

    var sparse = try skar.solve(allocator, &A5_RES0_CORNERS, .{});
    defer sparse.deinit();
    try std.testing.expect(std.meta.activeTag(sparse) == .converged);
    try std.testing.expect(sparse.converged.gap <= 1e-6);
    // The sparse cell (nw = 5 ≤ the boost threshold) stays on the bit-identical
    // 1-step path and must remain trivial (observed 1 outer iteration); guard
    // against the boost mistakenly engaging here or the small-input path
    // regressing.
    try std.testing.expect(sparse.converged.outer_iters <= 4);

    // Same cell as dense.A5_RES0_CELLS[0] (the 320-point boundary of
    // 200000000000000) → same minimum-volume enclosing cone, so the aspect
    // ratio must agree. AR is input-precision-limited (~7 digits); loose guard.
    var full = try skar.solve(allocator, dense.A5_RES0_CELLS[0], .{});
    defer full.deinit();
    try std.testing.expect(std.meta.activeTag(full) == .converged);
    try std.testing.expectApproxEqAbs(
        full.converged.aspectRatio(),
        sparse.converged.aspectRatio(),
        1e-4,
    );
}
