//! Solver-contract tests that don't depend on case fixtures.
//! Cases-driven tests live in `tests/cases/cases_test.zig`; this
//! file holds synthetic / property tests of `solve`'s contract.

const std = @import("std");
const sphar = @import("../src/root.zig");

test "max_outer cap forces DidNotConverge on any input" {
    // Deterministic DNC: an unattainably tight gap_tol with max_outer
    // = 1 guarantees the outer loop hits its cap without closing the
    // gap, regardless of input geometry. Pins the contract that solve
    // returns `.did_not_converge` with `outer_iters == max_outer` and
    // a real (non-degenerate) last-iterate gap. No real case in
    // tests/cases/ currently DNCs at default tolerance, so this is
    // the only end-to-end DNC coverage.
    const allocator = std.testing.allocator;
    // Deliberately IRREGULAR triangle: a perfectly symmetric frame
    // certifies with gap ≈ 0 exactly (measured ~5e-32 on the trust
    // path's eager certificate), which slips under even a 1e-20
    // tolerance. Irregularity keeps the achievable gap at the f64
    // noise floor (~1e-16) ≫ 1e-20 for both paths.
    const pts = [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0.1, 0.97, 0.2 },
        .{ -0.2, 0.3, 0.93 },
    };
    // The cap is a shared contract: both paths must respect it (the
    // trust path counts opening + TR + recert iterations against it).
    for ([_]sphar.Method{ .alternating, .trust }) |method| {
        var outcome = try sphar.solve(allocator, &pts, .{
            .max_outer = 1,
            .gap_tol = 1e-20,
            .method = method,
        });
        defer outcome.deinit();
        try std.testing.expect(std.meta.activeTag(outcome) == .did_not_converge);
        const p = outcome.did_not_converge;
        try std.testing.expectEqual(@as(u32, 1), p.diag.totalIters());
        // Convergence is `@abs(gap) <= gap_tol`. So DNC means `@abs(gap) > tol`,
        // not `gap > tol` — gap can be FP-noise-negative on near-converged inputs.
        try std.testing.expect(@abs(p.gap) > 1e-20);
    }
}

test "duplicate-heavy input with hull preprocessing disabled still converges" {
    // Regression for pre-release review finding 7: farthestPointSeed's
    // greedy pick loop could re-pick an already-chosen index when
    // fewer distinct positions than SEED_SPARSE_K remain (only
    // reachable with n_hull disabled — the hull dedupes exact
    // duplicates), collapsing two weight shares and seeding Σw = 0.8.
    // FW's pairwise steps and Newton's KKT both conserve the deficit,
    // which surfaced as a hard DNC (gap ~0.069) on an input whose
    // optimal AR is exactly 1.
    const allocator = std.testing.allocator;
    // 4 corners of a square on a small cap around +z, 5 copies each:
    // 20 points (> SEED_SPARSE_MIN_POINTS) but 4 distinct positions.
    const r = 0.1;
    const zc = @sqrt(1.0 - 2.0 * r * r);
    const corners = [_][3]f64{
        .{ r, r, zc }, .{ -r, r, zc }, .{ -r, -r, zc }, .{ r, -r, zc },
    };
    var pts: [20][3]f64 = undefined;
    for (0..20) |i| pts[i] = corners[i % 4];

    for ([_]sphar.Method{ .alternating, .trust }) |method| {
        var o = try sphar.solve(allocator, &pts, .{ .n_hull = -1, .method = method });
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .converged);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), o.converged.aspectRatio(), 1e-6);
    }
}
