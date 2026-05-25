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
    const pts = [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    };
    var outcome = try sphar.solve(allocator, &pts, .{
        .max_outer = 1,
        .gap_tol = 1e-20,
    });
    defer outcome.deinit();
    try std.testing.expect(std.meta.activeTag(outcome) == .did_not_converge);
    const p = outcome.did_not_converge;
    try std.testing.expectEqual(@as(u32, 1), p.outer_iters);
    // Convergence is `@abs(gap) <= gap_tol`. So DNC means `@abs(gap) > tol`,
    // not `gap > tol` — gap can be FP-noise-negative on near-converged inputs.
    try std.testing.expect(@abs(p.gap) > 1e-20);
}
