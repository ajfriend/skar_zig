//! Regressions for the consistency of non-converged outcomes
//! (2026-07-08 pre-release review, findings 2 and 3): DNC snapshots
//! must be self-consistent (one iterate's axis + eigenvectors + gap),
//! and the internal "certificate construction failed" sentinel must
//! never certify.

const std = @import("std");
const skar = @import("../src/root.zig");
const cases = @import("cases");

test "budget-limited trust DNC returns a self-consistent certified snapshot" {
    // Finding 2: with a small max_outer on a hard input, the trust
    // path used to exhaust its budget on far-field accepted steps
    // (certification is gated on pred), skip RECERT, and pair the
    // FINAL axis with the INITIAL axis's eigenvectors — a
    // non-orthonormal Q (column dots up to 1e-1 measured) and a stale
    // gap. The outcome must now be the last certified iterate:
    // Q orthonormal to roundoff.
    const allocator = std.testing.allocator;
    const pts = (cases.byName("wide_cap89") orelse unreachable).points;
    var o = try skar.solve(allocator, pts, .{ .method = .trust, .max_outer = 4, .gap_tol = 1e-9 });
    defer o.deinit();
    // Status may be either (converged if the budget suffices on some
    // platform); the invariant under test is snapshot consistency.
    const Qm = switch (o) {
        .converged => |c| c.Q,
        .did_not_converge => |p| p.Q,
        .infeasible => unreachable,
    };
    const c0 = Qm.col(0);
    const c1 = Qm.col(1);
    const c2 = Qm.col(2);
    try std.testing.expect(@abs(c0.dot(c1)) < 1e-10);
    try std.testing.expect(@abs(c0.dot(c2)) < 1e-10);
    try std.testing.expect(@abs(c1.dot(c2)) < 1e-10);
    try std.testing.expect(@abs(c0.dot(c0) - 1.0) < 1e-10);
}

test "the no-certificate sentinel never certifies, and absurd gap_tol is rejected" {
    // Finding 3: dualityGapConstructed signals "certificate
    // construction failed" with gap = 1e30; that sentinel used to
    // satisfy gapConverged for a legal-but-absurd gap_tol (1e300),
    // manufacturing Outcome.converged with an EMPTY cert. Now:
    // gap_tol >= 1e30 is InvalidTolerance, and the sentinel never
    // certifies regardless.
    const allocator = std.testing.allocator;
    // Coplanar great-circle points with the gate disabled: the
    // alternating path runs to max_outer with the sentinel gap.
    const half = (170.0 / 2.0) * std.math.pi / 180.0;
    const pts = [_][3]f64{
        .{ @cos(-half), @sin(-half), 0.0 },
        .{ 1.0, 0.0, 0.0 },
        .{ @cos(half), @sin(half), 0.0 },
    };

    try std.testing.expectError(
        skar.InputError.InvalidTolerance,
        skar.solve(allocator, &pts, .{ .gap_tol = 1e300, .coplanarity_tol = -1, .method = .alternating }),
    );

    // Just below the validation cap: legal, but the sentinel still
    // must not certify — the outcome stays DNC with the documented
    // sentinel value and an empty cert.
    var o = try skar.solve(allocator, &pts, .{ .gap_tol = 1e29, .coplanarity_tol = -1, .max_outer = 10, .method = .alternating });
    defer o.deinit();
    try std.testing.expect(std.meta.activeTag(o) == .did_not_converge);
    try std.testing.expectEqual(@as(f64, 1e30), o.did_not_converge.gap);
    try std.testing.expectEqual(@as(usize, 0), o.did_not_converge.cert.indices.len);
}

test "alternating DNC snapshot is also self-consistent" {
    // Same invariant on the alternating path (whose staleness was one
    // damped axis step rather than several TR steps — smaller, but the
    // same class). Wide caps DNC structurally under .alternating.
    const allocator = std.testing.allocator;
    const pts = (cases.byName("wide_cap82") orelse unreachable).points;
    var o = try skar.solve(allocator, pts, .{ .method = .alternating });
    defer o.deinit();
    try std.testing.expect(std.meta.activeTag(o) == .did_not_converge);
    const Qm = o.did_not_converge.Q;
    try std.testing.expect(@abs(Qm.col(0).dot(Qm.col(1))) < 1e-10);
    try std.testing.expect(@abs(Qm.col(0).dot(Qm.col(2))) < 1e-10);
}
