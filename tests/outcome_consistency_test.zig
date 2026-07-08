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
