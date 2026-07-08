//! Tests for the EXPERIMENTAL alternative solver paths
//! (`SolveOptions.method`; `src/trust.zig`), incl. the away-step FW
//! solver kept for the record and the wide-cap fixtures with their
//! Clarabel reference aspect ratios.
//!
//! Coverage:
//!  - the wide-cap fixtures (tests/wide_cap_cells.zig) that the fast
//!    path limit-cycles on: `.joint` and `.auto` must converge, with
//!    the AR matching the Clarabel SDP cross-check;
//!  - easy-case agreement: `.joint` reproduces `.alternating`'s aspect ratio
//!    on a spread of bundled manifest cases;
//!  - `.auto` never changes the result on inputs where the alternating path
//!    already converges;
//!  - certificate sanity on a joint solve (λ ≥ 0, certified gap in
//!    [−NEG_GAP, gap_tol], primal feasibility ≤ roundoff).

const std = @import("std");
const sphar = @import("../src/root.zig");
const cases = @import("cases");
const wide = @import("wide_cap_cells.zig");

const GAP_TOL: f64 = 1e-6;
/// The certified gap bounds primal suboptimality, but AR is a ratio of
/// eigenvalues of a near-optimal iterate — allow a few × 1e-4 relative
/// against the Clarabel reference (which has its own ~1e-8 tolerance).
const AR_REF_REL_TOL: f64 = 1e-3;
/// Fast and joint converge to the same optimum; both carry ~gap-sized
/// slack in the AR.
const AR_AGREE_REL_TOL: f64 = 1e-4;

fn expectJointConverges(pts: []const [3]f64, method: sphar.Method, ref_ar: f64) !void {
    const allocator = std.testing.allocator;
    var outcome = try sphar.solve(allocator, pts, .{ .method = method });
    defer outcome.deinit();
    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    const c = outcome.converged;
    try std.testing.expect(@abs(c.gap) <= GAP_TOL);
    // Structured-cone primal feasibility: within roundoff of 0.
    try std.testing.expect(sphar.checkFeasibility(c, pts) <= 1e-12);
    // Cross-check against the Clarabel reference.
    try std.testing.expect(@abs(c.aspectRatio() - ref_ar) <= AR_REF_REL_TOL * ref_ar);
}

test "trust: wide-cap fixtures converge and match the Clarabel reference AR" {
    try expectJointConverges(&wide.CAP82_S1, .trust, wide.AR_CAP82_S1);
    try expectJointConverges(&wide.CAP85_S1, .trust, wide.AR_CAP85_S1);
    try expectJointConverges(&wide.CAP89_S3, .trust, wide.AR_CAP89_S3);
}

test "trust: wide-cap fixture iteration ceilings (CANARY-style)" {
    // Trust-region iteration guard on the wide-angle frontier (same
    // flag-don't-bump policy as the dggs canaries). Observed: 20 / 34 /
    // 14; ceilings leave headroom for FP drift across platforms while
    // catching a trust-region or oracle regression that turns the
    // frontier slow again.
    const allocator = std.testing.allocator;
    const fixtures = [_]struct { pts: []const [3]f64, ceiling: u32 }{
        .{ .pts = &wide.CAP82_S1, .ceiling = 30 },
        .{ .pts = &wide.CAP85_S1, .ceiling = 50 },
        .{ .pts = &wide.CAP89_S3, .ceiling = 25 },
    };
    for (fixtures) |f| {
        var o = try sphar.solve(allocator, f.pts, .{ .method = .trust });
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .converged);
        try std.testing.expect(o.converged.outer_iters <= f.ceiling);
    }
}

test "trust: agrees with alternating on bundled cases incl. extreme-kappa cells" {
    const allocator = std.testing.allocator;
    // Superset of the joint agreement list: the trust path certifies
    // in the scaled chart, so it must ALSO handle the finest-resolution
    // extreme-kappa cells that pure .joint floors on.
    const names = [_][]const u8{
        "hex",           "h3_res09",      "np20",        "np400",
        "ha_05",         "ha_14",         "dnc_small_wide",
        "h3_r12_ring10", "h3_r15_midLat", "h3_r15_pent", "h3_r15_ring10",
    };
    for (names) |name| {
        const case = cases.byName(name) orelse unreachable;
        var fast_out = try sphar.solve(allocator, case.points, .{});
        defer fast_out.deinit();
        var red_out = try sphar.solve(allocator, case.points, .{ .method = .trust });
        defer red_out.deinit();
        try std.testing.expect(std.meta.activeTag(fast_out) == .converged);
        try std.testing.expect(std.meta.activeTag(red_out) == .converged);
        const ar_f = fast_out.converged.aspectRatio();
        const ar_r = red_out.converged.aspectRatio();
        if (@abs(ar_f - ar_r) > AR_AGREE_REL_TOL * ar_f) {
            std.debug.print("trust/alternating AR mismatch case={s}: alternating={d:.10} trust={d:.10}\n", .{ name, ar_f, ar_r });
            return error.TrustAlternatingArMismatch;
        }
        try std.testing.expect(@abs(red_out.converged.gap) <= GAP_TOL);
    }
}

test "auto: falls back to trust on the wide-cap fixtures" {
    // Fast alone DNCs on these (pinned below); .auto must rescue them.
    try expectJointConverges(&wide.CAP82_S1, .auto, wide.AR_CAP82_S1);
    try expectJointConverges(&wide.CAP85_S1, .auto, wide.AR_CAP85_S1);
    try expectJointConverges(&wide.CAP89_S3, .auto, wide.AR_CAP89_S3);
}

test "alternating: wide-cap fixtures still DNC (the gap .auto exists to close)" {
    // Pins the motivating failure. If the alternating path ever starts
    // converging here, celebrate — and re-evaluate whether the joint
    // fallback is still needed (see docs/wide-cap-dnc-report.md).
    const allocator = std.testing.allocator;
    for ([_][]const [3]f64{ &wide.CAP82_S1, &wide.CAP85_S1, &wide.CAP89_S3 }) |pts| {
        var outcome = try sphar.solve(allocator, pts, .{});
        defer outcome.deinit();
        try std.testing.expect(std.meta.activeTag(outcome) == .did_not_converge);
    }
}

test "auto: identical to alternating when alternating converges" {
    const allocator = std.testing.allocator;
    const names = [_][]const u8{ "hex", "h3_res09", "np100" };
    for (names) |name| {
        const case = cases.byName(name) orelse unreachable;
        var fast_out = try sphar.solve(allocator, case.points, .{});
        defer fast_out.deinit();
        var auto_out = try sphar.solve(allocator, case.points, .{ .method = .auto });
        defer auto_out.deinit();
        // Same path executed ⇒ bit-identical outcome.
        try std.testing.expectEqual(fast_out.converged.outer_iters, auto_out.converged.outer_iters);
        try std.testing.expectEqual(fast_out.converged.gap, auto_out.converged.gap);
        try std.testing.expectEqual(fast_out.converged.sigma, auto_out.converged.sigma);
    }
}

test "mveeFwAway: converges the design and keeps weights in the simplex" {
    // Bit-rot guard for the away-step FW solver, kept in-tree for the
    // record after the stage-1 experiment (docs/away-step-fw.md
    // "Stage 1 findings"): hazard-free by construction but slower than
    // pairwise as the trust oracle. This pins its correctness so the
    // recorded findings stay reproducible.
    const skar_core = @import("../src/skar.zig");
    // Slightly irregular quad in the chart: optimal design weights are
    // non-uniform, support is all 4 points.
    const P = [_][2]f64{ .{ 1.0, 0.1 }, .{ -0.9, 0.2 }, .{ 0.15, 1.1 }, .{ -0.1, -1.0 } };
    var Ql: [4]sphar.Vec3 = undefined;
    var w = [_]f64{ 0.25, 0.25, 0.25, 0.25 };
    skar_core.mveeFwAway(&P, 200, 1e-10, &Ql, &w);

    var sum: f64 = 0;
    for (w) |wi| {
        try std.testing.expect(wi >= 0);
        sum += wi;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);

    // Design optimality: g_i = q_i' S^-1 q_i within tol of 3 on the
    // support (Kiefer-Wolfowitz).
    var S = sphar.Mat3.zero;
    for (Ql, 0..) |q, i| S.addSymRank1(w[i], q);
    const L = S.cholesky().?;
    for (Ql, 0..) |q, i| {
        if (w[i] > 1e-9) {
            const gi = q.dot(L.solve(q));
            try std.testing.expect(@abs(gi - 3.0) < 1e-6);
        }
    }
}

test "trust: certificate sanity on a wide-cap solve" {
    const allocator = std.testing.allocator;
    var outcome = try sphar.solve(allocator, &wide.CAP85_S1, .{ .method = .trust });
    defer outcome.deinit();
    const c = outcome.converged;
    // Weak duality: certified gap is non-negative up to FP noise.
    try std.testing.expect(c.gap >= -1e-10);
    // Dual multipliers are non-negative and the cert carries at least
    // the >= 3 points any non-degenerate cone needs.
    try std.testing.expect(c.cert.indices.len >= 3);
    for (c.cert.lambdas) |lam| try std.testing.expect(lam >= 0);
    // Indices point into the caller's array.
    for (c.cert.indices) |idx| try std.testing.expect(idx < wide.CAP85_S1.len);
}
