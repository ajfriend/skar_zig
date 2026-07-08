//! Tests for solver path selection
//! (`SolveOptions.method`; `src/trust.zig`), incl. the away-step FW
//! solver kept for the record.
//!
//! Coverage:
//!  - the wide-cap manifest cases (tests/cases/zon/wide_cap*.zon) that
//!    the alternating path limit-cycles on: `.trust` must converge
//!    within iteration ceilings, matching the Clarabel SDP cross-check;
//!  - easy-case agreement: `.trust` reproduces `.alternating`'s aspect ratio
//!    on a spread of bundled manifest cases;
//!  - `.auto` is a pure alias for `Method.recommended` (currently
//!    `.trust`) — identical outcomes, and the resolution pinned;
//!  - certificate sanity on a trust solve (λ ≥ 0, certified gap in
//!    [−NEG_GAP, gap_tol], primal feasibility ≤ roundoff).

const std = @import("std");
const sphar = @import("../src/root.zig");
const cases = @import("cases");

const GAP_TOL: f64 = 1e-6;
/// The certified gap bounds primal suboptimality, but AR is a ratio of
/// eigenvalues of a near-optimal iterate — allow a few × 1e-4 relative
/// against the Clarabel reference (which has its own ~1e-8 tolerance).
const AR_REF_REL_TOL: f64 = 1e-3;
/// Both solvers converge to the same optimum; both carry ~gap-sized
/// slack in the AR.
const AR_AGREE_REL_TOL: f64 = 1e-4;

/// Points of a wide-cap manifest case (tests/cases/zon/wide_cap*.zon —
/// the single home of the fixture data).
fn wideCap(name: []const u8) []const [3]f64 {
    return (cases.byName(name) orelse unreachable).points;
}

test "trust: wide-cap iteration ceilings (CANARY-style) + Clarabel cross-check" {
    // Trust-region iteration guard on the wide-angle frontier (same
    // flag-don't-bump policy as the dggs canaries). Observed: 20 / 34 /
    // 14; ceilings leave headroom for FP drift across platforms while
    // catching a trust-region or oracle regression that turns the
    // frontier slow again.
    //
    // The AR is cross-checked against the CLARABEL reference from the
    // wide-cap investigation's SDP probe — independent provenance from
    // the solver-derived `.expected.ar` pins the manifest loop
    // (tests/cases/cases_test.zig) checks on the same cases. Explicit
    // `.method = .trust` (not the default), so this keeps guarding the
    // trust path even if `.auto` is ever re-pointed.
    const allocator = std.testing.allocator;
    const fixtures = [_]struct { name: []const u8, ceiling: u32, clarabel_ar: f64 }{
        .{ .name = "wide_cap82", .ceiling = 30, .clarabel_ar = 1.159634 },
        .{ .name = "wide_cap85", .ceiling = 50, .clarabel_ar = 1.269181 },
        .{ .name = "wide_cap89", .ceiling = 25, .clarabel_ar = 1.542028 },
    };
    for (fixtures) |f| {
        const pts = wideCap(f.name);
        var o = try sphar.solve(allocator, pts, .{ .method = .trust });
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .converged);
        const c = o.converged;
        try std.testing.expect(@abs(c.gap) <= GAP_TOL);
        // Structured-cone primal feasibility: within roundoff of 0.
        try std.testing.expect(sphar.checkFeasibility(c, pts) <= 1e-12);
        try std.testing.expect(@abs(c.aspectRatio() - f.clarabel_ar) <= AR_REF_REL_TOL * f.clarabel_ar);
        try std.testing.expect(c.diag.totalIters() <= f.ceiling);
    }
}

test "trust: agrees with alternating on bundled cases incl. extreme-kappa cells" {
    const allocator = std.testing.allocator;
    // Includes the finest-resolution extreme-kappa cells: the trust
    // path certifies in the scaled chart, so it must handle the cells
    // whose raw-3D certification floors (the removed joint IPM's
    // failure regime).
    const names = [_][]const u8{
        "hex",           "h3_res09",      "np20",        "np400",
        "ha_05",         "ha_14",         "dnc_small_wide",
        "h3_r12_ring10", "h3_r15_midLat", "h3_r15_pent", "h3_r15_ring10",
    };
    for (names) |name| {
        const case = cases.byName(name) orelse unreachable;
        var fast_out = try sphar.solve(allocator, case.points, .{ .method = .alternating });
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

test "alternating: wide-cap fixtures still DNC (the gap the trust default closes)" {
    // Pins the motivating failure. If the alternating path ever starts
    // converging here, celebrate — and update the .alternating
    // doc-comment's caveat (see docs/wide-cap-dnc-report.md).
    const allocator = std.testing.allocator;
    for ([_][]const u8{ "wide_cap82", "wide_cap85", "wide_cap89" }) |name| {
        var outcome = try sphar.solve(allocator, wideCap(name), .{ .method = .alternating });
        defer outcome.deinit();
        try std.testing.expect(std.meta.activeTag(outcome) == .did_not_converge);
    }
}

test "auto: resolves to Method.recommended (pure alias, identical outcomes)" {
    // .auto is the "library's current recommendation" placeholder;
    // `Method.recommended` is the single source of truth for the
    // resolution, and this test is where a re-point gets recorded.
    try std.testing.expectEqual(sphar.Method.trust, sphar.Method.recommended);
    try std.testing.expectEqual(sphar.Method.trust, sphar.Method.auto.resolved());

    // Behavioral half: same dispatch target ⇒ identical outcomes,
    // including the diag tag (the expectEqual on `.diag.trust` panics
    // on a wrong active tag, so a silent re-point trips loudly).
    const allocator = std.testing.allocator;
    for ([_][]const u8{ "hex", "h3_res09" }) |name| {
        const case = cases.byName(name) orelse unreachable;
        var trust_out = try sphar.solve(allocator, case.points, .{ .method = .trust });
        defer trust_out.deinit();
        var auto_out = try sphar.solve(allocator, case.points, .{ .method = .auto });
        defer auto_out.deinit();
        try std.testing.expectEqual(
            trust_out.converged.diag.trust,
            auto_out.converged.diag.trust,
        );
        try std.testing.expectEqual(trust_out.converged.gap, auto_out.converged.gap);
        try std.testing.expectEqual(trust_out.converged.sigma, auto_out.converged.sigma);
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
    const pts = wideCap("wide_cap85");
    var outcome = try sphar.solve(allocator, pts, .{ .method = .trust });
    defer outcome.deinit();
    const c = outcome.converged;
    // Weak duality: certified gap is non-negative up to FP noise.
    try std.testing.expect(c.gap >= -1e-10);
    // Dual multipliers are non-negative and the cert carries at least
    // the >= 3 points any non-degenerate cone needs.
    try std.testing.expect(c.cert.indices.len >= 3);
    for (c.cert.lambdas) |lam| try std.testing.expect(lam >= 0);
    // Indices point into the caller's array.
    for (c.cert.indices) |idx| try std.testing.expect(idx < pts.len);
}
