//! Tests for the EXPERIMENTAL alternative solver paths
//! (`SolveOptions.method`; `src/joint.zig` and `src/reduced.zig`).
//!
//! Coverage:
//!  - the wide-cap fixtures (tests/wide_cap_cells.zig) that the fast
//!    path limit-cycles on: `.joint` and `.auto` must converge, with
//!    the AR matching the Clarabel SDP cross-check;
//!  - easy-case agreement: `.joint` reproduces `.fast`'s aspect ratio
//!    on a spread of bundled manifest cases;
//!  - `.auto` never changes the result on inputs where the fast path
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

test "joint: wide-cap fixtures converge and match the Clarabel reference AR" {
    try expectJointConverges(&wide.CAP82_S1, .joint, wide.AR_CAP82_S1);
    try expectJointConverges(&wide.CAP85_S1, .joint, wide.AR_CAP85_S1);
    try expectJointConverges(&wide.CAP89_S3, .joint, wide.AR_CAP89_S3);
}

test "reduced: wide-cap fixtures converge and match the Clarabel reference AR" {
    try expectJointConverges(&wide.CAP82_S1, .reduced, wide.AR_CAP82_S1);
    try expectJointConverges(&wide.CAP85_S1, .reduced, wide.AR_CAP85_S1);
    try expectJointConverges(&wide.CAP89_S3, .reduced, wide.AR_CAP89_S3);
}

test "reduced: agrees with fast on bundled cases incl. extreme-kappa cells" {
    const allocator = std.testing.allocator;
    // Superset of the joint agreement list: the reduced path certifies
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
        var red_out = try sphar.solve(allocator, case.points, .{ .method = .reduced });
        defer red_out.deinit();
        try std.testing.expect(std.meta.activeTag(fast_out) == .converged);
        try std.testing.expect(std.meta.activeTag(red_out) == .converged);
        const ar_f = fast_out.converged.aspectRatio();
        const ar_r = red_out.converged.aspectRatio();
        if (@abs(ar_f - ar_r) > AR_AGREE_REL_TOL * ar_f) {
            std.debug.print("reduced/fast AR mismatch case={s}: fast={d:.10} reduced={d:.10}\n", .{ name, ar_f, ar_r });
            return error.ReducedFastArMismatch;
        }
        try std.testing.expect(@abs(red_out.converged.gap) <= GAP_TOL);
    }
}

test "auto: falls back to reduced on the wide-cap fixtures" {
    // Fast alone DNCs on these (pinned below); .auto must rescue them.
    try expectJointConverges(&wide.CAP82_S1, .auto, wide.AR_CAP82_S1);
    try expectJointConverges(&wide.CAP85_S1, .auto, wide.AR_CAP85_S1);
    try expectJointConverges(&wide.CAP89_S3, .auto, wide.AR_CAP89_S3);
}

test "fast: wide-cap fixtures still DNC (the gap .auto exists to close)" {
    // Pins the motivating failure. If the fast path ever starts
    // converging here, celebrate — and re-evaluate whether the joint
    // fallback is still needed (see docs/wide-cap-dnc-report.md).
    const allocator = std.testing.allocator;
    for ([_][]const [3]f64{ &wide.CAP82_S1, &wide.CAP85_S1, &wide.CAP89_S3 }) |pts| {
        var outcome = try sphar.solve(allocator, pts, .{});
        defer outcome.deinit();
        try std.testing.expect(std.meta.activeTag(outcome) == .did_not_converge);
    }
}

test "joint: agrees with fast on bundled cases across regimes" {
    const allocator = std.testing.allocator;
    // Hex (symmetric small), DGGS hexagon, random caps small and large,
    // the widest converging ha case, and a small wide irregular case.
    const names = [_][]const u8{ "hex", "h3_res09", "np20", "np400", "ha_05", "ha_14", "dnc_small_wide" };
    for (names) |name| {
        const case = cases.byName(name) orelse unreachable;
        var fast_out = try sphar.solve(allocator, case.points, .{});
        defer fast_out.deinit();
        var joint_out = try sphar.solve(allocator, case.points, .{ .method = .joint });
        defer joint_out.deinit();
        try std.testing.expect(std.meta.activeTag(fast_out) == .converged);
        try std.testing.expect(std.meta.activeTag(joint_out) == .converged);
        const ar_f = fast_out.converged.aspectRatio();
        const ar_j = joint_out.converged.aspectRatio();
        if (@abs(ar_f - ar_j) > AR_AGREE_REL_TOL * ar_f) {
            std.debug.print("joint/fast AR mismatch case={s}: fast={d:.10} joint={d:.10}\n", .{ name, ar_f, ar_j });
            return error.JointFastArMismatch;
        }
        try std.testing.expect(@abs(joint_out.converged.gap) <= GAP_TOL);
        try std.testing.expect(sphar.checkFeasibility(joint_out.converged, case.points) <= 1e-12);
    }
}

test "auto: identical to fast when fast converges" {
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

test "joint: certificate sanity on a wide-cap solve" {
    const allocator = std.testing.allocator;
    var outcome = try sphar.solve(allocator, &wide.CAP85_S1, .{ .method = .joint });
    defer outcome.deinit();
    const c = outcome.converged;
    // Weak duality: certified gap is non-negative up to FP noise.
    try std.testing.expect(c.gap >= -1e-10);
    // Dual multipliers are non-negative and the interior-point cert
    // carries at least the ≥3 points any non-degenerate cone needs.
    try std.testing.expect(c.cert.indices.len >= 3);
    for (c.cert.lambdas) |lam| try std.testing.expect(lam >= 0);
    // Indices point into the caller's array.
    for (c.cert.indices) |idx| try std.testing.expect(idx < wide.CAP85_S1.len);
}
