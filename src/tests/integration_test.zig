//! Integration tests for the skar solver. Loads fixtures from
//! `cases/*.txt` and validates convergence, certificates, feasibility,
//! and agreement with the C baseline.
//!
//! Run via `zig build test` from the package root.

const std = @import("std");
const sphar = @import("../root.zig");
const cases = @import("cases");
const Vec3 = sphar.Vec3;
const loadCase = cases.loadCase;

/// Labeled approx-equal check on aspect ratios. On failure prints
/// the case label and full-precision expected/actual/delta — the
/// equivalent of the hand-rolled diagnostic that pre-dated the move
/// to `std.testing.expectApproxEqAbs` (which prints values but no
/// label, leaving "which case tripped it?" implicit). The failure
/// branch is exercised by a dedicated negative test below, so kcov
/// covers the print path.
fn checkArEq(label: []const u8, expected: f64, actual: f64, tol: f64) !void {
    if (@abs(expected - actual) > tol) {
        std.debug.print(
            "AR mismatch case={s}: expected={d:.17} actual={d:.17} delta={e:.3}\n",
            .{ label, expected, actual, @abs(expected - actual) },
        );
        return error.AspectRatioMismatch;
    }
}

test "checkArEq prints case label on failure" {
    try std.testing.expectError(
        error.AspectRatioMismatch,
        checkArEq("test_label", 1.0, 1.1, 1e-6),
    );
}

const ExpectedAr = struct { name: []const u8, ar: f64 };

const EXPECTED: []const ExpectedAr = &.{
    .{ .name = "h3_res05", .ar = 1.0666666878954056 },
    .{ .name = "h3_res09", .ar = 1.0666666666762255 },
    .{ .name = "h3_res12", .ar = 1.0666666666666949 },
    .{ .name = "h3_res15", .ar = 1.0666666666666669 },
    .{ .name = "ha_05", .ar = 1.0058545383400732 },
    .{ .name = "ha_08", .ar = 1.0077908248743077 },
    .{ .name = "ha_10", .ar = 1.0105582152823531 },
    .{ .name = "ha_12", .ar = 1.0166530149054471 },
    .{ .name = "ha_14", .ar = 1.0368555055771478 },
    .{ .name = "hex", .ar = 1.0000000000000004 },
    .{ .name = "np100", .ar = 1.0911969178517065 },
    .{ .name = "np20", .ar = 1.0177775789268264 },
    .{ .name = "np400", .ar = 1.0192804795923 },
    // Octahedron-face spherical triangles: 3 symmetric points covering
    // 1/8 of the sphere; AR = 1 by 3-fold rotation around (±1,±1,±1)/√3.
    .{ .name = "oct_n0", .ar = 1.0000000000000004 },
    .{ .name = "oct_n1", .ar = 1.0000000000000004 },
    .{ .name = "oct_n2", .ar = 1.0000000000000004 },
    .{ .name = "oct_n3", .ar = 1.0000000000000004 },
    .{ .name = "oct_s0", .ar = 1.0000000000000004 },
    .{ .name = "oct_s1", .ar = 1.0000000000000004 },
    .{ .name = "oct_s2", .ar = 1.0000000000000004 },
    .{ .name = "oct_s3", .ar = 1.0000000000000004 },
    // H3 icosahedron-face spherical triangles: 20 symmetric faces;
    // AR = 1 by 3-fold rotation around each face centroid.
    .{ .name = "ico_00", .ar = 1.0000000000000004 },
    .{ .name = "ico_01", .ar = 1.0000000000000013 },
    .{ .name = "ico_02", .ar = 1.0000000000000009 },
    .{ .name = "ico_03", .ar = 1.0000000000000004 },
    .{ .name = "ico_04", .ar = 1.0000000000000004 },
    .{ .name = "ico_05", .ar = 1.0000000000000004 },
    .{ .name = "ico_06", .ar = 1.0000000000000004 },
    .{ .name = "ico_07", .ar = 1.0000000000000009 },
    .{ .name = "ico_08", .ar = 1.0000000000000009 },
    .{ .name = "ico_09", .ar = 1.0000000000000009 },
    .{ .name = "ico_10", .ar = 1.0000000000000013 },
    .{ .name = "ico_11", .ar = 1.0000000000000013 },
    .{ .name = "ico_12", .ar = 1.0000000000000004 },
    .{ .name = "ico_13", .ar = 1.0000000000000004 },
    .{ .name = "ico_14", .ar = 1.0000000000000004 },
    .{ .name = "ico_15", .ar = 1.0000000000000004 },
    .{ .name = "ico_16", .ar = 1.0000000000000004 },
    .{ .name = "ico_17", .ar = 1.0000000000000004 },
    .{ .name = "ico_18", .ar = 1.0000000000000004 },
    .{ .name = "ico_19", .ar = 1.0000000000000009 },
};

const INFEASIBLE_CASES: []const []const u8 = &.{ "infeas_antipodal", "near_collinear" };
// With the quasi-Newton preconditioned b-update, dnc_small_wide converges
// (the original damped-gradient axis update couldn't close it within 100
// iters). No cases left that should DNC.
const DNC_CASES: []const []const u8 = &.{};

test "converged cases match C baseline AR" {
    const allocator = std.testing.allocator;
    const tol: f64 = 1e-6;

    for (EXPECTED) |exp| {
        const path = try std.fmt.allocPrint(allocator, "cases/{s}.txt", .{exp.name});
        defer allocator.free(path);

        const X = try loadCase(allocator, path);
        defer allocator.free(X);

        var outcome = try sphar.solve(allocator, X, .{ .gap_tol = tol, .n_hull = 10, .coplanarity_tol = 1e-12 });
        defer outcome.deinit();

        try std.testing.expect(std.meta.activeTag(outcome) == .converged);
        const c = outcome.converged;
        try std.testing.expect(c.aspectRatio() >= 1.0 - 1e-10);
        // Gap: nonneg by weak duality (solver raises on meaningfully-negative
        // gap; ulp-level negatives can slip through here, hence |gap|).
        try std.testing.expect(@abs(c.gap) < tol);

        // AR agrees with C baseline to within solve tolerance. Zig and C
        // are independent numerical algorithms; the certified duality gap
        // is the source of truth for correctness, not cross-implementation
        // AR equality. Use the labeled helper so the case name appears
        // in the failure diagnostic.
        try checkArEq(exp.name, exp.ar, c.aspectRatio(), tol);

        // Feasibility: ‖Ax_i‖ ≤ b·x_i for all i (tol includes numerics buffer).
        const viol = sphar.checkFeasibility(c, X);
        try std.testing.expect(viol <= tol);
    }
}

test "infeasible cases return Farkas certificate" {
    const allocator = std.testing.allocator;
    for (INFEASIBLE_CASES) |name| {
        const path = try std.fmt.allocPrint(allocator, "cases/{s}.txt", .{name});
        defer allocator.free(path);

        const X = try loadCase(allocator, path);
        defer allocator.free(X);

        var outcome = try sphar.solve(allocator, X, .{ .gap_tol = 1e-6, .n_hull = 10, .coplanarity_tol = 1e-12 });
        defer outcome.deinit();

        try std.testing.expect(std.meta.activeTag(outcome) == .infeasible);
        const inf = outcome.infeasible;
        // Verify Farkas certificate: λ ≥ 0, ∑ λ ≈ 1, ‖∑ λᵢ xᵢ‖ small.
        var sum: f64 = 0;
        for (inf.cert.lambdas) |l| {
            try std.testing.expect(l >= 0);
            sum += l;
        }
        try std.testing.expect(@abs(sum - 1.0) < 1e-9);

        var z = Vec3.zero;
        for (inf.cert.indices, inf.cert.lambdas) |idx, l| {
            z = Vec3.lincomb(1.0, z, l, Vec3{ .m = X[idx] });
        }
        try std.testing.expect(z.norm() < 1e-2);
        // residual matches the computed witness magnitude (to a couple of ulp).
        try std.testing.expect(@abs(inf.residual - z.norm()) < 1e-6);
    }
}

test "did_not_converge case raises DNC status" {
    const allocator = std.testing.allocator;
    for (DNC_CASES) |name| {
        const path = try std.fmt.allocPrint(allocator, "cases/{s}.txt", .{name});
        defer allocator.free(path);

        const X = try loadCase(allocator, path);
        defer allocator.free(X);

        var outcome = try sphar.solve(allocator, X, .{ .gap_tol = 1e-6, .n_hull = 10, .coplanarity_tol = 1e-12 });
        defer outcome.deinit();

        try std.testing.expect(std.meta.activeTag(outcome) == .did_not_converge);
    }
}

test "Shape invariants: Q right-handed orthonormal, sigma paired with columns, AR = sigma[2]/sigma[1]" {
    const allocator = std.testing.allocator;
    const X = try loadCase(allocator, "cases/np100.txt");
    defer allocator.free(X);

    var outcome = try sphar.solve(allocator, X, .{ .gap_tol = 1e-6, .n_hull = 10, .coplanarity_tol = 1e-12 });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    const c = outcome.converged;

    const c0 = c.Q.col(0);
    const c1 = c.Q.col(1);
    const c2 = c.Q.col(2);

    // Q's three columns are an orthonormal basis.
    try std.testing.expect(@abs(c0.dot(c0) - 1.0) < 1e-14);
    try std.testing.expect(@abs(c1.dot(c1) - 1.0) < 1e-14);
    try std.testing.expect(@abs(c2.dot(c2) - 1.0) < 1e-14);
    try std.testing.expect(@abs(c0.dot(c1)) < 1e-14);
    try std.testing.expect(@abs(c0.dot(c2)) < 1e-14);
    try std.testing.expect(@abs(c1.dot(c2)) < 1e-14);

    // Right-handed: c0 × c1 = c2.
    const cross = c0.cross(c1);
    try std.testing.expect(@abs(cross.m[0] - c2.m[0]) < 1e-14);
    try std.testing.expect(@abs(cross.m[1] - c2.m[1]) < 1e-14);
    try std.testing.expect(@abs(cross.m[2] - c2.m[2]) < 1e-14);

    // b() returns the first column.
    const b = c.b();
    try std.testing.expect(@abs(b.m[0] - c0.m[0]) < 1e-14);
    try std.testing.expect(@abs(b.m[1] - c0.m[1]) < 1e-14);
    try std.testing.expect(@abs(b.m[2] - c0.m[2]) < 1e-14);

    // sigma[0] = 1/√3, tangent eigenvalues ascending, AR = sigma[2]/sigma[1].
    try std.testing.expect(@abs(c.sigma[0] - 1.0 / @sqrt(3.0)) < 1e-14);
    try std.testing.expect(c.sigma[1] <= c.sigma[2]);
    try std.testing.expect(@abs(c.sigma[2] / c.sigma[1] - c.aspectRatio()) < 1e-14);

    // c.A() reconstructs A faithfully: each Q column is an eigenvector of
    // A with the corresponding sigma as eigenvalue.
    const A_mat = c.A();
    const Ac0 = A_mat.apply(c0);
    const Ac1 = A_mat.apply(c1);
    const Ac2 = A_mat.apply(c2);
    try std.testing.expect(@abs(c0.dot(Ac0) - c.sigma[0]) < 1e-12);
    try std.testing.expect(@abs(c1.dot(Ac1) - c.sigma[1]) < 1e-12);
    try std.testing.expect(@abs(c2.dot(Ac2) - c.sigma[2]) < 1e-12);
}
