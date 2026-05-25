//! Tests driven by the bundled case manifest in `cases.zig`.
//!
//! This file lives in `tests/cases/` next to the `cases.zig` module
//! it exercises, but it is NOT compiled as part of the cases module.
//! It's a stand-alone test file reached by the test target's root
//! (`test_root.zig` → `tests/all.zig` → here).

const std = @import("std");
const sphar = @import("../../src/root.zig");
const cases = @import("cases");
const Vec3 = sphar.Vec3;

/// Labeled approx-equal check on aspect ratios. On failure prints
/// the case label + full-precision expected/actual/delta — useful
/// because the all-cases test loop has dozens of iterations and a
/// plain `expect` failure wouldn't say which case tripped it. The
/// failure branch is exercised by a dedicated negative test below,
/// so kcov covers the print path.
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

test "cases.byName: found and not-found" {
    try std.testing.expect(cases.byName("hex") != null);
    try std.testing.expectEqual(@as(?cases.Case, null), cases.byName("definitely_not_a_case"));
}

test "all cases match expected outcome" {
    const allocator = std.testing.allocator;
    const tol: f64 = 1e-6;

    for (cases.all) |entry| {
        var outcome = try sphar.solve(allocator, entry.case.points, .{
            .gap_tol = tol,
            .n_hull = 10,
            .coplanarity_tol = 1e-12,
        });
        defer outcome.deinit();

        switch (entry.case.expected) {
            .converged => |exp| {
                try std.testing.expect(std.meta.activeTag(outcome) == .converged);
                const c = outcome.converged;
                try std.testing.expect(c.aspectRatio() >= 1.0 - 1e-10);
                // Gap: nonneg by weak duality (solver raises on meaningfully-negative
                // gap; ulp-level negatives can slip through here, hence |gap|).
                try std.testing.expect(@abs(c.gap) < tol);

                // AR agrees with the per-case expected value to within solve
                // tolerance. The certified duality gap is the source of truth
                // for correctness; AR agreement is a cross-implementation /
                // cross-version sanity check.
                try checkArEq(entry.name, exp.ar, c.aspectRatio(), tol);

                // Feasibility: ‖Ax_i‖ ≤ b·x_i for all i (tol includes numerics buffer).
                const viol = sphar.checkFeasibility(c, entry.case.points);
                try std.testing.expect(viol <= tol);
            },
            .infeasible => {
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
                    z = Vec3.lincomb(1.0, z, l, Vec3{ .m = entry.case.points[idx] });
                }
                try std.testing.expect(z.norm() < 1e-2);
                // residual matches the computed witness magnitude (to a couple of ulp).
                try std.testing.expect(@abs(inf.residual - z.norm()) < 1e-6);
            },
        }
    }
}

test "Shape invariants: Q right-handed orthonormal, sigma paired with columns, AR = sigma[2]/sigma[1]" {
    const allocator = std.testing.allocator;
    const case = cases.byName("np100").?;

    var outcome = try sphar.solve(allocator, case.points, .{ .gap_tol = 1e-6, .n_hull = 10, .coplanarity_tol = 1e-12 });
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
