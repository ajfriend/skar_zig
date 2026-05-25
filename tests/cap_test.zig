//! Randomized stress test on spherical caps with equispaced
//! boundary points.
//!
//! Geometry: N points equally spaced around the boundary circle of
//! a spherical cap form a regular N-gon inscribed in that circle.
//! The minimum-AR enclosing cone of a regular N-gon is its
//! circumscribed circle (rotational symmetry forces isotropy), so
//! the optimal `aspectRatio()` is exactly 1.0 for every N ≥ 3.
//!
//! This test catches regressions where the solver biases away from
//! circular solutions, or where some combination of (cap center,
//! half-angle, N, phase) trips a numerical edge case.

const std = @import("std");
const sphar = @import("../src/root.zig");
const test_options = @import("test_options");
const Vec3 = sphar.Vec3;

/// Fill `out` with N equispaced points on the boundary of a
/// spherical cap of half-angle `half_angle` centered at the unit
/// vector `center`, rotated by `phase` around the cap axis.
fn capBoundary(center: Vec3, half_angle: f64, phase: f64, out: []Vec3) void {
    const Q = center.orthoBasis(); // Mat3x2 tangent basis at center
    const cos_a = @cos(half_angle);
    const sin_a = @sin(half_angle);
    const n_f = @as(f64, @floatFromInt(out.len));
    for (out, 0..) |*p, i| {
        const theta = phase + 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / n_f;
        // tangent component: sin_a * (cos θ · e1 + sin θ · e2)
        const tan = Vec3.lincomb(sin_a * @cos(theta), Q.e1, sin_a * @sin(theta), Q.e2);
        // full point = cos_a * center + tan, normalized for FP cleanup
        p.* = Vec3.lincomb(cos_a, center, 1.0, tan).normalize();
    }
}

test "random spherical caps: AR == 1 across center / size / N / phase" {
    // Slow randomized stress test. Skipped by `just test` (fast tier);
    // runs under `just test-slow` (full suite + coverage gate).
    if (!test_options.slow) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const tol: f64 = 1e-6;
    const ar_tol: f64 = 1e-6;
    const n_trials: u32 = 100;

    var prng = std.Random.DefaultPrng.init(0xCA9); // distinct from rotation seed
    const rng = prng.random();

    // Sizes range from the solver's minimum (3) up to a stress level
    // (500). For points equispaced on a circle, every input point is
    // a hull vertex, so the n_hull preprocessing finds N hull
    // vertices regardless of N — the inner loop sees the full size
    // every iteration. Hence the weighting below: tiny N dominates
    // the trial budget (cheap, broad coverage of geometry edge cases),
    // while stress sizes get a handful of runs each (debug-mode solve
    // on N=500 is ~500ms per trial; running 12 of those would dominate
    // the suite's runtime for diminishing bug-finding return).
    // Half-angle range avoids both edges: < 5° risks coplanarity
    // false positives on the tangent projection; > 80° approaches
    // the great-circle / hemisphere boundary where infeasibility or
    // coplanarity rejection take over.
    const min_half = std.math.degreesToRadians(5.0);
    const max_half = std.math.degreesToRadians(80.0);

    var t: u32 = 0;
    while (t < n_trials) : (t += 1) {
        const center = Vec3.randomUnit(rng);
        const half_angle = min_half + rng.float(f64) * (max_half - min_half);
        const n: usize = switch (rng.uintLessThan(u32, 16)) {
            0, 1, 2, 3 => 3, // 4/16
            4, 5, 6 => 5, // 3/16
            7, 8 => 10, // 2/16
            9, 10 => 20, // 2/16
            11, 12 => 50, // 2/16
            13 => 100, // 1/16
            14 => 200, // 1/16
            else => 500, // 1/16 (15)
        };
        const phase = rng.float(f64) * 2.0 * std.math.pi;

        const pts_v = try allocator.alloc(Vec3, n);
        defer allocator.free(pts_v);
        capBoundary(center, half_angle, phase, pts_v);
        // Vec3 is `extern struct` over [3]f64 so a []Vec3 reinterprets
        // freely as [][3]f64 — same trick the solver uses internally.
        const pts: [][3]f64 = @ptrCast(pts_v);

        var outcome = try sphar.solve(allocator, pts, .{});
        defer outcome.deinit();

        try std.testing.expect(std.meta.activeTag(outcome) == .converged);
        const c = outcome.converged;
        try std.testing.expect(c.aspectRatio() < 1.0 + ar_tol);
        const viol = sphar.checkFeasibility(c, pts);
        try std.testing.expect(viol <= tol);
    }
}
