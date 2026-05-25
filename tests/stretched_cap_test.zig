//! Stretched-cap test: equispaced boundary points of a spherical
//! cap, with one tangent-plane axis stretched by 5×. After
//! renormalization to the unit sphere, the points still lie on an
//! axis-aligned ellipse in the tangent plane at the cap center,
//! with aspect ratio exactly 5.
//!
//! Why exact: gnomonic projection (x/z, y/z) commutes with the
//! pre-projection stretch. The stretched-unstretched ratio in z
//! cancels in numerator and denominator. So the recovered AR is
//! independent of cap size (within the safe θ range).
//!
//! Why 4 cardinal points suffice: the D₂ symmetry (x → −x, y → −y
//! independently) forces the optimal cone axis to lie along the
//! cap axis and the cross-section to be axis-aligned. The 4
//! cardinal points then pin both semi-axes uniquely.

const std = @import("std");
const sphar = @import("../src/root.zig");
const Vec3 = sphar.Vec3;

/// Fill `out` with N equispaced points on a stretched cap. Points
/// are generated in the canonical frame (cap axis along +z), with
/// the x tangent component stretched by `stretch`, then renormalized
/// to the unit sphere. N must be a multiple of 4 with phase 0, so
/// the four cardinal directions are always sampled.
fn stretchedCap(n: usize, half_angle: f64, stretch: f64, out: []Vec3) void {
    const s = @sin(half_angle);
    const c = @cos(half_angle);
    const n_f = @as(f64, @floatFromInt(n));
    for (out, 0..) |*p, i| {
        const phi = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / n_f;
        const v = Vec3{ .m = .{ stretch * s * @cos(phi), s * @sin(phi), c } };
        p.* = v.normalize();
    }
}

test "stretched cap recovers stretch ratio as AR" {
    const allocator = std.testing.allocator;
    const stretch: f64 = 5.0;
    const ar_tol: f64 = 1e-6;
    const feas_tol: f64 = 1e-6;
    const n_trials: u32 = 50;

    var prng = std.Random.DefaultPrng.init(0xCAB);
    const rng = prng.random();

    // N must be a multiple of 4 so the four cardinal directions are
    // always among the sampled points. Range kept small — the test
    // is about correctness, not stress; cap_test covers large-N.
    const n_choices = [_]usize{ 4, 8, 12, 16 };
    // Half-angle range stays in the "tangent projection well-behaved"
    // band. Below ~3° the cap is tiny but the ratio is still exact;
    // above ~60° the stretched x-component approaches the equator
    // and gnomonic projection numerics get unpleasant.
    const min_half = std.math.degreesToRadians(5.0);
    const max_half = std.math.degreesToRadians(30.0);

    var t: u32 = 0;
    while (t < n_trials) : (t += 1) {
        const n = n_choices[rng.uintLessThan(usize, n_choices.len)];
        const half_angle = min_half + rng.float(f64) * (max_half - min_half);

        const pts_v = try allocator.alloc(Vec3, n);
        defer allocator.free(pts_v);
        stretchedCap(n, half_angle, stretch, pts_v);

        // Random rotation over SO(3).
        var R = sphar.Mat3.randomNormal(rng);
        R.orthonormalize();
        for (pts_v) |*p| p.* = R.apply(p.*);

        const pts: [][3]f64 = @ptrCast(pts_v);
        var outcome = try sphar.solve(allocator, pts, .{});
        defer outcome.deinit();

        try std.testing.expect(std.meta.activeTag(outcome) == .converged);
        const c = outcome.converged;
        try std.testing.expect(@abs(c.aspectRatio() - stretch) < ar_tol);
        const viol = sphar.checkFeasibility(c, pts);
        try std.testing.expect(viol <= feas_tol);
    }
}
