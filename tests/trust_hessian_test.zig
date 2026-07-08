//! Finite-difference validation of the trust path's envelope Hessian
//! (config.trust.EXACT_HESSIAN — the fixed-w term plus the dw/db
//! correction). The oracle's h-values are the ground truth: for tangent
//! directions u, the 1-D second difference of h along the retraction
//! b(t) = normalize(b + t·Q·u) must match uᵀBu. Value-based FD avoids
//! the tangent-basis transport problem entirely (h is a scalar; g/B
//! would need frame alignment between evaluations).
//!
//! Two regimes:
//!  - irregular small support (k ≤ 6 active): the KKT solve is exact —
//!    tight agreement expected;
//!  - dense near-circular support (k ≫ 6, degenerate C∘C): the
//!    Tikhonov-regularized correction is approximate — looser check,
//!    plus PD sanity.
//! The gradient is FD-checked in the same sweep (first difference vs
//! g·u), which validates the envelope story end to end.

const std = @import("std");
const trust = @import("../src/trust.zig");
const core = @import("../src/skar.zig");
const linalg = @import("../src/linalg.zig");
const halfspace = @import("../src/halfspace.zig");

const Vec2 = linalg.Vec2;
const Vec3 = linalg.Vec3;
const Mat3 = linalg.Mat3;

const FD_T: f64 = 3e-3;

/// h(b) via the full oracle, restoring the caller's weights afterward
/// so every FD evaluation starts from the same warm state.
fn hAt(b: Vec3, Xw: []const Vec3, wb: *trust.Buffers, w_base: []const f64) f64 {
    @memcpy(wb.w, w_base);
    const e = trust.evalH(b, Xw, wb, -std.math.inf(f64)) orelse unreachable;
    return e.h;
}

const FdCheck = struct {
    /// max |uᵀBu − FD| over the probed directions
    hess_abs_err: f64,
    /// scale of the Hessian entries (max |uᵀBu|), for relative checks
    hess_scale: f64,
    /// max |g·u − FD| over the probed directions
    grad_abs_err: f64,
    b_is_pd: bool,
};

fn fdCheck(alloc: std.mem.Allocator, points: []const [3]f64) !FdCheck {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    const n = points.len;
    const Xw = try a.alloc(Vec3, n);
    for (points, 0..) |p, i| Xw[i] = (Vec3{ .m = p }).normalize();

    // Axis: normalized centroid (strictly feasible for cap-like sets).
    var c = Vec3.zero;
    for (Xw) |x| c = Vec3.lincomb(1.0, c, 1.0, x);
    const b = c.normalize();

    var wb = try trust.Buffers.init(a, n);

    // Seed weights the way solveTrust's opening does, then refine to
    // oracle quality once so the base state is inner-optimal.
    const Q = b.orthoBasis();
    try std.testing.expect(halfspace.projectGnomonic(Xw, b, Q, wb.P_buf, -std.math.inf(f64)));
    _ = core.rescaleP(wb.P_buf, wb.Ps);
    core.initWeights(wb.Ps, wb.w);
    const w_base = try a.alloc(f64, n);
    @memcpy(w_base, wb.w);

    const e0 = trust.evalH(b, Xw, &wb, -std.math.inf(f64)) orelse unreachable;
    // Re-baseline on the refined weights: FD evaluations warm-start
    // from the oracle-quality state, minimizing refinement noise.
    @memcpy(w_base, wb.w);
    const h0 = hAt(b, Xw, &wb, w_base);

    const dirs = [_]Vec2{
        .{ .m = .{ 1, 0 } },
        .{ .m = .{ 0, 1 } },
        .{ .m = .{ std.math.sqrt1_2, std.math.sqrt1_2 } },
    };
    var res = FdCheck{
        .hess_abs_err = 0,
        .hess_scale = 0,
        .grad_abs_err = 0,
        .b_is_pd = e0.B.det() > 0 and e0.B.m[0] > 0,
    };
    for (dirs) |u| {
        const du = e0.Q.apply(u);
        const bp = Vec3.lincomb(1.0, b, FD_T, du).normalize();
        const bm = Vec3.lincomb(1.0, b, -FD_T, du).normalize();
        const hp = hAt(bp, Xw, &wb, w_base);
        const hm = hAt(bm, Xw, &wb, w_base);

        const fd_grad = (hp - hm) / (2.0 * FD_T);
        const fd_hess = (hp - 2.0 * h0 + hm) / (FD_T * FD_T);
        const model_grad = e0.g.dot(u);
        const model_hess = u.dot(e0.B.apply(u));

        res.grad_abs_err = @max(res.grad_abs_err, @abs(fd_grad - model_grad));
        res.hess_abs_err = @max(res.hess_abs_err, @abs(fd_hess - model_hess));
        res.hess_scale = @max(res.hess_scale, @abs(model_hess));
    }
    return res;
}

/// n points in a spherical cap of angular radius cap_deg around a
/// rotated axis (same construction as the wide-cap probes).
fn capPoints(a: std.mem.Allocator, rng: std.Random, n: usize, cap_deg: f64, ring: bool) ![][3]f64 {
    const pts = try a.alloc([3]f64, n);
    const cos_max = @cos(cap_deg * std.math.pi / 180.0);
    var R = Mat3.randomNormal(rng);
    R.orthonormalize();
    for (pts, 0..) |*p, i| {
        // ring=true puts every point ON the cap boundary (dense
        // near-circular support, k ≫ 6); ring=false fills the cap.
        const z = if (ring)
            cos_max + (1.0 - cos_max) * 0.02 * rng.float(f64)
        else
            cos_max + rng.float(f64) * (1.0 - cos_max);
        const phi = 2.0 * std.math.pi * (@as(f64, @floatFromInt(i)) + 0.5 * rng.float(f64)) / @as(f64, @floatFromInt(n));
        const s = @sqrt(1.0 - z * z);
        const v = Vec3{ .m = .{ s * @cos(phi), s * @sin(phi), z } };
        p.* = R.apply(v).m;
    }
    return pts;
}

test "envelope Hessian matches FD on irregular small-support caps" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    // Several seeds × cap widths; small supports keep the KKT solve
    // exact. FD tolerance: the oracle's stall exit leaves h noise
    // ~1e-9·(1+|h|), amplified by /t² to ~1e-3 absolute; truncation
    // adds O(t²)·(higher derivs). 2% relative (floored at 0.02 abs)
    // cleanly separates exact-vs-fixed-w (corrections are O(1)).
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for ([_]f64{ 20.0, 46.0, 70.0 }) |cap| {
        for (0..3) |_| {
            const pts = try capPoints(a, rng, 20, cap, false);
            const r = try fdCheck(alloc, pts);
            try std.testing.expect(r.b_is_pd);
            try std.testing.expect(r.grad_abs_err <= 1e-3 * (1.0 + r.hess_scale));
            try std.testing.expect(r.hess_abs_err <= 0.02 * (1.0 + r.hess_scale));
        }
    }
}

test "envelope Hessian on dense near-circular supports: regularized correction stays sane" {
    const alloc = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    // The ha_*-shaped regime: many active points, rank-deficient C∘C,
    // Tikhonov-regularized dw/db. The correction is approximate here —
    // require PD + the right order of magnitude (within 30%), which is
    // still far tighter than the fixed-w model's far-field error.
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for ([_]f64{ 60.0, 80.0 }) |cap| {
        const pts = try capPoints(a, rng, 60, cap, true);
        const r = try fdCheck(alloc, pts);
        try std.testing.expect(r.b_is_pd);
        try std.testing.expect(r.grad_abs_err <= 1e-2 * (1.0 + r.hess_scale));
        try std.testing.expect(r.hess_abs_err <= 0.3 * (1.0 + r.hess_scale));
    }
}
