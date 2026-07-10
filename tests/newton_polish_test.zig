//! Unit tests for the range-space Newton polish path
//! (k ≥ `newton.RANGE_SPACE_MIN_K` active points): for k ≥ 8 the dense
//! bordered KKT matrix is exactly singular (the polish Hessian
//! H = VᵀV has rank ≤ 6), and `newtonPolish` routes through the 6-dim
//! range-space solve. These tests pin that path's correctness on cases
//! with closed-form optima, plus the load-bearing invariants (Σw
//! conservation — see the farthestPointSeed notes in src/skar.zig —
//! and weight nonnegativity).
//!
//! Analytic anchor: for n ≥ 3 points equally spaced in angle on a
//! circle of radius 1, the lifted D-optimal design is uniform
//! (w ≡ 1/n): S = diag(1/2, 1/2, 1), g ≡ 3, log det S = log(1/4).
//! Affine invariance extends the anchor to ellipses: mapping the 2D
//! points by p → A·p + t maps the lifted problem by T = [A, t; 0, 1],
//! so S → T·S·Tᵀ, the gᵢ and the optimal weights are unchanged, and
//! the optimal value shifts to log(1/4) + 2·log|det A|.

const std = @import("std");
const newton = @import("../src/newton.zig");
const linalg = @import("../src/linalg.zig");
const config = @import("../src/config.zig");

const Vec3 = linalg.Vec3;
const Mat3 = linalg.Mat3;
const algo = config.algo;
const tol = config.tol;

/// Post-polish state summary. g statistics are over the points still
/// active (w > algo.ACTIVE_THRESH) — the set polish's own convergence
/// criterion ranges over; w statistics are over all points.
const PolishResult = struct {
    ok: bool,
    w_sum: f64,
    w_min: f64,
    g_max: f64,
    g_spread: f64,
    logdet: f64,
};

fn runPolish(a: std.mem.Allocator, Ql: []const Vec3, w: []f64) !PolishResult {
    var scratch = try newton.NewtonScratch.init(a, Ql.len);
    const ok = newton.newtonPolish(Ql, w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &scratch);

    var S = Mat3.zero;
    for (Ql, 0..) |qi, i| S.addSymRank1(w[i], qi);
    const L = S.cholesky().?;

    var r = PolishResult{
        .ok = ok,
        .w_sum = 0,
        .w_min = std.math.inf(f64),
        .g_max = -std.math.inf(f64),
        .g_spread = 0,
        .logdet = L.logDet(),
    };
    var g_min: f64 = std.math.inf(f64);
    for (Ql, 0..) |qi, i| {
        r.w_sum += w[i];
        if (w[i] < r.w_min) r.w_min = w[i];
        if (w[i] > algo.ACTIVE_THRESH) {
            const g = qi.dot(L.solve(qi));
            if (g > r.g_max) r.g_max = g;
            if (g < g_min) g_min = g;
        }
    }
    r.g_spread = r.g_max - g_min;
    return r;
}

/// Lifted ring points qᵢ = [cos θᵢ, sin θᵢ, 1], θᵢ = 2πi/n + phase,
/// then mapped by the affine T = [A, t; 0, 1].
fn liftAffineRing(
    a: std.mem.Allocator,
    n: usize,
    A2: [4]f64,
    t: [2]f64,
) ![]Vec3 {
    const Ql = try a.alloc(Vec3, n);
    for (Ql, 0..) |*q, i| {
        const th = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)) + 0.3;
        const x = @cos(th);
        const y = @sin(th);
        q.* = .{ .m = .{
            A2[0] * x + A2[1] * y + t[0],
            A2[2] * x + A2[3] * y + t[1],
            1.0,
        } };
    }
    return Ql;
}

/// Perturbed-from-uniform start: wᵢ ∝ 1 + 0.3·sin(2πi/n + 0.7),
/// normalized to Σw = 1. Strictly positive, so every point is active
/// and polish sees k = n.
fn perturbedWeights(a: std.mem.Allocator, n: usize) ![]f64 {
    const w = try a.alloc(f64, n);
    var sum: f64 = 0;
    for (w, 0..) |*wi, i| {
        const th = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)) + 0.7;
        wi.* = 1.0 + 0.3 * @sin(th);
        sum += wi.*;
    }
    for (w) |*wi| wi.* /= sum;
    return w;
}

test "range-space polish: unit-circle rings converge to the analytic optimum" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const eye: [4]f64 = .{ 1, 0, 0, 1 };
    const analytic = @log(0.25); // det S* = (1/2)·(1/2)·1

    // 8: the smallest range-path k; 12: mid; 60: the dense
    // near-circular regime (κ(G)² territory) the trust-path Hessian
    // work flagged as the case to watch.
    for ([_]usize{ 8, 12, 60 }) |n| {
        const Ql = try liftAffineRing(a, n, eye, .{ 0, 0 });
        const w = try perturbedWeights(a, n);
        const r = try runPolish(a, Ql, w);

        try std.testing.expect(r.ok);
        try std.testing.expect(@abs(r.w_sum - 1.0) <= 1e-13);
        try std.testing.expect(r.w_min >= 0);
        try std.testing.expect(r.g_spread <= 1e-9);
        try std.testing.expect(@abs(r.logdet - analytic) <= 1e-9);
    }
}

test "range-space polish: anisotropic translated ellipse matches the affine-shifted optimum" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // p → A·p + t with det A = 1.3·0.4 = 0.52: anisotropic G, non-
    // centered chart — the general-position shape of a real polish
    // call. Optimal weights stay uniform (affine invariance); value
    // shifts by 2·log|det A|.
    const A2: [4]f64 = .{ 1.3, 0.2, 0.0, 0.4 };
    const det_a = 0.52;
    const analytic = @log(0.25) + 2.0 * @log(det_a);

    const n: usize = 10;
    const Ql = try liftAffineRing(a, n, A2, .{ 0.35, -0.15 });
    const w = try perturbedWeights(a, n);
    const r = try runPolish(a, Ql, w);

    try std.testing.expect(r.ok);
    try std.testing.expect(@abs(r.w_sum - 1.0) <= 1e-13);
    try std.testing.expect(r.w_min >= 0);
    try std.testing.expect(r.g_spread <= 1e-9);
    try std.testing.expect(@abs(r.logdet - analytic) <= 1e-9);
}

test "range-space polish: jittered annulus keeps invariants and reaches primal optimality" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Deterministic jitter in radius and angle: an irregular k = 16
    // input whose optimal support may be a strict subset. Polish
    // cannot drive weights exactly to zero (fraction-to-boundary), so
    // the pinned assertions are the invariants plus g_max → 3 (the
    // primal optimality signal; Σwᵢgᵢ ≡ 3 identically, so g_max ≥ 3
    // always and its excess measures distance from the optimum).
    const n: usize = 16;
    const Ql = try a.alloc(Vec3, n);
    for (Ql, 0..) |*q, i| {
        const fi = @as(f64, @floatFromInt(i));
        const th = 2.0 * std.math.pi * fi / @as(f64, @floatFromInt(n)) + 0.11 * @sin(3.7 * fi + 1.3);
        const rad = 1.0 + 0.2 * @sin(2.3 * fi + 0.5);
        q.* = .{ .m = .{ rad * @cos(th), rad * @sin(th), 1.0 } };
    }
    const w = try a.alloc(f64, n);
    for (w) |*wi| wi.* = 1.0 / @as(f64, @floatFromInt(n));

    const r = try runPolish(a, Ql, w);

    try std.testing.expect(r.ok);
    try std.testing.expect(@abs(r.w_sum - 1.0) <= 1e-13);
    try std.testing.expect(r.w_min >= 0);
    try std.testing.expect(r.g_max - 3.0 <= 1e-6);
}
