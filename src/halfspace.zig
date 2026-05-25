//! Geometric preprocessing for the solver:
//!
//! - `halfspaceCheck`: Frank-Wolfe pairwise on the convex hull of the
//!   input points to find a feasible cone axis `b` (or a Farkas
//!   certificate of infeasibility).
//! - `convexHull2d`: Andrew's monotone chain over 2D-projected points,
//!   used to reduce large inputs to their hull boundary.
//! - `projectGnomonic`: tangent-plane projection at a feasible axis.
//!
//! All three operate on linear-algebra primitives from `linalg.zig`
//! and read tolerances/constants from `config.zig`. None of them
//! capture solver outer-loop state, so they can live cleanly outside
//! `skar.zig`.

const std = @import("std");
const linalg = @import("linalg.zig");
const config = @import("config.zig");

const Vec3 = linalg.Vec3;
const Mat3x2 = linalg.Mat3x2;
const tol = config.tol;

pub const HalfspaceResult = struct {
    /// If found: unit vector b with x_i · b > 0 for all i.
    b: ?Vec3,
    /// If infeasible: lambda weights on the input points (λ ≥ 0, ∑ λ = 1).
    lam: []f64,
    /// ‖∑ λᵢ xᵢ‖ — small = sharp Farkas certificate; large = FW stalled.
    residual: f64,
};

pub fn halfspaceCheck(allocator: std.mem.Allocator, X: []const Vec3) !HalfspaceResult {
    const n = X.len;
    var z = Vec3.zero;
    for (X) |xi| z = z.add(xi);
    z = z.scale(1.0 / @as(f64, @floatFromInt(n)));

    const lam = try allocator.alloc(f64, n);
    errdefer allocator.free(lam);
    for (lam) |*l| l.* = 1.0 / @as(f64, @floatFromInt(n));

    var b_out: ?Vec3 = null;

    var it: u32 = 0;
    while (it < 2000) : (it += 1) {
        var j: usize = 0;
        var k: ?usize = null;
        var g_min: f64 = 1e30;
        var g_max_active: f64 = -1e30;
        var all_positive = true;

        for (X, 0..) |xi, i| {
            const gi = xi.dot(z);
            if (gi <= 0) all_positive = false;
            if (gi < g_min) {
                g_min = gi;
                j = i;
            }
            if (lam[i] > tol.WEIGHT_ACTIVE and gi > g_max_active) {
                g_max_active = gi;
                k = i;
            }
        }

        if (all_positive) {
            const nz = z.norm();
            if (nz > tol.NEAR_SING) {
                b_out = z.scale(1.0 / nz);
            }
            break;
        }
        if (z.dot(z) < tol.FW_Z_EXHAUSTED) break;
        const ki = k orelse break;
        if (ki == j) break;

        const w = X[j].sub(X[ki]);
        const ww = w.dot(w);
        if (ww < tol.TINY) break;

        var gamma = -w.dot(z) / ww;
        if (gamma < 0) gamma = 0;
        if (gamma > lam[ki]) gamma = lam[ki];

        lam[j] += gamma;
        lam[ki] -= gamma;
        z = Vec3.lincomb(1.0, z, gamma, w);
    }

    return .{ .b = b_out, .lam = lam, .residual = z.norm() };
}

// ---- 2D convex hull (Andrew's monotone chain) ----

fn cross2(O: [2]f64, A: [2]f64, B: [2]f64) f64 {
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
}

const HullCtx = struct {
    P: []const [2]f64,
    pub fn lessThan(ctx: HullCtx, a: u32, b: u32) bool {
        const pa = ctx.P[a];
        const pb = ctx.P[b];
        if (pa[0] != pb[0]) return pa[0] < pb[0];
        return pa[1] < pb[1];
    }
};

/// Andrew's monotone-chain convex hull on `P`. Writes the hull
/// vertices (as indices into `P`) into `hull_idx[0..return_value]`.
///
/// PRECONDITION: `hull_idx.len >= 2 * P.len`. The buffer is used as
/// scratch during construction — lower and upper hull passes can
/// each fill up to `P.len` entries before the trailing dedup. On
/// inputs where most points lie on the hull (e.g. points on a
/// circle), an `hull_idx.len == P.len` buffer overflows.
pub fn convexHull2d(allocator: std.mem.Allocator, P: []const [2]f64, hull_idx: []u32) !u32 {
    const n = @as(u32, @intCast(P.len));
    const idx = try allocator.alloc(u32, n);
    defer allocator.free(idx);
    for (0..n) |i| idx[i] = @intCast(i);

    std.mem.sort(u32, idx, HullCtx{ .P = P }, HullCtx.lessThan);

    var h: u32 = 0;
    for (0..n) |i| {
        while (h >= 2 and cross2(P[hull_idx[h - 2]], P[hull_idx[h - 1]], P[idx[i]]) <= 0) h -= 1;
        hull_idx[h] = idx[i];
        h += 1;
    }
    const lower_size = h + 1;
    var i: isize = @as(isize, @intCast(n)) - 2;
    while (i >= 0) : (i -= 1) {
        while (h >= lower_size and cross2(P[hull_idx[h - 2]], P[hull_idx[h - 1]], P[idx[@intCast(i)]]) <= 0) h -= 1;
        hull_idx[h] = idx[@intCast(i)];
        h += 1;
    }
    h -= 1;
    return h;
}

/// Projection is well-defined iff every `b·xᵢ ≥ feas_margin`. Returns
/// `false` and short-circuits on the first violator; the trailing
/// `P[i..]` is left unspecified. Callers that already know feasibility
/// (e.g. post-`halfspaceCheck` initial projection) can pass
/// `-std.math.inf(f64)` to bypass the check.
pub fn projectGnomonic(X: []const Vec3, b: Vec3, Q: Mat3x2, P: [][2]f64, feas_margin: f64) bool {
    for (X, 0..) |xi, i| {
        const ci = b.dot(xi);
        if (ci < feas_margin) return false;
        const p = Q.applyT(xi);
        P[i] = .{ p.m[0] / ci, p.m[1] / ci };
    }
    return true;
}
