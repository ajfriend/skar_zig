//! Minimum-volume ellipsoidal cone (spherical aspect ratio) solver.
//!
//! Idiomatic Zig port of csrc/sphar.c. Uses @Vector(N, f64) + generic helpers
//! for 3D math. The algorithm is: Farkas halfspace check, optional convex
//! hull preprocessing, FW step + Newton polish + constructed dual certificate
//! in a single outer loop.
//!
//! Allocator convention:
//!   - solve() takes any std.mem.Allocator. The returned Info.cert lives on
//!     that allocator (caller frees via Info.deinit).
//!   - Internally, solve() wraps an ArenaAllocator over the caller's
//!     allocator for transient scratch (O(10) small+medium buffers per call);
//!     the arena's single deinit at function exit replaces per-buffer frees.
//!   - Recommended parent allocators:
//!       * Tests:      std.testing.allocator (leak detection on teardown)
//!       * Production: std.heap.smp_allocator (fast, thread-safe; beats
//!                     std.heap.c_allocator on this workload by ~1.5-3×
//!                     on mid-size cases due to arena-friendly growth)

const std = @import("std");

// ----------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------

const config = @import("config.zig");
const algo = config.algo;
const tol = config.tol;
const SIGMA_0 = config.SIGMA_0;


const linalg = @import("linalg.zig");
const Vec2 = linalg.Vec2;
const Vec3 = linalg.Vec3;
const Mat2 = linalg.Mat2;
const Mat3 = linalg.Mat3;
const Mat3x2 = linalg.Mat3x2;
const Chol3 = linalg.Chol3;
const Eig2 = linalg.Eig2;
const eig2 = linalg.eig2;

// ----------------------------------------------------------------
// Geometric preprocessing (halfspaceCheck, convex hull, projection)
// ----------------------------------------------------------------

const halfspace = @import("halfspace.zig");
const HalfspaceResult = halfspace.HalfspaceResult;
const halfspaceCheck = halfspace.halfspaceCheck;
const convexHull2d = halfspace.convexHull2d;
const projectGnomonic = halfspace.projectGnomonic;

// Public API surface (types, methods, `checkFeasibility`) lives in
// `api.zig`. `solve` is defined below and constructs `api.Outcome`
// variants directly.
const api = @import("api.zig");
const Cert = api.Cert;
const Outcome = api.Outcome;
const SolveError = api.SolveError;
const InputError = api.InputError;
const SolveOptions = api.SolveOptions;

// ----------------------------------------------------------------
// Outer-loop primitives: rescale / moments / damp.
// Each is a thin wrapper so the outer loop reads close to pseudocode.
// All inline → zero runtime cost vs hand-rolled arithmetic.
// ----------------------------------------------------------------

/// Rescale P_buf into Ps so max ‖Ps‖ = 1 (numerical hygiene for FW).
/// Returns the scale factor so callers can lift moments back to
/// unscaled coordinates.
pub inline fn rescaleP(P_buf: []const [2]f64, Ps: [][2]f64) f64 {
    var s2_max: f64 = 0;
    for (P_buf) |p| {
        const sq = @mulAdd(f64, p[1], p[1], p[0] * p[0]);
        if (sq > s2_max) s2_max = sq;
    }
    var s_scale = @sqrt(s2_max);
    if (s_scale < tol.UNDERFLOW) s_scale = 1.0;
    const inv_s = 1.0 / s_scale;
    for (P_buf, 0..) |p, i| Ps[i] = .{ p[0] * inv_s, p[1] * inv_s };
    return s_scale;
}

/// Weighted 2D moments of the scaled projected points, lifted back to
/// original (unscaled) coordinates: center = Σ w·P, M = Σ w·P·Pᵀ.
pub const Moments = struct { center: Vec2, M: Mat2 };

pub inline fn computeMoments(Ps: []const [2]f64, w: []const f64, s_scale: f64) Moments {
    var center_s = Vec2.zero;
    var M_s = Mat2.zero;
    for (Ps, 0..) |p_arr, i| {
        const p = Vec2{ .m = p_arr };
        center_s = Vec2.lincomb(1.0, center_s, w[i], p);
        M_s.addSymRank1(w[i], p);
    }
    return .{ .center = center_s.scale(s_scale), .M = M_s.scale(s_scale * s_scale) };
}

/// Damping controller for the axis update. Shrinks the step when |c|
/// grew, grows it when |c| shrank, bounded in [algo.DAMP_MIN, algo.DAMP_MAX].
/// (`pub` for the trust path's alternating-cadence opening rounds.)
pub const DampState = struct {
    alpha: f64 = 1.0,
    prev_c_norm: f64 = 1e30,

    pub inline fn tick(self: *DampState, c_norm: f64) void {
        if (c_norm > self.prev_c_norm) {
            self.alpha *= algo.DAMP_SHRINK;
            if (self.alpha < algo.DAMP_MIN) self.alpha = algo.DAMP_MIN;
        } else {
            self.alpha *= algo.DAMP_GROW;
            if (self.alpha > algo.DAMP_MAX) self.alpha = algo.DAMP_MAX;
        }
        self.prev_c_norm = c_norm;
    }
};

/// Quasi-Newton axis-update direction in the tangent plane. Returns u =
/// M⁻¹·center (preconditioned by the 2D moment) when M is anisotropic
/// enough to benefit; else u = center. u's magnitude is renormalized to
/// ‖center‖, so on isotropic M the step is bit-identical to the old
/// damped gradient, and the damping signal (`c_norm`, returned alongside)
/// is ‖center‖ either way.
///
/// Skip the whole check for the first algo.AXIS_WARMUP iters — easy cases
/// converge inside the warmup and pay zero preconditioner cost. See
/// docs/mvee_derivation.md "Quasi-Newton axis update" appendix for history.
pub const AxisStep = struct { u: Vec2, c_norm: f64 };

pub inline fn quasiNewtonAxisDirection(outer: u32, M: Mat2, center: Vec2) AxisStep {
    const c_norm = center.norm();
    var u: Vec2 = center;
    if (outer >= algo.AXIS_WARMUP and c_norm > tol.TINY) {
        const eigM = eig2(M.m);
        const eig_lo = eigM.vals[0];
        const eig_hi = eigM.vals[1];
        // cond(M) > algo.PRECOND_COND_MIN  ⟺  eig_hi > algo.PRECOND_COND_MIN · eig_lo,
        // division-free and robust to a near-zero eig_lo that we'd
        // otherwise guard separately.
        if (eig_hi > algo.PRECOND_COND_MIN * eig_lo) {
            const u_p = M.inverse().apply(center);
            const u_norm_2d = u_p.norm();
            if (u_norm_2d > tol.TINY) {
                u = u_p.scale(c_norm / u_norm_2d);
            }
        }
    }
    return .{ .u = u, .c_norm = c_norm };
}

/// Feasibility-safeguarded b update, fused with the projection that the
/// next cycle will consume. The raw step b + α·Q·u can walk b out of the
/// cone {v : v·xᵢ > 0 ∀i}; once outside, projectGnomonic divides by a
/// negative b·xᵢ and the iteration locks onto a spurious cm=0 fixed
/// point (observed on ha_12 rotations). `projectGnomonic` short-circuits
/// on any violator, so each backtrack is one trial projection — on
/// acceptance the next cycle's P_buf/Ps/s_scale are already in place.
///
/// On full rejection, the last rejected trial partially overwrote
/// P_buf; restore it (and Ps/s_scale) against the input (b, Q) so the
/// caller's loop invariant still holds.
pub const BStep = struct { b: Vec3, Q: Mat3x2, s_scale: f64 };

pub fn acceptBUpdate(
    Xw: []const Vec3,
    b: Vec3,
    Q: Mat3x2,
    u: Vec2,
    alpha0: f64,
    P_buf: [][2]f64,
    Ps: [][2]f64,
) BStep {
    const dQc = Q.apply(u);
    var alpha_try: f64 = alpha0;
    var bt: u32 = 0;
    while (bt < algo.MAX_BACKTRACKS) : (bt += 1) {
        const b_trial = Vec3.lincomb(1.0, b, alpha_try, dQc).normalize();
        const Q_trial = b_trial.orthoBasis();
        if (projectGnomonic(Xw, b_trial, Q_trial, P_buf, algo.FEAS_MARGIN)) {
            const s_scale = rescaleP(P_buf, Ps);
            return .{ .b = b_trial, .Q = Q_trial, .s_scale = s_scale };
        }
        alpha_try *= 0.5;
    }
    _ = projectGnomonic(Xw, b, Q, P_buf, -std.math.inf(f64));
    const s_scale = rescaleP(P_buf, Ps);
    return .{ .b = b, .Q = Q, .s_scale = s_scale };
}

// ----------------------------------------------------------------
// MVEE inner: pairwise FW on lifted points [P; 1]
// ----------------------------------------------------------------

pub fn mveeFw(
    P: []const [2]f64,
    max_iter: u32,
    inner_tol: f64,
    Ql: []Vec3,
    w: []f64,
) void {
    for (P, 0..) |p, i| Ql[i] = .{ .m = .{ p[0], p[1], 1.0 } };

    var it: u32 = 0;
    while (it < max_iter) : (it += 1) {
        var S = Mat3.zero;
        // Inlined `addSymRank1` minus the mirror writes — we mirror
        // once after the loop instead of per-iteration. Shared
        // `wq_i = wi · qi.m[i]` precompute + chained @mulAdd: 3 muls
        // + 6 FMAs = 9 rounds per iter, vs the prior 12 muls + 6 adds
        // = 18 ops / 12 rounds. Multiplication order `(wi·qi_r)·qi_c`
        // preserved.
        for (Ql, 0..) |qi, i| {
            const wi = w[i];
            const wq0 = wi * qi.m[0];
            const wq1 = wi * qi.m[1];
            const wq2 = wi * qi.m[2];
            S.m[0] = @mulAdd(f64, wq0, qi.m[0], S.m[0]);
            S.m[1] = @mulAdd(f64, wq0, qi.m[1], S.m[1]);
            S.m[2] = @mulAdd(f64, wq0, qi.m[2], S.m[2]);
            S.m[4] = @mulAdd(f64, wq1, qi.m[1], S.m[4]);
            S.m[5] = @mulAdd(f64, wq1, qi.m[2], S.m[5]);
            S.m[8] = @mulAdd(f64, wq2, qi.m[2], S.m[8]);
        }
        S.m[3] = S.m[1];
        S.m[6] = S.m[2];
        S.m[7] = S.m[5];

        const L = S.cholesky() orelse break;

        var j_max: usize = 0;
        var j_min: ?usize = null;
        var g_max: f64 = -1e30;
        var g_min: f64 = 1e30;
        var x_min: Vec3 = undefined;
        for (Ql, 0..) |qi, i| {
            const x = L.solve(qi);
            const gi = qi.dot(x);
            if (gi > g_max) {
                g_max = gi;
                j_max = i;
            }
            if (w[i] > tol.WEIGHT_ACTIVE and gi < g_min) {
                g_min = gi;
                j_min = i;
                x_min = x;
            }
        }

        if (g_max - 3.0 < inner_tol) break;

        if (j_min) |jm| {
            if (jm != j_max) {
                const g_cross = Ql[j_max].dot(x_min);
                const a = g_max - g_min;
                const det_G = linalg.diff_of_products(g_max, g_min, g_cross, g_cross);
                var step: f64 = if (det_G > tol.NEAR_SING) a / (2.0 * det_G) else w[jm];
                if (step > w[jm]) step = w[jm];
                // Drop guard. A full-mass step (step = w[jm], zeroing the
                // donor) is only justified when it genuinely improves the
                // design. The exact log-det change of the rank-2 update
                // S' = S + γ(q_a·q_aᵀ − q_b·q_bᵀ) is available in closed
                // form from quantities already in hand:
                //   det S'/det S = (1 + γ·g_max)(1 − γ·g_min) + γ²·g_cross²
                // Take the drop only if that ratio exceeds 1 — a
                // threshold-free real-improvement test. Without it, the
                // near-singular fallback (and the cap when det_G is
                // small-but-positive under an anisotropic inner metric)
                // fires full drops on noise-level descent signals at
                // converged designs, zeroing support points that
                // newtonPolish cannot resurrect (its active set is
                // w-thresholded) — the hazard that bit the trust path's
                // oracle four times (docs/trust-solver.md). Interior
                // steps (step < w[jm]) are exact 1-D line-search optima
                // and need no guard. On a blocked drop, fall through to
                // the vanilla FW step below.
                var take = true;
                if (step == w[jm]) {
                    const ratio = (1.0 + step * g_max) * (1.0 - step * g_min) + step * step * g_cross * g_cross;
                    take = ratio > 1.0;
                }
                if (take) {
                    w[j_max] += step;
                    w[jm] -= step;
                    continue;
                }
            }
        }
        // Vanilla FW fallback.
        const step = (g_max - 3.0) / (3.0 * (g_max - 1.0));
        for (w) |*wi| wi.* *= (1.0 - step);
        w[j_max] += step;
    }
}

/// Noise-floor exit for `mveeFwAway`: stop when its optimality gap has
/// not improved by AWAY_GAP_IMPR within AWAY_STALL_ITERS iterations.
/// Local to the (unwired) experimental solver — deliberately NOT in
/// `config.algo`, which is the audited tuning surface for shipping
/// paths.
const AWAY_GAP_IMPR: f64 = 0.9;
const AWAY_STALL_ITERS: u32 = 24;

/// Away-step Frank–Wolfe for the lifted D-optimal design. NOT wired
/// into any solver path: it was stage 1 of docs/away-step-fw.md, tried
/// as the trust path's oracle and reverted (measurably slower than
/// pairwise on large near-circular supports — see the "Stage 1
/// findings" there). Kept in-tree for the record with a bit-rot guard
/// test in tests/methods_test.zig.
///
/// Same per-iteration quantities as `mveeFw` (gradients gᵢ = qᵢᵀS⁻¹qᵢ,
/// toward-vertex j_max, away-vertex j_min), different decision: pick
/// the direction with more first-order progress (toward: g_max − 3;
/// away: 3 − g_min) and take the EXACT 1-D line-search step of the
/// log-det objective along it:
///
///   toward  w ← (1−γ)w + γ·e_j,  γ* = (g−3)/(3(g−1)), capped at 1
///   away    w ← (1+γ)w − γ·e_j,  γ* = (3−g)/(3(g−1)), capped at
///           γmax = w[j]/(1−w[j]) — the drop boundary, where w[j]
///           hits exactly 0. For g ≤ 1 (deep interior point) the
///           objective is monotone along the away ray, so the full
///           drop is optimal.
///
/// Because the step length is proportional to the gap it closes, a
/// noise-level away gap produces a noise-level step — this solver
/// cannot fire the full-mass drop `mveeFw`'s near-singular pairwise
/// fallback takes on noise at converged designs (the hazard that bit
/// the trust path four times; see docs/trust-solver.md).
pub fn mveeFwAway(
    P: []const [2]f64,
    max_iter: u32,
    inner_tol: f64,
    Ql: []Vec3,
    w: []f64,
) void {
    for (P, 0..) |p, i| Ql[i] = .{ .m = .{ p[0], p[1], 1.0 } };

    var gap_best: f64 = std.math.inf(f64);
    var since_best: u32 = 0;
    var it: u32 = 0;
    while (it < max_iter) : (it += 1) {
        var S = Mat3.zero;
        for (Ql, 0..) |qi, i| {
            const wi = w[i];
            const wq0 = wi * qi.m[0];
            const wq1 = wi * qi.m[1];
            const wq2 = wi * qi.m[2];
            S.m[0] = @mulAdd(f64, wq0, qi.m[0], S.m[0]);
            S.m[1] = @mulAdd(f64, wq0, qi.m[1], S.m[1]);
            S.m[2] = @mulAdd(f64, wq0, qi.m[2], S.m[2]);
            S.m[4] = @mulAdd(f64, wq1, qi.m[1], S.m[4]);
            S.m[5] = @mulAdd(f64, wq1, qi.m[2], S.m[5]);
            S.m[8] = @mulAdd(f64, wq2, qi.m[2], S.m[8]);
        }
        S.m[3] = S.m[1];
        S.m[6] = S.m[2];
        S.m[7] = S.m[5];

        const L = S.cholesky() orelse break;

        var j_max: usize = 0;
        var j_min: ?usize = null;
        var g_max: f64 = -1e30;
        var g_min: f64 = 1e30;
        for (Ql, 0..) |qi, i| {
            const x = L.solve(qi);
            const gi = qi.dot(x);
            if (gi > g_max) {
                g_max = gi;
                j_max = i;
            }
            if (w[i] > tol.WEIGHT_ACTIVE and gi < g_min) {
                g_min = gi;
                j_min = i;
            }
        }

        const toward_gap = g_max - 3.0;
        const away_gap = if (j_min != null) 3.0 - g_min else -1e30;
        const gap = @max(toward_gap, away_gap);
        if (gap < inner_tol) break;
        // Noise-floor exit on the solver's OWN optimality measure: when
        // the gap stops improving geometrically it has hit the f64
        // floor for this input (κ-limited cells never reach any fixed
        // inner_tol) — stop instead of random-walking the budget away.
        // This replaces the caller-side burst/stall machinery: the gap
        // is a sound convergence signal; an h-sample window was a proxy
        // that misread slow-but-genuine descent phases as stalls.
        if (gap < AWAY_GAP_IMPR * gap_best) {
            gap_best = gap;
            since_best = 0;
        } else {
            since_best += 1;
            if (since_best >= AWAY_STALL_ITERS) break;
        }

        if (toward_gap >= away_gap) {
            const denom = 3.0 * (g_max - 1.0);
            if (denom < tol.NEAR_SING) break;
            var gamma = toward_gap / denom;
            if (gamma > 1.0) gamma = 1.0;
            for (w) |*wi| wi.* *= (1.0 - gamma);
            w[j_max] += gamma;
        } else {
            const jm = j_min.?;
            const one_minus = 1.0 - w[jm];
            if (one_minus < tol.NEAR_SING) break; // sole support point
            const gamma_max = w[jm] / one_minus;
            var gamma: f64 = gamma_max;
            if (g_min > 1.0 + tol.NEAR_SING) {
                const g_star = away_gap / (3.0 * (g_min - 1.0));
                if (g_star < gamma_max) gamma = g_star;
            }
            for (w) |*wi| wi.* *= (1.0 + gamma);
            w[jm] -= gamma;
            if (w[jm] < 0 or gamma == gamma_max) w[jm] = 0;
        }
    }
}

/// Uniform FW weights `w_i = 1/len`. The maximum-entropy start, optimal for
/// near-circular inputs whose enclosing ellipse touches every point.
fn uniformWeights(w: []f64) void {
    const inv = 1.0 / @as(f64, @floatFromInt(w.len));
    for (w) |*wi| wi.* = inv;
}

/// Sparse farthest-point seed of the FW weights: pick up to `k_req` well-spread
/// points and put weight 1/m on them (m = #picks), 0 elsewhere, so the inner FW
/// *grows* the support instead of *draining* a full active set. The first three
/// picks (farthest-from-centroid, farthest-from-that, farthest-from-their-line)
/// are non-collinear given a 2D-spanning scatter — guaranteed upstream by
/// `isCoplanarInput` — so the lifted [P;1] simplex is full rank and `mveeFw`'s
/// first Cholesky won't break. Falls back to uniform if the scatter is
/// degenerate. O(k·n), once. Gated on input size — see
/// `algo.SEED_SPARSE_MIN_POINTS` for the rationale and the a5_res0 story.
fn farthestPointSeed(P: []const [2]f64, w: []f64, k_req: usize) void {
    const max_seeds = 16; // buffer bound; k_req is small (algo.SEED_SPARSE_K = 5)
    const n = P.len;
    const k = @min(@min(k_req, n), max_seeds);
    var picks: [max_seeds]usize = undefined;

    // Centroid of P.
    var c = Vec2.zero;
    for (P) |p| c = c.add(.{ .m = p });
    c = c.scale(1.0 / @as(f64, @floatFromInt(n)));

    // pick0: farthest from centroid.
    var p0: usize = 0;
    var d0_max: f64 = -1;
    for (P, 0..) |p, i| {
        const d = (Vec2{ .m = p }).sub(c).norm();
        if (d > d0_max) {
            d0_max = d;
            p0 = i;
        }
    }
    if (d0_max < tol.TINY or k < 3) {
        uniformWeights(w); // degenerate (or tiny): nothing to seed
        return;
    }
    picks[0] = p0;

    // pick1: farthest from pick0.
    const a = Vec2{ .m = P[p0] };
    var p1: usize = p0;
    var d1_max: f64 = -1;
    for (P, 0..) |p, i| {
        const d = (Vec2{ .m = p }).sub(a).norm();
        if (d > d1_max) {
            d1_max = d;
            p1 = i;
        }
    }
    picks[1] = p1;

    // pick2: farthest (perpendicular) from the line pick0–pick1. The divisor
    // ‖b−a‖ is constant over i, so maximizing |cross| suffices.
    const bma = (Vec2{ .m = P[p1] }).sub(a);
    var p2: usize = p0;
    var cr_max: f64 = -1;
    for (P, 0..) |p, i| {
        const pma = (Vec2{ .m = p }).sub(a);
        const cr = @abs(linalg.diff_of_products(bma.m[0], pma.m[1], bma.m[1], pma.m[0]));
        if (cr > cr_max) {
            cr_max = cr;
            p2 = i;
        }
    }
    picks[2] = p2;

    // Remaining picks: farthest-from-the-chosen-set (max-min distance).
    var m: usize = 3;
    while (m < k) : (m += 1) {
        var best: usize = 0;
        var best_mindist: f64 = -1;
        for (P, 0..) |p, i| {
            const pv = Vec2{ .m = p };
            var mindist: f64 = std.math.inf(f64);
            for (0..m) |j| {
                const d = pv.sub(.{ .m = P[picks[j]] }).norm();
                if (d < mindist) mindist = d;
            }
            if (mindist > best_mindist) {
                best_mindist = mindist;
                best = i;
            }
        }
        picks[m] = best;
    }

    // Weights: 1/m on the picks, 0 elsewhere.
    for (w) |*wi| wi.* = 0;
    const wval = 1.0 / @as(f64, @floatFromInt(m));
    for (0..m) |j| w[picks[j]] = wval;
}

/// Initialize the inner-FW weight vector, choosing the regime by working-set
/// size: large/dense inputs get a sparse farthest-point seed (so FW grows the
/// support instead of draining it — the a5_res0 DNC fix, also faster on genuine
/// medium/large inputs); small inputs get the uniform start (already optimal for
/// near-circular cells, where a sparse seed would break symmetry and slow them).
/// `P` and `w` index the same working set. See `algo.SEED_SPARSE_MIN_POINTS`.
pub fn initWeights(P: []const [2]f64, w: []f64) void {
    if (P.len > algo.SEED_SPARSE_MIN_POINTS) {
        farthestPointSeed(P, w, algo.SEED_SPARSE_K);
    } else {
        uniformWeights(w);
    }
}

// ----------------------------------------------------------------
// Solution recovery: 2D shape M → 3D A
// ----------------------------------------------------------------

/// Recovers the 2×2 tangent-plane shape A_perp from the weights' moment matrix M.
/// A_perp is Minv_half scaled by √(2/(3·g_max)), where g_max = max_i pᵢᵀ·M⁻¹·pᵢ
/// enforces the budget max_i ‖A_perp·pᵢ‖² = 2/3 that pins the axial eigenvalue
/// of A to SIGMA_0.
pub fn recoverAPerp(P: []const [2]f64, M: Mat2) SolveError!Mat2 {
    const Minv = M.inverse();

    // Max of pᵀ M⁻¹ p over points (used for scaling).
    var g_max: f64 = 0;
    for (P) |p_arr| {
        const p = Vec2{ .m = p_arr };
        const g = p.dot(Minv.apply(p));
        if (g > g_max) g_max = g;
    }

    // Closed-form sqrt of symmetric SPD 2×2 Minv:
    //   sqrt(S) = (S + √det(S)·I) / √(tr(S) + 2√det(S))
    // avoids eigendecomposition when eigenvalues are nearly equal.
    // Minv is PSD by construction (M is PD ⇒ Minv is PD), so det(Minv)
    // and tr(Minv) are both ≥ 0 in exact arithmetic. Roundoff can push
    // det negative when M is near-singular; clip ulp-scale noise and
    // raise SingularMoment beyond that. tr is a sum of squared FMAs,
    // bounded below by 0 structurally, but we guard it the same way
    // for completeness.
    const tr = Minv.m[0] + Minv.m[3];
    const det = Minv.det();
    if (det < -tol.PSD_NEG_REL * tr * tr) return SolveError.SingularMoment;
    const s_det = @sqrt(@max(det, 0));
    const denom = @sqrt(@mulAdd(f64, 2.0, s_det, tr));
    const eye2: Mat2 = .{ .m = .{ 1, 0, 0, 1 } };
    const Minv_half = Mat2.lincomb(1.0 / denom, Minv, s_det / denom, eye2);

    const budget: f64 = 2.0 / 3.0;
    return Minv_half.scale(@sqrt(budget / g_max));
}

// ----------------------------------------------------------------
// Newton polish (extracted to newton.zig)
// ----------------------------------------------------------------

const newton = @import("newton.zig");
const NewtonScratch = newton.NewtonScratch;
const newtonPolish = newton.newtonPolish;

// The trust solver path (`SolveOptions.method`).
const trust = @import("trust.zig");

// ----------------------------------------------------------------
// Dual-certificate gap scratch
// ----------------------------------------------------------------

/// Scratch for `dualityGapConstructed` (constructed dual certificate + gap).
pub const GapScratch = struct {
    active_idx: []usize, // [nmax]  points with w > thresh
    lam: []f64, // [nmax]  dual lambdas: 3 w_i / (b·x_i)
    xa: []Vec3, // [nmax]  active x_i (from X_work)
    za: []Vec3, // [nmax]  normalized A x_i / ‖A x_i‖

    pub fn init(allocator: std.mem.Allocator, nmax: usize) !GapScratch {
        return .{
            .active_idx = try allocator.alloc(usize, nmax),
            .lam = try allocator.alloc(f64, nmax),
            .xa = try allocator.alloc(Vec3, nmax),
            .za = try allocator.alloc(Vec3, nmax),
        };
    }
};

/// Per-call working buffers backing the outer loop. All allocations
/// live on the scratch arena passed to `init`, so there's no `deinit` —
/// `solve` frees the arena once at the end. The fields are mutable
/// slices; methods that take them (`mveeFw`, `newtonPolish`,
/// `dualityGapConstructed`, etc.) read or write directly.
const WorkBuffers = struct {
    P_buf: [][2]f64,
    Ps: [][2]f64,
    Ql: []Vec3,
    w: []f64,
    cert_active: []usize,
    cert_lambdas: []f64,
    newton_scratch: NewtonScratch,
    gap_scratch: GapScratch,

    fn init(scratch: std.mem.Allocator, nw: usize) !WorkBuffers {
        return .{
            .P_buf = try scratch.alloc([2]f64, nw),
            .Ps = try scratch.alloc([2]f64, nw),
            .Ql = try scratch.alloc(Vec3, nw),
            .w = try scratch.alloc(f64, nw),
            .cert_active = try scratch.alloc(usize, nw),
            .cert_lambdas = try scratch.alloc(f64, nw),
            .newton_scratch = try NewtonScratch.init(scratch, nw),
            .gap_scratch = try GapScratch.init(scratch, nw),
        };
    }
};


// ----------------------------------------------------------------
// Dual-certificate gap
// ----------------------------------------------------------------

pub const GapResult = struct {
    gap: f64,
    cert_n: usize,
    /// A's tangent-plane eigenvectors (lifted to 3D) and eigenvalues. Valid
    /// only when gap < tol.GAP_UNCERTIFIED; `solve` reuses these to fill `info.Q`/`info.sigma`,
    /// skipping a redundant eig2 + lift at the end of the outer loop.
    v1: Vec3,
    v2: Vec3,
    sigma: [2]f64,
};

/// Assemble A from its eigendecomposition: A = (1/√3)·b·bᵀ + σ₁·v₁·v₁ᵀ
/// + σ₂·v₂·v₂ᵀ. Used internally by the gap computation; consumers
/// should call `Converged.A()` instead (in `api.zig`).
fn buildA(b: Vec3, v1: Vec3, v2: Vec3, sigma1: f64, sigma2: f64) Mat3 {
    var m = Mat3.zero;
    m.addSymRank1(SIGMA_0, b);
    m.addSymRank1(sigma1, v1);
    m.addSymRank1(sigma2, v2);
    return m;
}

/// Structural dual gap on (b, A_perp, Q_ortho). A's eigendecomposition falls out
/// of eig(A_perp) + lifting through Q_ortho, so we build L = V·√Λ directly — no
/// Cholesky with fallback.
pub fn dualityGapConstructed(
    w: []const f64,
    b: Vec3,
    X_work: []const Vec3,
    A_perp: Mat2,
    Q_ortho: Mat3x2,
    s: *GapScratch,
    cert_active_out: []usize,
    cert_lambdas_out: []f64,
) SolveError!GapResult {
    // A's eigendecomposition: V = [b | v₁ | v₂], Λ = diag(SIGMA_0, σ₁, σ₂).
    // Always valid (depends only on A_perp and Q_ortho); returned in GapResult
    // so `solve`'s finalization reuses it without re-decomposing.
    const eAPerp = eig2(A_perp.m);
    // A_perp is PSD by construction; eig2 can produce ulp-scale negative
    // eigenvalues from FP noise. Clip noise to 0 (so the sqrt below is
    // well-defined and downstream M = LᵀZL routes through the Cholesky
    // null guard as "no progress"), but raise NegativeEigenvalue when
    // the negative value is meaningful — that signals Newton polish
    // landed on a non-PSD iterate or eig2 has a bug.
    const sigma_raw: [2]f64 = eAPerp.vals;
    const sigma_neg_thr = tol.PSD_NEG_REL * @max(sigma_raw[1], 1.0);
    if (sigma_raw[0] < -sigma_neg_thr) return SolveError.NegativeEigenvalue;
    const sigma: [2]f64 = .{ @max(sigma_raw[0], 0), @max(sigma_raw[1], 0) };
    const v1 = Vec3.lincomb(eAPerp.vecs.m[0], Q_ortho.e1, eAPerp.vecs.m[1], Q_ortho.e2);
    const v2 = Vec3.lincomb(eAPerp.vecs.m[2], Q_ortho.e1, eAPerp.vecs.m[3], Q_ortho.e2);

    const active_idx = s.active_idx;
    const lam = s.lam;
    const xa = s.xa;
    const za = s.za;
    var k: usize = 0;
    for (w, 0..) |wi, i| {
        if (wi > algo.ACTIVE_THRESH) {
            active_idx[k] = i;
            k += 1;
        }
    }
    if (k == 0) return .{ .gap = tol.GAP_UNCERTIFIED, .cert_n = 0, .v1 = v1, .v2 = v2, .sigma = sigma };

    // Materialize A once; per-point matvec in the zᵢ loop is cheaper than a
    // structural A·x decomposition once there are ≥ 2 points.
    const A = buildA(b, v1, v2, sigma[0], sigma[1]);

    for (0..k) |i| {
        const idx = active_idx[i];
        xa[i] = X_work[idx];
        lam[i] = 3.0 * w[idx] / b.dot(xa[i]);
        za[i] = A.apply(xa[i]).normalize();
    }

    // Z = Σᵢ λᵢ · (xᵢ zᵢᵀ + zᵢ xᵢᵀ) / 2
    var Z = Mat3.zero;
    for (0..k) |i| {
        Z.addSymRank2(lam[i], xa[i], za[i]);
    }

    // L = V·√Λ so L·Lᵀ = A. Non-triangular, but we only use it in the
    // symmetric similarity Lᵀ·Z·L — any square root of A works there.
    const L0 = b.scale(@sqrt(SIGMA_0));
    const L1 = v1.scale(@sqrt(sigma[0]));
    const L2 = v2.scale(@sqrt(sigma[1]));
    const L = Mat3{ .m = .{
        L0.m[0], L1.m[0], L2.m[0],
        L0.m[1], L1.m[1], L2.m[1],
        L0.m[2], L1.m[2], L2.m[2],
    } };

    // M = Lᵀ · Z · L. eig(M) = eig(A·Z); eigenvalues cluster near 1 at
    // convergence, so Cholesky on M is well-conditioned. A failed pivot
    // is the indefinite-dual guard — Z not PSD enough for log det.
    const M = L.transpose().mul(Z).mul(L).symmetrize();
    const Lm = M.cholesky() orelse
        return .{ .gap = tol.GAP_UNCERTIFIED, .cert_n = 0, .v1 = v1, .v2 = v2, .sigma = sigma };

    var w_sum = Vec3.zero;
    for (0..k) |i| {
        w_sum = Vec3.lincomb(1.0, w_sum, lam[i], xa[i]);
    }

    for (0..k) |i| {
        cert_active_out[i] = active_idx[i];
        cert_lambdas_out[i] = lam[i];
    }

    // gap = (−log det Z − 3 + ‖w‖) − log det A, and via the similarity
    //   log det Z = log det M − log det A,
    // so the two log det A terms cancel:  gap = ‖w‖ − 3 − log det M.
    // Routing through M (eigenvalues near 1 at convergence) avoids the
    // ~1e-3 error that sum-of-logs on Z's own ill-conditioned eigenvalues
    // would suffer (hex-degenerate cases, κ(Z) ~ 1e7).
    const gap = w_sum.norm() - 3.0 - Lm.logDet();
    return .{
        .gap = gap,
        .cert_n = k,
        .v1 = v1,
        .v2 = v2,
        .sigma = sigma,
    };
}

// ----------------------------------------------------------------
// Preprocessing helpers used by `solve`
// ----------------------------------------------------------------

/// Build the infeasibility certificate from the halfspace result.
/// Keeps only the nonzero (above-threshold) λ entries with their
/// original indices. The witness magnitude ‖Σ λᵢ xᵢ‖ lives on the
/// enclosing `Infeasible` variant as `residual`, not on the cert.
fn buildFarkasCert(allocator: std.mem.Allocator, hs: HalfspaceResult) !Cert {
    var k: u32 = 0;
    for (hs.lam) |l| if (l > algo.ACTIVE_THRESH) {
        k += 1;
    };
    const indices = try allocator.alloc(u32, k);
    errdefer allocator.free(indices);
    const lambdas = try allocator.alloc(f64, k);
    var j: u32 = 0;
    for (hs.lam, 0..) |l, i| {
        if (l > algo.ACTIVE_THRESH) {
            indices[j] = @intCast(i);
            lambdas[j] = l;
            j += 1;
        }
    }
    return .{ .indices = indices, .lambdas = lambdas };
}

/// Build the active-set certificate for a converged (or DNC-best-effort)
/// solve. Translates work-set indices back to the caller's original
/// `X[]` indexing via `work_to_orig` (`null` when no hull reduction
/// happened). The scalar quality measurement (the `gap` field) lives
/// on the enclosing variant.
fn buildPrimalCert(
    allocator: std.mem.Allocator,
    cert_active: []const usize,
    cert_lambdas: []const f64,
    cert_n: usize,
    work_to_orig: ?[]const u32,
) !Cert {
    const indices = try allocator.alloc(u32, cert_n);
    errdefer allocator.free(indices);
    const lambdas = try allocator.alloc(f64, cert_n);
    for (0..cert_n) |i| {
        const idx_in_work = cert_active[i];
        indices[i] = if (work_to_orig) |wto| wto[idx_in_work] else @intCast(idx_in_work);
        lambdas[i] = cert_lambdas[i];
    }
    return .{ .indices = indices, .lambdas = lambdas };
}

const HullResult = struct {
    /// The working point set the solver should iterate on. Either the hull
    /// subset (when reduction fired) or the original input (when disabled,
    /// skipped, or the hull collapsed to < 3 vertices).
    Xw: []const Vec3,
    /// Indices into the original X[] for each point in Xw. `null` when no
    /// reduction was performed (Xw == Xv); the solver uses the identity
    /// mapping in that case.
    work_to_orig: ?[]const u32,
};

/// Optional convex-hull preprocessing. If `n_hull >= 0` and there are more
/// than `n_hull` points, project to the tangent plane at b, run Andrew's
/// monotone chain, and keep only the hull vertices. Falls back to the
/// original input on disable, small n, or hull-collapse (< 3 vertices).
fn hullPreprocess(
    scratch: std.mem.Allocator,
    Xv: []const Vec3,
    b: Vec3,
    n_hull: i32,
) !HullResult {
    var result = HullResult{ .Xw = Xv, .work_to_orig = null };
    if (n_hull < 0 or Xv.len <= @as(usize, @intCast(n_hull))) return result;

    const Qh = b.orthoBasis();
    const P2 = try scratch.alloc([2]f64, Xv.len);
    for (Xv, 0..) |xi, i| {
        P2[i] = Qh.applyT(xi).m;
    }
    // 2·n: Andrew's monotone chain uses `hull_idx` as scratch for the
    // lower and upper passes; on inputs where most points are on the
    // hull (e.g. equispaced on a circle), both passes can write up to
    // n entries before the final dedup. Allocating only n overflows.
    const hull_idx = try scratch.alloc(u32, 2 * Xv.len);
    const nh = try convexHull2d(scratch, P2, hull_idx);
    if (nh >= 3) {
        const Xhull = try scratch.alloc(Vec3, nh);
        for (0..nh) |i| Xhull[i] = Xv[hull_idx[i]];
        result.Xw = Xhull;
        result.work_to_orig = hull_idx[0..nh];
    }
    return result;
}

/// True iff the points lie (approximately) in a 2D subspace through the
/// origin — i.e., on a single great circle. Projects to the tangent plane
/// at b and tests the 2×2 centered scatter via `4·det(C) < tol · trace(C)²`.
/// That's the cancellation-safe form of `λ_min/λ_max` for ill-conditioned C
/// (the literal `(T − √(T² − 4D))/2` form loses precision exactly where the
/// check needs to fire). Scale-invariant "fraction of isotropic" ∈ [0, 1]:
/// 1 for a circular scatter, → 0 for a perfect line. Tight clusters on the
/// sphere (e.g. H3 res-15) have full-rank 2D scatter regardless of absolute
/// scale, so this correctly distinguishes them from genuinely rank-deficient
/// input.
///
/// Implementation: two-pass accumulator. Pass 1 computes the mean; pass 2
/// accumulates squared deviations from the mean. The textbook one-pass form
/// (`Σx² − (Σx)²/n`) is cancellation-prone when the projection cluster sits
/// far from the tangent-plane origin (mean comparable in magnitude to spread).
/// Two-pass avoids the subtraction entirely — each deviation term is small
/// and non-negative, so `tr ≥ 0` is structural rather than a roundoff
/// coincidence.
fn isCoplanarInput(points: []const Vec3, b: Vec3, threshold: f64) bool {
    const Qh = b.orthoBasis();

    // Pass 1: mean of the 2D projections.
    var ps0: f64 = 0;
    var ps1: f64 = 0;
    for (points) |xi| {
        const p = Qh.applyT(xi);
        ps0 += p.m[0];
        ps1 += p.m[1];
    }
    const inv_n = 1.0 / @as(f64, @floatFromInt(points.len));
    const m0 = ps0 * inv_n;
    const m1 = ps1 * inv_n;

    // Pass 2: squared deviations from mean — no cancellation.
    var c00: f64 = 0;
    var c01: f64 = 0;
    var c11: f64 = 0;
    for (points) |xi| {
        const p = Qh.applyT(xi);
        const d0 = p.m[0] - m0;
        const d1 = p.m[1] - m1;
        c00 += d0 * d0;
        c01 += d0 * d1;
        c11 += d1 * d1;
    }

    const tr = c00 + c11;
    const det = linalg.diff_of_products(c00, c11, c01, c01);
    return tr <= 0 or 4.0 * det < threshold * tr * tr;
}

/// Preprocessed problem handed to a solver path: a strictly feasible
/// axis, the (possibly hull-reduced) working point set, and the map
/// back to the caller's original indices (`null` = identity).
pub const Prep = struct {
    b0: Vec3,
    Xw: []const Vec3,
    work_to_orig: ?[]const u32,
};

const PrepResult = union(enum) {
    /// Input is infeasible; carries the ready-to-return outcome
    /// (Farkas certificate already on the parent allocator).
    infeasible: Outcome,
    ready: Prep,
};

/// Steps shared by every solver path: input validation, Farkas
/// feasibility check, optional hull reduction, coplanarity rejection.
/// `scratch_alloc` backs `Xw`/`work_to_orig` (arena, freed by `solve`);
/// `allocator` backs the Farkas cert on the infeasible branch.
fn preprocess(
    scratch_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
    Xv: []const Vec3,
    opts: SolveOptions,
) !PrepResult {
    // 0) Input validation. Catch malformed caller inputs at the boundary
    //    so they propagate as typed errors instead of slipping into the
    //    algorithm where they manifest as NaN-tainted statuses or silent
    //    perf cliffs. See the InputError doc-comments in api.zig for the
    //    contract on each tolerance.
    if (Xv.len < 3) return InputError.InsufficientPoints;
    if (!std.math.isFinite(opts.gap_tol) or opts.gap_tol <= 0 or opts.gap_tol >= tol.GAP_UNCERTIFIED) return InputError.InvalidTolerance;
    if (std.math.isNan(opts.coplanarity_tol)) return InputError.InvalidTolerance;

    // 1) Feasibility via Farkas FW.
    const hs = try halfspaceCheck(scratch_alloc, Xv);
    var b: Vec3 = undefined;
    if (hs.b) |bb| {
        b = bb;
    } else {
        // Infeasible: Farkas cert lives on the parent allocator since it's
        // returned to the caller.
        const farkas = try buildFarkasCert(allocator, hs);
        return .{ .infeasible = .{ .infeasible = .{
            .cert = farkas,
            .residual = hs.residual,
            .allocator = allocator,
        } } };
    }

    // 2) Optional hull preprocessing.
    const hp = try hullPreprocess(scratch_alloc, Xv, b, opts.n_hull);

    // 2.5) Coplanarity check on the hulled subset — an input whose hull is
    //      collinear in the tangent plane drives the SDP to a degenerate
    //      cone (one tangent eigenvalue → 0) and produces NaN downstream.
    //      Signaled as `InputError.CoplanarInput`, symmetric with
    //      `InsufficientPoints` — both are "X is structurally bad."
    if (opts.coplanarity_tol > 0 and isCoplanarInput(hp.Xw, b, opts.coplanarity_tol)) {
        return InputError.CoplanarInput;
    }

    return .{ .ready = .{ .b0 = b, .Xw = hp.Xw, .work_to_orig = hp.work_to_orig } };
}

/// The alternating path: alternating axis/MVEE outer loop (FW + Newton polish
/// + constructed dual certificate). This is the original `solve` body;
/// see the module doc-comment for the algorithm.
fn solveAlternating(
    allocator: std.mem.Allocator,
    scratch_alloc: std.mem.Allocator,
    prep: Prep,
    opts: SolveOptions,
) !Outcome {
    var b = prep.b0;
    const Xw = prep.Xw;
    const work_to_orig = prep.work_to_orig;
    const nw = Xw.len;

    // 3) Working buffers — all backed by the arena, freed once at the
    //    end of `solve`.
    var wb = try WorkBuffers.init(scratch_alloc, nw);

    var damp = DampState{};
    var outer_count: u32 = 0;
    var converged = false;
    var newton_polish_failures: u32 = 0;

    // Eigen-data from the last gap call — feeds the converged/partial
    // outcome's Q/sigma at finalization without a redundant eig2 + lift.
    var last_gap = GapResult{ .gap = tol.GAP_UNCERTIFIED, .cert_n = 0, .v1 = Vec3.zero, .v2 = Vec3.zero, .sigma = .{ 0, 0 } };
    // Axis at which last_gap was computed. The outer loop steps b AFTER
    // certifying, so on DNC the final b is one step past the last
    // certificate — returning (b_cert, last_gap) keeps the outcome's
    // Q/sigma/gap a consistent snapshot of one iterate.
    var b_cert = b;

    // Orthonormal tangent basis at the current b. Rebuilt after each
    // accepted step in the outer loop (trivial: one project-and-normalize
    // plus one cross-and-normalize; see `Vec3.orthoBasis`).
    var Q: Mat3x2 = b.orthoBasis();

    // Seed P_buf/Ps/s_scale so the loop invariant holds on entry to the
    // first cycle. `halfspaceCheck` guarantees b·xᵢ > 0 strictly (not
    // necessarily ≥ algo.FEAS_MARGIN), so bypass the feasibility check here.
    _ = projectGnomonic(Xw, b, Q, wb.P_buf, -std.math.inf(f64));
    var s_scale: f64 = rescaleP(wb.P_buf, wb.Ps);

    // FW weight init (sparse seed vs uniform, chosen by size — see initWeights).
    initWeights(wb.Ps, wb.w);

    // 4) Hybrid outer loop. Each outer iteration runs algo.FW_PER_NEWTON cycles
    //    of (FW + b-update); only the last cycle also runs Newton polish +
    //    gap check. Extra cheap cycles buy more b-motion per Newton call.
    //    At algo.FW_PER_NEWTON = 1 this is the original one-FW-per-Newton
    //    schedule; larger values amortise Newton's cost across more b-motion.
    //
    //    Loop invariant: on entry to each cycle, P_buf/Ps/s_scale correspond
    //    to the current (b, Q). The accepted b-update at cycle end also
    //    produces the next cycle's projection in one sweep.
    var outer: u32 = 0;
    outer_loop: while (outer < opts.max_outer) : (outer += 1) {
        outer_count += 1;
        var cycle: u32 = 0;
        while (cycle < algo.FW_PER_NEWTON) : (cycle += 1) {
            const is_full = (cycle == algo.FW_PER_NEWTON - 1);

            mveeFw(wb.Ps, 1, 0.0, wb.Ql, wb.w);

            if (is_full) {
                if (!newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch)) {
                    newton_polish_failures += 1;
                }
            }

            const m = computeMoments(wb.Ps, wb.w, s_scale);

            if (is_full) {
                const A_perp = try recoverAPerp(wb.P_buf, m.M);
                last_gap = try dualityGapConstructed(wb.w, b, Xw, A_perp, Q, &wb.gap_scratch, wb.cert_active, wb.cert_lambdas);
                b_cert = b;
                // Convergence + broken-certificate guard (see
                // gapConverged for the load-bearing ordering).
                if (try gapConverged(last_gap.gap, opts.gap_tol)) {
                    converged = true;
                    break :outer_loop;
                }
            }

            const axis = quasiNewtonAxisDirection(outer, m.M, m.center);
            damp.tick(axis.c_norm);
            const step = acceptBUpdate(Xw, b, Q, axis.u, damp.alpha, wb.P_buf, wb.Ps);
            b = step.b;
            Q = step.Q;
            s_scale = step.s_scale;
        }
    }

    // 5) Build final cert (translate work indices back to original X indices)
    //    and bundle the outcome. Shared with the trust path.
    return buildOutcome(
        allocator,
        converged,
        b_cert,
        last_gap,
        .{ .alternating = .{
            .outer_iters = outer_count,
            .newton_polish_failures = newton_polish_failures,
        } },
        wb.cert_active,
        wb.cert_lambdas,
        work_to_orig,
    );
}

/// Classify a freshly computed certificate gap. Returns true when the
/// solve is converged at `gap_tol`. ORDER MATTERS and is shared by
/// every certification site in both solver paths: a converged-at-noise
/// gap can be slightly negative (seen on H3 r15 cells, gap ~ −5e-9
/// from κ·ε noise) and must be ACCEPTED before the hard NegGap guard
/// fires; anything meaningfully negative beyond `tol.NEG_GAP` is a
/// broken certificate and errors loudly.
pub fn gapConverged(gap: f64, gap_tol: f64) SolveError!bool {
    // The no-certificate sentinel is not a measured gap and must never
    // certify, no matter how loose gap_tol is (validation additionally
    // caps gap_tol below it, so this guard is belt-and-braces).
    if (gap >= tol.GAP_UNCERTIFIED) return false;
    if (@abs(gap) <= gap_tol) return true;
    if (gap < -tol.NEG_GAP) return SolveError.NegativeDualityGap;
    return false;
}

/// Shared finalization for the alternating and trust paths: translate the
/// work-set certificate back to caller indices, bundle the full
/// eigendecomposition (Q's columns are (b, v1, v2) with eigenvalues
/// (SIGMA_0, sigma[0], sigma[1]); v2 flipped if needed so det Q = +1),
/// and wrap as Converged / DidNotConverge.
/// `b` MUST be the axis at which `last_gap` was computed: Q's
/// orthonormality (and the meaning of gap/sigma) depends on v1/v2
/// being tangent to this exact axis. Callers track a `b_cert`
/// alongside `last_gap` for this reason.
pub fn buildOutcome(
    allocator: std.mem.Allocator,
    converged: bool,
    b: Vec3,
    last_gap: GapResult,
    diag: api.Diagnostics,
    cert_active: []const usize,
    cert_lambdas: []const f64,
    work_to_orig: ?[]const u32,
) !Outcome {
    const cert = try buildPrimalCert(allocator, cert_active, cert_lambdas, last_gap.cert_n, work_to_orig);

    var v1 = last_gap.v1;
    var v2 = last_gap.v2;
    if (v1.cross(v2).dot(b) < 0) v2 = v2.scale(-1.0);
    const Qmat = Mat3.fromCols(b, v1, v2);
    const sigma: [3]f64 = .{ SIGMA_0, last_gap.sigma[0], last_gap.sigma[1] };

    if (converged) {
        return .{ .converged = .{
            .Q = Qmat,
            .sigma = sigma,
            .gap = last_gap.gap,
            .diag = diag,
            .cert = cert,
            .allocator = allocator,
        } };
    } else {
        return .{ .did_not_converge = .{
            .Q = Qmat,
            .sigma = sigma,
            .gap = last_gap.gap,
            .diag = diag,
            .cert = cert,
            .allocator = allocator,
        } };
    }
}

/// Main solver. Returns an `Outcome` tagged union — switch on the tag
/// to dispatch (`converged` carries the cone's eigendecomposition +
/// primal certificate; `infeasible` carries the Farkas certificate;
/// `did_not_converge` carries the last iterate for diagnostics).
/// Structural input problems (too few points, bad tolerance,
/// rank-deficient X) propagate as `InputError` via `try`. `opts`
/// controls convergence, preprocessing, validation, and solver-path
/// knobs — see `SolveOptions` for per-field docs and defaults.
///
/// Preprocessing (validation, Farkas feasibility, hull reduction,
/// coplanarity rejection) is shared; `opts.method` selects the solver
/// path that runs on the preprocessed working set (see `api.Method`).
pub fn solve(
    allocator: std.mem.Allocator,
    X: []const [3]f64,
    opts: SolveOptions,
) !Outcome {
    // Arena for all transient scratch allocations in this solve call.
    // Single backing alloc (bumped) + single free-all on deinit — vastly
    // cheaper than per-buffer alloc/free. The returned cert (for the
    // Converged / Infeasible / DidNotConverge variants) lives on the
    // parent `allocator` so it outlives the arena.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch_alloc = arena.allocator();

    // Cast once: Vec3 is an extern struct over [3]f64, so layout is shared.
    // All internal routines work in []const Vec3.
    const Xv: []const Vec3 = @ptrCast(X);

    const prep = switch (try preprocess(scratch_alloc, allocator, Xv, opts)) {
        .infeasible => |outcome| return outcome,
        .ready => |p| p,
    };

    switch (opts.method.resolved()) {
        .alternating => return solveAlternating(allocator, scratch_alloc, prep, opts),
        .trust => return trust.solveTrust(allocator, scratch_alloc, prep, opts),
        .auto => unreachable, // resolved() maps .auto to Method.recommended
    }
}

