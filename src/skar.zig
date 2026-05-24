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

/// Structural axial eigenvalue: A·b = SIGMA_0·b, where b is the cone axis.
/// Derived in `recoverAPerp` via the budget/g_max rescaling: λ_b = √(1 − 2/3).
/// Not tunable — it's geometry, not a knob.
const SIGMA_0: f64 = 1.0 / @sqrt(3.0);

/// Algorithm tuning parameters — internal knobs tuned together for the
/// solver to converge cleanly. Not exposed to callers because they
/// interact subtly: changing one without coordinated changes to others
/// can break convergence. Adjust here if you're working on the algorithm
/// itself; user-facing tuning is in `SolveOptions`.
const algo = struct {
    /// Number of (project + FW + b-update) cycles per outer iteration.
    /// Only the final cycle of each outer iteration runs Newton polish
    /// + gap check. FW_PER_NEWTON = 1 is the original behaviour.
    const FW_PER_NEWTON: u32 = 2;

    /// Damping curve for the b-update: shrink alpha when |c| grew,
    /// grow when |c| shrank, bounded in [DAMP_MIN, DAMP_MAX].
    const DAMP_SHRINK: f64 = 0.5;
    const DAMP_GROW: f64 = 1.2;
    const DAMP_MIN: f64 = 0.05;
    const DAMP_MAX: f64 = 1.0;

    /// Certificate active-set cutoff: weights below this are dropped
    /// from `Info.cert`. Distinct from (and tighter than) the FW inner
    /// loops' `tol.WEIGHT_ACTIVE`.
    const ACTIVE_THRESH: f64 = 1e-6;

    /// Feasibility-cone margin for the backtracking b-update. Each
    /// outer step requires `min_i(b_new · xᵢ) ≥ FEAS_MARGIN`; α is
    /// halved up to MAX_BACKTRACKS times until the new b satisfies it.
    const FEAS_MARGIN: f64 = 1e-8;
    const MAX_BACKTRACKS: u32 = 30;

    /// Quasi-Newton b-update gate: only precondition the axis step by
    /// M⁻¹ when cond(M) exceeds this. For near-isotropic M (hex, DGGS
    /// cells, rotations near coordinate axes) the preconditioner adds
    /// sub-ULP direction noise that interacts badly with damping after
    /// Newton polish; the plain gradient step is used instead.
    const PRECOND_COND_MIN: f64 = 1.2;

    /// Skip the quasi-Newton machinery for the first `AXIS_WARMUP`
    /// outer iterations. Easy cases (hex, most DGGS cells) converge
    /// in ≤ this, so they pay zero preconditioner overhead.
    const AXIS_WARMUP: u32 = 2;
};

/// Numerical tolerances — the "how small is small" guards.
/// These guard against divide-by-zero, underflow, and spurious convergence.
/// Tuned to f64 precision; not exposed to callers.
const tol = struct {
    /// Newton polish inner loop: stop when max-min of gradient components < this.
    const NEWTON_INNER: f64 = 1e-14;
    /// Newton polish: fraction-to-boundary step-size floor; below, declare stuck.
    const NEWTON_STEP_MIN: f64 = 1e-12;
    /// Hard floor for SolveError.NegativeDualityGap (FP noise below, bug above).
    const NEG_GAP: f64 = 1e-10;
    /// FW inner loops: minimum w_i to participate in the pairwise-swap candidate set.
    /// Distinct from (and looser than) algo.ACTIVE_THRESH, which is the *cert* cutoff.
    const WEIGHT_ACTIVE: f64 = 1e-14;
    /// Tiny-magnitude zero guard for norms and dot-products (`< tol ⇒ treat as 0`).
    const TINY: f64 = 1e-30;
    /// 2D det / scalar singular guard (denominator-is-zero cutoff).
    const NEAR_SING: f64 = 1e-15;
    /// halfspaceCheck: z.dot(z) ceiling below which FW cannot make progress.
    const FW_Z_EXHAUSTED: f64 = 1e-12;
    /// Underflow floor: pivot / scale / log argument.
    const UNDERFLOW: f64 = 1e-300;
    /// Relative cutoff for "FP noise" vs. "theorem violation" on values
    /// that should be ≥ 0 by PSD invariant (eigenvalues of A_perp,
    /// det of Minv). Below the threshold ⇒ silent clip; above ⇒ loud
    /// SolveError. Mirrors NEG_GAP's role for the gap.
    const PSD_NEG_REL: f64 = 1e-12;
};


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
// Geometric preprocessing
// ----------------------------------------------------------------

const HalfspaceResult = struct {
    /// If found: unit vector b with x_i · b > 0 for all i.
    b: ?Vec3,
    /// If infeasible: lambda weights on the input points (λ ≥ 0, ∑ λ = 1).
    lam: []f64,
    /// ‖∑ λᵢ xᵢ‖ — small = sharp Farkas certificate; large = FW stalled.
    residual: f64,
};

fn halfspaceCheck(allocator: std.mem.Allocator, X: []const Vec3) !HalfspaceResult {
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

fn convexHull2d(allocator: std.mem.Allocator, P: []const [2]f64, hull_idx: []u32) !u32 {
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
fn projectGnomonic(X: []const Vec3, b: Vec3, Q: Mat3x2, P: [][2]f64, feas_margin: f64) bool {
    for (X, 0..) |xi, i| {
        const ci = b.dot(xi);
        if (ci < feas_margin) return false;
        const p = Q.applyT(xi);
        P[i] = .{ p.m[0] / ci, p.m[1] / ci };
    }
    return true;
}

// ----------------------------------------------------------------
// Outer-loop primitives: rescale / moments / damp.
// Each is a thin wrapper so the outer loop reads close to pseudocode.
// All inline → zero runtime cost vs hand-rolled arithmetic.
// ----------------------------------------------------------------

/// Rescale P_buf into Ps so max ‖Ps‖ = 1 (numerical hygiene for FW).
/// Returns the scale factor so callers can lift moments back to
/// unscaled coordinates.
inline fn rescaleP(P_buf: []const [2]f64, Ps: [][2]f64) f64 {
    var s2_max: f64 = 0;
    for (P_buf) |p| {
        const sq = p[0] * p[0] + p[1] * p[1];
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
const Moments = struct { center: Vec2, M: Mat2 };

inline fn computeMoments(Ps: []const [2]f64, w: []const f64, s_scale: f64) Moments {
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
const DampState = struct {
    alpha: f64 = 1.0,
    prev_c_norm: f64 = 1e30,

    inline fn tick(self: *DampState, c_norm: f64) void {
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
const AxisStep = struct { u: Vec2, c_norm: f64 };

inline fn quasiNewtonAxisDirection(outer: u32, M: Mat2, center: Vec2) AxisStep {
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
const BStep = struct { b: Vec3, Q: Mat3x2, s_scale: f64 };

fn acceptBUpdate(
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

fn mveeFw(
    P: []const [2]f64,
    max_iter: u32,
    inner_tol: f64,
    Ql: []Vec3,
    w: []f64,
    warm: bool,
) void {
    const n = P.len;
    for (P, 0..) |p, i| Ql[i] = .{ .m = .{ p[0], p[1], 1.0 } };
    if (!warm) {
        const inv_n = 1.0 / @as(f64, @floatFromInt(n));
        for (w) |*wi| wi.* = inv_n;
    }

    var it: u32 = 0;
    while (it < max_iter) : (it += 1) {
        var S = Mat3.zero;
        for (Ql, 0..) |qi, i| {
            const wi = w[i];
            S.m[0] += wi * qi.m[0] * qi.m[0];
            S.m[1] += wi * qi.m[0] * qi.m[1];
            S.m[2] += wi * qi.m[0] * qi.m[2];
            S.m[4] += wi * qi.m[1] * qi.m[1];
            S.m[5] += wi * qi.m[1] * qi.m[2];
            S.m[8] += wi * qi.m[2] * qi.m[2];
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
                const det_G = g_max * g_min - g_cross * g_cross;
                var step: f64 = if (det_G > tol.NEAR_SING) a / (2.0 * det_G) else w[jm];
                if (step > w[jm]) step = w[jm];
                w[j_max] += step;
                w[jm] -= step;
                continue;
            }
        }
        // Vanilla FW fallback.
        const step = (g_max - 3.0) / (3.0 * (g_max - 1.0));
        for (w) |*wi| wi.* *= (1.0 - step);
        w[j_max] += step;
    }
}

// ----------------------------------------------------------------
// Solution recovery: 2D shape M → 3D A
// ----------------------------------------------------------------

/// Recovers the 2×2 tangent-plane shape A_perp from the weights' moment matrix M.
/// A_perp is Minv_half scaled by √(2/(3·g_max)), where g_max = max_i pᵢᵀ·M⁻¹·pᵢ
/// enforces the budget max_i ‖A_perp·pᵢ‖² = 2/3 that pins the axial eigenvalue
/// of A to SIGMA_0.
fn recoverAPerp(P: []const [2]f64, M: Mat2) SolveError!Mat2 {
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
    const denom = @sqrt(tr + 2.0 * s_det);
    const eye2: Mat2 = .{ .m = .{ 1, 0, 0, 1 } };
    const Minv_half = Mat2.lincomb(1.0 / denom, Minv, s_det / denom, eye2);

    const budget: f64 = 2.0 / 3.0;
    return Minv_half.scale(@sqrt(budget / g_max));
}

// ----------------------------------------------------------------
// Newton polish scratch + KKT solver + Newton iteration
// ----------------------------------------------------------------

/// Scratch for `newtonPolish` + `solveBorderedKkt` (active-set Newton's
/// method on the D-optimal dual). All fields are owned by the caller's
/// allocator (typically an arena scoped to one solve call) — no deinit.
const NewtonScratch = struct {
    active_idx: []usize, // [nmax]      points with w > thresh
    q: []Vec3, // [nmax]      active lifted points [P; 1]
    w_a: []f64, // [nmax]      active weights
    Y: []Vec3, // [nmax]      W⁻¹ q_i  (W = Σ w_i q_i q_iᵀ)
    g: []f64, // [nmax]      gradient q_iᵀ W⁻¹ q_i  (→ 3 at optimum)
    H: []f64, // [nmax²]     Hessian (q_iᵀ W⁻¹ q_j)²
    delta_w: []f64, // [nmax]      Newton step
    KKT: []f64, // [(nmax+1)²] bordered KKT [H, 1; 1ᵀ, 0]
    rhs: []f64, // [nmax+1]    KKT RHS
    piv: []usize, // [nmax+1]    LU pivot indices

    fn init(allocator: std.mem.Allocator, nmax: usize) !NewtonScratch {
        const n1 = nmax + 1;
        return .{
            .active_idx = try allocator.alloc(usize, nmax),
            .q = try allocator.alloc(Vec3, nmax),
            .w_a = try allocator.alloc(f64, nmax),
            .Y = try allocator.alloc(Vec3, nmax),
            .g = try allocator.alloc(f64, nmax),
            .H = try allocator.alloc(f64, nmax * nmax),
            .delta_w = try allocator.alloc(f64, nmax),
            .KKT = try allocator.alloc(f64, n1 * n1),
            .rhs = try allocator.alloc(f64, n1),
            .piv = try allocator.alloc(usize, n1),
        };
    }
};

/// Scratch for `dualityGapConstructed` (constructed dual certificate + gap).
const GapScratch = struct {
    active_idx: []usize, // [nmax]  points with w > thresh
    lam: []f64, // [nmax]  dual lambdas: 3 w_i / (b·x_i)
    xa: []Vec3, // [nmax]  active x_i (from X_work)
    za: []Vec3, // [nmax]  normalized A x_i / ‖A x_i‖

    fn init(allocator: std.mem.Allocator, nmax: usize) !GapScratch {
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

/// LU factorization with partial pivoting. Storage (`data`, `piv`) is
/// borrowed from the caller — `factorize` mutates `data` in place to hold
/// the packed L\U factors. The returned handle just binds the dimension
/// to those slices so `solve` can't mismatch them.
const LU = struct {
    data: []f64, // n·n, row-major; L (strict lower, unit diag) + U (upper)
    piv: []usize, // n
    n: usize,

    /// In-place factorization. Returns null on singular.
    fn factorize(data: []f64, n: usize, piv: []usize) ?LU {
        for (0..n) |kk| {
            var pmax = kk;
            var vmax = @abs(data[kk * n + kk]);
            for (kk + 1..n) |i| {
                const v = @abs(data[i * n + kk]);
                if (v > vmax) {
                    vmax = v;
                    pmax = i;
                }
            }
            if (vmax < tol.UNDERFLOW) return null;
            piv[kk] = pmax;
            if (pmax != kk) {
                for (0..n) |j| {
                    const t = data[kk * n + j];
                    data[kk * n + j] = data[pmax * n + j];
                    data[pmax * n + j] = t;
                }
            }
            const inv = 1.0 / data[kk * n + kk];
            for (kk + 1..n) |i| {
                data[i * n + kk] *= inv;
                for (kk + 1..n) |j| {
                    data[i * n + j] -= data[i * n + kk] * data[kk * n + j];
                }
            }
        }
        return .{ .data = data, .piv = piv, .n = n };
    }

    /// In-place solve: overwrites b with the solution of (P·L·U)·x = b.
    fn solve(self: LU, b: []f64) void {
        const n = self.n;
        const data = self.data;
        const piv = self.piv;
        for (0..n) |kk| {
            const p = piv[kk];
            if (p != kk) {
                const t = b[kk];
                b[kk] = b[p];
                b[p] = t;
            }
        }
        for (1..n) |i| {
            for (0..i) |j| b[i] -= data[i * n + j] * b[j];
        }
        var i: usize = n;
        while (i > 0) {
            i -= 1;
            var j = i + 1;
            while (j < n) : (j += 1) b[i] -= data[i * n + j] * b[j];
            b[i] /= data[i * n + i];
        }
    }
};

/// Bordered KKT [H, 1; 1', 0] [Δw; -ν] = [g; 0] via LU on the (k+1)×(k+1)
/// symmetric indefinite system.
fn solveBorderedKkt(H: []const f64, k: usize, g: []const f64, delta_w: []f64, s: *NewtonScratch) bool {
    const n = k + 1;
    const K = s.KKT;
    for (0..k) |i| {
        for (0..k) |j| K[i * n + j] = H[i * k + j];
        K[i * n + k] = 1.0;
        K[k * n + i] = 1.0;
    }
    K[k * n + k] = 0.0;

    const rhs = s.rhs;
    for (0..k) |i| rhs[i] = g[i];
    rhs[k] = 0.0;

    const lu = LU.factorize(K, n, s.piv) orelse return false;
    lu.solve(rhs);
    for (0..k) |i| delta_w[i] = rhs[i];
    return true;
}

/// Newton polish on the D-optimal dual restricted to {i : w_i > active_thresh}.
/// Mutates w in place; inactive entries reset to 0 on exit.
/// Returns false on failure (<3 active, Cholesky breakdown, or KKT singular).
fn newtonPolish(Ql: []const Vec3, w: []f64, active_thresh: f64, max_iter: u32, inner_tol: f64, s: *NewtonScratch) bool {
    const active_idx = s.active_idx;
    var k: usize = 0;
    for (w, 0..) |wi, i| {
        if (wi > active_thresh) {
            active_idx[k] = i;
            k += 1;
        }
    }
    if (k < 3) return false;

    const q = s.q;
    const w_a = s.w_a;
    for (0..k) |i| {
        const idx = active_idx[i];
        q[i] = Ql[idx];
        w_a[i] = w[idx];
    }

    const Y = s.Y;
    const g = s.g;
    const H = s.H;
    const delta_w = s.delta_w;

    var it: u32 = 0;
    while (it < max_iter) : (it += 1) {
        // S = Σ wᵢ qᵢ qᵢᵀ
        var S = Mat3.zero;
        for (0..k) |i| S.addSymRank1(w_a[i], q[i]);

        const L_W = S.cholesky() orelse return false;

        // yᵢ = W⁻¹ qᵢ,  gᵢ = qᵢ · yᵢ
        for (0..k) |i| {
            Y[i] = L_W.solve(q[i]);
            g[i] = q[i].dot(Y[i]);
        }

        var g_max: f64 = -1e30;
        var g_min: f64 = 1e30;
        for (0..k) |i| {
            if (g[i] > g_max) g_max = g[i];
            if (g[i] < g_min) g_min = g[i];
        }
        if (g_max - g_min < inner_tol) break;

        // H is symmetric: H_ij = (qᵢ · W⁻¹ qⱼ)² = (qᵢ · yⱼ)²
        for (0..k) |i| {
            for (i..k) |j| {
                const dij = q[i].dot(Y[j]);
                H[i * k + j] = dij * dij;
                H[j * k + i] = H[i * k + j];
            }
        }

        if (!solveBorderedKkt(H, k, g, delta_w, s)) return false;

        var alpha: f64 = 1.0;
        for (0..k) |i| {
            if (delta_w[i] < 0) {
                const a = 0.99 * (-w_a[i] / delta_w[i]);
                if (a < alpha) alpha = a;
            }
        }
        if (alpha < tol.NEWTON_STEP_MIN) break;
        for (0..k) |i| w_a[i] += alpha * delta_w[i];
    }

    for (w) |*wi| wi.* = 0;
    for (0..k) |i| w[active_idx[i]] = w_a[i];
    return true;
}

// ----------------------------------------------------------------
// Dual-certificate gap
// ----------------------------------------------------------------

const GapResult = struct {
    gap: f64,
    cert_n: usize,
    /// A's tangent-plane eigenvectors (lifted to 3D) and eigenvalues. Valid
    /// only when gap < 1e30; `solve` reuses these to fill `info.Q`/`info.sigma`,
    /// skipping a redundant eig2 + lift at the end of the outer loop.
    v1: Vec3,
    v2: Vec3,
    sigma: [2]f64,
};

/// Structural dual gap on (b, A_perp, Q_ortho). A's eigendecomposition falls out
/// of eig(A_perp) + lifting through Q_ortho, so we build L = V·√Λ directly — no
/// Cholesky with fallback.
fn dualityGapConstructed(
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
    if (k == 0) return .{ .gap = 1e30, .cert_n = 0, .v1 = v1, .v2 = v2, .sigma = sigma };

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
        return .{ .gap = 1e30, .cert_n = 0, .v1 = v1, .v2 = v2, .sigma = sigma };

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
    const log_det_M = 2.0 * (@log(Lm.m[0]) + @log(Lm.m[4]) + @log(Lm.m[8]));
    const gap = w_sum.norm() - 3.0 - log_det_M;
    return .{
        .gap = gap,
        .cert_n = k,
        .v1 = v1,
        .v2 = v2,
        .sigma = sigma,
    };
}

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------
//
// Two-axis result model for `solve`:
//
//   - Errors (signaled via the `!` in the return type, an inferred
//     union over `SolveError || InputError || Allocator.Error`) mean
//     the call could not produce a meaningful `Info`. Three sources:
//     the host couldn't cooperate (`OutOfMemory`); the caller passed
//     invalid arguments (`InputError`); or the library miscomputed
//     something internally (`SolveError`, each variant signalling a
//     PSD or duality theorem violation beyond floating-point noise).
//     `try` propagation is the right default for all three — the
//     caller cooperates with allocation / fixes their input / files a
//     bug against the library, respectively.
//
//   - `Info.status` describes what the algorithm *found* on the input.
//     Callers switch on it to dispatch — use the certificate, ask the
//     user to fix the input, retry with more iterations, etc. Every
//     status variant corresponds to a meaningful (possibly partial)
//     `Info` the caller can inspect.
//
// In short: errors = "couldn't run"; status = "ran, here's the answer."

pub const Status = enum {
    /// Solver closed the duality gap within `gap_tol`. `Info.cert` holds
    /// the primal certificate; `Info.Q` and `Info.sigma` hold the
    /// eigendecomposition of A.
    converged,
    /// No feasible cone exists for the input. `Info.cert` holds the
    /// Farkas certificate (`λ ≥ 0`, `Σλ = 1`, `‖Σ λᵢ xᵢ‖` small) and
    /// `claimed_gap` is the Farkas residual.
    infeasible,
    /// Solver hit max iterations without closing the gap. `Info.Q` and
    /// `Info.sigma` reflect the last iterate (near-feasible but
    /// uncertified).
    did_not_converge,
    /// Coplanarity check rejected the input before iteration: the
    /// points' tangent-plane projections at the feasible axis form a
    /// near-collinear 2D scatter, so the SDP would be degenerate
    /// (one tangent eigenvalue → 0). The literal "coplanar with the
    /// origin" case (all points on a great circle) is the dominant
    /// instance, but the check is slightly broader — short arcs on
    /// non-equatorial latitude circles can also project to a near-line
    /// in the tangent plane and trigger this. Either way the solver
    /// can't produce a meaningful cone. `Info` is otherwise empty.
    /// Disable the check by passing `coplanarity_tol ≤ 0` to `solve`
    /// if you want to handle this case yourself.
    coplanar_input,
};

/// Internal-correctness errors. Distinct from `Allocator.Error` (the
/// host couldn't allocate) — these mean the library produced a result
/// that violates a theorem and the bug needs to be surfaced loudly.
/// All three variants share the same tolerance-band shape: ulp-level
/// negatives on PSD-invariant values are float noise and silently
/// clipped; anything beyond `tol.NEG_GAP` / `tol.PSD_NEG_REL`
/// propagates as a typed error.
pub const SolveError = error{
    /// The duality-gap computation produced a meaningfully negative
    /// value — either the dual certificate is not actually feasible,
    /// or the log-det was computed on ill-conditioned input. Weak
    /// duality (`gap ≥ 0`) is a theorem, so this signals a bug.
    /// ulp-level negatives are float noise and silently ignored;
    /// anything beyond that propagates as this error.
    NegativeDualityGap,
    /// `eig2(A_perp)` produced a smaller eigenvalue below the
    /// PSD-noise threshold. A_perp is PSD by construction (it's the
    /// perpendicular block of the dual ellipsoid), so a meaningfully
    /// negative eigenvalue means either Newton polish landed on an
    /// infeasible iterate or `eig2` has a bug. ulp-level negatives
    /// are clipped to 0; anything beyond `tol.PSD_NEG_REL · max_eig`
    /// propagates as this error.
    NegativeEigenvalue,
    /// `recoverAPerp` saw `det(Minv) < 0` beyond float noise. M is
    /// PSD by construction (weighted sum of outer products with
    /// non-negative weights), so its inverse should also be PSD and
    /// its determinant non-negative. A meaningfully negative det
    /// signals that M is numerically singular and `recoverAPerp`
    /// can't proceed.
    SingularMoment,
};

/// Errors signalling the caller passed invalid arguments to `solve`.
/// Distinct from `SolveError` (which signals internal-correctness
/// bugs) — these are recoverable from the caller's side by passing
/// better input.
pub const InputError = error{
    /// `X.len < 3`. The SDP is structurally degenerate for fewer
    /// than 3 input points (the algorithm needs at least one point
    /// per tangent dimension to define a non-degenerate cone).
    /// Caller should aggregate / dedupe upstream or fall back to a
    /// trivial bounding-cone routine.
    InsufficientPoints,
    /// A tolerance argument (`gap_tol` or `coplanarity_tol`) was not
    /// finite, or had an invalid sign. See the parameter docs on
    /// `solve` for the contract on each.
    InvalidTolerance,
};

/// User-tunable solver options. Pass `.{}` to use defaults; override
/// individual fields with named-field syntax: `.{ .gap_tol = 1e-9 }`.
///
/// These are the knobs a typical caller might legitimately want to
/// twist (perf-vs-accuracy trade-offs, behavior toggles). Deeper
/// tuning constants — Frank-Wolfe inner cycles, damping curve,
/// backtracking, preconditioner gates — are kept internal in `algo`
/// because they interact subtly with each other.
pub const SolveOptions = struct {
    /// Convergence threshold on the duality gap. Must be finite and
    /// positive. Smaller = tighter solution but more iterations.
    gap_tol: f64 = 1e-6,

    /// Convex-hull preprocessing threshold. If `X.len > n_hull`,
    /// reduce input to its 2D hull at the feasible axis before
    /// iterating. `-1` disables; `0` always hulls. Default 10 is a
    /// good break-even point on typical inputs.
    n_hull: i32 = 10,

    /// Coplanarity check threshold (see `Status.coplanar_input`).
    /// `4·det(C) < tol · trace(C)²` on the centered 2D scatter
    /// triggers rejection. ≤ 0 disables the check; tighter positive
    /// values catch only essentially-exact coplanarity; looser
    /// values also reject near-coplanar inputs the solver would
    /// otherwise NaN on.
    coplanarity_tol: f64 = 1e-12,

    /// Outer iteration cap before returning `Status.did_not_converge`.
    /// Each outer iteration runs `algo.FW_PER_NEWTON` inner cycles +
    /// one Newton polish + one gap check.
    max_outer: u32 = 100,
};

pub const Cert = struct {
    indices: []u32,
    lambdas: []f64,
    /// On `.converged`: the duality gap |primal − dual| (≤ `gap_tol`).
    /// On `.infeasible`: the Farkas residual ‖∑ λᵢ xᵢ‖.
    /// On `.did_not_converge`: the last computed gap from the final
    /// outer iteration. May be near zero (almost converged) or large
    /// (the solver gave up far from optimal); inspect alongside
    /// `status` and `outer_iters` rather than as a uniform quality
    /// metric.
    /// On `.coplanar_input`: 0 (no certificate was constructed).
    claimed_gap: f64,
};

pub const Info = struct {
    status: Status,
    /// Full eigenbasis of A as columns of a 3×3 orthonormal matrix:
    ///   Q[:,0] = b (cone axis, structural eigenvalue λ_b = 1/√3)
    ///   Q[:,1], Q[:,2] = tangent-plane eigenvectors
    /// Right-handed: det(Q) = +1.
    Q: Mat3,
    /// Eigenvalues of A pairing with Q's columns (A·Q[:,i] = sigma[i]·Q[:,i]):
    ///   sigma[0] = 1/√3 (SIGMA_0): structural axial eigenvalue
    ///   sigma[1] ≤ sigma[2]: tangent-plane eigenvalues
    /// Aspect ratio of the cone cross-section = sigma[2] / sigma[1].
    sigma: [3]f64,
    outer_iters: u32,
    /// Count of outer iterations where Newton polish bailed and FW weights were used.
    newton_polish_failures: u32,
    cert: Cert,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Info) void {
        self.allocator.free(self.cert.indices);
        self.allocator.free(self.cert.lambdas);
    }

    /// Aspect ratio of the cone cross-section = sigma[2] / sigma[1] ≥ 1.
    /// NaN (via 0/0) on any non-converged status — sigma stays at its
    /// zero-initialized value on `infeasible` and `coplanar_input`, and
    /// `did_not_converge` may have partial sigma but no guarantee of
    /// meaningful aspect ratio. Callers that care should gate on
    /// `status == .converged` before reading.
    pub fn aspectRatio(self: Info) f64 {
        return self.sigma[2] / self.sigma[1];
    }

    /// Cone axis: first column of Q. Only meaningful on `.converged` —
    /// on other statuses Q stays at its zero-initialized value and
    /// this silently returns `Vec3.zero`. Callers should gate on
    /// `status == .converged` before reading.
    pub fn b(self: Info) Vec3 {
        return self.Q.col(0);
    }

    /// Materialize A from its eigendecomposition: Σᵢ sigma[i] · Q[:,i] · Q[:,i]ᵀ.
    /// Cheap (three symmetric rank-1 updates). For a loop applying A to many
    /// vectors, call once and reuse. Only meaningful on `.converged` — on
    /// other statuses sigma and Q stay at zero and this silently returns
    /// the zero matrix.
    pub fn A(self: Info) Mat3 {
        var m = Mat3.zero;
        m.addSymRank1(self.sigma[0], self.Q.col(0));
        m.addSymRank1(self.sigma[1], self.Q.col(1));
        m.addSymRank1(self.sigma[2], self.Q.col(2));
        return m;
    }
};

/// Assemble A from its eigendecomposition: A = (1/√3)·b·bᵀ + σ₁·v₁·v₁ᵀ
/// + σ₂·v₂·v₂ᵀ. Used internally; consumers should call `Info.A()` instead.
fn buildA(b: Vec3, v1: Vec3, v2: Vec3, sigma1: f64, sigma2: f64) Mat3 {
    var m = Mat3.zero;
    m.addSymRank1(SIGMA_0, b);
    m.addSymRank1(sigma1, v1);
    m.addSymRank1(sigma2, v2);
    return m;
}

/// Max primal violation `‖A·xᵢ‖ − b·xᵢ` over all input points. Negative
/// or zero means every point sits inside the cone defined by `info`;
/// positive means the certificate doesn't cover at least one point.
///
/// Returns `+inf` on any non-converged status — `info.A()` and `info.b()`
/// are not meaningful when the solver didn't produce a certificate, so
/// "violation = 0 for every point" would be misleading apparent
/// feasibility. The `+inf` sentinel composes correctly with the typical
/// `checkFeasibility(info, X) <= tol` gate: it always rejects.
pub fn checkFeasibility(info: Info, X: []const [3]f64) f64 {
    if (info.status != .converged) return std.math.inf(f64);
    const A = info.A();
    const bv = info.b();
    var max_viol: f64 = -1e30;
    const Xv: []const Vec3 = @ptrCast(X);
    for (Xv) |xi| {
        const viol = A.apply(xi).norm() - bv.dot(xi);
        if (viol > max_viol) max_viol = viol;
    }
    return max_viol;
}

// ----------------------------------------------------------------
// Preprocessing helpers used by `solve`
// ----------------------------------------------------------------

/// Build a Farkas infeasibility certificate from the halfspace result.
/// Keeps only the nonzero (above-threshold) λ entries with their original
/// indices. `claimed_gap` holds the Farkas residual ‖Σ λᵢ xᵢ‖.
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
    return .{ .indices = indices, .lambdas = lambdas, .claimed_gap = hs.residual };
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
    const hull_idx = try scratch.alloc(u32, Xv.len);
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
    const det = c00 * c11 - c01 * c01;
    return tr <= 0 or 4.0 * det < threshold * tr * tr;
}

/// Main solver. Returns an `Info` carrying the result `Status`, the
/// cone's eigendecomposition (when converged), and a certificate.
/// `opts` controls convergence, preprocessing, and validation knobs —
/// see `SolveOptions` for per-field docs and defaults.
pub fn solve(
    allocator: std.mem.Allocator,
    X: []const [3]f64,
    opts: SolveOptions,
) !Info {
    var info = Info{
        .status = .did_not_converge,
        .Q = Mat3.zero,
        .sigma = .{ 0, 0, 0 },  // aspectRatio() returns NaN via 0/0 on any non-converged status
        .outer_iters = 0,
        .newton_polish_failures = 0,
        .cert = .{
            .indices = &[_]u32{},
            .lambdas = &[_]f64{},
            .claimed_gap = 0,
        },
        .allocator = allocator,
    };

    // Arena for all transient scratch allocations in this solve call.
    // Single backing alloc (bumped) + single free-all on deinit — vastly
    // cheaper than per-buffer alloc/free. The returned Info.cert lives on
    // the parent `allocator` so it outlives the arena.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch_alloc = arena.allocator();

    // Cast once: Vec3 is an extern struct over [3]f64, so layout is shared.
    // All internal routines work in []const Vec3.
    const Xv: []const Vec3 = @ptrCast(X);

    // 0) Input validation. Catch malformed caller inputs at the boundary
    //    so they propagate as typed errors instead of slipping into the
    //    algorithm where they manifest as NaN-tainted statuses or silent
    //    perf cliffs. See the InputError doc-comments above for the
    //    contract on each tolerance.
    if (Xv.len < 3) return InputError.InsufficientPoints;
    if (!std.math.isFinite(opts.gap_tol) or opts.gap_tol <= 0) return InputError.InvalidTolerance;
    if (std.math.isNan(opts.coplanarity_tol)) return InputError.InvalidTolerance;

    // 1) Feasibility via Farkas FW.
    const hs = try halfspaceCheck(scratch_alloc, Xv);
    var b: Vec3 = undefined;
    if (hs.b) |bb| {
        b = bb;
    } else {
        // Infeasible: Farkas cert lives on the parent allocator since it's
        // returned to the caller.
        info.status = .infeasible;
        info.cert = try buildFarkasCert(allocator, hs);
        return info;
    }

    // 2) Optional hull preprocessing.
    const hp = try hullPreprocess(scratch_alloc, Xv, b, opts.n_hull);
    const Xw = hp.Xw;
    const work_to_orig = hp.work_to_orig;
    const nw = Xw.len;

    // 2.5) Coplanarity check on the hulled subset — an input whose hull is
    //      collinear in the tangent plane drives the SDP to a degenerate
    //      cone (one tangent eigenvalue → 0) and produces NaN downstream.
    if (opts.coplanarity_tol > 0 and isCoplanarInput(Xw, b, opts.coplanarity_tol)) {
        // Allocate empty cert slices on the parent allocator so Info.deinit
        // is uniform across statuses — never frees a static-literal pointer.
        const indices = try allocator.alloc(u32, 0);
        errdefer allocator.free(indices);
        const lambdas = try allocator.alloc(f64, 0);
        info.cert = .{ .indices = indices, .lambdas = lambdas, .claimed_gap = 0 };
        info.status = .coplanar_input;
        return info;
    }

    // 3) Working buffers — all backed by the arena, freed once at the
    //    end of `solve`.
    var wb = try WorkBuffers.init(scratch_alloc, nw);
    var cert_n: usize = 0;
    var final_gap: f64 = 1e30;

    const inv_nw = 1.0 / @as(f64, @floatFromInt(nw));
    for (wb.w) |*wi| wi.* = inv_nw;

    var damp = DampState{};
    var outer_count: u32 = 0;
    var converged = false;
    var newton_polish_failures: u32 = 0;

    // Eigen-data from the last gap call — feeds info.Q/info.sigma at finalization
    // without a redundant eig2 + lift.
    var last_gap = GapResult{ .gap = 1e30, .cert_n = 0, .v1 = Vec3.zero, .v2 = Vec3.zero, .sigma = .{ 0, 0 } };

    // Orthonormal tangent basis at the current b. Rebuilt after each
    // accepted step in the outer loop (trivial: one project-and-normalize
    // plus one cross-and-normalize; see `Vec3.orthoBasis`).
    var Q: Mat3x2 = b.orthoBasis();

    // Seed P_buf/Ps/s_scale so the loop invariant holds on entry to the
    // first cycle. `halfspaceCheck` guarantees b·xᵢ > 0 strictly (not
    // necessarily ≥ algo.FEAS_MARGIN), so bypass the feasibility check here.
    _ = projectGnomonic(Xw, b, Q, wb.P_buf, -std.math.inf(f64));
    var s_scale: f64 = rescaleP(wb.P_buf, wb.Ps);

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

            mveeFw(wb.Ps, 1, 0.0, wb.Ql, wb.w, true);

            if (is_full) {
                if (!newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch)) {
                    newton_polish_failures += 1;
                }
            }

            const m = computeMoments(wb.Ps, wb.w, s_scale);

            if (is_full) {
                const A_perp = try recoverAPerp(wb.P_buf, m.M);
                last_gap = try dualityGapConstructed(wb.w, b, Xw, A_perp, Q, &wb.gap_scratch, wb.cert_active, wb.cert_lambdas);
                final_gap = last_gap.gap;
                cert_n = last_gap.cert_n;
                // Convergence: |gap| ≤ tol. FP noise can push the gap
                // slightly negative when the iteration has converged to
                // a near-zero gap (seen on h3_r15_pent: gap = -8.5e-10
                // with tol = 1e-6). Accept those as converged before the
                // hard NegGap guard kicks in.
                if (@abs(last_gap.gap) <= opts.gap_tol) {
                    converged = true;
                    break :outer_loop;
                }
                // Anything else negative is a broken certificate.
                if (last_gap.gap < -tol.NEG_GAP) return SolveError.NegativeDualityGap;
            }

            const axis = quasiNewtonAxisDirection(outer, m.M, m.center);
            damp.tick(axis.c_norm);
            const step = acceptBUpdate(Xw, b, Q, axis.u, damp.alpha, wb.P_buf, wb.Ps);
            b = step.b;
            Q = step.Q;
            s_scale = step.s_scale;
        }
    }

    // 5) Build final cert (translate work indices back to original X indices).
    const indices = try allocator.alloc(u32, cert_n);
    errdefer allocator.free(indices);
    const lambdas = try allocator.alloc(f64, cert_n);
    for (0..cert_n) |i| {
        const idx_in_work = wb.cert_active[i];
        indices[i] = if (work_to_orig) |wto| wto[idx_in_work] else @intCast(idx_in_work);
        lambdas[i] = wb.cert_lambdas[i];
    }

    info.outer_iters = outer_count;
    info.newton_polish_failures = newton_polish_failures;
    info.cert = .{ .indices = indices, .lambdas = lambdas, .claimed_gap = final_gap };

    // Bundle the full eigendecomposition: Q's columns are (b, v1, v2) with
    // eigenvalues (SIGMA_0, sigma[0], sigma[1]). Flip v2 if needed so (b, v1, v2) is
    // right-handed (det Q = +1).
    var v1 = last_gap.v1;
    var v2 = last_gap.v2;
    if (v1.cross(v2).dot(b) < 0) v2 = v2.scale(-1.0);
    info.Q = Mat3.fromCols(b, v1, v2);
    info.sigma = .{ SIGMA_0, last_gap.sigma[0], last_gap.sigma[1] };
    info.status = if (converged) .converged else .did_not_converge;
    return info;
}

