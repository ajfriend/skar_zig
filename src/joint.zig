//! EXPERIMENTAL joint convex solver path (`SolveOptions.method = .joint`).
//!
//! Barrier-Newton interior-point method on the primal SDP, solved
//! *jointly* in (A, b) — no axis/shape alternation:
//!
//!   minimize   −log det A
//!   subject to ‖A·xᵢ‖₂ ≤ bᵀxᵢ,   i = 1..n
//!              ‖b‖₂ ≤ 1
//!
//! over A ∈ S³₊₊ (6 parameters) and b ∈ R³ — 9 unknowns total. The
//! problem is jointly convex (paper, eq. primal), so log-barrier
//! path-following with damped Newton is globally convergent for every
//! feasible input — including the dense wide-angle inputs where the
//! fast path's axis fixed-point iteration limit-cycles (see
//! docs/wide-cap-dnc-report.md).
//!
//! Central-path objective at barrier parameter t:
//!
//!   F_t(A, b) = t·(−log det A) + Σᵢ −log(sᵢ² − ‖uᵢ‖²) − log(1 − bᵀb)
//!
//! with sᵢ = bᵀxᵢ, uᵢ = A·xᵢ, rᵢ = sᵢ² − ‖uᵢ‖² > 0. −log det A is both
//! the objective and the PSD barrier; −log(s²−‖u‖²) is the standard
//! θ=2 self-concordant barrier for the second-order cone.
//!
//! Certification: after each centering stage the iterate is projected
//! to the structured form the rest of the library speaks — b̂ = b/‖b‖,
//! A_perp = Q̂ᵀAQ̂ rescaled so max_i ‖A_perp·pᵢ‖² = 2/3 over the gnomonic
//! projections pᵢ (exact primal feasibility, mirroring `recoverAPerp`'s
//! budget), weights wᵢ = λᵢ·sᵢ/3 from the barrier multipliers
//! λᵢ = 2sᵢ/(t·rᵢ) — and handed to the *existing*
//! `skar.dualityGapConstructed`, which builds the dual-feasible
//! certificate (paper §cs) and the certified gap. Convergence is
//! declared on |gap| ≤ gap_tol, same as the fast path; the barrier
//! parameter schedule is only a driver.
//!
//! Iteration counts: `outer_iters` on the returned outcome is the
//! total number of Newton steps (not directly comparable to the fast
//! path's outer-loop count).

const std = @import("std");

const linalg = @import("linalg.zig");
const Vec3 = linalg.Vec3;
const Mat2 = linalg.Mat2;
const Mat3 = linalg.Mat3;
const Mat3x2 = linalg.Mat3x2;
const LU = linalg.LU;

const config = @import("config.zig");
const jc = config.joint;
const tol = config.tol;
const algo = config.algo;

const halfspace = @import("halfspace.zig");
const projectGnomonic = halfspace.projectGnomonic;

const api = @import("api.zig");
const Outcome = api.Outcome;
const SolveOptions = api.SolveOptions;
const SolveError = api.SolveError;

const core = @import("skar.zig");
const Prep = core.Prep;
const GapScratch = core.GapScratch;
const GapResult = core.GapResult;

// ----------------------------------------------------------------
// Symmetric-A parametrization
// ----------------------------------------------------------------

/// Variable layout: z[0..6] = A params, z[6..9] = b.
/// A params index (row, col) pairs of the symmetric matrix; off-diagonal
/// params move BOTH mirror entries (basis matrix E_p = eᵢeⱼᵀ + eⱼeᵢᵀ).
const PAIRS = [6][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } };

inline fn matFromParams(a: [6]f64) Mat3 {
    return .{ .m = .{
        a[0], a[3], a[4],
        a[3], a[1], a[5],
        a[4], a[5], a[2],
    } };
}

inline fn matAt(M: Mat3, r: usize, c: usize) f64 {
    return M.m[r * 3 + c];
}

/// v_p = E_p·x for A-param p.
inline fn basisApply(p: usize, x: Vec3) Vec3 {
    const i = PAIRS[p][0];
    const j = PAIRS[p][1];
    var v = Vec3.zero;
    if (i == j) {
        v.m[i] = x.m[i];
    } else {
        v.m[i] = x.m[j];
        v.m[j] = x.m[i];
    }
    return v;
}

/// tr(S·E_p·S·E_q) for symmetric S — the (p,q) entry of the −log det
/// Hessian in the E-basis. Expands E_p = Σ eₐe_bᵀ over its 1 or 2 index
/// tuples; tr(S·eₐe_bᵀ·S·e_ce_dᵀ) = S[b][c]·S[d][a].
fn trSEpSEq(S: Mat3, p: usize, q: usize) f64 {
    var tuples_p: [2][2]usize = undefined;
    var np: usize = 1;
    tuples_p[0] = PAIRS[p];
    if (PAIRS[p][0] != PAIRS[p][1]) {
        tuples_p[1] = .{ PAIRS[p][1], PAIRS[p][0] };
        np = 2;
    }
    var tuples_q: [2][2]usize = undefined;
    var nq: usize = 1;
    tuples_q[0] = PAIRS[q];
    if (PAIRS[q][0] != PAIRS[q][1]) {
        tuples_q[1] = .{ PAIRS[q][1], PAIRS[q][0] };
        nq = 2;
    }
    var total: f64 = 0;
    for (tuples_p[0..np]) |ab| {
        for (tuples_q[0..nq]) |cd| {
            total += matAt(S, ab[1], cd[0]) * matAt(S, cd[1], ab[0]);
        }
    }
    return total;
}

// ----------------------------------------------------------------
// Barrier evaluation and Newton-system assembly
// ----------------------------------------------------------------

/// F_t at (A, b), or null if (A, b) is outside the barrier domain.
fn evalF(A: Mat3, b: Vec3, Xw: []const Vec3, t: f64) ?f64 {
    const L = A.cholesky() orelse return null;
    const logdet_a = 2.0 * (@log(L.m[0]) + @log(L.m[4]) + @log(L.m[8]));
    const q = b.dot(b);
    if (q >= 1.0) return null;
    var f = -t * logdet_a - @log(1.0 - q);
    for (Xw) |x| {
        const s = b.dot(x);
        if (s <= 0) return null;
        const u = A.apply(x);
        const r = s * s - u.dot(u);
        if (r <= 0) return null;
        f -= @log(r);
    }
    return f;
}

/// Assemble ∇F_t (9) and ∇²F_t (9×9 row-major) at (A, b). Returns false
/// if A is not SPD (caller treats as stage failure — the line search
/// domain check should make this unreachable from a feasible iterate).
fn assembleNewton(
    A: Mat3,
    b: Vec3,
    Xw: []const Vec3,
    t: f64,
    grad: *[9]f64,
    H: *[81]f64,
) bool {
    const L = A.cholesky() orelse return false;

    // S = A⁻¹, built column-by-column from the Cholesky factor.
    var S = Mat3.zero;
    for (0..3) |c| {
        var e = Vec3.zero;
        e.m[c] = 1.0;
        const col = L.solve(e);
        S.m[0 * 3 + c] = col.m[0];
        S.m[1 * 3 + c] = col.m[1];
        S.m[2 * 3 + c] = col.m[2];
    }

    @memset(grad, 0);
    @memset(H, 0);

    // Objective t·(−log det A): gradient −t·S in the E-basis
    // (off-diagonal params see both mirror entries → factor 2), Hessian
    // t·tr(S·E_p·S·E_q).
    for (0..6) |p| {
        const i = PAIRS[p][0];
        const j = PAIRS[p][1];
        grad[p] -= t * (if (i == j) matAt(S, i, i) else 2.0 * matAt(S, i, j));
        for (p..6) |q| {
            const h = t * trSEpSEq(S, p, q);
            H[p * 9 + q] += h;
            if (q != p) H[q * 9 + p] += h;
        }
    }

    // SOC barriers: φᵢ = −log(rᵢ). With ∇r = (−2·g_p ; 2s·x):
    //   ∇φ = −∇r/r,
    //   ∇²φ = (∇r·∇rᵀ)/r² + (2/r)·diag-block(v_pᵀv_q, −x·xᵀ).
    for (Xw) |x| {
        const s = b.dot(x);
        const u = A.apply(x);
        const r = s * s - u.dot(u);
        if (r <= 0) return false;
        const inv_r = 1.0 / r;

        var v: [6]Vec3 = undefined;
        var dr: [9]f64 = undefined;
        for (0..6) |p| {
            v[p] = basisApply(p, x);
            dr[p] = -2.0 * u.dot(v[p]);
        }
        for (0..3) |m| dr[6 + m] = 2.0 * s * x.m[m];

        for (0..9) |k| grad[k] -= dr[k] * inv_r;

        // Rank-1 term (∇r·∇rᵀ)/r².
        for (0..9) |k| {
            for (k..9) |l| {
                const h = dr[k] * dr[l] * inv_r * inv_r;
                H[k * 9 + l] += h;
                if (l != k) H[l * 9 + k] += h;
            }
        }
        // Curvature of r itself: A-block +(2/r)·v_pᵀv_q, b-block −(2/r)·x·xᵀ.
        for (0..6) |p| {
            for (p..6) |q| {
                const h = 2.0 * inv_r * v[p].dot(v[q]);
                H[p * 9 + q] += h;
                if (q != p) H[q * 9 + p] += h;
            }
        }
        for (0..3) |m| {
            for (m..3) |m2| {
                const h = -2.0 * inv_r * x.m[m] * x.m[m2];
                H[(6 + m) * 9 + (6 + m2)] += h;
                if (m2 != m) H[(6 + m2) * 9 + (6 + m)] += h;
            }
        }
    }

    // Ball barrier −log(1 − bᵀb): ∇ = 2b/d, ∇² = (2/d)·I + (4/d²)·b·bᵀ,
    // d = 1 − bᵀb.
    const d = 1.0 - b.dot(b);
    for (0..3) |m| {
        grad[6 + m] += 2.0 * b.m[m] / d;
        for (m..3) |m2| {
            var h = 4.0 * b.m[m] * b.m[m2] / (d * d);
            if (m2 == m) h += 2.0 / d;
            H[(6 + m) * 9 + (6 + m2)] += h;
            if (m2 != m) H[(6 + m2) * 9 + (6 + m)] += h;
        }
    }

    return true;
}

// ----------------------------------------------------------------
// Certificate extraction
// ----------------------------------------------------------------

/// Scratch buffers for the per-stage certificate extraction, all on
/// the solve arena.
const CertScratch = struct {
    P_buf: [][2]f64,
    w: []f64,
    cert_active: []usize,
    cert_lambdas: []f64,
    best_active: []usize,
    best_lambdas: []f64,
    gap_scratch: GapScratch,

    fn init(scratch: std.mem.Allocator, nw: usize) !CertScratch {
        return .{
            .P_buf = try scratch.alloc([2]f64, nw),
            .w = try scratch.alloc(f64, nw),
            .cert_active = try scratch.alloc(usize, nw),
            .cert_lambdas = try scratch.alloc(f64, nw),
            .best_active = try scratch.alloc(usize, nw),
            .best_lambdas = try scratch.alloc(f64, nw),
            .gap_scratch = try GapScratch.init(scratch, nw),
        };
    }
};

const CertAttempt = struct {
    gap_result: GapResult,
    b_hat: Vec3,
};

/// Project the interior-point iterate (A, b, t) onto the structured
/// primal the library certifies — b̂ unit, A_perp budget-tight — and run
/// the shared constructed-dual gap. See the module doc-comment.
fn extractCert(
    A: Mat3,
    b: Vec3,
    t: f64,
    Xw: []const Vec3,
    cs: *CertScratch,
) SolveError!CertAttempt {
    const b_hat = b.normalize();
    const Q = b_hat.orthoBasis();

    // A_perp = Q̂ᵀ·A·Q̂ (2×2 symmetric).
    const Ae1 = A.apply(Q.e1);
    const Ae2 = A.apply(Q.e2);
    var A_perp = Mat2{ .m = .{
        Q.e1.dot(Ae1), Q.e1.dot(Ae2),
        Q.e2.dot(Ae1), Q.e2.dot(Ae2),
    } };

    // Gnomonic projections at b̂ (b̂ strictly feasible ⇒ no margin check).
    _ = projectGnomonic(Xw, b_hat, Q, cs.P_buf, -std.math.inf(f64));

    // Budget rescale for exact primal feasibility of the structured cone:
    // ‖A_struct·x‖ ≤ b̂ᵀx  ⟺  ‖A_perp·p‖² ≤ 2/3 in the gnomonic chart, so
    // scale A_perp to make the max exactly 2/3 (cf. recoverAPerp).
    var g_max: f64 = 0;
    for (cs.P_buf) |p_arr| {
        const px = p_arr[0];
        const py = p_arr[1];
        const c0 = A_perp.m[0] * px + A_perp.m[1] * py;
        const c1 = A_perp.m[2] * px + A_perp.m[3] * py;
        const g = c0 * c0 + c1 * c1;
        if (g > g_max) g_max = g;
    }
    if (g_max < tol.TINY) {
        // Degenerate A_perp (shouldn't happen from an interior iterate);
        // report an uncertified attempt.
        return .{ .gap_result = .{ .gap = 1e30, .cert_n = 0, .v1 = Vec3.zero, .v2 = Vec3.zero, .sigma = .{ 0, 0 } }, .b_hat = b_hat };
    }
    A_perp = A_perp.scale(@sqrt((2.0 / 3.0) / g_max));

    // Weights from the barrier multipliers λᵢ = 2sᵢ/(t·rᵢ), converted to
    // the library convention wᵢ = λᵢ·sᵢ/3 (dualityGapConstructed
    // reconstructs λᵢ = 3wᵢ/(b̂ᵀxᵢ)).
    for (Xw, 0..) |x, i| {
        const s = b.dot(x);
        const u = A.apply(x);
        const r = s * s - u.dot(u);
        cs.w[i] = if (r > 0) 2.0 * s * s / (3.0 * t * r) else 0;
    }

    const gr = try core.dualityGapConstructed(
        cs.w,
        b_hat,
        Xw,
        A_perp,
        Q,
        &cs.gap_scratch,
        cs.cert_active,
        cs.cert_lambdas,
    );
    return .{ .gap_result = gr, .b_hat = b_hat };
}

// ----------------------------------------------------------------
// Solver entry point
// ----------------------------------------------------------------

/// Solve the preprocessed problem with the joint barrier method.
/// Mirrors `solveFast`'s contract: returns Converged or DidNotConverge
/// (never Infeasible — feasibility was established in preprocessing);
/// certs live on `allocator`, everything else on `scratch_alloc`.
pub fn solveJoint(
    allocator: std.mem.Allocator,
    scratch_alloc: std.mem.Allocator,
    prep: Prep,
    opts: SolveOptions,
) !Outcome {
    const Xw = prep.Xw;
    const nw = Xw.len;

    var cs = try CertScratch.init(scratch_alloc, nw);

    // Strictly interior start (see config.joint.B0_SHRINK/A0_SHRINK).
    var b = prep.b0.scale(jc.B0_SHRINK);
    var s_min: f64 = 1e30;
    for (Xw) |x| s_min = @min(s_min, b.dot(x));
    var a_params: [6]f64 = .{ jc.A0_SHRINK * s_min, jc.A0_SHRINK * s_min, jc.A0_SHRINK * s_min, 0, 0, 0 };
    var A = matFromParams(a_params);

    var t: f64 = jc.T0;
    var newton_total: u32 = 0;
    var converged = false;

    // Best certified attempt so far (returned on DNC as the diagnostic
    // iterate, mirroring the fast path's last-iterate reporting).
    var best_gap: f64 = 1e30;
    var best = GapResult{ .gap = 1e30, .cert_n = 0, .v1 = Vec3.zero, .v2 = Vec3.zero, .sigma = .{ 0, 0 } };
    var best_b = prep.b0;
    var best_n: usize = 0;

    var grad: [9]f64 = undefined;
    var H: [81]f64 = undefined;
    var Hwork: [81]f64 = undefined;
    var rhs: [9]f64 = undefined;
    var piv: [9]usize = undefined;

    const mu = if (jc.probe_mu > 0) jc.probe_mu else jc.MU;
    stages: while (t <= jc.T_MAX and newton_total < jc.MAX_NEWTON_TOTAL) : (t *= mu) {
        // Centering: damped Newton on F_t.
        var it: u32 = 0;
        while (it < jc.MAX_NEWTON_PER_STAGE and newton_total < jc.MAX_NEWTON_TOTAL) : (it += 1) {
            if (!assembleNewton(A, b, Xw, t, &grad, &H)) break;

            @memcpy(&Hwork, &H);
            for (0..9) |k| rhs[k] = -grad[k];
            const lu = LU.factorize(&Hwork, 9, &piv, tol.UNDERFLOW) orelse break;
            lu.solve(&rhs);
            newton_total += 1;

            // Newton decrement λ² = −∇FᵀΔ; centered when small.
            var dec: f64 = 0;
            for (0..9) |k| dec -= grad[k] * rhs[k];
            if (dec < jc.NEWTON_DEC_TOL) break;

            // Backtracking line search: domain first, then Armijo.
            const f0 = evalF(A, b, Xw, t) orelse break;
            const slope = -dec;
            var alpha: f64 = 1.0;
            var accepted = false;
            var ls: u32 = 0;
            while (ls < jc.MAX_LS) : (ls += 1) {
                var a_try: [6]f64 = undefined;
                for (0..6) |k| a_try[k] = a_params[k] + alpha * rhs[k];
                const A_try = matFromParams(a_try);
                const b_try = Vec3{ .m = .{
                    b.m[0] + alpha * rhs[6],
                    b.m[1] + alpha * rhs[7],
                    b.m[2] + alpha * rhs[8],
                } };
                if (evalF(A_try, b_try, Xw, t)) |f_try| {
                    if (f_try <= f0 + jc.ARMIJO * alpha * slope) {
                        a_params = a_try;
                        A = A_try;
                        b = b_try;
                        accepted = true;
                        break;
                    }
                }
                alpha *= jc.LS_BETA;
            }
            if (!accepted) break;
        }

        // Certify the (approximately) centered iterate.
        const attempt = try extractCert(A, b, t, Xw, &cs);
        const gap = attempt.gap_result.gap;
        if (gap < best_gap) {
            best_gap = gap;
            best = attempt.gap_result;
            best_b = attempt.b_hat;
            best_n = attempt.gap_result.cert_n;
            @memcpy(cs.best_active[0..best_n], cs.cert_active[0..best_n]);
            @memcpy(cs.best_lambdas[0..best_n], cs.cert_lambdas[0..best_n]);
        }
        if (@abs(gap) <= opts.gap_tol) {
            converged = true;
            break :stages;
        }
        if (gap < -tol.NEG_GAP) return SolveError.NegativeDualityGap;
    }

    return core.buildOutcome(
        allocator,
        converged,
        best_b,
        best,
        best_gap,
        newton_total,
        0, // newton_polish_failures: the joint path has no polish step
        cs.best_active,
        cs.best_lambdas,
        best_n,
        prep.work_to_orig,
    );
}
