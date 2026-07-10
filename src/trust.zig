//! The trust solver path (`SolveOptions.method = .trust`, the default).
//!
//! Trust-region descent on the *reduced* convex objective
//!
//!   h(b) = min_A { −log det A : ‖A·xᵢ‖₂ ≤ bᵀxᵢ }
//!
//! over the unit sphere. Partial minimization of the jointly convex
//! primal (paper eq. primal) makes h convex on the unit ball and
//! radially non-increasing, so its sphere minimum is the joint optimum
//! and every sphere point has a non-increasing normalized-segment path
//! to it — no spurious strict local minima for a descent method to
//! fall into.
//!
//! The inner minimization at fixed b is EXACTLY the 2D lifted MVEE the
//! alternating path already solves in the gnomonic chart: the lifted points
//! qᵢ = [pᵢ; 1] are the coordinates of zᵢ = xᵢ/(bᵀxᵢ) in the [Q̂ | b]
//! basis, so the centered 3D D-optimal design on the zᵢ and the chart
//! MVEE coincide, and D-optimal design values transform predictably
//! under the chart's rescaling. Everything is therefore *reused*, not
//! re-derived:
//!
//!  - oracle:   `initWeights` + `mveeFw` + `newtonPolish` on the
//!              rescaled chart (numerically the well-behaved chart —
//!              this path inherits none of the joint IPM's raw-3D
//!              extreme-κ certification floor);
//!  - value:    h(b) = ½·(log det S + 3·ln 3) + 2·ln s_scale, with
//!              S = Σ wᵢ·qᵢ·qᵢᵀ the design moment in the scaled chart;
//!  - gradient: by the envelope theorem ∇h(b) = −3·Σ wᵢ·zᵢ, whose
//!              tangent component is −3·c where c is the weighted
//!              centroid the alternating path already computes — i.e. the
//!              alternating path IS gradient descent on h, minus the merit
//!              function. This path adds the merit function and a
//!              second-order (envelope-Hessian) model;
//!  - cert:     `recoverAPerp` + `dualityGapConstructed`, identical to
//!              the alternating path; convergence is declared on the same
//!              certified |gap| ≤ gap_tol.
//!
//! Diagnostics: outcomes carry `Diagnostics.trust` (typed per-path —
//! see api.zig): `eager_certified`, `tr_iters` (accepted + rejected
//! trials, each one inner-oracle evaluation), `recert_attempts`, and
//! `polish_failures`. Nothing here overloads the alternating path's
//! counters.

const std = @import("std");

const linalg = @import("linalg.zig");
const Vec2 = linalg.Vec2;
const Vec3 = linalg.Vec3;
const Mat2 = linalg.Mat2;
const Mat3 = linalg.Mat3;
const Mat3x2 = linalg.Mat3x2;

const config = @import("config.zig");
const tc = config.trust;
const algo = config.algo;
const tol = config.tol;

const halfspace = @import("halfspace.zig");
const projectGnomonic = halfspace.projectGnomonic;

const newton = @import("newton.zig");
const NewtonScratch = newton.NewtonScratch;
const newtonPolish = newton.newtonPolish;

const api = @import("api.zig");
const Outcome = api.Outcome;
const SolveOptions = api.SolveOptions;
const SolveError = api.SolveError;

const core = @import("skar.zig");
const Prep = core.Prep;
const GapScratch = core.GapScratch;
const GapResult = core.GapResult;

/// Per-solve working buffers, all on the solve arena. (`pub` so the
/// FD Hessian-validation test can drive `evalH` directly.)
pub const Buffers = struct {
    P_buf: [][2]f64,
    Ps: [][2]f64,
    Ql: []Vec3,
    w: []f64,
    w_bak: []f64,
    cert_active: []usize,
    cert_lambdas: []f64,
    /// Scratch for the model-Hessian computation: active indices and
    /// forward-solved design vectors yᵢ = L⁻¹qᵢ.
    act_idx: []usize,
    Yf: []Vec3,
    /// Per-active envelope cross-derivatives mᵢ for the exact-Hessian
    /// dw/db correction (the 7×7 range-space solve itself lives on the
    /// stack).
    m_buf: [][2]f64,
    newton_scratch: NewtonScratch,
    gap_scratch: GapScratch,

    pub fn init(scratch: std.mem.Allocator, nw: usize) !Buffers {
        return .{
            .P_buf = try scratch.alloc([2]f64, nw),
            .Ps = try scratch.alloc([2]f64, nw),
            .Ql = try scratch.alloc(Vec3, nw),
            .w = try scratch.alloc(f64, nw),
            .w_bak = try scratch.alloc(f64, nw),
            .cert_active = try scratch.alloc(usize, nw),
            .cert_lambdas = try scratch.alloc(f64, nw),
            .act_idx = try scratch.alloc(usize, nw),
            .Yf = try scratch.alloc(Vec3, nw),
            .m_buf = try scratch.alloc([2]f64, nw),
            .newton_scratch = try NewtonScratch.init(scratch, nw),
            .gap_scratch = try GapScratch.init(scratch, nw),
        };
    }
};

/// Certify a structured iterate: recover the budget-tight A_perp from
/// the chart moment matrix and run the shared constructed-dual gap.
/// One recipe for all four certification sites (eager, initial, TR
/// accept, RECERT); reads the chart buffers (`P_buf`, `w`) that the
/// preceding oracle/moments computation left in place.
fn certifyAt(
    M: Mat2,
    Q: Mat3x2,
    b: Vec3,
    Xw: []const Vec3,
    wb: *Buffers,
) SolveError!GapResult {
    const A_perp = try core.recoverAPerp(wb.P_buf, M);
    return core.dualityGapConstructed(wb.w, b, Xw, A_perp, Q, &wb.gap_scratch, wb.cert_active, wb.cert_lambdas);
}

/// One h-oracle evaluation at a trial axis. `evalH` returns null when
/// the trial leaves the barrier domain (a point below the feasibility
/// margin or a rank-deficient design) — the trust region treats that
/// as an unconditionally rejected step.
const Eval = struct {
    h: f64,
    /// Tangent gradient of h at b in the Q basis: g = −3·c.
    g: Vec2,
    /// Model Hessian: the envelope Hessian of h on the sphere at b
    /// (fixed-w term + dw/db correction when EXACT_HESSIAN), in the
    /// same tangent basis. See evalH.
    B: Mat2,
    Q: Mat3x2,
    moments: core.Moments,
    polish_failed: bool,
};

/// Design state of the current weights in the scaled chart:
/// S = Σ wᵢ·qᵢ·qᵢᵀ factored, the design value
/// h = ½(log det S + 3·ln 3) + 2·ln s_scale (D-optimal values shift by
/// 2·ln s_scale under the rescaling), and the chart moments read
/// directly off S — with qᵢ = [pᵢ; 1], S's last column IS Σ wᵢ·pᵢ and
/// its upper-left block IS Σ wᵢ·pᵢpᵢᵀ, so no separate
/// `computeMoments` pass over the points is needed. Null on a
/// rank-deficient design.
const DesignState = struct { L: linalg.Chol3, h: f64, moments: core.Moments };

fn designState(Ql: []const Vec3, w: []const f64, s_scale: f64) ?DesignState {
    var S = Mat3.zero;
    for (Ql, 0..) |q, i| S.addSymRank1(w[i], q);
    const L = S.cholesky() orelse return null;
    return .{
        .L = L,
        .h = 0.5 * (L.logDet() + 3.0 * @log(3.0)) + 2.0 * @log(s_scale),
        .moments = .{
            .center = (Vec2{ .m = .{ S.m[2], S.m[5] } }).scale(s_scale),
            .M = (Mat2{ .m = .{ S.m[0], S.m[1], S.m[1], S.m[4] } }).scale(s_scale * s_scale),
        },
    };
}

/// Evaluate h(b), its tangent gradient, and the chart data needed for
/// certification. Mutates wb.w (warm start) — caller snapshots/restores
/// around rejected trials. `margin` guards the projection (weights are
/// always warm here: the eager phase seeds them via initWeights before
/// the first call). (`pub` for the FD Hessian-validation test.)
pub fn evalH(
    b: Vec3,
    Xw: []const Vec3,
    wb: *Buffers,
    margin: f64,
) ?Eval {
    const Q = b.orthoBasis();
    if (!projectGnomonic(Xw, b, Q, wb.P_buf, margin)) return null;
    const s_scale = core.rescaleP(wb.P_buf, wb.Ps);

    // Inner solve: pairwise FW in bursts with a stall exit on the
    // design value, then ONE Newton polish. The stall exit stops
    // κ-limited inputs (g-noise above any reachable tolerance) from
    // grinding the budget at noise amplitude; the single final polish
    // keeps the returned state inner-(near-)optimal, so the envelope
    // gradient −3·c below is the gradient of the h reported. mveeFw's
    // drop step is ratio-guarded at the source, so a burst can no
    // longer end on a corrupted state; the burst stays LARGE anyway —
    // measured (docs/away-step-fw.md findings): small windows misread
    // mid-drain flat stretches as the noise floor and return
    // under-refined states (New York DNC at burst 8/16/32, with or
    // without the guard, with or without flat-burst patience), and
    // don't even save time (np400 69 → 95 µs at burst 8). An
    // away-step oracle (`mveeFwAway`) was also tried and reverted —
    // hazard-free by construction but measurably slower than pairwise
    // on large near-circular supports (ha_05 56 → 261 µs).
    var spent: u32 = 0;
    var h_prev: f64 = std.math.inf(f64);
    while (spent < tc.INNER_ITERS) : (spent += tc.INNER_BURST) {
        core.mveeFw(wb.Ps, tc.INNER_BURST, tc.INNER_TOL, wb.Ql, wb.w);
        // Cheap progress check on the design value (one 3×3 Cholesky).
        const chk = designState(wb.Ql, wb.w, s_scale) orelse break;
        if (chk.h >= h_prev - tc.INNER_STALL_REL * (1.0 + @abs(chk.h))) break;
        h_prev = chk.h;
    }
    const polished = newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch);

    const ds = designState(wb.Ql, wb.w, s_scale) orelse return null;
    const L = ds.L;

    // Model Hessian, part 1 — the fixed-w Hessian of h̃_w on the
    // sphere, entirely in chart quantities. For frozen weights,
    // h̃_w(b) = ½·log det(3·Σ wᵢzᵢzᵢᵀ) with zᵢ = xᵢ/(bᵀxᵢ) is a global
    // MINORANT of h touching at the current point: the inner FW/polish
    // MAXIMIZES the design value over w, so h(b) = max_w h̃_w(b) ≥
    // h̃_w(b) for the frozen w — the envelope curves upward relative to
    // every member. Its curvature therefore UNDER-estimates the
    // envelope's away from the touch point (the measured ρ ≈ 0.15
    // far-field creep on dense near-circular caps), which the dw/db
    // correction in part 2 restores. Differentiating twice and
    // retracting to the sphere (b(u) = normalize(b + Q̂u)) gives, with
    // gᵢ = qᵢᵀS⁻¹qᵢ and cᵢⱼ = qᵢᵀS⁻¹qⱼ (both chart-scale invariant)
    // and unscaled chart points pᵢ = s_scale·Ps[i]:
    //
    //   B̃ = 3·Σᵢ wᵢgᵢ·pᵢpᵢᵀ − 2·Σᵢⱼ wᵢwⱼcᵢⱼ²·pᵢpⱼᵀ + (Σᵢ wᵢgᵢ)·I₂
    //
    // (the identity term is the spherical retraction correction
    // −(∇h̃ᵀb)·I with ∇h̃ᵀb = −Σwᵢgᵢ, via zᵢᵀb = 1). The gᵢ/cᵢⱼ come
    // from forward-solves against the design Cholesky already in hand.
    // At a circular optimum this reduces to ≈ 3·I — the old BFGS seed
    // B0, now derived rather than fitted. Restricted to the active set
    // (w > ACTIVE_THRESH): inactive points carry no design mass.
    var k: usize = 0;
    for (wb.w, 0..) |wi, i| {
        if (wi > algo.ACTIVE_THRESH) {
            wb.act_idx[k] = i;
            wb.Yf[k] = L.forwardSolve(wb.Ql[i]);
            k += 1;
        }
    }
    var sum_wg: f64 = 0;
    var Bm = Mat2.zero;
    for (0..k) |i| {
        const wi = wb.w[wb.act_idx[i]];
        const gi = wb.Yf[i].dot(wb.Yf[i]);
        sum_wg += wi * gi;
        const p = Vec2{ .m = .{ wb.Ps[wb.act_idx[i]][0] * s_scale, wb.Ps[wb.act_idx[i]][1] * s_scale } };
        Bm.addSymRank1(3.0 * wi * gi, p);
        // Seed the envelope cross-derivative mᵢ = −gᵢpᵢ + Σⱼ wⱼcᵢⱼ²pⱼ
        // (part 2); the Σⱼ term accumulates in the loop below.
        wb.m_buf[i] = .{ -gi * p.m[0], -gi * p.m[1] };
    }
    for (0..k) |i| {
        const wi = wb.w[wb.act_idx[i]];
        const pi0 = wb.Ps[wb.act_idx[i]][0] * s_scale;
        const pi1 = wb.Ps[wb.act_idx[i]][1] * s_scale;
        for (0..k) |j| {
            const cij = wb.Yf[i].dot(wb.Yf[j]);
            const cij2 = cij * cij;
            const wj = wb.w[wb.act_idx[j]];
            const coef = -2.0 * wi * wj * cij2;
            const pj0 = wb.Ps[wb.act_idx[j]][0] * s_scale;
            const pj1 = wb.Ps[wb.act_idx[j]][1] * s_scale;
            Bm.m[0] += coef * pi0 * pj0;
            Bm.m[1] += coef * pi0 * pj1;
            Bm.m[2] += coef * pi1 * pj0;
            Bm.m[3] += coef * pi1 * pj1;
            wb.m_buf[i][0] += wj * cij2 * pj0;
            wb.m_buf[i][1] += wj * cij2 * pj1;
        }
    }
    Bm.m[0] += sum_wg;
    Bm.m[3] += sum_wg;

    // Model Hessian, part 2 — the exact envelope correction (see
    // config.trust.EXACT_HESSIAN). Danskin at second order: with
    // φ(w, u) the design value, ∂φ/∂wᵢ = gᵢ/2, ∂²φ/∂wᵢ∂wⱼ = −½cᵢⱼ²,
    // and ∂²φ/∂wᵢ∂u = mᵢ = −gᵢpᵢ + Σⱼ wⱼcᵢⱼ²pⱼ. Differentiating the
    // inner KKT conditions along the active face (Σ dw = 0) gives the
    // weight response dw/du from
    //
    //   ½·(C∘C)·dw − ν·1 = m,   1ᵀ·dw = 0,
    //
    // and ∇²h = B̃ + Mᵀ·(dw/du) — a PSD add-on (C∘C is PSD by the
    // Schur product theorem; the envelope of maxima curves at least as
    // much as any member).
    //
    // C∘C has rank ≤ 6 exactly: (C∘C)ᵢⱼ = (yᵢᵀyⱼ)² = ⟨vᵢ, vⱼ⟩ with
    // vᵢ ∈ R⁶ the symmetric outer product of yᵢ (√2 on off-diagonal
    // slots). So for degenerate supports (k > 6, dense near-circular
    // caps) the system is singular, and a Tikhonov-regularized k×k
    // solve AMPLIFIES the null-space component of m (active-set and
    // oracle-stall noise) by 1/ε — measured blowup to ~1e5 on a
    // 60-point ring. The pseudo-inverse (null components projected
    // out — flat directions of the optimal face carry no curvature)
    // is computed exactly in the 6-dim range space: substituting
    // dw = Vᵀα and projecting by V gives the bordered 7×7 system
    //
    //   [ ½·G²   −V·1 ] [α ]   [V·m]
    //   [ (V·1)ᵀ   0  ] [ν ] = [ 0 ],    G = V·Vᵀ (6×6),
    //
    // whose RHS lies in range(G) by construction, and
    // corr = (V·m_a)ᵀ·α_b per axis pair. O(k·36) build + 7³ solve —
    // cheaper than the k×k LU it replaces. A pivot failure just keeps
    // the fixed-w model; the ρ test guards model quality either way.
    if (tc.EXACT_HESSIAN and k >= 2) {
        var G = [_]f64{0} ** 36; // V·Vᵀ, row-major 6×6
        var v1 = [_]f64{0} ** 6; // V·1
        var vmx = [_]f64{0} ** 6; // V·m_x
        var vmy = [_]f64{0} ** 6; // V·m_y
        for (0..k) |i| {
            const v = wb.Yf[i].svec();
            for (0..6) |a| {
                v1[a] += v[a];
                vmx[a] += v[a] * wb.m_buf[i][0];
                vmy[a] += v[a] * wb.m_buf[i][1];
                for (0..6) |bb| G[a * 6 + bb] += v[a] * v[bb];
            }
        }
        // A = [½G², −v1; v1ᵀ, 0] with relative Tikhonov mass on the
        // ½G² block (benign here: the RHS has no null component to
        // amplify).
        var A = [_]f64{0} ** 49;
        var tr_g2: f64 = 0;
        for (0..6) |a| {
            for (0..6) |bb| {
                var s: f64 = 0;
                for (0..6) |cidx| s += G[a * 6 + cidx] * G[cidx * 6 + bb];
                A[a * 7 + bb] = 0.5 * s;
            }
            tr_g2 += A[a * 7 + a];
        }
        const reg = tc.HESS_REG * (1.0 + tr_g2 / 6.0);
        for (0..6) |a| {
            A[a * 7 + a] += reg;
            A[a * 7 + 6] = -v1[a];
            A[6 * 7 + a] = v1[a];
        }
        A[6 * 7 + 6] = 0.0;

        var piv: [7]usize = undefined;
        if (linalg.LU.factorize(&A, 7, &piv, tc.HESS_PIVOT_MIN)) |lu| {
            var ax: [7]f64 = .{ vmx[0], vmx[1], vmx[2], vmx[3], vmx[4], vmx[5], 0 };
            var ay: [7]f64 = .{ vmy[0], vmy[1], vmy[2], vmy[3], vmy[4], vmy[5], 0 };
            lu.solve(&ax);
            lu.solve(&ay);
            var corr00: f64 = 0;
            var corr01: f64 = 0;
            var corr10: f64 = 0;
            var corr11: f64 = 0;
            for (0..6) |a| {
                corr00 += vmx[a] * ax[a];
                corr01 += vmx[a] * ay[a];
                corr10 += vmy[a] * ax[a];
                corr11 += vmy[a] * ay[a];
            }
            const corr01s = 0.5 * (corr01 + corr10);
            Bm.m[0] += corr00;
            Bm.m[1] += corr01s;
            Bm.m[2] += corr01s;
            Bm.m[3] += corr11;
        }
    }

    return .{
        .h = ds.h,
        .g = ds.moments.center.scale(-3.0),
        .B = Bm,
        .Q = Q,
        .moments = ds.moments,
        .polish_failed = !polished,
    };
}

/// Exact 2D trust-region step (dogleg): Newton point if inside the
/// radius, else the dogleg path's boundary intersection, else the
/// scaled steepest-descent point. B must be PD (caller resets it
/// otherwise). Returns the step u and the predicted decrease.
const TrStep = struct { u: Vec2, pred: f64 };

fn doglegStep(B: Mat2, g: Vec2, delta: f64) TrStep {
    const model = struct {
        fn pred(B_: Mat2, g_: Vec2, u: Vec2) f64 {
            return -(g_.dot(u) + 0.5 * u.dot(B_.apply(u)));
        }
    };

    const pn = B.inverse().apply(g).scale(-1.0); // Newton point
    if (pn.norm() <= delta) return .{ .u = pn, .pred = model.pred(B, g, pn) };

    const gBg = g.dot(B.apply(g));
    const pu = g.scale(-g.dot(g) / gBg); // Cauchy point
    const pu_norm = pu.norm();
    if (pu_norm >= delta) {
        const u = g.scale(-delta / g.norm());
        return .{ .u = u, .pred = model.pred(B, g, u) };
    }

    // Dogleg segment pu → pn: find τ with ‖pu + τ·(pn − pu)‖ = delta.
    const d = pn.sub(pu);
    const a = d.dot(d);
    const bq = 2.0 * pu.dot(d);
    const cq = pu.dot(pu) - delta * delta;
    const disc = @max(bq * bq - 4.0 * a * cq, 0);
    const tau = (-bq + @sqrt(disc)) / (2.0 * a);
    const u = Vec2.lincomb(1.0, pu, tau, d);
    return .{ .u = u, .pred = model.pred(B, g, u) };
}

/// Solve the preprocessed problem by trust-region descent on h(b) over
/// the sphere. Same contract as `solveAlternating`.
pub fn solveTrust(
    allocator: std.mem.Allocator,
    scratch_alloc: std.mem.Allocator,
    prep: Prep,
    opts: SolveOptions,
) !Outcome {
    const Xw = prep.Xw;
    var wb = try Buffers.init(scratch_alloc, Xw.len);

    var b = prep.b0;
    var tr_iters: u32 = 0;
    var recert_attempts: u32 = 0;
    var polish_failures: u32 = 0;
    var converged = false;
    var eager_certified = false;

    var last_gap: GapResult = undefined;
    // Axis at which last_gap was computed (see buildOutcome's contract):
    // TR-loop certification is gated on pred, and the RECERT loop can be
    // budget-skipped, so on DNC the final b may be several accepted
    // steps past the last certificate.
    var b_cert = b;

    // Eager first certificate — the alternating path's exact opening cadence
    // (two FW steps, one polish, certify) BEFORE any full-precision
    // oracle work. On the DGGS hot path the certificate passes right
    // here and the solve ends having done essentially what the fast
    // path's first outer iteration would have done; the full oracle
    // (stall-quality FW refinement, gradient, envelope Hessian, trust
    // region) runs only when this certificate fails. Safe w.r.t. the
    // oracle-consistency lesson: this certificate is a pure
    // upper-bound check — it never feeds the trust-region model, and
    // the (h, g, B̃) triple is only ever constructed from the
    // fully-refined state on the path that continues.
    var open_iters: u32 = 0;
    {
        var Q = b.orthoBasis();
        // b0 comes from halfspaceCheck (strictly feasible), so the
        // projection cannot fail.
        _ = projectGnomonic(Xw, b, Q, wb.P_buf, -std.math.inf(f64));
        var s_scale = core.rescaleP(wb.P_buf, wb.Ps);
        core.initWeights(wb.Ps, wb.w);
        core.mveeFw(wb.Ps, algo.FW_PER_NEWTON, 0.0, wb.Ql, wb.w);
        if (!newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch)) {
            polish_failures += 1;
        }
        var m = core.computeMoments(wb.Ps, wb.w, s_scale);
        last_gap = try certifyAt(m.M, Q, b, Xw, &wb);
        b_cert = b;
        if (try core.gapConverged(last_gap.gap, opts.gap_tol)) {
            converged = true;
            eager_certified = true;
        }

        // Opening rounds: continue the alternating path's outer-loop cadence
        // (see solveAlternating in skar.zig — axis step from the current
        // moments, cheap FW cycle, polish + certificate every
        // FW_PER_NEWTON-th cycle) for up to OPEN_ROUNDS certified
        // rounds. Mid-size DGGS cells converge right here at
        // alternating-path cost; hard inputs fall through to the trust
        // region having spent a bounded, cheap prefix. See
        // config.trust.OPEN_ROUNDS.
        var damp = core.DampState{};
        const max_rounds = @min(tc.OPEN_ROUNDS, opts.max_outer);
        var cycle: u32 = 0;
        while (!converged and open_iters < max_rounds) : (cycle += 1) {
            const axis = core.quasiNewtonAxisDirection(cycle / algo.FW_PER_NEWTON, m.M, m.center);
            damp.tick(axis.c_norm);
            const st = core.acceptBUpdate(Xw, b, Q, axis.u, damp.alpha, wb.P_buf, wb.Ps);
            b = st.b;
            Q = st.Q;
            s_scale = st.s_scale;

            core.mveeFw(wb.Ps, 1, 0.0, wb.Ql, wb.w);
            const is_full = (cycle % algo.FW_PER_NEWTON == algo.FW_PER_NEWTON - 1);
            if (is_full) {
                if (!newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch)) {
                    polish_failures += 1;
                }
            }
            m = core.computeMoments(wb.Ps, wb.w, s_scale);
            if (is_full) {
                open_iters += 1;
                last_gap = try certifyAt(m.M, Q, b, Xw, &wb);
                b_cert = b;
                if (try core.gapConverged(last_gap.gap, opts.gap_tol)) converged = true;
            }
        }
    }

    // Full-precision evaluation at the initial axis (warm-started from
    // the eager phase's weights), certified again at oracle quality.
    // Reuses the alternating path's certification wholesale.
    var cur: Eval = undefined;
    if (!converged) {
        // The projection cannot fail: b is the halfspace axis or an
        // opening-round axis accepted at FEAS_MARGIN. A rank-deficient
        // design here means the input slipped past the coplanarity
        // gate — surface it as the same error the alternating path's
        // recoverAPerp would raise.
        cur = evalH(b, Xw, &wb, -std.math.inf(f64)) orelse return SolveError.SingularMoment;
        if (cur.polish_failed) polish_failures += 1;

        last_gap = try certifyAt(cur.moments.M, cur.Q, b, Xw, &wb);
        b_cert = b;
        converged = try core.gapConverged(last_gap.gap, opts.gap_tol);
    }

    // Trust-region state. The model Hessian is per-evaluation (the
    // envelope Hessian computed by evalH) — no quasi-Newton history,
    // no transport between tangent bases.
    var delta: f64 = tc.DELTA0;

    while (!converged and open_iters + tr_iters < opts.max_outer) {
        tr_iters += 1;

        // The model Hessian is PSD-in-exact-arithmetic near inner
        // optimality but can go indefinite from roundoff or far-field
        // states; fall back to the derived isotropic value (≈ its own
        // circular-optimum limit) so the dogleg's prediction is
        // positive whenever g ≠ 0.
        var B = cur.B;
        if (!(B.det() > 0) or !(B.m[0] > 0)) B = .{ .m = .{ tc.B0, 0, 0, tc.B0 } }; // negated form: NaN falls back too

        var step = doglegStep(B, cur.g, delta);
        if (step.pred <= 0) {
            B = .{ .m = .{ tc.B0, 0, 0, tc.B0 } };
            step = doglegStep(B, cur.g, delta);
        }
        if (step.pred <= 0 or !(step.u.norm() > 0)) break; // stationary: g ≈ 0
        // Below merit resolution the ratio test can never verify a
        // step — hand off to the re-cert phase instead of rejecting
        // the same unresolvable step until Δ hits its floor. See
        // config.trust.PRED_NOISE_REL.
        if (step.pred <= tc.PRED_NOISE_REL * (1.0 + @abs(cur.h))) break;

        const b_trial = Vec3.lincomb(1.0, b, 1.0, cur.Q.apply(step.u)).normalize();

        @memcpy(wb.w_bak, wb.w);
        const trial = evalH(b_trial, Xw, &wb, algo.FEAS_MARGIN);

        const rho: f64 = if (trial) |t| (cur.h - t.h) / step.pred else -1.0;
        // !(rho >= ETA), not rho < ETA: a NaN ratio (h from a poisoned
        // state) must REJECT the trial, not accept it.
        if (!(rho >= tc.ETA)) {
            // Reject: restore the warm-start weights, shrink the radius
            // (relative to the step actually attempted, so interior
            // Newton steps shrink meaningfully too).
            @memcpy(wb.w, wb.w_bak);
            delta = @min(delta, step.u.norm()) * tc.SHRINK;
            if (delta < tc.DELTA_MIN) break;
            continue;
        }

        // Accept. The trial evaluation carries its own model Hessian in
        // its own tangent basis — nothing to transport.
        if (trial.?.polish_failed) polish_failures += 1;
        b = b_trial;
        cur = trial.?;

        // Certify the accepted iterate (alternating-path machinery) — but only
        // once the accepted step's predicted decrease is within a
        // couple of orders of gap_tol; while the model still predicts
        // ≫ gap_tol of remaining descent no certificate can pass. See
        // config.trust.CERT_PRED_FACTOR.
        if (step.pred <= tc.CERT_PRED_FACTOR * opts.gap_tol) {
            last_gap = try certifyAt(cur.moments.M, cur.Q, b, Xw, &wb);
            b_cert = b;
            if (try core.gapConverged(last_gap.gap, opts.gap_tol)) {
                converged = true;
                break;
            }
        }

        if (rho < tc.RHO_POOR) {
            // Accepted, but the quadratic model over-promised (higher
            // order terms dominate over this radius) — shrink gently so
            // the model regains fidelity instead of creeping at
            // ρ ≈ 0.15 or oscillating across the fidelity boundary.
            delta = @min(delta, step.u.norm()) * tc.SHRINK_POOR;
            if (delta < tc.DELTA_MIN) break;
        } else if (rho >= tc.ETA_GOOD and step.u.norm() >= 0.8 * delta) {
            delta = @min(delta * tc.GROW, tc.DELTA_MAX);
        }
    }

    // Re-certification phase: the trust region found h stationary but
    // the certificate hasn't reached tol (or failed outright). Near the
    // f64 floor the constructed cert is sensitive to the incidental
    // numerical state at noise amplitude, and everything the TR loop
    // can do at a bit-frozen axis is idempotent: the h-guarded oracle
    // restores weights on no-improvement, a raw FW step is a no-op once
    // g_max < 3 numerically, and polish is at its fixed point — so
    // retrying at fixed b certifies the identical state forever. The
    // alternating path escapes this because its axis moves a little every
    // outer iteration, re-projecting the points and re-sampling the
    // whole numerical state (measured on A5 res-30: fast's first cert
    // fails the same M-Cholesky; its second passes). So this phase IS
    // a few alternating-path outer iterations warm-started at the TR optimum:
    // FW step → polish → certify → damped axis micro-step. TR for the
    // global descent, fast iteration for the terminal certification.
    if (!converged) {
        var Q = cur.Q;
        // The last trial may have been rejected, leaving the projection
        // buffers at the rejected axis; re-project at the accepted b.
        _ = projectGnomonic(Xw, b, Q, wb.P_buf, -std.math.inf(f64));
        var s_scale = core.rescaleP(wb.P_buf, wb.Ps);
        while (recert_attempts < tc.RECERT_MAX and open_iters + tr_iters + recert_attempts < opts.max_outer) {
            recert_attempts += 1;
            core.mveeFw(wb.Ps, 1, 0.0, wb.Ql, wb.w);
            if (!newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch)) {
                polish_failures += 1;
            }
            const m = core.computeMoments(wb.Ps, wb.w, s_scale);
            last_gap = try certifyAt(m.M, Q, b, Xw, &wb);
            b_cert = b;
            if (try core.gapConverged(last_gap.gap, opts.gap_tol)) {
                converged = true;
                break;
            }
            // Axis micro-step along the h-gradient (plain, undamped —
            // ‖center‖ is at noise scale here). This is the numerical
            // re-sample the alternating path gets for free each iteration.
            const bstep = core.acceptBUpdate(Xw, b, Q, m.center, 1.0, wb.P_buf, wb.Ps);
            b = bstep.b;
            Q = bstep.Q;
            s_scale = bstep.s_scale;
        }
    }

    return core.buildOutcome(
        allocator,
        converged,
        b_cert,
        last_gap,
        .{ .trust = .{
            .eager_certified = eager_certified,
            .open_iters = open_iters,
            .tr_iters = tr_iters,
            .recert_attempts = recert_attempts,
            .polish_failures = polish_failures,
        } },
        wb.cert_active,
        wb.cert_lambdas,
        prep.work_to_orig,
    );
}
