//! EXPERIMENTAL reduced solver path (`SolveOptions.method = .reduced`).
//!
//! Trust-region BFGS on the *reduced* convex objective
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
//! fast path already solves in the gnomonic chart: the lifted points
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
//!              centroid the fast path already computes — i.e. the
//!              fast path IS gradient descent on h, minus the merit
//!              function. This path adds the merit function and a
//!              second-order (BFGS) model;
//!  - cert:     `recoverAPerp` + `dualityGapConstructed`, identical to
//!              the fast path; convergence is declared on the same
//!              certified |gap| ≤ gap_tol.
//!
//! `outer_iters` on the returned outcome counts trust-region
//! iterations (accepted + rejected trials), each costing one inner
//! oracle evaluation — directly comparable to the fast path's outer
//! count in per-iteration cost.

const std = @import("std");

const linalg = @import("linalg.zig");
const Vec2 = linalg.Vec2;
const Vec3 = linalg.Vec3;
const Mat2 = linalg.Mat2;
const Mat3 = linalg.Mat3;
const Mat3x2 = linalg.Mat3x2;

const config = @import("config.zig");
const rc = config.reduced;
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

/// Per-solve working buffers, all on the solve arena.
const Buffers = struct {
    P_buf: [][2]f64,
    Ps: [][2]f64,
    Ql: []Vec3,
    w: []f64,
    w_bak: []f64,
    w_round: []f64,
    cert_active: []usize,
    cert_lambdas: []f64,
    newton_scratch: NewtonScratch,
    gap_scratch: GapScratch,

    fn init(scratch: std.mem.Allocator, nw: usize) !Buffers {
        return .{
            .P_buf = try scratch.alloc([2]f64, nw),
            .Ps = try scratch.alloc([2]f64, nw),
            .Ql = try scratch.alloc(Vec3, nw),
            .w = try scratch.alloc(f64, nw),
            .w_bak = try scratch.alloc(f64, nw),
            .w_round = try scratch.alloc(f64, nw),
            .cert_active = try scratch.alloc(usize, nw),
            .cert_lambdas = try scratch.alloc(f64, nw),
            .newton_scratch = try NewtonScratch.init(scratch, nw),
            .gap_scratch = try GapScratch.init(scratch, nw),
        };
    }
};

/// One h-oracle evaluation at a trial axis. `ok = false` means the
/// trial left the barrier domain (a point below the feasibility margin
/// or a rank-deficient design) — the trust region treats it as an
/// unconditionally rejected step.
const Eval = struct {
    ok: bool,
    h: f64,
    /// Tangent gradient of h at b in the Q basis: g = −3·c.
    g: Vec2,
    Q: Mat3x2,
    moments: core.Moments,
    polish_failed: bool,
};

const EVAL_FAIL = Eval{ .ok = false, .h = 0, .g = Vec2.zero, .Q = undefined, .moments = undefined, .polish_failed = false };

/// Evaluate h(b), its tangent gradient, and the chart data needed for
/// certification. Mutates wb.w (warm start) — caller snapshots/restores
/// around rejected trials. `margin` guards the projection (−inf for the
/// initial axis, which halfspaceCheck only guarantees strictly feasible).
fn evalH(
    b: Vec3,
    Xw: []const Vec3,
    wb: *Buffers,
    margin: f64,
    first: bool,
) Eval {
    const Q = b.orthoBasis();
    if (!projectGnomonic(Xw, b, Q, wb.P_buf, margin)) return EVAL_FAIL;
    const s_scale = core.rescaleP(wb.P_buf, wb.Ps);

    if (first) core.initWeights(wb.Ps, wb.w);

    // Inner solve: one FW run to tolerance + one Newton polish — the
    // proven-simple oracle. Its states are inner-(near-)optimal, so the
    // envelope gradient −3·c below is the gradient of the h reported.
    // Floor-regime pathologies (certificates that fail at noise level
    // no matter how the inner state is refined) are NOT the oracle's
    // job — they're handled by the re-certification phase in
    // solveReduced, which is where the fast path handles them too.
    core.mveeFw(wb.Ps, rc.INNER_ITERS, rc.INNER_TOL, wb.Ql, wb.w);
    const polished = newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch);

    // Design value in the scaled chart: h = ½(log det S + 3·ln 3) +
    // 2·ln s_scale, S = Σ wᵢ·qᵢ·qᵢᵀ (D-optimal design values shift by
    // 2·ln s_scale under the rescaling).
    var S = Mat3.zero;
    for (wb.Ql, 0..) |q, i| S.addSymRank1(wb.w[i], q);
    const L = S.cholesky() orelse return EVAL_FAIL;
    const logdet_s = 2.0 * (@log(L.m[0]) + @log(L.m[4]) + @log(L.m[8]));
    const h = 0.5 * (logdet_s + 3.0 * @log(3.0)) + 2.0 * @log(s_scale);

    const m = core.computeMoments(wb.Ps, wb.w, s_scale);
    return .{
        .ok = true,
        .h = h,
        .g = m.center.scale(-3.0),
        .Q = Q,
        .moments = m,
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

/// TEMPORARY probe knob: per-iteration trace (revert before merge).
pub var probe_trace: bool = false;

/// Solve the preprocessed problem by trust-region BFGS on h(b) over
/// the sphere. Same contract as `solveFast` / `solveJoint`.
pub fn solveReduced(
    allocator: std.mem.Allocator,
    scratch_alloc: std.mem.Allocator,
    prep: Prep,
    opts: SolveOptions,
) !Outcome {
    const Xw = prep.Xw;
    var wb = try Buffers.init(scratch_alloc, Xw.len);

    var b = prep.b0;
    var outer_count: u32 = 0;
    var polish_failures: u32 = 0;
    var converged = false;

    var cur = evalH(b, Xw, &wb, -std.math.inf(f64), true);
    if (cur.polish_failed) polish_failures += 1;
    // The initial axis comes from halfspaceCheck, so the projection
    // cannot fail; a rank-deficient design here means the input slipped
    // past the coplanarity gate — surface it as the same error the
    // fast path's recoverAPerp would raise.
    if (!cur.ok) return SolveError.SingularMoment;

    // Certified gap at the current (always-accepted) iterate. Reuses
    // the fast path's certification wholesale.
    var last_gap = GapResult{ .gap = 1e30, .cert_n = 0, .v1 = Vec3.zero, .v2 = Vec3.zero, .sigma = .{ 0, 0 } };
    var final_gap: f64 = 1e30;

    const certify = struct {
        fn run(cur_: Eval, b_: Vec3, Xw_: []const Vec3, wb_: *Buffers) SolveError!GapResult {
            const A_perp = try core.recoverAPerp(wb_.P_buf, cur_.moments.M);
            return core.dualityGapConstructed(wb_.w, b_, Xw_, A_perp, cur_.Q, &wb_.gap_scratch, wb_.cert_active, wb_.cert_lambdas);
        }
    };

    last_gap = try certify.run(cur, b, Xw, &wb);
    final_gap = last_gap.gap;
    // Order matters (mirrors the fast path's break-before-guard): a
    // converged-at-noise-level gap can be slightly negative (seen on
    // H3 r15 cells, gap ~ −5e-9 from κ·ε noise) and must be accepted
    // before the hard NegGap guard fires.
    if (@abs(final_gap) <= opts.gap_tol) {
        converged = true;
    } else if (final_gap < -tol.NEG_GAP) {
        return SolveError.NegativeDualityGap;
    }

    // Trust-region state: 2×2 BFGS model in the current tangent basis.
    var B = Mat2{ .m = .{ rc.B0, 0, 0, rc.B0 } };
    var delta: f64 = rc.DELTA0;

    while (!converged and outer_count < opts.max_outer) {
        outer_count += 1;

        // Reset a non-PD model (BFGS skips guarantee PD in exact
        // arithmetic; this is the roundoff backstop).
        if (B.det() <= 0 or B.m[0] <= 0) B = .{ .m = .{ rc.B0, 0, 0, rc.B0 } };

        var step = doglegStep(B, cur.g, delta);
        if (step.pred <= 0) {
            // Non-positive predicted decrease means the BFGS model went
            // bad (near active-set kinks the curvature pairs can leave B
            // so ill-conditioned that the closed-form inverse fails its
            // own model). Reset to the well-scaled initial model and
            // recompute — with B = B0·I the dogleg's prediction is
            // positive whenever g ≠ 0.
            B = .{ .m = .{ rc.B0, 0, 0, rc.B0 } };
            step = doglegStep(B, cur.g, delta);
        }
        if (probe_trace) {
            std.debug.print("  it={d:3} h={e:12.5} gap={e:9.2} |g|={e:9.2} delta={e:9.2} pred={e:9.2}\n", .{ outer_count, cur.h, final_gap, cur.g.norm(), delta, step.pred });
        }
        if (step.pred <= 0 or !(step.u.norm() > 0)) break; // stationary: g ≈ 0

        const b_trial = Vec3.lincomb(1.0, b, 1.0, cur.Q.apply(step.u)).normalize();

        @memcpy(wb.w_bak, wb.w);
        const trial = evalH(b_trial, Xw, &wb, algo.FEAS_MARGIN, false);

        const rho: f64 = if (trial.ok) (cur.h - trial.h) / step.pred else -1.0;
        if (probe_trace) {
            std.debug.print("         trial ok={} h={e:12.5} rho={e:9.2}\n", .{ trial.ok, trial.h, rho });
        }
        if (rho < rc.ETA) {
            // Reject: restore the warm-start weights, shrink the radius.
            @memcpy(wb.w, wb.w_bak);
            delta *= rc.SHRINK;
            if (delta < rc.DELTA_MIN) break;
            continue;
        }

        // Accept. Transport the 2D model/gradient into the trial
        // tangent basis: T = Q_trialᵀ·Q_cur (near-rotation for the
        // small steps BFGS feeds on).
        if (trial.polish_failed) polish_failures += 1;
        const T = Mat2{ .m = .{
            trial.Q.e1.dot(cur.Q.e1), trial.Q.e1.dot(cur.Q.e2),
            trial.Q.e2.dot(cur.Q.e1), trial.Q.e2.dot(cur.Q.e2),
        } };
        const g_old_t = T.apply(cur.g);
        const s_t = T.apply(step.u);
        B = .{ .m = .{
            T.m[0] * (B.m[0] * T.m[0] + B.m[1] * T.m[1]) + T.m[1] * (B.m[2] * T.m[0] + B.m[3] * T.m[1]),
            T.m[0] * (B.m[0] * T.m[2] + B.m[1] * T.m[3]) + T.m[1] * (B.m[2] * T.m[2] + B.m[3] * T.m[3]),
            T.m[2] * (B.m[0] * T.m[0] + B.m[1] * T.m[1]) + T.m[3] * (B.m[2] * T.m[0] + B.m[3] * T.m[1]),
            T.m[2] * (B.m[0] * T.m[2] + B.m[1] * T.m[3]) + T.m[3] * (B.m[2] * T.m[2] + B.m[3] * T.m[3]),
        } };

        b = b_trial;
        cur = trial;

        // Certify the accepted iterate (fast-path machinery).
        last_gap = try certify.run(cur, b, Xw, &wb);
        final_gap = last_gap.gap;
        if (@abs(final_gap) <= opts.gap_tol) {
            converged = true;
            break;
        }
        if (final_gap < -tol.NEG_GAP) return SolveError.NegativeDualityGap;

        // BFGS update with the transported pair (s, y); skip on flat
        // curvature to keep B PD.
        const y = cur.g.sub(g_old_t);
        const curv = y.dot(s_t);
        if (curv > rc.CURV_MIN_REL * y.norm() * s_t.norm()) {
            const Bs = B.apply(s_t);
            const sBs = s_t.dot(Bs);
            if (sBs > 0) {
                B = Mat2.lincomb(1.0, B, -1.0 / sBs, Mat2.outer(Bs, Bs));
                B = Mat2.lincomb(1.0, B, 1.0 / curv, Mat2.outer(y, y));
            }
        }

        if (rho >= rc.ETA_GOOD and step.u.norm() >= 0.8 * delta) {
            delta = @min(delta * rc.GROW, rc.DELTA_MAX);
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
    // fast path escapes this because its axis moves a little every
    // outer iteration, re-projecting the points and re-sampling the
    // whole numerical state (measured on A5 res-30: fast's first cert
    // fails the same M-Cholesky; its second passes). So this phase IS
    // a few fast-path outer iterations warm-started at the TR optimum:
    // FW step → polish → certify → damped axis micro-step. TR for the
    // global descent, fast iteration for the terminal certification.
    if (!converged) {
        var Q = cur.Q;
        // The last trial may have been rejected, leaving the projection
        // buffers at the rejected axis; re-project at the accepted b.
        _ = projectGnomonic(Xw, b, Q, wb.P_buf, -std.math.inf(f64));
        var s_scale = core.rescaleP(wb.P_buf, wb.Ps);
        var attempts: u32 = 0;
        while (attempts < rc.RECERT_MAX and outer_count < opts.max_outer) : (attempts += 1) {
            outer_count += 1;
            core.mveeFw(wb.Ps, 1, 0.0, wb.Ql, wb.w);
            if (!newtonPolish(wb.Ql, wb.w, algo.ACTIVE_THRESH, 20, tol.NEWTON_INNER, &wb.newton_scratch)) {
                polish_failures += 1;
            }
            const m = core.computeMoments(wb.Ps, wb.w, s_scale);
            const A_perp = try core.recoverAPerp(wb.P_buf, m.M);
            last_gap = try core.dualityGapConstructed(wb.w, b, Xw, A_perp, Q, &wb.gap_scratch, wb.cert_active, wb.cert_lambdas);
            final_gap = last_gap.gap;
            if (probe_trace) std.debug.print("  recert attempt={d:2} gap={e:10.3}\n", .{ attempts, final_gap });
            if (@abs(final_gap) <= opts.gap_tol) {
                converged = true;
                break;
            }
            if (final_gap < -tol.NEG_GAP) return SolveError.NegativeDualityGap;
            // Axis micro-step along the h-gradient (plain, undamped —
            // ‖center‖ is at noise scale here). This is the numerical
            // re-sample the fast path gets for free each iteration.
            const bstep = core.acceptBUpdate(Xw, b, Q, m.center, 1.0, wb.P_buf, wb.Ps);
            b = bstep.b;
            Q = bstep.Q;
            s_scale = bstep.s_scale;
        }
    }

    return core.buildOutcome(
        allocator,
        converged,
        b,
        last_gap,
        final_gap,
        outer_count,
        polish_failures,
        wb.cert_active,
        wb.cert_lambdas,
        last_gap.cert_n,
        prep.work_to_orig,
    );
}
