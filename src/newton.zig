//! Newton polish for the D-optimal dual restricted to the active set.
//!
//! Pieces:
//! - `NewtonScratch`: per-call scratch (allocated on the solve arena);
//!   exposed because `WorkBuffers` in `skar.zig` aggregates it.
//! - `newtonPolish`: the inner iteration. Mutates the weight vector
//!   `w` in place; inactive entries reset to 0 on exit.
//!
//! The bordered KKT linear solve is private to this module; the
//! generic LU it rides on lives in `linalg.zig`. For k ≥
//! `RANGE_SPACE_MIN_K` active points the dense (k+1)×(k+1) system is
//! exactly singular (the Hessian has rank ≤ 6) and the solve routes
//! through a 6-dim range-space system instead — see
//! `solveKktRangeSpace`.

const std = @import("std");
const linalg = @import("linalg.zig");
const config = @import("config.zig");

const Vec3 = linalg.Vec3;
const Mat3 = linalg.Mat3;
const LU = linalg.LU;
const tol = config.tol;

/// Scratch for `newtonPolish` + `solveBorderedKkt` (active-set Newton's
/// method on the D-optimal dual). All fields are owned by the caller's
/// allocator (typically an arena scoped to one solve call) — no deinit.
pub const NewtonScratch = struct {
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

    pub fn init(allocator: std.mem.Allocator, nmax: usize) !NewtonScratch {
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

/// Active-set count at which `newtonPolish` switches from the dense
/// bordered KKT solve to the range-space solve. The polish Hessian
/// H_ij = (qᵢᵀW⁻¹qⱼ)² is the Schur square of a rank-3 Gram, so
/// rank(H) ≤ 6: for k ≥ 8, null(H) ∩ 1⊥ is nonempty and the dense
/// bordered matrix [H, 1; 1ᵀ, 0] is EXACTLY singular — its LU
/// "succeeds" only via roundoff-scale pivots (the floor is
/// tol.UNDERFLOW), amplifying a noise component along the optimal
/// face by O(1/ε). That component is model-flat and sum-preserving,
/// but it inflates |Δwᵢ| and can drive the fraction-to-boundary step
/// below NEWTON_STEP_MIN — a premature break with an under-polished
/// state. k = 7 stays on the dense path (generically nonsingular;
/// identical to prior behavior except when a boundary drop fires —
/// see the drop rule in `newtonPolish`); its known asymptotic
/// ill-conditioning — 1 → g/3 approaches range(H) as the g-spread
/// → 0, since gᵢ = vᵢᵀe identically — is bounded by the inner_tol
/// break firing first. Pre-existing, documented, unchanged.
pub const RANGE_SPACE_MIN_K: usize = 8;

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

    const lu = LU.factorize(K, n, s.piv, tol.UNDERFLOW) orelse return false;
    lu.solve(rhs);
    for (0..k) |i| delta_w[i] = rhs[i];
    return true;
}

/// Range-space KKT solve for k ≥ RANGE_SPACE_MIN_K. Exploits
/// H = VᵀV with vᵢ ∈ R⁶ the √2-weighted symmetric outer product of
/// the forward-solved design vectors yᵢ = L⁻¹qᵢ — the same rank-≤ 6
/// structure as the trust path's exact-Hessian correction
/// (src/trust.zig, evalH part 2). Substituting Δw = Vᵀα into the
/// bordered system [H, 1; 1ᵀ, 0][Δw; s] = [g; 0] and projecting the
/// first block by V:
///
///   [G² + reg·I  V·1] [α]   [V·g]
///   [(V·1)ᵀ       0 ] [s] = [ 0 ],    G = V·Vᵀ (6×6).
///
/// Near the inner optimum all gᵢ → 3, so 1 → g/3 ∈ range(Vᵀ)
/// (gᵢ = vᵢᵀe identically, with e the vectorization of I₃) and this
/// is the EXACT minimum-norm Newton step, with multiplier s → 3 — a
/// borderless "s = 0" pseudo-inverse would inflate Σw every
/// iteration. Away from the optimum it is the model minimizer
/// restricted to range(Vᵀ) ∩ 1⊥: still strict ascent, and it loses no
/// ability to move the design moment W, because null(V) is exactly
/// the set of weight directions with ΣΔwᵢ·yᵢyᵢᵀ = 0. O(k·36) build +
/// one 7×7 LU per call, replacing the dense O(k³) — and, unlike the
/// dense system at k ≥ 8, provably nonsingular (see
/// tol.NEWTON_RANGE_PIVOT_MIN).
fn solveKktRangeSpace(Y: []const Vec3, g: []const f64, delta_w: []f64) bool {
    const k = Y.len;
    const SQRT2 = std.math.sqrt2;
    var G = [_]f64{0} ** 36; // V·Vᵀ, row-major 6×6
    var b1 = [_]f64{0} ** 6; // V·1
    var bg = [_]f64{0} ** 6; // V·g
    for (0..k) |i| {
        const y = Y[i].m;
        const v = [6]f64{ y[0] * y[0], y[1] * y[1], y[2] * y[2], SQRT2 * y[0] * y[1], SQRT2 * y[0] * y[2], SQRT2 * y[1] * y[2] };
        for (0..6) |a| {
            b1[a] += v[a];
            bg[a] = @mulAdd(f64, g[i], v[a], bg[a]);
            for (0..6) |b| G[a * 6 + b] = @mulAdd(f64, v[a], v[b], G[a * 6 + b]);
        }
    }

    // A = [G² + reg·I, V·1; (V·1)ᵀ, 0], relative Tikhonov mass on the
    // G² block (benign: the projected RHS V·g lies in range(G) by
    // construction — see tol.NEWTON_RANGE_REG).
    var A = [_]f64{0} ** 49;
    var tr_g2: f64 = 0;
    for (0..6) |a| {
        for (0..6) |b| {
            var acc: f64 = 0;
            for (0..6) |c| acc = @mulAdd(f64, G[a * 6 + c], G[c * 6 + b], acc);
            A[a * 7 + b] = acc;
        }
        tr_g2 += A[a * 7 + a];
    }
    const reg = tol.NEWTON_RANGE_REG * (1.0 + tr_g2 / 6.0);
    for (0..6) |a| {
        A[a * 7 + a] += reg;
        A[a * 7 + 6] = b1[a];
        A[6 * 7 + a] = b1[a];
    }
    A[6 * 7 + 6] = 0.0;

    var piv: [7]usize = undefined;
    const lu = LU.factorize(&A, 7, &piv, tol.NEWTON_RANGE_PIVOT_MIN) orelse return false;
    var rhs: [7]f64 = .{ bg[0], bg[1], bg[2], bg[3], bg[4], bg[5], 0 };
    lu.solve(&rhs);

    // Δw = Vᵀα (the multiplier rhs[6] is discarded).
    for (0..k) |i| {
        const y = Y[i].m;
        const v = [6]f64{ y[0] * y[0], y[1] * y[1], y[2] * y[2], SQRT2 * y[0] * y[1], SQRT2 * y[0] * y[2], SQRT2 * y[1] * y[2] };
        var dw = v[0] * rhs[0];
        for (1..6) |a| dw = @mulAdd(f64, v[a], rhs[a], dw);
        delta_w[i] = dw;
    }
    return true;
}

/// Newton polish on the D-optimal dual restricted to {i : w_i > active_thresh}.
/// Mutates w in place; inactive entries reset to 0 on exit. Boundary-
/// limited Newton steps shed the blocking weight and continue on the
/// reduced active set (the drop rule below) — so unlike the historical
/// behavior, weights CAN reach exactly 0 during polish.
/// Returns false on failure (<3 active, Cholesky breakdown, or KKT singular).
pub fn newtonPolish(Ql: []const Vec3, w: []f64, active_thresh: f64, max_iter: u32, inner_tol: f64, s: *NewtonScratch) bool {
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

    // k ≥ RANGE_SPACE_MIN_K routes the KKT solve through the rank-6
    // range space (see solveKktRangeSpace); k below it keeps the dense
    // path of prior behavior. Re-evaluated after boundary drops.
    var use_range = k >= RANGE_SPACE_MIN_K;

    var it: u32 = 0;
    while (it < max_iter) : (it += 1) {
        // S = Σ wᵢ qᵢ qᵢᵀ
        var S = Mat3.zero;
        for (0..k) |i| S.addSymRank1(w_a[i], q[i]);

        const L_W = S.cholesky() orelse return false;

        // Dense path: yᵢ = W⁻¹ qᵢ (full solve), gᵢ = qᵢ · yᵢ.
        // Range path: yᵢ = L⁻¹ qᵢ (forward solve only), gᵢ = ‖yᵢ‖² —
        // the same value up to rounding (qᵢᵀW⁻¹qᵢ = ‖L⁻¹qᵢ‖²), and
        // the form the rank-6 factorization H = VᵀV is built from.
        if (use_range) {
            for (0..k) |i| {
                Y[i] = L_W.forwardSolve(q[i]);
                g[i] = Y[i].dot(Y[i]);
            }
        } else {
            for (0..k) |i| {
                Y[i] = L_W.solve(q[i]);
                g[i] = q[i].dot(Y[i]);
            }
        }

        var g_max: f64 = -1e30;
        var g_min: f64 = 1e30;
        for (0..k) |i| {
            if (g[i] > g_max) g_max = g[i];
            if (g[i] < g_min) g_min = g[i];
        }
        if (g_max - g_min < inner_tol) break;

        var solved = false;
        if (use_range) solved = solveKktRangeSpace(Y[0..k], g[0..k], delta_w);
        if (!solved) {
            // Dense bordered KKT — the k ≤ 7 primary path, and the
            // safety net for a range-solve pivot failure (provably
            // shouldn't happen; see tol.NEWTON_RANGE_PIVOT_MIN). H is
            // symmetric: H_ij = (qᵢᵀW⁻¹qⱼ)². On the dense path Y
            // holds full solves, so the original qᵢ·yⱼ form applies
            // (bit-identical); on the range path Y holds forward
            // solves, so the equal-value form (yᵢ·yⱼ)² must be used.
            for (0..k) |i| {
                for (i..k) |j| {
                    const dij = if (use_range) Y[i].dot(Y[j]) else q[i].dot(Y[j]);
                    H[i * k + j] = dij * dij;
                    H[j * k + i] = H[i * k + j];
                }
            }
            if (!solveBorderedKkt(H, k, g, delta_w, s)) return false;
        }

        // Ratio to the positivity boundary: r_min = min over shrinking
        // weights of −wᵢ/Δwᵢ, with its argmin. r_min > 1 means the full
        // Newton step is interior.
        var r_min: f64 = std.math.inf(f64);
        var blocker: usize = 0;
        for (0..k) |i| {
            if (delta_w[i] < 0) {
                const r = -w_a[i] / delta_w[i];
                if (r < r_min) {
                    r_min = r;
                    blocker = i;
                }
            }
        }

        // Boundary drop (the active-set update polish historically
        // lacked): when the Newton step is boundary-limited (r_min ≤ 1),
        // the blocking weight is headed for zero — take the step exactly
        // TO the boundary, zero it, remove it from the active set, and
        // continue on the reduced set. Without this, the iteration pins:
        // the same boundary-crossing step recurs while the
        // fraction-to-boundary alpha collapses geometrically to the
        // NEWTON_STEP_MIN floor, returning an under-polished state
        // (measured: g_max − 3 stuck at ~1e-2 on post-FW oracle states
        // whose active subset wants to shed a point — the old singular
        // dense KKT merely blurred this pinning with null-space noise).
        // Safety vs the FW drop-step hazard (docs/trust-solver.md): a
        // polish drop cannot fire at a converged design (the g-spread
        // break above exits first) nor on an interior step (r_min > 1),
        // and a wrong drop is recoverable — the next FW step re-adds the
        // max-gradient point, unlike mveeFw's full-mass noise drop.
        // Steps with r_min > 1 are bit-identical to prior behavior
        // (fl(0.99·r) is monotone, so min-of-products = product-of-min).
        if (r_min <= 1.0 and k > 3) {
            for (0..k) |i| w_a[i] += r_min * delta_w[i];
            // Swap-remove the blocker; its caller-side weight is zeroed
            // by the final writeback (it left the active list).
            w_a[blocker] = 0;
            k -= 1;
            q[blocker] = q[k];
            w_a[blocker] = w_a[k];
            active_idx[blocker] = active_idx[k];
            use_range = k >= RANGE_SPACE_MIN_K;
            continue;
        }

        const alpha = @min(1.0, 0.99 * r_min);
        if (alpha < tol.NEWTON_STEP_MIN) break;
        for (0..k) |i| w_a[i] += alpha * delta_w[i];
    }

    for (w) |*wi| wi.* = 0;
    for (0..k) |i| w[active_idx[i]] = w_a[i];
    return true;
}
