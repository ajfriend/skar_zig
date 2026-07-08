//! Newton polish for the D-optimal dual restricted to the active set.
//!
//! Pieces:
//! - `NewtonScratch`: per-call scratch (allocated on the solve arena);
//!   exposed because `WorkBuffers` in `skar.zig` aggregates it.
//! - `newtonPolish`: the inner iteration. Mutates the weight vector
//!   `w` in place; inactive entries reset to 0 on exit.
//!
//! The bordered KKT linear solve is private to this module; the
//! generic LU it rides on lives in `linalg.zig` (also used by the
//! joint barrier solver's 9×9 Newton system).

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

/// Newton polish on the D-optimal dual restricted to {i : w_i > active_thresh}.
/// Mutates w in place; inactive entries reset to 0 on exit.
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
