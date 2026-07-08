//! Public API surface of the `skar` package.
//!
//! This file is the canonical "what does this library expose?" doc:
//! every public type, every public method, the `checkFeasibility`
//! free function, and the rationale connecting them. The algorithm
//! itself lives in `src/skar.zig`; consumers don't need to read it.
//!
//! Two-axis result model for `solve` (defined in `src/skar.zig`):
//!
//!   - Errors (signaled via the `!` in the return type, an inferred
//!     union over `SolveError || InputError || Allocator.Error`) mean
//!     the call could not produce a meaningful `Outcome`. Three sources:
//!     the host couldn't cooperate (`OutOfMemory`); the caller passed
//!     invalid arguments (`InputError`); or the library miscomputed
//!     something internally (`SolveError`, each variant signalling a
//!     PSD or duality theorem violation beyond floating-point noise).
//!     `try` propagation is the right default for all three — the
//!     caller cooperates with allocation / fixes their input / files a
//!     bug against the library, respectively.
//!
//!   - `Outcome` is a tagged union over what the algorithm *found* on
//!     the input. Callers switch on it to dispatch — use the certificate,
//!     ask the user to fix the input, retry with more iterations, etc.
//!     Each variant carries only the data meaningful for it; the type
//!     system prevents reading `aspectRatio()` etc. on a non-converged
//!     outcome (you have to switch first and reach it through the
//!     `Converged` payload).
//!
//! In short: errors = "couldn't run"; outcome = "ran, here's the answer."

const std = @import("std");

const linalg = @import("linalg.zig");
const Vec3 = linalg.Vec3;
const Mat3 = linalg.Mat3;

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
    /// The input is rank-deficient at the feasible axis: the points'
    /// tangent-plane projections form a near-collinear 2D scatter, so
    /// the SDP would be degenerate (one tangent eigenvalue → 0). The
    /// literal "all points on a great circle" case is the dominant
    /// instance, but short arcs on non-equatorial latitude circles can
    /// also project to a near-line in the tangent plane and trigger
    /// this. Disable the check by passing `coplanarity_tol ≤ 0` to
    /// `solve` if you want to handle this case yourself.
    CoplanarInput,
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
    ///
    /// Conditioning floor: the gap has an f64 precision floor at
    /// O(κ(A)·ε) ≈ O(σ_max·ε). For well-conditioned inputs this is far
    /// below the 1e-6 default. But very small, far-from-origin scatters
    /// (e.g. sub-meter DGGS cells at finest resolution, where σ_max ~ 1e9)
    /// floor at ~1e-4–1e-3 and will return `.did_not_converge` at the
    /// default — correctly, since f64 cannot certify a tighter bound
    /// (the optimal cone axis is a sub-ulp rotation away). Pass a looser
    /// `gap_tol` (e.g. 1e-3) for such inputs; the aspect ratio is
    /// input-precision-limited and accurate regardless of the gap.
    gap_tol: f64 = 1e-6,

    /// Convex-hull preprocessing threshold. If `X.len > n_hull`,
    /// reduce input to its 2D hull at the feasible axis before
    /// iterating. `-1` disables; `0` always hulls. Default 10 is a
    /// good break-even point on typical inputs.
    n_hull: i32 = 10,

    /// Coplanarity check threshold (see `InputError.CoplanarInput`).
    /// `4·det(C) < tol · trace(C)²` on the centered 2D scatter
    /// triggers rejection. ≤ 0 disables the check; tighter positive
    /// values catch only essentially-exact coplanarity; looser
    /// values also reject near-coplanar inputs the solver would
    /// otherwise NaN on.
    coplanarity_tol: f64 = 1e-12,

    /// Outer iteration cap before returning `Outcome.did_not_converge`.
    /// Each outer iteration runs `algo.FW_PER_NEWTON` inner cycles +
    /// one Newton polish + one gap check.
    max_outer: u32 = 100,

    /// EXPERIMENTAL — solver path selection (prototype on the
    /// `investigate/wide-cap-dnc` branch; see docs/wide-cap-dnc-report.md).
    ///
    ///   .fast  — today's exact behavior: the alternating axis/MVEE
    ///            outer loop. Fast on the DGGS hot path, but limit-cycles
    ///            on dense inputs spanning ≳ 81° from the optimal axis.
    ///   .reduced — trust-region descent on the reduced convex
    ///            objective h(b) = min_A(−log det A) over the sphere,
    ///            using the fast path's inner MVEE machinery as the
    ///            oracle and the same certification. Converges on the
    ///            wide-angle/elongated inputs the fast path cannot
    ///            (see src/reduced.zig and docs/reduced-solver.md).
    ///   .auto  — .fast first; if it returns `did_not_converge`, retry
    ///            with .reduced on the same preprocessed working set and
    ///            return the better outcome.
    ///
    /// Default .fast keeps existing behavior bit-identical while the
    /// prototype is evaluated.
    method: Method = .fast,
};

/// Solver path selector for `SolveOptions.method` (see that field's
/// doc-comment for the semantics of each variant).
pub const Method = enum { fast, reduced, auto };

/// Active-set certificate. `indices` / `lambdas` are paired arrays:
/// the indices into the caller's `X[]` of the active input points and
/// their dual weights (λ ≥ 0, ∑λ = 1). Shared across all three
/// outcome variants; the per-variant scalar (gap / residual) lives on
/// the variant itself.
pub const Cert = struct {
    indices: []u32,
    lambdas: []f64,
};

/// Successful solve. Carries the full eigendecomposition of A, the
/// certified duality gap, and the active-set certificate. Methods
/// (`aspectRatio`, `b`, `A`) are defined here, not on `Outcome`, so a
/// caller must switch on the union tag first — by construction they
/// can't accidentally read these on a non-converged outcome.
pub const Converged = struct {
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
    /// Certified duality gap |primal − dual| (≤ `gap_tol`).
    gap: f64,
    outer_iters: u32,
    /// Count of outer iterations where Newton polish bailed and FW weights were used.
    newton_polish_failures: u32,
    cert: Cert,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Converged) void {
        self.allocator.free(self.cert.indices);
        self.allocator.free(self.cert.lambdas);
    }

    /// Aspect ratio of the cone cross-section = sigma[2] / sigma[1] ≥ 1.
    pub fn aspectRatio(self: Converged) f64 {
        return self.sigma[2] / self.sigma[1];
    }

    /// Cone axis: first column of Q.
    pub fn b(self: Converged) Vec3 {
        return self.Q.col(0);
    }

    /// Materialize A from its eigendecomposition: Σᵢ sigma[i] · Q[:,i] · Q[:,i]ᵀ.
    /// Cheap (three symmetric rank-1 updates). For a loop applying A to many
    /// vectors, call once and reuse.
    pub fn A(self: Converged) Mat3 {
        var m = Mat3.zero;
        m.addSymRank1(self.sigma[0], self.Q.col(0));
        m.addSymRank1(self.sigma[1], self.Q.col(1));
        m.addSymRank1(self.sigma[2], self.Q.col(2));
        return m;
    }
};

/// Proven-infeasible outcome. Carries the active-set certificate
/// (the λ on the inputs whose convex combination is near zero) and
/// the witness magnitude.
pub const Infeasible = struct {
    cert: Cert,
    /// Witness magnitude: ‖∑ λᵢ xᵢ‖. Near zero by construction.
    residual: f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Infeasible) void {
        self.allocator.free(self.cert.indices);
        self.allocator.free(self.cert.lambdas);
    }
};

/// Solver hit `max_outer` without closing the gap. Last iterate is
/// available for warm-start / inspection; not a certified cone, so no
/// `aspectRatio`/`b`/`A` methods. Raw `Q`, `sigma`, `gap`, and
/// iteration counters are exposed for diagnostics.
pub const DidNotConverge = struct {
    Q: Mat3,
    sigma: [3]f64,
    /// Last computed gap from the final outer iteration. May be near
    /// zero (almost converged) or large (the solver gave up far from
    /// optimal); inspect alongside `outer_iters` rather than as a
    /// uniform quality metric — unlike `Converged.gap`, this value is
    /// not certified to be below `gap_tol`.
    gap: f64,
    outer_iters: u32,
    newton_polish_failures: u32,
    /// Active-set cert from the last iterate (uncertified — solver
    /// did not close the gap).
    cert: Cert,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DidNotConverge) void {
        self.allocator.free(self.cert.indices);
        self.allocator.free(self.cert.lambdas);
    }
};

/// Tagged union over the possible outcomes of `solve`. Switch on it
/// before reading the payload — there is no top-level `aspectRatio`
/// to accidentally call. Use `defer outcome.deinit()` to free the
/// per-variant allocations. Rank-deficient inputs (great-circle
/// scatter) don't appear here: they're signaled as
/// `InputError.CoplanarInput`, alongside `InsufficientPoints`.
pub const Outcome = union(enum) {
    /// A valid cone was found; full eigendecomposition + primal certificate.
    converged: Converged,
    /// Proven infeasible — no hemisphere contains all input points.
    infeasible: Infeasible,
    /// Solver hit `max_outer` without closing the gap. Last iterate
    /// is available for inspection; no certified cone.
    did_not_converge: DidNotConverge,

    pub fn deinit(self: *Outcome) void {
        switch (self.*) {
            .converged => |*c| c.deinit(),
            .infeasible => |*i| i.deinit(),
            .did_not_converge => |*p| p.deinit(),
        }
    }
};

/// Max primal violation `‖A·xᵢ‖ − b·xᵢ` over all input points. Negative
/// or zero means every point sits inside the cone defined by `c`;
/// positive means the certificate doesn't cover at least one point.
///
/// Takes `Converged` rather than `Outcome` — feasibility is only
/// meaningful for a certified cone, so the type system gates the call
/// at the switch site. Callers reach this via `outcome.converged` after
/// switching on the union tag.
pub fn checkFeasibility(c: Converged, X: []const [3]f64) f64 {
    const A = c.A();
    const bv = c.b();
    var max_viol: f64 = -1e30;
    const Xv: []const Vec3 = @ptrCast(X);
    for (Xv) |xi| {
        const viol = A.apply(xi).norm() - bv.dot(xi);
        if (viol > max_viol) max_viol = viol;
    }
    return max_viol;
}
