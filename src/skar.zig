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

const MAX_OUTER: u32 = 100;
/// Number of (project + FW + b-update) cycles per outer iteration. Only
/// the final cycle of each outer iteration runs Newton polish + gap
/// check. FW_PER_NEWTON = 1 is the original behaviour.
const FW_PER_NEWTON: u32 = 2;
const DAMP_SHRINK: f64 = 0.5;
const DAMP_GROW: f64 = 1.2;
const DAMP_MIN: f64 = 0.05;
const DAMP_MAX: f64 = 1.0;
const ACTIVE_THRESH: f64 = 1e-6;

/// Feasibility-cone margin for the backtracking b-update. Each outer
/// step requires min_i(b_new · xᵢ) ≥ FEAS_MARGIN; α is halved up to
/// MAX_BACKTRACKS times until the new b satisfies it. See the axis-
/// update block in `solve` for context.
const FEAS_MARGIN: f64 = 1e-8;
const MAX_BACKTRACKS: u32 = 30;

/// Quasi-Newton b-update gate: only precondition the axis step by M⁻¹
/// when cond(M) exceeds this threshold. For near-isotropic M (hex,
/// DGGS cells, rotations near coordinate axes) the preconditioner adds
/// sub-ULP direction noise that interacts badly with damping after
/// Newton polish; the plain gradient step is used instead.
const PRECOND_COND_MIN: f64 = 1.2;

/// Skip the quasi-Newton machinery for the first `AXIS_WARMUP` outer
/// iterations. Easy cases (hex, most DGGS cells) converge in ≤ this,
/// so they pay zero preconditioner overhead. The active set also tends
/// to settle in the first few iters, after which the 2D moment `M`
/// becomes a more meaningful proxy Hessian for the b-update.
const AXIS_WARMUP: u32 = 2;

/// Structural axial eigenvalue: A·b = SIGMA_0·b, where b is the cone axis.
/// Derived in `recoverAPerp` via the budget/g_max rescaling: λ_b = √(1 − 2/3).
const SIGMA_0: f64 = 1.0 / @sqrt(3.0);

/// Numerical tolerances — the "how small is small" guards. Algorithm
/// parameters (MAX_OUTER, DAMP_*, ACTIVE_THRESH, FEAS_MARGIN, AXIS_WARMUP,
/// PRECOND_COND_MIN, MAX_BACKTRACKS) are above; they tune behaviour.
/// These guard against divide-by-zero, underflow, and spurious convergence.
const tol = struct {
    /// Newton polish inner loop: stop when max-min of gradient components < this.
    const NEWTON_INNER: f64 = 1e-14;
    /// Newton polish: fraction-to-boundary step-size floor; below, declare stuck.
    const NEWTON_STEP_MIN: f64 = 1e-12;
    /// Hard floor for SolveError.NegativeDualityGap (FP noise below, bug above).
    const NEG_GAP: f64 = 1e-10;
    /// FW inner loops: minimum w_i to participate in the pairwise-swap candidate set.
    /// Distinct from (and looser than) ACTIVE_THRESH, which is the *cert* cutoff.
    const WEIGHT_ACTIVE: f64 = 1e-14;
    /// Tiny-magnitude zero guard for norms and dot-products (`< tol ⇒ treat as 0`).
    const TINY: f64 = 1e-30;
    /// 2D det / scalar singular guard (denominator-is-zero cutoff).
    const NEAR_SING: f64 = 1e-15;
    /// halfspaceCheck: z.dot(z) ceiling below which FW cannot make progress.
    const FW_Z_EXHAUSTED: f64 = 1e-12;
    /// Underflow floor: pivot / scale / log argument.
    const UNDERFLOW: f64 = 1e-300;
    /// eig2: |b|/scale relative threshold for closed-form vs. axis-aligned fallback.
    const EIG2_REL: f64 = 1e-8;
};

// ----------------------------------------------------------------
// Vec3: 3-vector with method-style operations, backed by [3]f64. Matches
// Mat3's storage convention so vec/mat code reads uniformly.
// Construct with `Vec3{ .m = arr }`; raw-array access via `v.m`.
// All methods are static inline — zero runtime cost vs hand-rolled arithmetic.
// ----------------------------------------------------------------

pub const Vec3 = extern struct {
    m: [3]f64,

    pub const zero: Vec3 = .{ .m = .{ 0, 0, 0 } };

    pub inline fn dot(a: Vec3, b: Vec3) f64 {
        return a.m[0] * b.m[0] + a.m[1] * b.m[1] + a.m[2] * b.m[2];
    }
    pub inline fn norm(v: Vec3) f64 {
        return @sqrt(v.dot(v));
    }
    pub inline fn normalize(v: Vec3) Vec3 {
        return v.scale(1.0 / v.norm());
    }
    pub inline fn scale(v: Vec3, s: f64) Vec3 {
        return .{ .m = .{ s * v.m[0], s * v.m[1], s * v.m[2] } };
    }
    pub inline fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .m = .{ a.m[0] + b.m[0], a.m[1] + b.m[1], a.m[2] + b.m[2] } };
    }
    pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .m = .{ a.m[0] - b.m[0], a.m[1] - b.m[1], a.m[2] - b.m[2] } };
    }
    /// Linear combination: s_a·a + s_b·b.
    pub inline fn lincomb(s_a: f64, a: Vec3, s_b: f64, b: Vec3) Vec3 {
        return .{ .m = .{
            s_a * a.m[0] + s_b * b.m[0],
            s_a * a.m[1] + s_b * b.m[1],
            s_a * a.m[2] + s_b * b.m[2],
        } };
    }
    /// Cross product a × b.
    pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{ .m = .{
            a.m[1] * b.m[2] - a.m[2] * b.m[1],
            a.m[2] * b.m[0] - a.m[0] * b.m[2],
            a.m[0] * b.m[1] - a.m[1] * b.m[0],
        } };
    }

    /// Pick the standard axis (ex, ey, or ez) least aligned with self.
    /// Used as the seed for `orthoBasis`.
    pub fn pickRefAxis(self: Vec3) Vec3 {
        const b = self;
        const ax = @abs(b.m[0]);
        const ay = @abs(b.m[1]);
        const az = @abs(b.m[2]);
        if (ax <= ay and ax <= az) return .{ .m = .{ 1, 0, 0 } };
        if (ay <= az) return .{ .m = .{ 0, 1, 0 } };
        return .{ .m = .{ 0, 0, 1 } };
    }

    /// Orthonormal tangent basis (e1, e2) at the unit vector self.
    /// e1 = normalize(ax − (ax·b)·b) where ax = pickRefAxis(b) is the
    /// least-aligned standard axis, so |ax·b| ≤ 1/√3 and |e1| ≥ √(2/3)
    /// — catastrophic cancellation is impossible; no fallback needed.
    /// e2 = b × e1 (right-handed, unit).
    pub fn orthoBasis(b: Vec3) Mat3x2 {
        const ax = b.pickRefAxis();
        const e1 = Vec3.lincomb(1.0, ax, -ax.dot(b), b).normalize();
        const e2 = b.cross(e1).normalize();
        return .{ .e1 = e1, .e2 = e2 };
    }
};

// ----------------------------------------------------------------
// Vec2: 2-vector (used for the 2D gnomonic projection plane).
// ----------------------------------------------------------------

pub const Vec2 = extern struct {
    m: [2]f64,

    pub const zero: Vec2 = .{ .m = .{ 0, 0 } };

    pub inline fn dot(a: Vec2, b: Vec2) f64 { return a.m[0] * b.m[0] + a.m[1] * b.m[1]; }
    pub inline fn norm(v: Vec2) f64 { return @sqrt(v.dot(v)); }
    pub inline fn scale(v: Vec2, s: f64) Vec2 { return .{ .m = .{ s * v.m[0], s * v.m[1] } }; }
    pub inline fn add(a: Vec2, b: Vec2) Vec2 { return .{ .m = .{ a.m[0] + b.m[0], a.m[1] + b.m[1] } }; }
    pub inline fn sub(a: Vec2, b: Vec2) Vec2 { return .{ .m = .{ a.m[0] - b.m[0], a.m[1] - b.m[1] } }; }
    pub inline fn lincomb(s_a: f64, a: Vec2, s_b: f64, b: Vec2) Vec2 {
        return .{ .m = .{ s_a * a.m[0] + s_b * b.m[0], s_a * a.m[1] + s_b * b.m[1] } };
    }
};

// ----------------------------------------------------------------
// Mat2: 2x2 matrix (row-major).
// ----------------------------------------------------------------

pub const Mat2 = struct {
    m: [4]f64,

    pub const zero: Mat2 = .{ .m = .{ 0, 0, 0, 0 } };

    pub inline fn apply(self: Mat2, v: Vec2) Vec2 {
        return .{ .m = .{
            self.m[0] * v.m[0] + self.m[1] * v.m[1],
            self.m[2] * v.m[0] + self.m[3] * v.m[1],
        } };
    }

    pub inline fn det(self: Mat2) f64 {
        return self.m[0] * self.m[3] - self.m[1] * self.m[2];
    }

    pub inline fn scale(self: Mat2, s: f64) Mat2 {
        return .{ .m = .{ s * self.m[0], s * self.m[1], s * self.m[2], s * self.m[3] } };
    }

    pub inline fn lincomb(s_a: f64, a: Mat2, s_b: f64, b: Mat2) Mat2 {
        return .{ .m = .{
            s_a * a.m[0] + s_b * b.m[0],
            s_a * a.m[1] + s_b * b.m[1],
            s_a * a.m[2] + s_b * b.m[2],
            s_a * a.m[3] + s_b * b.m[3],
        } };
    }

    /// Outer product x yᵀ (2x2).
    pub inline fn outer(x: Vec2, y: Vec2) Mat2 {
        return .{ .m = .{
            x.m[0] * y.m[0], x.m[0] * y.m[1],
            x.m[1] * y.m[0], x.m[1] * y.m[1],
        } };
    }

    /// Symmetric rank-1 update: self ← self + w · p · pᵀ.
    pub inline fn addSymRank1(self: *Mat2, w: f64, p: Vec2) void {
        const wp0 = w * p.m[0];
        const wp1 = w * p.m[1];
        self.m[0] += wp0 * p.m[0];
        self.m[3] += wp1 * p.m[1];
        self.m[1] += wp0 * p.m[1];
        self.m[2] = self.m[1];
    }

    /// Inverse of a 2x2 matrix (caller should ensure det ≠ 0).
    pub inline fn inverse(self: Mat2) Mat2 {
        const inv_det = 1.0 / self.det();
        return .{ .m = .{
            self.m[3] * inv_det, -self.m[1] * inv_det,
            -self.m[2] * inv_det, self.m[0] * inv_det,
        } };
    }
};

// ----------------------------------------------------------------
// 3x2 orthonormal basis (two named 3-vectors)
// ----------------------------------------------------------------

pub const Mat3x2 = struct {
    e1: Vec3,
    e2: Vec3,

    /// 3×2 · 2-vector = 3-vector: v[0]·e1 + v[1]·e2.
    pub fn apply(self: Mat3x2, v: Vec2) Vec3 {
        return Vec3.lincomb(v.m[0], self.e1, v.m[1], self.e2);
    }

    /// (3×2)ᵀ · 3-vector = 2-vector: (e1·x, e2·x).
    pub fn applyT(self: Mat3x2, x: Vec3) Vec2 {
        return .{ .m = .{ self.e1.dot(x), self.e2.dot(x) } };
    }
};

// ----------------------------------------------------------------
// Mat3: 3x3 matrix (row-major) with methods for the operations we need.
// Used for both general and symmetric matrices (symmetry is a caller
// invariant on methods like addSymRank1). The Cholesky factor lives in
// the separate `Chol3` type, which only exposes forward/back-solve —
// so the lower-triangular invariant is compile-enforced.
// ----------------------------------------------------------------

pub const Mat3 = struct {
    m: [9]f64,

    pub const zero: Mat3 = .{ .m = .{0} ** 9 };

    pub inline fn apply(self: Mat3, v: Vec3) Vec3 {
        const a = self.m;
        return .{
            .m = .{
                a[0] * v.m[0] + a[1] * v.m[1] + a[2] * v.m[2],
                a[3] * v.m[0] + a[4] * v.m[1] + a[5] * v.m[2],
                a[6] * v.m[0] + a[7] * v.m[1] + a[8] * v.m[2],
            },
        };
    }

    pub inline fn det(self: Mat3) f64 {
        const a = self.m;
        return a[0] * (a[4] * a[8] - a[5] * a[7]) -
            a[1] * (a[3] * a[8] - a[5] * a[6]) +
            a[2] * (a[3] * a[7] - a[4] * a[6]);
    }

    pub inline fn scale(self: Mat3, s: f64) Mat3 {
        var r: Mat3 = undefined;
        for (0..9) |i| r.m[i] = s * self.m[i];
        return r;
    }

    /// Linear combination: s_a·A + s_b·B.
    pub inline fn lincomb(s_a: f64, a: Mat3, s_b: f64, b: Mat3) Mat3 {
        var r: Mat3 = undefined;
        for (0..9) |i| r.m[i] = s_a * a.m[i] + s_b * b.m[i];
        return r;
    }

    pub inline fn transpose(self: Mat3) Mat3 {
        const a = self.m;
        return .{ .m = .{
            a[0], a[3], a[6],
            a[1], a[4], a[7],
            a[2], a[5], a[8],
        } };
    }

    /// Matrix product self · other.
    pub inline fn mul(self: Mat3, other: Mat3) Mat3 {
        const a = self.m;
        const b = other.m;
        var r: Mat3 = undefined;
        for (0..3) |row| {
            for (0..3) |c| {
                r.m[row * 3 + c] =
                    a[row * 3 + 0] * b[0 * 3 + c] +
                    a[row * 3 + 1] * b[1 * 3 + c] +
                    a[row * 3 + 2] * b[2 * 3 + c];
            }
        }
        return r;
    }

    /// Build a Mat3 from its three columns. Row-major storage; `c0` populates
    /// indices {0,3,6}, `c1` populates {1,4,7}, `c2` populates {2,5,8}.
    pub inline fn fromCols(c0: Vec3, c1: Vec3, c2: Vec3) Mat3 {
        return .{ .m = .{
            c0.m[0], c1.m[0], c2.m[0],
            c0.m[1], c1.m[1], c2.m[1],
            c0.m[2], c1.m[2], c2.m[2],
        } };
    }

    /// Extract column `i` (0..2) as a Vec3.
    pub inline fn col(self: Mat3, i: usize) Vec3 {
        return .{ .m = .{ self.m[i], self.m[3 + i], self.m[6 + i] } };
    }

    /// Outer product x yᵀ.
    pub inline fn outer(x: Vec3, y: Vec3) Mat3 {
        var r: Mat3 = undefined;
        for (0..3) |row| {
            for (0..3) |c| {
                r.m[row * 3 + c] = x.m[row] * y.m[c];
            }
        }
        return r;
    }

    /// Symmetric rank-1 update: self ← self + w · q · qᵀ. Updates all 9
    /// entries (upper triangle computed, mirrored to lower). Multiply order
    /// is (w·q_r)·q_c — keeping this fixed matters for degenerate problems
    /// where Newton polish would otherwise land on different points of the
    /// KKT manifold under FP reassociation.
    pub inline fn addSymRank1(self: *Mat3, w: f64, q: Vec3) void {
        const wq0 = w * q.m[0];
        const wq1 = w * q.m[1];
        const wq2 = w * q.m[2];
        self.m[0] += wq0 * q.m[0];
        self.m[4] += wq1 * q.m[1];
        self.m[8] += wq2 * q.m[2];
        self.m[1] += wq0 * q.m[1];
        self.m[3] = self.m[1];
        self.m[2] += wq0 * q.m[2];
        self.m[6] = self.m[2];
        self.m[5] += wq1 * q.m[2];
        self.m[7] = self.m[5];
    }

    /// Symmetric rank-2 update: self ← self + λ · (x zᵀ + z xᵀ) / 2.
    /// Upper triangle computed, mirrored to lower.
    pub inline fn addSymRank2(self: *Mat3, lam: f64, x: Vec3, z: Vec3) void {
        self.m[0] += lam * x.m[0] * z.m[0];
        self.m[4] += lam * x.m[1] * z.m[1];
        self.m[8] += lam * x.m[2] * z.m[2];
        const half = 0.5 * lam;
        self.m[1] += half * (x.m[0] * z.m[1] + z.m[0] * x.m[1]);
        self.m[3] = self.m[1];
        self.m[2] += half * (x.m[0] * z.m[2] + z.m[0] * x.m[2]);
        self.m[6] = self.m[2];
        self.m[5] += half * (x.m[1] * z.m[2] + z.m[1] * x.m[2]);
        self.m[7] = self.m[5];
    }

    /// Symmetric outer product: (x zᵀ + z xᵀ) / 2.
    pub inline fn symOuter(x: Vec3, z: Vec3) Mat3 {
        var r: Mat3 = undefined;
        for (0..3) |row| {
            for (0..3) |c| {
                r.m[row * 3 + c] = (x.m[row] * z.m[c] + z.m[row] * x.m[c]) * 0.5;
            }
        }
        return r;
    }

    /// Symmetrize: returns (self + selfᵀ) / 2.
    pub inline fn symmetrize(self: Mat3) Mat3 {
        return Mat3.lincomb(0.5, self, 0.5, self.transpose());
    }

    /// Cholesky: returns L (lower-triangular, upper zeroed) such that
    /// self = L · Lᵀ, or null on non-SPD.
    pub fn cholesky(self: Mat3) ?Chol3 {
        const S = self.m;
        var L: [9]f64 = .{0} ** 9;
        var s = S[0];
        if (s <= 0) return null;
        L[0] = @sqrt(s);
        L[3] = S[1] / L[0];
        L[6] = S[2] / L[0];
        s = S[4] - L[3] * L[3];
        if (s <= 0) return null;
        L[4] = @sqrt(s);
        L[7] = (S[5] - L[6] * L[3]) / L[4];
        s = S[8] - L[6] * L[6] - L[7] * L[7];
        if (s <= 0) return null;
        L[8] = @sqrt(s);
        return Chol3{ .m = L };
    }
};

/// Lower-triangular Cholesky factor of a 3×3 SPD matrix. Produced only
/// via `Mat3.cholesky`; the restricted API (forward/back/solve only)
/// prevents accidentally treating the factor as a general 3×3.
pub const Chol3 = struct {
    m: [9]f64,

    /// Forward-solve self·y = b (self is lower-triangular).
    pub inline fn forwardSolve(self: Chol3, b: Vec3) Vec3 {
        const L = self.m;
        const y0 = b.m[0] / L[0];
        const y1 = (b.m[1] - L[3] * y0) / L[4];
        const y2 = (b.m[2] - L[6] * y0 - L[7] * y1) / L[8];
        return .{ .m = .{ y0, y1, y2 } };
    }

    /// Back-solve selfᵀ·x = y.
    pub inline fn backSolve(self: Chol3, y: Vec3) Vec3 {
        const L = self.m;
        const x2 = y.m[2] / L[8];
        const x1 = (y.m[1] - L[7] * x2) / L[4];
        const x0 = (y.m[0] - L[3] * x1 - L[6] * x2) / L[0];
        return .{ .m = .{ x0, x1, x2 } };
    }

    /// Solve (self · selfᵀ) · x = b via forward- then back-substitution.
    pub inline fn solve(self: Chol3, b: Vec3) Vec3 {
        return self.backSolve(self.forwardSolve(b));
    }
};

/// Eigenvalues (ascending) and eigenvectors (columns) of a 2x2 symmetric M.
///
/// Numerically robust at the near-isotropic boundary: when |b| is small
/// *relative to the matrix scale* (not just absolutely), the closed-form
/// eigenvector formula `v = (vals[0] - d, b)` cancels catastrophically
/// and the resulting direction is dominated by FP noise in `vals[0] - d`.
/// In that regime we fall back to the standard-axis eigenvectors (sorted
/// to match vals' ascending order) — for a near-isotropic matrix any
/// orthonormal basis is a valid eigenbasis, and the standard axes give
/// a well-defined deterministic choice.
pub const Eig2 = struct { vals: [2]f64, vecs: Mat2 };
pub fn eig2(M: [4]f64) Eig2 {
    const a = M[0];
    const b = M[1];
    const d = M[3];
    const tr = a + d;
    const disc = @sqrt((a - d) * (a - d) + 4.0 * b * b);
    var result: Eig2 = undefined;
    result.vals[0] = (tr - disc) / 2.0;
    result.vals[1] = (tr + disc) / 2.0;
    // Relative threshold: |b| must be large enough that `vals[0] - d`
    // doesn't lose all its precision in cancellation. The cancellation
    // magnitude is ~|a-d| - disc which is O(b²/(a-d)) for small b. We
    // want b² / max(|a|,|d|) ≫ ulp · max(|a|,|d|), i.e.
    // |b| ≫ sqrt(ulp) · max(|a|,|d|) ≈ 1.5e-8 · scale.
    const scale = @max(@abs(a), @abs(d));
    if (@abs(b) > tol.EIG2_REL * scale) {
        var v0 = result.vals[0] - d;
        var v1 = b;
        const nrm = @sqrt(v0 * v0 + v1 * v1);
        v0 /= nrm;
        v1 /= nrm;
        result.vecs = .{ .m = .{ v0, v1, -v1, v0 } };
    } else if (a <= d) {
        // M nearly diagonal, M[0,0] is the smaller (or equal) eigenvalue.
        // vals[0] = a → eigenvector (1, 0). vals[1] = d → (0, 1).
        result.vecs = .{ .m = .{ 1, 0, 0, 1 } };
    } else {
        // M nearly diagonal, M[1,1] is the smaller eigenvalue.
        // vals[0] = d → eigenvector (0, 1). Right-handed: vals[1] → (-1, 0).
        result.vecs = .{ .m = .{ 0, 1, -1, 0 } };
    }
    return result;
}

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
/// grew, grows it when |c| shrank, bounded in [DAMP_MIN, DAMP_MAX].
const DampState = struct {
    alpha: f64 = 1.0,
    prev_c_norm: f64 = 1e30,

    inline fn tick(self: *DampState, c_norm: f64) void {
        if (c_norm > self.prev_c_norm) {
            self.alpha *= DAMP_SHRINK;
            if (self.alpha < DAMP_MIN) self.alpha = DAMP_MIN;
        } else {
            self.alpha *= DAMP_GROW;
            if (self.alpha > DAMP_MAX) self.alpha = DAMP_MAX;
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
/// Skip the whole check for the first AXIS_WARMUP iters — easy cases
/// converge inside the warmup and pay zero preconditioner cost. See
/// docs/mvee_derivation.md "Quasi-Newton axis update" appendix for history.
const AxisStep = struct { u: Vec2, c_norm: f64 };

inline fn quasiNewtonAxisDirection(outer: u32, M: Mat2, center: Vec2) AxisStep {
    const c_norm = center.norm();
    var u: Vec2 = center;
    if (outer >= AXIS_WARMUP and c_norm > tol.TINY) {
        const eigM = eig2(M.m);
        const eig_lo = eigM.vals[0];
        const eig_hi = eigM.vals[1];
        // cond(M) > PRECOND_COND_MIN  ⟺  eig_hi > PRECOND_COND_MIN · eig_lo,
        // division-free and robust to a near-zero eig_lo that we'd
        // otherwise guard separately.
        if (eig_hi > PRECOND_COND_MIN * eig_lo) {
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
    while (bt < MAX_BACKTRACKS) : (bt += 1) {
        const b_trial = Vec3.lincomb(1.0, b, alpha_try, dQc).normalize();
        const Q_trial = b_trial.orthoBasis();
        if (projectGnomonic(Xw, b_trial, Q_trial, P_buf, FEAS_MARGIN)) {
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
fn recoverAPerp(P: []const [2]f64, M: Mat2) Mat2 {
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
    const s_det = @sqrt(Minv.det());
    const tr = Minv.m[0] + Minv.m[3];
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
) GapResult {
    // A's eigendecomposition: V = [b | v₁ | v₂], Λ = diag(SIGMA_0, σ₁, σ₂).
    // Always valid (depends only on A_perp and Q_ortho); returned in GapResult
    // so `solve`'s finalization reuses it without re-decomposing.
    const eAPerp = eig2(A_perp.m);
    const sigma: [2]f64 = eAPerp.vals;
    const v1 = Vec3.lincomb(eAPerp.vecs.m[0], Q_ortho.e1, eAPerp.vecs.m[1], Q_ortho.e2);
    const v2 = Vec3.lincomb(eAPerp.vecs.m[2], Q_ortho.e1, eAPerp.vecs.m[3], Q_ortho.e2);

    const active_idx = s.active_idx;
    const lam = s.lam;
    const xa = s.xa;
    const za = s.za;
    var k: usize = 0;
    for (w, 0..) |wi, i| {
        if (wi > ACTIVE_THRESH) {
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
//   - Errors (`SolveError || Allocator.Error`, signaled via the `!` in
//     the return type) mean the call could not produce a meaningful
//     `Info`. Either the host couldn't cooperate (`OutOfMemory`) or the
//     library miscomputed something internally (`NegativeDualityGap`).
//     Neither is recoverable from the caller's side, so `try`
//     propagation is the right default behavior.
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
    /// Coplanarity check rejected the input before iteration: all
    /// points lie in a 2D subspace through the origin, so the SDP is
    /// degenerate (one tangent eigenvalue → 0). `Info` is otherwise
    /// empty. Disable the check by passing `coplanarity_tol < 0` to
    /// `solve` if you want to handle this case yourself.
    coplanar_input,
};

/// Internal-correctness errors. Distinct from `Allocator.Error` (the
/// host couldn't allocate) — these mean the library produced a result
/// that violates a theorem and the bug needs to be surfaced loudly.
pub const SolveError = error{
    /// The duality-gap computation produced a meaningfully negative
    /// value — either the dual certificate is not actually feasible,
    /// or the log-det was computed on ill-conditioned input. Weak
    /// duality (`gap ≥ 0`) is a theorem, so this signals a bug.
    /// ulp-level negatives are float noise and silently ignored;
    /// anything beyond that propagates as this error.
    NegativeDualityGap,
};

pub const Cert = struct {
    indices: []u32,
    lambdas: []f64,
    /// On CONVERGED: the duality gap |primal - dual|.
    /// On INFEASIBLE: the Farkas residual ‖∑ λᵢ xᵢ‖.
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
    /// NaN on INFEASIBLE (sigma initialized to zeros).
    pub fn aspectRatio(self: Info) f64 {
        return self.sigma[2] / self.sigma[1];
    }

    /// Cone axis: first column of Q.
    pub fn b(self: Info) Vec3 {
        return self.Q.col(0);
    }

    /// Materialize A from its eigendecomposition: Σᵢ sigma[i] · Q[:,i] · Q[:,i]ᵀ.
    /// Cheap (three symmetric rank-1 updates). For a loop applying A to many
    /// vectors, call once and reuse.
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

pub fn checkFeasibility(info: Info, X: []const [3]f64) f64 {
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

/// Main solver. `n_hull` convex-hull threshold: if n > n_hull, reduce to hull.
/// Pass -1 to disable, 0 to always hull, 10 for the default.
///
/// `coplanarity_tol`: rejects rank-deficient inputs (all points in a 2D
/// subspace through the origin) with `Status.coplanar_input`. After
/// projecting to the tangent plane at the feasible axis, the 2×2 centered
/// scatter C is checked against `4·det(C) < tol · trace(C)²` —
/// "fraction-of-isotropic" ∈ [0, 1] where 1 is a circular scatter and 0 is
/// a perfect line. tol = 1e-12 flags inputs whose scatter ellipse is roughly
/// >10⁶× longer than wide; tighter catches only essentially-exact
/// coplanarity, looser also rejects near-coplanar inputs the solver would
/// otherwise produce NaN on. Pass < 0 to disable the check entirely.
pub fn solve(
    allocator: std.mem.Allocator,
    X: []const [3]f64,
    gap_tol: f64,
    n_hull: i32,
    coplanarity_tol: f64,
) !Info {
    var info = Info{
        .status = .did_not_converge,
        .Q = Mat3.zero,
        .sigma = .{ 0, 0, 0 },  // aspectRatio() returns NaN via 0/0 on INFEASIBLE
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

    // 1) Feasibility via Farkas FW.
    const hs = try halfspaceCheck(scratch_alloc, Xv);
    var b: Vec3 = undefined;
    if (hs.b) |bb| {
        b = bb;
    } else {
        // Infeasible: populate Farkas cert from hs.lam. Allocate on the
        // parent allocator since it's returned to the caller.
        var k: u32 = 0;
        for (hs.lam) |l| if (l > ACTIVE_THRESH) {
            k += 1;
        };
        const indices = try allocator.alloc(u32, k);
        const lambdas = try allocator.alloc(f64, k);
        var j: u32 = 0;
        for (hs.lam, 0..) |l, i| {
            if (l > ACTIVE_THRESH) {
                indices[j] = @intCast(i);
                lambdas[j] = l;
                j += 1;
            }
        }
        info.status = .infeasible;
        info.cert = .{ .indices = indices, .lambdas = lambdas, .claimed_gap = hs.residual };
        return info;
    }

    // 2) Optional hull preprocessing.
    var Xw_storage: []const Vec3 = Xv;
    var work_to_orig: ?[]const u32 = null;

    if (n_hull >= 0 and X.len > @as(usize, @intCast(n_hull))) {
        const n = X.len;
        const Qh = b.orthoBasis();
        const P2 = try scratch_alloc.alloc([2]f64, n);
        for (Xv, 0..) |xi, i| {
            P2[i] = Qh.applyT(xi).m;
        }
        const hull_idx = try scratch_alloc.alloc(u32, n);
        const nh = try convexHull2d(scratch_alloc, P2, hull_idx);
        if (nh >= 3) {
            const Xhull = try scratch_alloc.alloc(Vec3, nh);
            for (0..nh) |i| Xhull[i] = Xv[hull_idx[i]];
            Xw_storage = Xhull;
            work_to_orig = hull_idx[0..nh];
        }
    }

    const Xw = Xw_storage;
    const nw = Xw.len;

    // 2.5) Coplanarity check. Reject inputs where the working set lies in a
    //      2D subspace through the origin (i.e., on a single great circle).
    //      Such inputs drive the optimum to a degenerate cone (one tangent
    //      eigenvalue → 0) and produce NaN downstream. Projecting to the
    //      tangent plane at b reduces this to a 2D rank check: the
    //      projected points are collinear iff the 3D points are coplanar
    //      with the origin. Scale-invariant "fraction of isotropic" via
    //      4·det / trace² ∈ [0, 1] on the 2×2 centered scatter C — 1 for
    //      isotropic (circular), → 0 for collinear. Equivalent to the
    //      cancellation-safe form of λ_min/λ_max for ill-conditioned C
    //      (where (T − √(T² − 4D))/2 loses precision). Tight clusters on
    //      the sphere (e.g. H3 res-15) have full-rank 2D scatter
    //      regardless of absolute scale, so this correctly distinguishes
    //      them from genuinely rank-deficient input. Runs on Xw (the
    //      hulled subset when hull preprocessing fired) so an input
    //      whose hull is collinear gets caught even if the full cloud
    //      jitter looked full-rank. Disabled by coplanarity_tol < 0.
    if (coplanarity_tol >= 0) {
        const Qh = b.orthoBasis();
        var ps0: f64 = 0;
        var ps1: f64 = 0;
        var s00: f64 = 0;
        var s01: f64 = 0;
        var s11: f64 = 0;
        for (Xw) |xi| {
            const p = Qh.applyT(xi);
            ps0 += p.m[0];
            ps1 += p.m[1];
            s00 += p.m[0] * p.m[0];
            s01 += p.m[0] * p.m[1];
            s11 += p.m[1] * p.m[1];
        }
        const inv_n = 1.0 / @as(f64, @floatFromInt(nw));
        const c00 = s00 - ps0 * ps0 * inv_n;
        const c01 = s01 - ps0 * ps1 * inv_n;
        const c11 = s11 - ps1 * ps1 * inv_n;
        const tr = c00 + c11;
        const det = c00 * c11 - c01 * c01;
        if (tr <= 0 or 4.0 * det < coplanarity_tol * tr * tr) {
            info.status = .coplanar_input;
            return info;
        }
    }

    // 3) Working buffers (all in the arena).
    const P_buf = try scratch_alloc.alloc([2]f64, nw);
    const Ps = try scratch_alloc.alloc([2]f64, nw);
    const Ql = try scratch_alloc.alloc(Vec3, nw);
    const w = try scratch_alloc.alloc(f64, nw);

    var newton_scratch = try NewtonScratch.init(scratch_alloc, nw);
    var gap_scratch = try GapScratch.init(scratch_alloc, nw);
    // No deinit: arena frees everything at end.

    const cert_active = try scratch_alloc.alloc(usize, nw);
    const cert_lambdas = try scratch_alloc.alloc(f64, nw);
    var cert_n: usize = 0;
    var final_gap: f64 = 1e30;

    const inv_nw = 1.0 / @as(f64, @floatFromInt(nw));
    for (w) |*wi| wi.* = inv_nw;

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
    // necessarily ≥ FEAS_MARGIN), so bypass the feasibility check here.
    _ = projectGnomonic(Xw, b, Q, P_buf, -std.math.inf(f64));
    var s_scale: f64 = rescaleP(P_buf, Ps);

    // 4) Hybrid outer loop. Each outer iteration runs FW_PER_NEWTON cycles
    //    of (FW + b-update); only the last cycle also runs Newton polish +
    //    gap check. Extra cheap cycles buy more b-motion per Newton call.
    //    At FW_PER_NEWTON = 1 this is the original one-FW-per-Newton
    //    schedule; larger values amortise Newton's cost across more b-motion.
    //
    //    Loop invariant: on entry to each cycle, P_buf/Ps/s_scale correspond
    //    to the current (b, Q). The accepted b-update at cycle end also
    //    produces the next cycle's projection in one sweep.
    var outer: u32 = 0;
    outer_loop: while (outer < MAX_OUTER) : (outer += 1) {
        outer_count += 1;
        var cycle: u32 = 0;
        while (cycle < FW_PER_NEWTON) : (cycle += 1) {
            const is_full = (cycle == FW_PER_NEWTON - 1);

            mveeFw(Ps, 1, 0.0, Ql, w, true);

            if (is_full) {
                if (!newtonPolish(Ql, w, ACTIVE_THRESH, 20, tol.NEWTON_INNER, &newton_scratch)) {
                    newton_polish_failures += 1;
                }
            }

            const m = computeMoments(Ps, w, s_scale);

            if (is_full) {
                const A_perp = recoverAPerp(P_buf, m.M);
                last_gap = dualityGapConstructed(w, b, Xw, A_perp, Q, &gap_scratch, cert_active, cert_lambdas);
                final_gap = last_gap.gap;
                cert_n = last_gap.cert_n;
                // Convergence: |gap| ≤ tol. FP noise can push the gap
                // slightly negative when the iteration has converged to
                // a near-zero gap (seen on h3_r15_pent: gap = -8.5e-10
                // with tol = 1e-6). Accept those as converged before the
                // hard NegGap guard kicks in.
                if (@abs(last_gap.gap) <= gap_tol) {
                    converged = true;
                    break :outer_loop;
                }
                // Anything else negative is a broken certificate.
                if (last_gap.gap < -tol.NEG_GAP) return SolveError.NegativeDualityGap;
            }

            const axis = quasiNewtonAxisDirection(outer, m.M, m.center);
            damp.tick(axis.c_norm);
            const step = acceptBUpdate(Xw, b, Q, axis.u, damp.alpha, P_buf, Ps);
            b = step.b;
            Q = step.Q;
            s_scale = step.s_scale;
        }
    }

    // 5) Build final cert (translate work indices back to original X indices).
    const indices = try allocator.alloc(u32, cert_n);
    const lambdas = try allocator.alloc(f64, cert_n);
    for (0..cert_n) |i| {
        const idx_in_work = cert_active[i];
        indices[i] = if (work_to_orig) |wto| wto[idx_in_work] else @intCast(idx_in_work);
        lambdas[i] = cert_lambdas[i];
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

