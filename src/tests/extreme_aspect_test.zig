//! Stress tests for extreme-aspect-ratio inputs under rotation.
//!
//! Motivation. The recent symEigvals → Cholesky swap in
//! `dualityGapConstructed` (src/skar.zig) tightened the indefinite-dual
//! guard: the prior eig path tolerated a smallest M-eigenvalue in
//! `[-CS_PSD_REL · max_eig, 0]` and clamped the log via UNDERFLOW; the
//! Cholesky path bails on any non-positive pivot. The existing 48-case
//! integration suite covers aspect ratios in roughly [1.0, 1.09], which
//! never exercises the regime where M can plausibly be near-singular.
//!
//! At KKT optimality M = LᵀZL = Lᵀ A⁻¹ L = I exactly, so A's aspect
//! ratio cancels and κ(M) → 1 regardless of how elongated the optimal
//! cone is. The actual risk is intermediate iterations: a Cholesky
//! failure that previously didn't bail under the lenient eig guard
//! could now wedge the solver. These tests check the solver still
//! converges on inputs whose optimal A has a large aspect ratio.
//!
//! Three case geometries, each with three points and a small
//! perturbation that breaks rank-2 degeneracy. The three geometries
//! stress different parts of the solver:
//!  - **arc 174°**: two points near-antipodal, one between. Stresses
//!    the gnomonic projection at large angular spans where spherical
//!    curvature is significant. Solver hits a pre-existing
//!    convergence-plateau wall around AR ≈ 80–100 (verified
//!    independent of this change — reproduces on HEAD~1), so we cap
//!    this case at AR ≈ 63.
//!  - **arc 90°**: moderate angular span; less spherical distortion,
//!    higher achievable AR before the wall. AR ≈ 143.
//!  - **small patch**: three points clustered in a small angular
//!    region near a pole, arranged as a long-thin triangle. The
//!    tangent-plane approximation holds, so spherical curvature
//!    barely affects conditioning — pure in-plane aspect, AR scales
//!    as √3 · L / eps. AR ≈ 17 000 here.
//!
//! Each case is rerun under 20 random SO(3) rotations using a fixed
//! PRNG seed (same 20 rotations applied to every case for
//! reproducibility — if a (case, k) pair fails, k identifies which
//! rotation to reproduce). Rotation breaks alignment with the
//! coordinate axes, which otherwise can mask numerical issues in
//! eig2, the gnomonic projection, and any branch that compares
//! against the standard basis.
//!
//! Invariants asserted per (case, rotation):
//!   - status == converged within max_outer iterations
//!   - |claimed_gap| < tol (the converged certificate is valid)
//!   - aspect ratio matches the canonical (unrotated) AR to within
//!     `ar_rel_tol · canon_ar` (relative tolerance: ulp-level rotation
//!     noise gets amplified by the optimum's ill-conditioning, so an
//!     absolute tolerance fixed against AR=1 cases would be too tight)
//!
//! Known limitations (both verified pre-existing on HEAD~1, identical
//! numbers — neither is a Cholesky regression):
//!  - 3 points exactly coplanar with the origin (great-circle inputs,
//!    z = 0 in canonical orientation) caused the solver to diverge to
//!    NaN. Now caught by the coplanarity preprocessing check in
//!    `solve` (returns `InputError.CoplanarInput`); see the
//!    "coplanarity check flags great-circle inputs" test below. The
//!    three high-AR cases here include a z-perturbation that keeps
//!    them safely above the rank-deficiency threshold.
//!  - Near-antipodal (174°) cases hit a convergence plateau around
//!    AR ≈ 80–100: some rotations plateau at gap ~1e-4 and don't
//!    tighten in 1000 outer iterations. Each chosen `eps` here is
//!    one notch back from the empirical edge so future numeric
//!    drift doesn't make the test flaky.

const std = @import("std");
const sphar = @import("../root.zig");
// Internals reached by filesystem path — these aren't part of the
// public API exposed via root.zig, but tests inside src/ have direct
// access to sibling files.
const halfspace = @import("../halfspace.zig");
const linalg = @import("../linalg.zig");
const skar = @import("../skar.zig");

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

/// Three unit points roughly spanning `span_deg` of a great-circle
/// arc, perturbed ±eps in z (then renormalized) so the configuration
/// is full-rank 3D.
fn arcPoints(span_deg: f64, eps: f64) [3][3]f64 {
    const half = deg(span_deg / 2.0);
    var pts: [3][3]f64 = .{
        .{ std.math.cos(-half), std.math.sin(-half), eps },
        .{ 1.0, 0.0, -eps },
        .{ std.math.cos(half), std.math.sin(half), eps },
    };
    normalizeAll(&pts);
    return pts;
}

/// Three unit points in a small tangent-plane patch near (0,0,1):
/// two extremes at ±L in x, one apex at +eps in y. AR ≈ √3 · L / eps
/// when L is small enough that the patch is near-planar.
fn patchPoints(L: f64, eps: f64) [3][3]f64 {
    var pts: [3][3]f64 = .{
        .{ L, 0.0, 1.0 },
        .{ -L, 0.0, 1.0 },
        .{ 0.0, eps, 1.0 },
    };
    normalizeAll(&pts);
    return pts;
}

fn normalizeAll(pts: *[3][3]f64) void {
    for (pts) |*p| {
        const n = @sqrt(p.*[0] * p.*[0] + p.*[1] * p.*[1] + p.*[2] * p.*[2]);
        p.*[0] /= n;
        p.*[1] /= n;
        p.*[2] /= n;
    }
}

/// Tiny LCG returning f64 in [0, 1). Hand-rolled rather than std.Random
/// only so the seed/sequence is stable across Zig versions; we just
/// need reproducibility, not statistical guarantees.
fn nextU01(state: *u64) f64 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    const x: u32 = @truncate(state.* >> 32);
    return @as(f64, @floatFromInt(x)) / 4294967296.0;
}

/// Uniform random rotation matrix (row-major 3×3) via Shoemake's
/// three-uniform unit-quaternion construction.
fn randomRotation(state: *u64) [9]f64 {
    const r1 = nextU01(state);
    const r2 = nextU01(state);
    const r3 = nextU01(state);
    const sq1 = @sqrt(1.0 - r1);
    const sqr = @sqrt(r1);
    const two_pi = 2.0 * std.math.pi;
    const x = sq1 * std.math.sin(two_pi * r2);
    const y = sq1 * std.math.cos(two_pi * r2);
    const z = sqr * std.math.sin(two_pi * r3);
    const w = sqr * std.math.cos(two_pi * r3);
    return .{
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y),
    };
}

fn applyRot(R: [9]f64, p: [3]f64) [3]f64 {
    return .{
        R[0] * p[0] + R[1] * p[1] + R[2] * p[2],
        R[3] * p[0] + R[4] * p[1] + R[5] * p[2],
        R[6] * p[0] + R[7] * p[1] + R[8] * p[2],
    };
}

const Case = struct {
    name: []const u8,
    points: [3][3]f64,
};

/// Rotation-invariance assertion with case + rotation-index labels.
/// Three checks against a rotated `rot_outcome`: it must be converged,
/// have small gap, and match canonical AR. Each failure prints
/// case={s} k={d} + offending value so a future regression is easy to
/// localize. The three failure-print branches are exercised by negative
/// tests below.
fn checkRotationInvariance(
    label: []const u8,
    k: u32,
    canon_ar: f64,
    rot_outcome: sphar.Outcome,
    tol: f64,
    ar_rel_tol: f64,
) !void {
    const c = switch (rot_outcome) {
        .converged => |c| c,
        else => {
            std.debug.print(
                "rotation status mismatch case={s} k={d}: expected converged, got {s}\n",
                .{ label, k, @tagName(rot_outcome) },
            );
            return error.RotationNotConverged;
        },
    };
    const gap_abs = @abs(c.cert.claimed_gap);
    if (gap_abs >= tol) {
        std.debug.print(
            "rotation gap exceeds tol case={s} k={d}: |gap|={e:.3} tol={e:.3}\n",
            .{ label, k, gap_abs, tol },
        );
        return error.RotationGapTooLarge;
    }
    const ar_delta = @abs(canon_ar - c.aspectRatio());
    const ar_tol = ar_rel_tol * canon_ar;
    if (ar_delta > ar_tol) {
        std.debug.print(
            "rotation AR drift case={s} k={d}: canon={d:.10} rot={d:.10} delta={e:.3} tol={e:.3}\n",
            .{ label, k, canon_ar, c.aspectRatio(), ar_delta, ar_tol },
        );
        return error.RotationArMismatch;
    }
}

test "checkRotationInvariance: status branch prints case+k label" {
    const fake: sphar.Outcome = .{ .infeasible = .{
        .cert = .{ .indices = &[_]u32{}, .lambdas = &[_]f64{} },
        .residual = 0,
        .allocator = std.testing.allocator,
    } };
    try std.testing.expectError(
        error.RotationNotConverged,
        checkRotationInvariance("test_label", 7, 1.0, fake, 1e-6, 1e-4),
    );
}

test "checkRotationInvariance: gap branch prints case+k label" {
    const fake: sphar.Outcome = .{ .converged = .{
        .Q = sphar.Mat3.zero,
        .sigma = .{ 0, 1, 1 },
        .outer_iters = 0,
        .newton_polish_failures = 0,
        .cert = .{ .indices = &[_]u32{}, .lambdas = &[_]f64{}, .claimed_gap = 1.0 },
        .allocator = std.testing.allocator,
    } };
    try std.testing.expectError(
        error.RotationGapTooLarge,
        checkRotationInvariance("test_label", 3, 1.0, fake, 1e-6, 1e-4),
    );
}

test "checkRotationInvariance: AR branch prints case+k label" {
    // sigma[2]/sigma[1] = 2.0, which deviates from canon_ar = 1.0
    // by far more than ar_rel_tol * canon_ar = 1e-4.
    const fake: sphar.Outcome = .{ .converged = .{
        .Q = sphar.Mat3.zero,
        .sigma = .{ 0, 1, 2 },
        .outer_iters = 0,
        .newton_polish_failures = 0,
        .cert = .{ .indices = &[_]u32{}, .lambdas = &[_]f64{}, .claimed_gap = 0 },
        .allocator = std.testing.allocator,
    } };
    try std.testing.expectError(
        error.RotationArMismatch,
        checkRotationInvariance("test_label", 0, 1.0, fake, 1e-6, 1e-4),
    );
}

test "extreme aspect ratio: three geometries, rotation-invariant" {
    const allocator = std.testing.allocator;
    const tol: f64 = 1e-6;
    const ar_rel_tol: f64 = 1e-4;
    const max_outer: u32 = 1000;
    const n_rotations: u32 = 20;

    const cases = [_]Case{
        // Near-antipodal arc. Bounded by the 174°-span convergence
        // plateau (pre-existing) at AR ≈ 80; eps=0.025 keeps comfortably
        // below that ceiling.
        .{ .name = "arc_174deg_AR63", .points = arcPoints(174.0, 0.025) },
        // Moderate arc. Higher achievable AR than the 174° case.
        .{ .name = "arc_90deg_AR143", .points = arcPoints(90.0, 0.005) },
        // Small tangent-plane patch. Near-planar regime: spherical
        // curvature barely contributes, so AR scales cleanly with L/eps
        // and the solver handles AR in the tens of thousands.
        .{ .name = "patch_AR17320", .points = patchPoints(0.1, 1e-5) },
    };

    for (cases) |case| {
        // Canonical (unrotated) solve establishes the reference AR.
        var canon_pts: [3][3]f64 = case.points;
        var canon_outcome = try sphar.solve(allocator, canon_pts[0..], .{ .gap_tol = tol, .coplanarity_tol = 1e-12, .max_outer = max_outer });
        defer canon_outcome.deinit();

        try std.testing.expect(std.meta.activeTag(canon_outcome) == .converged);
        const canon_c = canon_outcome.converged;
        try std.testing.expect(@abs(canon_c.cert.claimed_gap) < tol);
        const canon_ar = canon_c.aspectRatio();
        // Each case here has AR far above the existing-suite max of 1.09.
        try std.testing.expect(canon_ar > 50.0);

        // Same seed across all cases for reproducibility — if a particular
        // (case, k) pair fails, k identifies which rotation to reproduce.
        var rng_state: u64 = 0xCA7;
        var k: u32 = 0;
        while (k < n_rotations) : (k += 1) {
            const R = randomRotation(&rng_state);
            var rot_pts: [3][3]f64 = undefined;
            for (case.points, 0..) |p, i| {
                rot_pts[i] = applyRot(R, p);
            }
            var rot_outcome = try sphar.solve(allocator, rot_pts[0..], .{ .gap_tol = tol, .coplanarity_tol = 1e-12, .max_outer = max_outer });
            defer rot_outcome.deinit();

            // Labeled check: status + gap + AR with case + rotation
            // index in the failure diagnostic. Negative tests below
            // exercise each print branch.
            try checkRotationInvariance(case.name, k, canon_ar, rot_outcome, tol, ar_rel_tol);
        }
    }
}

test "coplanarity check cutoff is near the parameter's value" {
    // For arcPoints(90°, eps), the 2D centered-scatter eigenvalue ratio
    // works out to 4·det/trace² ≈ 10.67·eps². So the cutoff against
    // tol=1e-12 lands at eps ≈ 3e-7 (10.67·eps² = tol). We test inputs
    // sitting ~40× above and ~37× below that cutoff: tight enough to
    // catch a missing/changed 4× factor in the trigger condition, loose
    // enough that ordinary numerical drift won't flip the result.
    const allocator = std.testing.allocator;
    const tol: f64 = 1e-6;
    const max_outer: u32 = 100;
    const coplanarity_tol: f64 = 1e-12;

    // Above cutoff: ratio ≈ 4.3e-11 (43× above tol). Must not flag.
    {
        var pts: [3][3]f64 = arcPoints(90.0, 2.0e-6);
        var outcome = try sphar.solve(allocator, pts[0..], .{ .gap_tol = tol, .coplanarity_tol = coplanarity_tol, .max_outer = max_outer });
        defer outcome.deinit();
    }

    // Below cutoff: ratio ≈ 2.7e-14 (37× below tol). Must flag.
    {
        var pts: [3][3]f64 = arcPoints(90.0, 5.0e-8);
        try std.testing.expectError(
            sphar.InputError.CoplanarInput,
            sphar.solve(allocator, pts[0..], .{ .gap_tol = tol, .coplanarity_tol = coplanarity_tol, .max_outer = max_outer }),
        );
    }

    // Same near-degenerate input, but tol tightened by 1e8: ratio
    // (2.7e-14) is now ~2.7e6× above the tighter threshold (1e-20),
    // so must not flag. Confirms the parameter actually drives the
    // cutoff rather than the threshold being baked in.
    {
        var pts: [3][3]f64 = arcPoints(90.0, 5.0e-8);
        var outcome = try sphar.solve(allocator, pts[0..], .{ .gap_tol = tol, .coplanarity_tol = 1e-20, .max_outer = max_outer });
        defer outcome.deinit();
    }
}

test "coplanarity check flags great-circle inputs" {
    const allocator = std.testing.allocator;
    const tol: f64 = 1e-6;
    const max_outer: u32 = 100;
    const coplanarity_tol: f64 = 1e-12;

    // Three points exactly on a great circle (z=0): mathematically
    // coplanar with the origin. Without the check, the solver diverges
    // to NaN (verified at the time the check was added).
    const half = deg(170.0 / 2.0);
    var canon_pts: [3][3]f64 = .{
        .{ std.math.cos(-half), std.math.sin(-half), 0.0 },
        .{ 1.0, 0.0, 0.0 },
        .{ std.math.cos(half), std.math.sin(half), 0.0 },
    };
    try std.testing.expectError(
        sphar.InputError.CoplanarInput,
        sphar.solve(allocator, canon_pts[0..], .{ .gap_tol = tol, .coplanarity_tol = coplanarity_tol, .max_outer = max_outer }),
    );
    // `checkFeasibility` is no longer callable on a non-converged outcome:
    // its signature takes `Converged`, so a caller who hasn't switched
    // can't reach it. The "no apparent feasibility on a rejected input"
    // guarantee is now structural (compile-time), not a runtime assertion.

    // Rotational invariance: should still be flagged after rotation.
    var rng_state: u64 = 0xCA7;
    var k: u32 = 0;
    while (k < 10) : (k += 1) {
        const R = randomRotation(&rng_state);
        var rot_pts: [3][3]f64 = undefined;
        for (canon_pts, 0..) |p, i| rot_pts[i] = applyRot(R, p);
        try std.testing.expectError(
            sphar.InputError.CoplanarInput,
            sphar.solve(allocator, rot_pts[0..], .{ .gap_tol = tol, .coplanarity_tol = coplanarity_tol, .max_outer = max_outer }),
        );
    }

    // Sanity: same input does NOT flag when the check is disabled.
    // We don't care whether it converges — only that we get past the
    // coplanarity gate and produce an Outcome (which may itself be
    // any variant).
    var unchecked = try sphar.solve(allocator, canon_pts[0..], .{ .gap_tol = tol, .coplanarity_tol = -1, .max_outer = max_outer });
    defer unchecked.deinit();
}

test "solve rejects malformed inputs with typed errors" {
    const allocator = std.testing.allocator;
    const max_outer: u32 = 10;
    const valid_tol: f64 = 1e-6;
    const valid_cop: f64 = 1e-12;
    const ok_pts = [_][3]f64{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    };

    const opts: sphar.SolveOptions = .{
        .gap_tol = valid_tol,
        .coplanarity_tol = valid_cop,
        .max_outer = max_outer,
    };

    // n < 3 → InsufficientPoints. Three cases: empty, one point, two points.
    try std.testing.expectError(
        sphar.InputError.InsufficientPoints,
        sphar.solve(allocator, &[_][3]f64{}, opts),
    );
    try std.testing.expectError(
        sphar.InputError.InsufficientPoints,
        sphar.solve(allocator, ok_pts[0..1], opts),
    );
    try std.testing.expectError(
        sphar.InputError.InsufficientPoints,
        sphar.solve(allocator, ok_pts[0..2], opts),
    );

    // gap_tol: must be finite and positive.
    try std.testing.expectError(
        sphar.InputError.InvalidTolerance,
        sphar.solve(allocator, ok_pts[0..], .{ .gap_tol = -1.0 }),
    );
    try std.testing.expectError(
        sphar.InputError.InvalidTolerance,
        sphar.solve(allocator, ok_pts[0..], .{ .gap_tol = 0.0 }),
    );
    try std.testing.expectError(
        sphar.InputError.InvalidTolerance,
        sphar.solve(allocator, ok_pts[0..], .{ .gap_tol = std.math.nan(f64) }),
    );
    try std.testing.expectError(
        sphar.InputError.InvalidTolerance,
        sphar.solve(allocator, ok_pts[0..], .{ .gap_tol = std.math.inf(f64) }),
    );

    // coplanarity_tol: NaN is the only flagged value (≤ 0 documented as
    // disable; +inf documented as "reject everything").
    try std.testing.expectError(
        sphar.InputError.InvalidTolerance,
        sphar.solve(allocator, ok_pts[0..], .{ .coplanarity_tol = std.math.nan(f64) }),
    );
}

test "convexHull2d tie-break sort: points sharing an x-coordinate" {
    // Four hull corners plus five points sharing x = 0; cross-product
    // collapse leaves the hull at 4. The five tied-on-x points
    // exercise the y-fallback branch in halfspace.HullCtx.lessThan.
    const allocator = std.testing.allocator;
    const convexHull2d = halfspace.convexHull2d;
    const P = [_][2]f64{
        .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, 1 },
        .{ 0, -0.5 }, .{ 0, 0.5 }, .{ 0, 0 }, .{ 0, -0.25 }, .{ 0, 0.25 },
    };
    const hull_idx = try allocator.alloc(u32, P.len);
    defer allocator.free(hull_idx);
    const nh = try convexHull2d(allocator, &P, hull_idx);
    try std.testing.expectEqual(@as(u32, 4), nh);
}

// FailingAllocator-based tests targeting the errdefer cleanup paths
// in buildPrimalCert (converged path) and buildFarkasCert (infeasible
// path). On both paths the LAST parent-allocator call is the cert's
// `lambdas = alloc(f64, k)` — the second of two back-to-back allocs
// in the respective cert builder. Failing that last alloc exercises
// the `errdefer allocator.free(indices)` immediately above it.
//
// `lastAllocFailIndex` runs solve once under a non-failing
// FailingAllocator to count parent-allocator calls, then returns
// `total - 1` for the test to use as fail_index. Robust to
// ArenaAllocator page-pull / halfspaceCheck scratch / future build
// changes that would shift a hard-coded fail_index off-target.

fn runWithFailIndex(fail_index: usize, X: []const [3]f64, opts: sphar.SolveOptions) !sphar.Outcome {
    var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
    return sphar.solve(fa.allocator(), X, opts);
}

fn lastAllocFailIndex(X: []const [3]f64, opts: sphar.SolveOptions) !usize {
    var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var outcome = try sphar.solve(fa.allocator(), X, opts);
    outcome.deinit();
    return fa.alloc_index - 1;
}

test "OOM in the last cert alloc hits buildPrimalCert's indices errdefer" {
    const half = std.math.pi / 4.0;
    var pts: [3][3]f64 = .{
        .{ @cos(-half), @sin(-half), 0.1 },
        .{ 1.0, 0.0, -0.1 },
        .{ @cos(half), @sin(half), 0.1 },
    };
    for (&pts) |*p| {
        const n = @sqrt(p.*[0] * p.*[0] + p.*[1] * p.*[1] + p.*[2] * p.*[2]);
        p.*[0] /= n;
        p.*[1] /= n;
        p.*[2] /= n;
    }
    const fi = try lastAllocFailIndex(pts[0..], .{});
    try std.testing.expectError(error.OutOfMemory, runWithFailIndex(fi, pts[0..], .{}));
}

test "OOM in the last cert alloc hits buildFarkasCert's indices errdefer" {
    // Three equiangular equatorial points (120° apart) — the convex
    // hull contains the origin, so no hemisphere fits all three;
    // solve takes the .infeasible branch and calls buildFarkasCert.
    const c120 = -0.5;
    const s120 = @sqrt(3.0) / 2.0;
    const pts = [_][3]f64{
        .{ 1.0, 0.0, 0.0 },
        .{ c120, s120, 0.0 },
        .{ c120, -s120, 0.0 },
    };
    const opts: sphar.SolveOptions = .{ .coplanarity_tol = -1 };
    const fi = try lastAllocFailIndex(&pts, opts);
    try std.testing.expectError(error.OutOfMemory, runWithFailIndex(fi, &pts, opts));
}



test "acceptBUpdate fallback: all backtracks fail when a point sits below FEAS_MARGIN" {
    // halfspaceCheck only guarantees `b·xᵢ > 0` strictly — not
    // `≥ FEAS_MARGIN`. So a single point at `b·x = ε < FEAS_MARGIN`
    // can survive into the outer loop. From there, every backtracked
    // b-trial = normalize(b + α·dQc) still has `b_trial·x ≈ ε` for
    // small enough α, so all 30 backtracks fail their feasibility
    // check and acceptBUpdate falls through to the "keep (b, Q)
    // unchanged + re-project" tail. This test crafts that exact
    // setup directly to hit lines 174-176 of src/skar.zig.
    const acceptBUpdate = skar.acceptBUpdate;
    const b = sphar.Vec3{ .m = .{ 1, 0, 0 } };
    const Q = b.orthoBasis();
    // x dotted with b equals 1e-9 — below FEAS_MARGIN = 1e-8.
    const x_eps: f64 = 1e-9;
    const x = sphar.Vec3{ .m = .{ x_eps, @sqrt(1.0 - x_eps * x_eps), 0 } };
    const Xw = [_]sphar.Vec3{x};
    // u sends Q·u in the -y direction with b=x̂, Q=(ŷ, ẑ); since x has
    // y ≈ 1, this pushes b_trial · x further below FEAS_MARGIN — every
    // backtrack stays infeasible.
    const u = linalg.Vec2{ .m = .{ -1, 0 } };
    var P_buf: [1][2]f64 = undefined;
    var Ps: [1][2]f64 = undefined;
    const step = acceptBUpdate(&Xw, b, Q, u, 1.0, &P_buf, &Ps);
    // Fallback returns the input (b, Q) unchanged and a freshly
    // recomputed s_scale from rescaleP on the re-projected P_buf.
    // Assert all three so a future change that drops the rescaleP
    // call (leaving s_scale undefined) or perturbs Q is caught.
    try std.testing.expectEqual(b.m[0], step.b.m[0]);
    try std.testing.expectEqual(b.m[1], step.b.m[1]);
    try std.testing.expectEqual(b.m[2], step.b.m[2]);
    try std.testing.expectEqual(Q.e1.m[0], step.Q.e1.m[0]);
    try std.testing.expectEqual(Q.e1.m[1], step.Q.e1.m[1]);
    try std.testing.expectEqual(Q.e1.m[2], step.Q.e1.m[2]);
    try std.testing.expectEqual(Q.e2.m[0], step.Q.e2.m[0]);
    try std.testing.expectEqual(Q.e2.m[1], step.Q.e2.m[1]);
    try std.testing.expectEqual(Q.e2.m[2], step.Q.e2.m[2]);
    try std.testing.expect(std.math.isFinite(step.s_scale));
    try std.testing.expect(step.s_scale > 0);
}
