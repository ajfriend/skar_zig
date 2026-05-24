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
//! Known limitations surfaced while writing these tests (both
//! verified pre-existing on HEAD~1, identical numbers — neither is a
//! Cholesky regression):
//!  - 3 points exactly coplanar with the origin (great-circle inputs,
//!    z = 0 in canonical orientation) make the solver diverge to NaN
//!    even at modest arc spans. All three cases here include a
//!    z-perturbation specifically to skirt this.
//!  - Near-antipodal (174°) cases hit a convergence plateau around
//!    AR ≈ 80–100: some rotations plateau at gap ~1e-4 and don't
//!    tighten in 1000 outer iterations. Each chosen `eps` here is
//!    one notch back from the empirical edge so future numeric
//!    drift doesn't make the test flaky.

const std = @import("std");
const sphar = @import("skar");

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
        var canon_info = try sphar.solve(allocator, canon_pts[0..], tol, max_outer);
        defer canon_info.deinit();

        try std.testing.expectEqual(sphar.Status.converged, canon_info.status);
        try std.testing.expect(@abs(canon_info.cert.claimed_gap) < tol);
        const canon_ar = canon_info.aspectRatio();
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
            var rot_info = try sphar.solve(allocator, rot_pts[0..], tol, max_outer);
            defer rot_info.deinit();

            const ar_delta = @abs(rot_info.aspectRatio() - canon_ar);
            const ar_tol_abs = ar_rel_tol * canon_ar;
            const status_ok = rot_info.status == sphar.Status.converged;
            const gap_ok = @abs(rot_info.cert.claimed_gap) < tol;
            const ar_ok = ar_delta < ar_tol_abs;

            if (!status_ok or !gap_ok or !ar_ok) {
                std.debug.print(
                    "FAIL case={s} k={d} status={any} gap={e} canon_ar={d:.6} rot_ar={d:.6} ar_delta={e}\n",
                    .{ case.name, k, rot_info.status, rot_info.cert.claimed_gap, canon_ar, rot_info.aspectRatio(), ar_delta },
                );
                return error.RotationInvarianceFailure;
            }
        }
    }
}
