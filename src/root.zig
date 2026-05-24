//! Public entry point for the `skar` package.
//!
//! `skar` solves the spherical aspect-ratio problem: given a point set on
//! the unit sphere, find the tightest ellipsoidal cone enclosing it. See
//! `skar.zig` for the algorithm.

const linalg = @import("linalg.zig");
const skar = @import("skar.zig");

// Linear algebra primitives (`src/linalg.zig`).
pub const Vec3 = linalg.Vec3;
pub const Vec2 = linalg.Vec2;
pub const Mat2 = linalg.Mat2;
pub const Mat3x2 = linalg.Mat3x2;
pub const Mat3 = linalg.Mat3;
pub const Chol3 = linalg.Chol3;
pub const Eig2 = linalg.Eig2;
pub const eig2 = linalg.eig2;

// Solver API (`src/skar.zig`).
pub const Status = skar.Status;
pub const Cert = skar.Cert;
pub const Info = skar.Info;
pub const SolveError = skar.SolveError;
pub const InputError = skar.InputError;
pub const SolveOptions = skar.SolveOptions;

pub const checkFeasibility = skar.checkFeasibility;
pub const solve = skar.solve;

/// ⚠️ INTERNAL — DO NOT USE FROM OUTSIDE THE PACKAGE.
///
/// Zig has no module-private `pub`, so this surface is technically
/// reachable from any consumer. **It is NOT part of the public API**:
/// no semver protection, no stability guarantee, no deprecation
/// window. Anything in here may be renamed, retyped, or deleted in
/// any commit without notice.
///
/// Why it exists: to give in-tree tests reach into helpers
/// (`convexHull2d` for the tie-break sort, `acceptBUpdate` for the
/// MAX_BACKTRACKS fallback) that aren't reachable through `solve`
/// for all inputs. The 100% coverage gate would otherwise force
/// either contrived inputs or dead-branch exemption — both worse
/// than this explicit "tests-only" surface.
///
/// If you find yourself wanting any of these from outside the
/// package, file an issue requesting promotion to the public API
/// instead — opening a stable surface is a deliberate decision, not
/// something that should happen by accident.
pub const _internal_for_tests = struct {
    pub const halfspace = @import("halfspace.zig");
    pub const newton = @import("newton.zig");
    pub const config = @import("config.zig");
    pub const acceptBUpdate = skar.acceptBUpdate;
    pub const BStep = skar.BStep;
};
