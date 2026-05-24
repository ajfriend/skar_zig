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

/// Internal modules surfaced for in-tree tests only. Not for external
/// callers — these aren't covered by the public-API stability promise
/// and may be renamed or removed without notice. Used to exercise
/// internal helpers (e.g. `convexHull2d` tie-break, `halfspaceCheck`
/// early-exits) that aren't reachable through `solve` for all inputs.
pub const _internal = struct {
    pub const halfspace = @import("halfspace.zig");
    pub const newton = @import("newton.zig");
    pub const config = @import("config.zig");
};
