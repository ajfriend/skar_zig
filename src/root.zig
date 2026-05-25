//! Public entry point for the `skar` package.
//!
//! `skar` solves the spherical aspect-ratio problem: given a point set on
//! the unit sphere, find the tightest ellipsoidal cone enclosing it. See
//! `skar.zig` for the algorithm.

const linalg = @import("linalg.zig");
const skar = @import("skar.zig");

// Linear-algebra types surfaced by the public solver API:
//   Vec3 — returned by `info.b()`.
//   Mat3 — `info.Q`, returned by `info.A()`.
// Other linalg primitives (Vec2, Mat2, Mat3x2, Chol3, Eig2, eig2) are
// internal — see `src/linalg.zig`.
pub const Vec3 = linalg.Vec3;
pub const Mat3 = linalg.Mat3;

// Solver API (`src/skar.zig`).
pub const Status = skar.Status;
pub const Cert = skar.Cert;
pub const Info = skar.Info;
pub const SolveError = skar.SolveError;
pub const InputError = skar.InputError;
pub const SolveOptions = skar.SolveOptions;

pub const checkFeasibility = skar.checkFeasibility;
pub const solve = skar.solve;

// Test discovery: `zig build test` uses this file as the test root,
// so the comptime import below pulls every test in `src/tests/` into
// the test binary. The body of this `test` block only compiles when
// Zig is building tests (the `test` declaration is skipped in
// non-test builds), so the library artifact itself sees no test
// dependencies.
test {
    _ = @import("tests/all.zig");
}
