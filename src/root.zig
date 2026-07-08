//! Public entry point for the `skar` package.
//!
//! `skar` solves the spherical aspect-ratio problem: given a point set on
//! the unit sphere, find the tightest ellipsoidal cone enclosing it.
//!
//! Where to look:
//!   - `src/api.zig` — public API surface: types, methods,
//!     `checkFeasibility`, and the errors-vs-outcome rationale. Read
//!     this file end-to-end to learn what the library exposes.
//!   - `src/skar.zig` — algorithm implementation; defines `solve`.
//!
//! This file is just a re-export shim so consumers can write
//! `@import("skar")` and reach everything from one namespace.

const linalg = @import("linalg.zig");
const api = @import("api.zig");
const skar = @import("skar.zig");

// Linear-algebra types surfaced by the public solver API:
//   Vec3 — returned by `Converged.b()`.
//   Mat3 — `Converged.Q`, returned by `Converged.A()`.
// Other linalg primitives (Vec2, Mat2, Mat3x2, Chol3, Eig2, eig2) are
// internal — see `src/linalg.zig`.
pub const Vec3 = linalg.Vec3;
pub const Mat3 = linalg.Mat3;

// Public API (`src/api.zig`).
pub const Outcome = api.Outcome;
pub const Converged = api.Converged;
pub const Infeasible = api.Infeasible;
pub const DidNotConverge = api.DidNotConverge;
pub const Cert = api.Cert;
pub const SolveError = api.SolveError;
pub const InputError = api.InputError;
pub const SolveOptions = api.SolveOptions;
pub const Method = api.Method;
pub const Diagnostics = api.Diagnostics;
pub const checkFeasibility = api.checkFeasibility;

// Solver entry point (`src/skar.zig`).
pub const solve = skar.solve;
