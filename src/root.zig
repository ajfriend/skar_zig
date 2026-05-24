//! Public entry point for the `skar` package.
//!
//! `skar` solves the spherical aspect-ratio problem: given a point set on
//! the unit sphere, find the tightest ellipsoidal cone enclosing it. See
//! `skar.zig` for the algorithm.

const skar = @import("skar.zig");

pub const Vec3 = skar.Vec3;
pub const Vec2 = skar.Vec2;
pub const Mat2 = skar.Mat2;
pub const Mat3x2 = skar.Mat3x2;
pub const Mat3 = skar.Mat3;
pub const Chol3 = skar.Chol3;
pub const Eig2 = skar.Eig2;

pub const Status = skar.Status;
pub const Cert = skar.Cert;
pub const Info = skar.Info;
pub const SolveError = skar.SolveError;
pub const InputError = skar.InputError;

pub const eig2 = skar.eig2;
pub const checkFeasibility = skar.checkFeasibility;
pub const solve = skar.solve;
