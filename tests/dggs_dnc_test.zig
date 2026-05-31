//! Regression tests for DGGS cells at the finest resolution, where the
//! duality-gap certificate hits a *fundamental f64 floor*.
//!
//! Source: scripts/dggs/aspect.zig, N=10_000 random cells per system at
//! finest resolution (H3 r15, S2 L30, A5 r30). At the strict default
//! `gap_tol = 1e-6`, ~22% of S2 L30 and ~47% of A5 r30 cells return
//! `.did_not_converge`. The two cells below are the first DNC encountered
//! per system on seed=0xC0FFEE.
//!
//! These are NOT a bug. The cells are sub-meter scatters at an O(1) point
//! on the unit sphere, so κ(A) ~ σ_max ~ 1e9 and the duality gap has an
//! f64 precision floor at O(κ·ε): the optimal cone axis sits a *sub-ulp*
//! rotation away from the best representable `b`, so the iterate cannot be
//! driven closer in f64 and the gap genuinely cannot reach 1e-6. Reporting
//! `.did_not_converge` at `gap_tol = 1e-6` is therefore the *correct*
//! behaviour — the solver honestly declines to certify a bound it cannot
//! achieve.
//!
//! What these tests pin: that the solver *does* converge — with an
//! accurate aspect ratio — once asked for a tolerance f64 can actually
//! deliver on these inputs. The observed gap floor across the finest
//! resolution is ~3.4e-4 (A5 r30, worst), so `gap_tol = 1e-3` certifies
//! the whole class with headroom. The AR itself is input-precision-limited
//! (~7 significant digits) and is accurate regardless of the gap — that's
//! the quantity callers actually want.

const std = @import("std");
const skar = @import("../src/root.zig");

// Tolerance that f64 can certify for finest-resolution DGGS cells (the
// gap floor is ~3.4e-4 at A5 r30; 1e-3 covers the class with headroom).
const DGGS_GAP_TOL: f64 = 1e-3;

// Pentagonal A5 r30 cell (id 2a08d74e8e79123c): five vertices agreeing to
// ~9 decimal places — a near-degenerate scatter, but not coplanar enough
// to trip skar's coplanarity guard.
const A5_CELL = [_][3]f64{
    .{ -8.76368008991394400e-1, 3.45295754150762360e-1, 3.35782600773052830e-1 },
    .{ -8.76368008698072600e-1, 3.45295754812974860e-1, 3.35782600857627600e-1 },
    .{ -8.76368008522131700e-1, 3.45295755483736640e-1, 3.35782600627055400e-1 },
    .{ -8.76368008823817700e-1, 3.45295755231014470e-1, 3.35782600099559000e-1 },
    .{ -8.76368009047065800e-1, 3.45295754541700400e-1, 3.35782600225741100e-1 },
};

// S2 L30 leaf cell (id 332c258c3f285f93): four vertices, same scale as A5.
const S2_CELL = [_][3]f64{
    .{ -6.84434006983608300e-1, 7.11477104991097700e-1, 1.59218149586812550e-1 },
    .{ -6.84434007909358400e-1, 7.11477104143007500e-1, 1.59218149397022360e-1 },
    .{ -6.84434007784890200e-1, 7.11477104013621300e-1, 1.59218150510246930e-1 },
    .{ -6.84434006859140100e-1, 7.11477104861711600e-1, 1.59218150700037110e-1 },
};

test "A5 r30 cell certifies at an f64-achievable tolerance (cell 2a08d74e8e79123c)" {
    // DNCs at gap_tol=1e-6 (gap ~2.6e-5, an f64 floor); converges at 1e-3
    // with an accurate aspect ratio.
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &A5_CELL, .{ .gap_tol = DGGS_GAP_TOL });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    // AR is input-precision-limited (~7 digits); pin loosely as a
    // correctness guard, not a bit-exact pin.
    try std.testing.expectApproxEqAbs(2.21164606, outcome.converged.aspectRatio(), 1e-4);
}

test "S2 L30 cell certifies at an f64-achievable tolerance (cell 332c258c3f285f93)" {
    // DNCs at gap_tol=1e-6 (gap ~2.9e-6); converges at 1e-3.
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &S2_CELL, .{ .gap_tol = DGGS_GAP_TOL });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expectApproxEqAbs(1.21362116, outcome.converged.aspectRatio(), 1e-4);
}

test "A5/S2 finest cells correctly DNC at the strict 1e-6 default" {
    // The companion assertion: at the strict default the solver honestly
    // declines to certify (the gap floor is above 1e-6). This is the key
    // regression guard — it pins that a future change can't silently make
    // these cells "converge" at 1e-6 via a non-certificate (the exact
    // failure mode this investigation hit). Both named cells are asserted,
    // symmetric with the 1e-3 convergence tests above.
    const allocator = std.testing.allocator;

    var oa = try skar.solve(allocator, &A5_CELL, .{}); // default gap_tol = 1e-6
    defer oa.deinit();
    try std.testing.expect(std.meta.activeTag(oa) == .did_not_converge);

    var os = try skar.solve(allocator, &S2_CELL, .{});
    defer os.deinit();
    try std.testing.expect(std.meta.activeTag(os) == .did_not_converge);
}
