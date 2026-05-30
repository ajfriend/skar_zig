//! Regression tests for DGGS cells that fail to converge under default
//! solver options. Source: scripts/dggs/aspect.zig, N=10_000 random
//! cells per system at finest resolution (H3 r15, S2 L30, A5 r30).
//!
//! At default `gap_tol = 1e-6` and `max_outer = 100`, ~47% of A5 r30
//! cells and ~22% of S2 L30 cells return `.did_not_converge`. The
//! vertices below are the *first* DNC encountered per system on
//! seed=0xC0FFEE; each test asserts the solver converges, so a failing
//! test is the bug to debug. Once fixed, leave the test in place —
//! these are real, in-the-wild cells, not synthetic adversarial input.
//!
//! Failure mode characterized in docs/dggs-dnc-investigation.md: the
//! gap formula `‖w‖ − 3 − log_det_M` has a precision floor at
//! O(κ(A) · ε_machine), which for tiny cells (σ_max(A) ~ 1e9) sits at
//! ~10^-6, just above default `gap_tol`.

const std = @import("std");
const skar = @import("../src/root.zig");

test "A5 r30 cell that DNCs at defaults (cell 2a08d74e8e79123c)" {
    // Pentagonal A5 cell at finest resolution. All five vertices agree
    // to ~9 decimal places — a near-degenerate scatter on the unit
    // sphere, but not coplanar enough to trip skar's coplanarity guard.
    const allocator = std.testing.allocator;
    const pts = [_][3]f64{
        .{ -8.76368008991394400e-1, 3.45295754150762360e-1, 3.35782600773052830e-1 },
        .{ -8.76368008698072600e-1, 3.45295754812974860e-1, 3.35782600857627600e-1 },
        .{ -8.76368008522131700e-1, 3.45295755483736640e-1, 3.35782600627055400e-1 },
        .{ -8.76368008823817700e-1, 3.45295755231014470e-1, 3.35782600099559000e-1 },
        .{ -8.76368009047065800e-1, 3.45295754541700400e-1, 3.35782600225741100e-1 },
    };

    var outcome = try skar.solve(allocator, &pts, .{});
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
}

test "S2 L30 cell that DNCs at defaults (cell 332c258c3f285f93)" {
    // S2 leaf cell at level 30. Four vertices agreeing to ~9 decimal
    // places; same scale as the A5 case above.
    const allocator = std.testing.allocator;
    const pts = [_][3]f64{
        .{ -6.84434006983608300e-1, 7.11477104991097700e-1, 1.59218149586812550e-1 },
        .{ -6.84434007909358400e-1, 7.11477104143007500e-1, 1.59218149397022360e-1 },
        .{ -6.84434007784890200e-1, 7.11477104013621300e-1, 1.59218150510246930e-1 },
        .{ -6.84434006859140100e-1, 7.11477104861711600e-1, 1.59218150700037110e-1 },
    };

    var outcome = try skar.solve(allocator, &pts, .{});
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
}
