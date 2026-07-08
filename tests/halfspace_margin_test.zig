//! Regression for the false-infeasibility proof on small-margin
//! feasible inputs (2026-07-08 pre-release review, finding 1).
//!
//! For a FEASIBLE input, halfspaceCheck's FW iterate satisfies
//! ‖z‖ ≥ margin, so the z-exhaustion break can only fire when the
//! margin is below the floor — but the old floor (‖z‖ < 1e-6) sat two
//! orders above the f64 witness limit (xᵢ·z* ≥ m² must clear ~1e-16
//! dot noise ⇒ m ≳ 1e-8), so strictly feasible inputs with margins in
//! (1e-8, 1e-6) were returned as "proven infeasible" — measured
//! 200/200 at margin 1e-7 before the fix. The floor now matches the
//! witness limit (tol.FW_Z_EXHAUSTED = 1e-16 on z·z).

const std = @import("std");
const skar = @import("../src/root.zig");

/// n unit points ringing the equator, all lifted to x·ẑ ≥ margin:
/// strictly feasible with hemisphere margin ≈ margin (the verifier's
/// reproduction recipe). Deterministic angles, no PRNG.
fn marginRing(comptime n: usize, margin: f64) [n][3]f64 {
    var pts: [n][3]f64 = undefined;
    const s = @sqrt(1.0 - margin * margin);
    for (0..n) |i| {
        const phi = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
        pts[i] = .{ s * @cos(phi), s * @sin(phi), margin };
    }
    return pts;
}

test "small-margin feasible rings are not declared infeasible (margins 1e-7..1e-5)" {
    const allocator = std.testing.allocator;
    inline for (.{ 1e-5, 1e-6, 1e-7 }) |margin| {
        const pts = marginRing(24, margin);
        var o = try skar.solve(allocator, &pts, .{ .gap_tol = 1e-3 });
        defer o.deinit();
        // The one forbidden answer is a false infeasibility proof.
        // (Converged is expected; DNC would be acceptable honesty on a
        // near-degenerate cone, and would still fail loudly here so we
        // learn about the behavior change.)
        try std.testing.expect(std.meta.activeTag(o) != .infeasible);
    }
}

test "genuinely infeasible input still proves infeasibility with a sharp witness" {
    // Companion guard: lowering the exhaustion floor must not weaken
    // real infeasibility detection. Antipodal pair + orthogonal point:
    // 0 is in the hull, no hemisphere works.
    const allocator = std.testing.allocator;
    const pts = [_][3]f64{
        .{ 1, 0, 0 },
        .{ -1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, -1, 0 },
    };
    var o = try skar.solve(allocator, &pts, .{});
    defer o.deinit();
    try std.testing.expect(std.meta.activeTag(o) == .infeasible);
    try std.testing.expect(o.infeasible.residual < 1e-8);
}
