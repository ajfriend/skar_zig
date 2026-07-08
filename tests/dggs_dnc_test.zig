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
const cases = @import("cases"); // bundled case manifest (for the H3 r15 fixture)

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

// H3 r9 cell 899f4d0cd47ffff — a near-circular hexagon (AR ~1.0195). Unlike
// the finest-resolution S2/A5 cells above, this is NOT an f64 floor: it's a
// mid-resolution cell whose D-optimal design is degenerate (alternating
// vertices sit on the enclosing ellipse with true dual weight ~1e-7). It used
// to DNC at the strict 1e-6 default because the old `ACTIVE_THRESH = 1e-6`
// dropped those binding constraints, flooring the gap at ~1.7e-6. With
// `ACTIVE_THRESH = 1e-12` it converges at 1e-6 (gap ~1.5e-7). See the
// `ACTIVE_THRESH` doc-comment in src/config.zig for the full mechanism.
const H3_R9_CELL = [_][3]f64{
    .{ -0.8586175701975843, 0.28761239723198995, -0.42432885490673883 },
    .{ -0.8586271933201559, 0.28762660191847433, -0.42429975342908594 },
    .{ -0.8586197375801148, 0.2876590246563569, -0.42429286085392487 },
    .{ -0.8586026585975493, 0.2876772430738286, -0.42431506980858175 },
    .{ -0.8585930353179254, 0.2876630384891841, -0.42434417162336724 },
    .{ -0.8586004911779209, 0.28763061538522544, -0.42435106414636176 },
};

// Two more cells from the same r7–r10 gap-floor band (worst-gap DNC found in
// an 8k-cell-per-resolution survey, seed 0xC0FFEE, under the old
// ACTIVE_THRESH = 1e-6). They broaden the regression beyond the single r9
// cell: r8 floored at gap 2.18e-6, r10 at 2.27e-6 before the fix. Both
// converge at the strict 1e-6 default now (r8 ~4.3e-7, r10 ~9.5e-7).
const H3_R8_CELL = [_][3]f64{
    .{ -0.43574038542520366, -0.7556981921521153, 0.48892796901744084 },
    .{ -0.43566886076762934, -0.7557046846412897, 0.48898166976753316 },
    .{ -0.4356641141281686, -0.7556581174519464, 0.4890578587344225 },
    .{ -0.43573089096666295, -0.755605059770775, 0.4890803454507262 },
    .{ -0.4358024132308145, -0.7555985683909756, 0.48902664556004144 },
    .{ -0.4358071610498452, -0.7556451335830144, 0.4889504580936421 },
};
const H3_R10_CELL = [_][3]f64{
    .{ 0.7971117446546273, -0.5749409727169088, -0.18454198553443296 },
    .{ 0.7971187643188198, -0.5749339069211487, -0.18453367780224242 },
    .{ 0.7971193129396517, -0.5749371412121672, -0.18452123073889923 },
    .{ 0.7971128418782646, -0.5749474413348267, -0.18451709139072378 },
    .{ 0.7971058221898187, -0.5749545071571371, -0.18452539914815702 },
    .{ 0.7971052735870137, -0.5749512728302374, -0.18453784622852284 },
};

// A second A5 r30 cell (id bac84da19e50dc29) — the rare case that needs more
// outer iterations (4, vs 2-3 for the bulk of A5). Used by the canary below.
const A5_CELL_4ITER = [_][3]f64{
    .{ 4.07328516791610530e-1, -5.01584867128560500e-1, 7.63214321456280700e-1 },
    .{ 4.07328517328366060e-1, -5.01584866612700500e-1, 7.63214321508837000e-1 },
    .{ 4.07328516822916650e-1, -5.01584866460808700e-1, 7.63214321878419400e-1 },
    .{ 4.07328516196555300e-1, -5.01584866834068200e-1, 7.63214321967403000e-1 },
    .{ 4.07328516148072970e-1, -5.01584867409401800e-1, 7.63214321615168600e-1 },
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

test "A5/S2 finest cells: honest decline at the strict 1e-6 tolerance" {
    // The companion assertion: at the strict 1e-6 tolerance the solver
    // honestly declines to certify when the f64 gap floor sits above it.
    // This is the key regression guard — it pins that a change can't
    // silently make floor cells "converge" at 1e-6 via a non-certificate
    // (the exact failure mode this investigation hit).
    //
    // Which cells sit above vs below the floor is PATH-DEPENDENT at
    // noise level (the certificate is noisier than the answer near the
    // floor): under .alternating both cells DNC; under .trust (the
    // default since the flip) this A5 cell still DNCs (gap ~3.3e-4)
    // while the S2 cell legitimately certifies (gap ~4.9e-7 ≤ 1e-6 — a
    // real certificate, not the guarded failure mode). Pin all three
    // facts.
    const allocator = std.testing.allocator;

    inline for (.{ &A5_CELL, &S2_CELL }) |cell| {
        var o = try skar.solve(allocator, cell, .{ .method = .alternating }); // default gap_tol = 1e-6
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .did_not_converge);
    }

    var oa = try skar.solve(allocator, &A5_CELL, .{}); // default method (.trust), gap_tol = 1e-6
    defer oa.deinit();
    try std.testing.expect(std.meta.activeTag(oa) == .did_not_converge);

    var os = try skar.solve(allocator, &S2_CELL, .{});
    defer os.deinit();
    try std.testing.expect(std.meta.activeTag(os) == .converged);
    try std.testing.expect(@abs(os.converged.gap) <= 1e-6);
}

test "H3 r9 near-circular hexagon converges at the strict 1e-6 default (cell 899f4d0cd47ffff)" {
    // Regression for the r7–r10 gap-floor bug: this cell DNC'd at 1e-6 under
    // the old ACTIVE_THRESH = 1e-6 (gap stalled ~1.7e-6) because Newton polish
    // zeroed the small-weight binding constraints. With ACTIVE_THRESH = 1e-12
    // those constraints survive and the gap reaches ~1.5e-7. This is the
    // counterpart to the S2/A5 DNC guard below: a *well-conditioned* DGGS cell
    // must honour the strict default, while the genuine f64-floor cells must
    // not be forced to.
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &H3_R9_CELL, .{}); // default gap_tol = 1e-6
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expect(outcome.converged.gap <= 1e-6);
    // AR is input-precision-limited (~7 digits); loose correctness guard.
    try std.testing.expectApproxEqAbs(1.0195139, outcome.converged.aspectRatio(), 1e-4);
}

test "H3 r8 and r10 band cells converge at the strict 1e-6 default" {
    // Breadth across the r7–r10 band (the r9 cell above is the headline; these
    // pin the band edges). Both DNC'd under the old threshold; both must now
    // certify the strict default.
    const allocator = std.testing.allocator;
    inline for (.{ &H3_R8_CELL, &H3_R10_CELL }) |cell| {
        var o = try skar.solve(allocator, cell, .{}); // default gap_tol = 1e-6
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .converged);
        try std.testing.expect(o.converged.gap <= 1e-6);
    }
}

test "H3 r9 cell: no artificial gap floor between gap_tol and ACTIVE_THRESH" {
    // Guards the fix's design rule (ACTIVE_THRESH = 1e-12 ≪ gap_tol). The bug
    // was a scale collision: the certificate cutoff equalled the certified
    // tolerance, so the active-set drop floored the gap right at gap_tol. With
    // the cutoff six orders below, this well-conditioned cell must keep
    // certifying as gap_tol tightens toward (but stays well above) 1e-12 — and
    // the achieved gap must keep shrinking, not stall at a fixed floor. If a
    // future change reintroduces a coarse cutoff, the tighter tolerances here
    // will DNC and trip this test.
    const allocator = std.testing.allocator;
    const tols = [_]f64{ 1e-6, 1e-7, 1e-8, 1e-9 };
    var prev_gap: f64 = 1e30;
    for (tols) |t| {
        var o = try skar.solve(allocator, &H3_R9_CELL, .{ .gap_tol = t });
        defer o.deinit();
        try std.testing.expect(std.meta.activeTag(o) == .converged);
        try std.testing.expect(o.converged.gap <= t);
        // Monotone: a tighter request yields a tighter (or equal) certificate,
        // never a higher floor.
        try std.testing.expect(o.converged.gap <= prev_gap);
        prev_gap = o.converged.gap;
    }
}

// ── Alternating-path CANARIES (informational, NOT hard requirements) ─────
//
// These pin the exact/near-exact outer-iteration count on a few cells. They
// are deliberately brittle: their job is to *flag the developer* when a
// solver/algorithm change shifts how many iterations these inputs take —
// the same spirit as a bit-exact snapshot.
//
// They are also the project's PERFORMANCE-regression guard for the hot/common
// path: small DGGS cells (4–10 points) solve in µs, where wall-time bench
// (`examples/bench.zig`, esp. its `TOTAL`) can't resolve a regression and
// over-indexes on the large synthetic cases. Iteration count is the
// deterministic signal — these canaries are how a small-cell slowdown gets
// caught (see CLAUDE.md "Performance & regression monitoring").
//
// A trip here is NOT a failure to
// force-fix; it's a signal to (a) understand what changed and (b) update the
// expected value if the change is intended. The b-iterate reaches its fixed
// point almost immediately, so at an achievable tolerance these cells settle
// in very few iterations (the work per cell is tiny); the counts below are
// from the N=10_000 survey. Tolerance distinction: H3 r15 hits no f64 gap
// floor so it converges at the strict 1e-6 default; S2/A5 use the achievable
// DGGS_GAP_TOL (they DNC at 1e-6, per the tests above).
//
// If you are updating these expected counts: that means solver behaviour
// changed — call it out explicitly rather than quietly bumping the number.

test "CANARY(alternating): H3 r15 cell converges in 1 outer iteration (strict default)" {
    const allocator = std.testing.allocator;
    const h3 = cases.byName("h3_r15_equator").?.points;
    var outcome = try skar.solve(allocator, h3, .{ .method = .alternating }); // default gap_tol = 1e-6
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expectEqual(@as(u32, 1), outcome.converged.diag.alternating.outer_iters);
}

test "CANARY(alternating): H3 r9 near-circular hexagon converges in 2 outer iterations (strict default)" {
    // Sister to the r15 canary: the well-conditioned mid-resolution band now
    // certifies the strict 1e-6 default. The degenerate design takes one extra
    // outer iteration to refine the small-weight binding constraints (2 vs r15's
    // 1). If this count shifts, solver behaviour changed — flag it, don't bump.
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &H3_R9_CELL, .{ .method = .alternating }); // default gap_tol = 1e-6
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expectEqual(@as(u32, 2), outcome.converged.diag.alternating.outer_iters);
}

test "CANARY(alternating): S2 L30 cell converges in 1 outer iteration" {
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &S2_CELL, .{ .gap_tol = DGGS_GAP_TOL, .method = .alternating });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expectEqual(@as(u32, 1), outcome.converged.diag.alternating.outer_iters);
}

test "CANARY(alternating): a common A5 r30 cell converges in exactly 2 outer iterations" {
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &A5_CELL, .{ .gap_tol = DGGS_GAP_TOL, .method = .alternating });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expectEqual(@as(u32, 2), outcome.converged.diag.alternating.outer_iters);
}

test "CANARY(alternating): a harder A5 r30 cell takes more than 2 outer iterations" {
    // This cell currently takes 4 (the rare tail of the A5 distribution);
    // we only assert > 2 so the canary is about "demonstrably more work than
    // the common case", not the exact tail value.
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &A5_CELL_4ITER, .{ .gap_tol = DGGS_GAP_TOL, .method = .alternating });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    try std.testing.expect(outcome.converged.diag.alternating.outer_iters > 2);
}

// ── Trust-path CANARIES (same cells, same spirit) ────────────────────────
//
// Iteration pins for the `.trust` path (the default) on the same five
// cells the alternating-path canaries pin. Same policy: a trip is a signal to
// understand what changed and call it out — never quietly bump.
//
// The counts read differently from the alternating path's. The pattern
// to know (see docs/trust-solver.md):
//  - eager_certified = the iteration-0 certificate at the halfspace
//    axis already passes (the axis is optimal on arrival);
//  - open_iters = 1, tr/recert = 0 = a cert-edge cell — the eager cert
//    lands just above tol and ONE alternating-cadence opening round
//    (axis motion + cheap FW + polish) certifies. Before the opening
//    rounds landed (perf/trust-losing-cases branch) these cells read
//    tr_iters = 1 + recert_attempts = 2: the trust region found h
//    stationary, exited via the pred-noise check, and the RECERT
//    phase certified on its 2nd fast-style attempt — the same
//    escape, paid for at full-oracle price. Those older pins are what
//    caught the 26-rejection Δ-collapse thrash that survey means had
//    smeared (H3 r9 read 27 before the pred-noise exit landed).

test "CANARY(trust): H3 r15 cell certifies eagerly at iteration 0 (strict default)" {
    const allocator = std.testing.allocator;
    const h3 = cases.byName("h3_r15_equator").?.points;
    var outcome = try skar.solve(allocator, h3, .{ .method = .trust });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    const d = outcome.converged.diag.trust;
    try std.testing.expect(d.eager_certified);
    try std.testing.expectEqual(@as(u32, 0), d.tr_iters);
    try std.testing.expectEqual(@as(u32, 0), d.recert_attempts);
}

test "CANARY(trust): H3 r9 is a cert-edge cell — certifies in 1 opening round" {
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &H3_R9_CELL, .{ .method = .trust });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    const d = outcome.converged.diag.trust;
    try std.testing.expect(!d.eager_certified);
    try std.testing.expectEqual(@as(u32, 1), d.open_iters);
    try std.testing.expectEqual(@as(u32, 0), d.tr_iters);
    try std.testing.expectEqual(@as(u32, 0), d.recert_attempts);
}

test "CANARY(trust): S2 L30 cell certifies eagerly at iteration 0" {
    const allocator = std.testing.allocator;
    var outcome = try skar.solve(allocator, &S2_CELL, .{ .gap_tol = DGGS_GAP_TOL, .method = .trust });
    defer outcome.deinit();

    try std.testing.expect(std.meta.activeTag(outcome) == .converged);
    const d = outcome.converged.diag.trust;
    try std.testing.expect(d.eager_certified);
    try std.testing.expectEqual(@as(u32, 0), d.tr_iters);
    try std.testing.expectEqual(@as(u32, 0), d.recert_attempts);
}

test "CANARY(trust): common and harder A5 r30 cells share the cert-edge signature" {
    // Under the trust path the alternating path's "hard tail" cell costs
    // the same as the common one: both certify in ONE opening round
    // (the eager cert lands just above tol; a single axis-motion round
    // re-samples the numerical state and passes) — even though the
    // harder cell needs 4 alternating iterations from a cold start.
    // The eager phase's FW+polish opening is a stronger warm start
    // than the alternating path's first cycles. The equality is the
    // interesting fact — pin both.
    const allocator = std.testing.allocator;
    var common = try skar.solve(allocator, &A5_CELL, .{ .gap_tol = DGGS_GAP_TOL, .method = .trust });
    defer common.deinit();
    var harder = try skar.solve(allocator, &A5_CELL_4ITER, .{ .gap_tol = DGGS_GAP_TOL, .method = .trust });
    defer harder.deinit();

    try std.testing.expect(std.meta.activeTag(common) == .converged);
    try std.testing.expect(std.meta.activeTag(harder) == .converged);
    for ([_]skar.Diagnostics{ common.converged.diag, harder.converged.diag }) |diag| {
        const d = diag.trust;
        try std.testing.expect(!d.eager_certified);
        try std.testing.expectEqual(@as(u32, 1), d.open_iters);
        try std.testing.expectEqual(@as(u32, 0), d.tr_iters);
        try std.testing.expectEqual(@as(u32, 0), d.recert_attempts);
    }
}
