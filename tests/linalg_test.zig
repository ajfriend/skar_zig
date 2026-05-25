//! Contract tests for linalg primitives. Solver-level tests cover
//! linalg indirectly via correctness assertions on AR / gap; this
//! file pins specific invariants that the solver relies on but
//! couldn't easily diagnose if broken.

const std = @import("std");
const linalg = @import("../src/linalg.zig");

test "addSymRank1: output matrix is bit-exactly symmetric" {
    // The function's contract: after the update, the output is
    // symmetric — m[1] == m[3], m[2] == m[6], m[5] == m[7] — to the
    // last bit, regardless of FP rounding. This is achieved by
    // computing the 6 upper-triangle entries and mirroring with
    // exact `=` assignment. If a future refactor (e.g. independent
    // FMAs per pair) breaks this, downstream code that treats the
    // matrix as symmetric (Cholesky, eig2) silently goes wrong.
    var prng = std.Random.DefaultPrng.init(0x5111);
    const rng = prng.random();
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        // Start from a non-symmetric M to verify the function
        // unconditionally produces symmetric output, not just
        // preserves a pre-symmetric input.
        var m = linalg.Mat3.randomNormal(rng);
        const w = rng.float(f64) * 10.0 - 5.0;
        const q = linalg.Vec3.randomUnit(rng).scale(rng.float(f64) * 3.0);
        m.addSymRank1(w, q);
        try std.testing.expectEqual(m.m[1], m.m[3]);
        try std.testing.expectEqual(m.m[2], m.m[6]);
        try std.testing.expectEqual(m.m[5], m.m[7]);
    }
}

test "addSymRank2: output matrix is bit-exactly symmetric" {
    var prng = std.Random.DefaultPrng.init(0x5222);
    const rng = prng.random();
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        var m = linalg.Mat3.randomNormal(rng);
        const lam = rng.float(f64) * 10.0 - 5.0;
        const x = linalg.Vec3.randomUnit(rng).scale(rng.float(f64) * 3.0);
        const z = linalg.Vec3.randomUnit(rng).scale(rng.float(f64) * 3.0);
        m.addSymRank2(lam, x, z);
        try std.testing.expectEqual(m.m[1], m.m[3]);
        try std.testing.expectEqual(m.m[2], m.m[6]);
        try std.testing.expectEqual(m.m[5], m.m[7]);
    }
}
