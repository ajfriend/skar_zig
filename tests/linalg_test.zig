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

test "Mat3.symmetrize: output matrix is bit-exactly symmetric" {
    // `symmetrize` returns (self + self.transpose()) / 2 via Mat3.lincomb.
    // Cell (i,j) = 0.5·m[i,j] + 0.5·m[j,i] (single FMA rounding).
    // Cell (j,i) = 0.5·m[j,i] + 0.5·m[i,j].
    // IEEE 754 addition is commutative and 0.5·x is exact, so the
    // two FMAs produce bit-identical results.
    var prng = std.Random.DefaultPrng.init(0x5333);
    const rng = prng.random();
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        const m = linalg.Mat3.randomNormal(rng);
        const sym = m.symmetrize();
        try std.testing.expectEqual(sym.m[1], sym.m[3]);
        try std.testing.expectEqual(sym.m[2], sym.m[6]);
        try std.testing.expectEqual(sym.m[5], sym.m[7]);
    }
}

test "Mat3.symOuter: output matrix is bit-exactly symmetric" {
    // Per cell: (x[i]·z[j] + z[i]·x[j]) * 0.5. The (i,j) and (j,i)
    // cells compute the same sum with summands in reverse order;
    // IEEE 754 addition commutativity makes them bit-equal.
    var prng = std.Random.DefaultPrng.init(0x5444);
    const rng = prng.random();
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        const x = linalg.Vec3.randomUnit(rng).scale(rng.float(f64) * 3.0);
        const z = linalg.Vec3.randomUnit(rng).scale(rng.float(f64) * 3.0);
        const m = linalg.Mat3.symOuter(x, z);
        try std.testing.expectEqual(m.m[1], m.m[3]);
        try std.testing.expectEqual(m.m[2], m.m[6]);
        try std.testing.expectEqual(m.m[5], m.m[7]);
    }
}

test "Mat2.addSymRank1: output matrix is bit-exactly symmetric" {
    // Mat2 has a single mirror pair (m[1], m[2]). Same "compute
    // upper, mirror to lower" pattern as the Mat3 versions.
    var prng = std.Random.DefaultPrng.init(0x5555);
    const rng = prng.random();
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        var m: linalg.Mat2 = .{ .m = .{
            rng.floatNorm(f64), rng.floatNorm(f64),
            rng.floatNorm(f64), rng.floatNorm(f64),
        } };
        const w = rng.float(f64) * 10.0 - 5.0;
        const p = linalg.Vec2{ .m = .{
            rng.floatNorm(f64), rng.floatNorm(f64),
        } };
        m.addSymRank1(w, p);
        try std.testing.expectEqual(m.m[1], m.m[2]);
    }
}
