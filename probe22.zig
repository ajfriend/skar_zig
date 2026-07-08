//! Temporary probe 22: CANARY cells through fast vs reduced.
const std = @import("std");
const sphar = @import("src/root.zig");
const cases = @import("tests/cases/cases.zig");

const DGGS_GAP_TOL: f64 = 1e-3;
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

fn run(allocator: std.mem.Allocator, name: []const u8, pts: []const [3]f64, tol_: f64) !void {
    var of = try sphar.solve(allocator, pts, .{ .gap_tol = tol_ });
    defer of.deinit();
    var orr = try sphar.solve(allocator, pts, .{ .gap_tol = tol_, .method = .reduced });
    defer orr.deinit();
    const fi = if (of == .converged) of.converged.outer_iters else 999;
    const ri = if (orr == .converged) orr.converged.outer_iters else 999;
    std.debug.print("{s:24} tol={e:8.0}  fast={d:3}  reduced={d:3}\n", .{ name, tol_, fi, ri });
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const h3r15 = cases.byName("h3_r15_equator").?.points;
    try run(allocator, "H3 r15 (pin: fast=1)", h3r15, 1e-6);
    try run(allocator, "H3 r9 (pin: fast=2)", &H3_R9_CELL, 1e-6);
    try run(allocator, "S2 L30 (pin: fast=1)", &S2_CELL, DGGS_GAP_TOL);
    try run(allocator, "A5 r30 (pin: fast=2)", &A5_CELL, DGGS_GAP_TOL);
    try run(allocator, "A5 hard (pin: fast>2)", &A5_CELL_4ITER, DGGS_GAP_TOL);
}

// trace entry added by probe22b
