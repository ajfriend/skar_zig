//! Temporary probe 5: dump failing wide-cap cases to JSON for cross-check.
//! Run: zig run probe5.zig
const std = @import("std");
const sphar = @import("src/root.zig");
const Vec3 = sphar.Vec3;

fn deg(d: f64) f64 {
    return d * std.math.pi / 180.0;
}

fn capPoints(allocator: std.mem.Allocator, rng: std.Random, n: usize, cap_deg: f64) ![][3]f64 {
    const pts = try allocator.alloc([3]f64, n);
    const cos_max = @cos(deg(cap_deg));
    var R = sphar.Mat3.randomNormal(rng);
    R.orthonormalize();
    for (pts) |*p| {
        const z = cos_max + rng.float(f64) * (1.0 - cos_max);
        const phi = 2.0 * std.math.pi * rng.float(f64);
        const s = @sqrt(1.0 - z * z);
        const v = Vec3{ .m = .{ s * @cos(phi), s * @sin(phi), z } };
        p.* = R.apply(v).m;
    }
    return pts;
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cases = [_]struct { w: f64, seed: u64, name: []const u8 }{
        .{ .w = 85, .seed = 1, .name = "cap85_s1" },
        .{ .w = 89, .seed = 3, .name = "cap89_s3" },
        .{ .w = 82, .seed = 1, .name = "cap82_s1" },
    };
    for (cases) |c| {
        var prng = std.Random.DefaultPrng.init(c.seed);
        const rng = prng.random();
        const pts = try capPoints(allocator, rng, 200, c.w);
        defer allocator.free(pts);
        var buf: [256]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}.json", .{c.name});
        var f = try std.fs.cwd().createFile(path, .{});
        defer f.close();
        var w = f.deprecatedWriter();
        try w.writeAll("[");
        for (pts, 0..) |p, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("[{d},{d},{d}]", .{ p[0], p[1], p[2] });
        }
        try w.writeAll("]\n");
        std.debug.print("wrote {s}\n", .{path});
    }
}
