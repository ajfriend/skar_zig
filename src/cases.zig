//! Shared case-file loader. Reads the `cases/*.txt` format documented in
//! the repo-root cases/README.md: leading `#`-lines are header metadata
//! (skipped here), the rest is whitespace-separated 3D points.

const std = @import("std");

pub fn loadCase(allocator: std.mem.Allocator, path: []const u8) ![][3]f64 {
    const content = try std.fs.cwd().readFileAlloc(allocator, path, 10 * 1024 * 1024);
    defer allocator.free(content);

    var pts = std.ArrayList([3]f64){};
    defer pts.deinit(allocator);

    var line_it = std.mem.tokenizeScalar(u8, content, '\n');
    while (line_it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;
        var tok_it = std.mem.tokenizeAny(u8, trimmed, " \t");
        var xyz: [3]f64 = undefined;
        var i: usize = 0;
        while (tok_it.next()) |tok| : (i += 1) {
            if (i >= 3) break;
            xyz[i] = try std.fmt.parseFloat(f64, tok);
        }
        if (i == 3) try pts.append(allocator, xyz);
    }
    return pts.toOwnedSlice(allocator);
}

/// Case "stem": filename minus directories minus the last extension.
/// `cases/np400.txt` → `"np400"`.
pub fn caseStem(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    const dot = std.mem.lastIndexOfScalar(u8, base, '.') orelse return base;
    return base[0..dot];
}
