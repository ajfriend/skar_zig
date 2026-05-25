const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const skar_mod = b.addModule("skar", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Case fixtures (tests/cases/*.zon) + their compile-time manifest.
    // The module lives inside tests/cases/ so the manifest can `@import`
    // the sibling .zon files directly without crossing module-path
    // boundaries.
    const cases_mod = b.addModule("cases", .{
        .root_source_file = b.path("tests/cases/cases.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "skar",
        .root_module = skar_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // Test runner roots at `test_root.zig` at the repo root, so the
    // test module's filesystem-import scope covers BOTH `src/` (for
    // the library under test, reached via `@import("../src/foo.zig")`
    // from test files) AND `tests/` (the test files themselves).
    // This lets tests reach internals like `acceptBUpdate` directly,
    // without re-exporting them through the public API.
    const test_mod = b.createModule(.{
        .root_source_file = b.path("test_root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("cases", cases_mod);
    const tests = b.addTest(.{ .name = "skar-test", .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    run_tests.setCwd(b.path(""));
    const test_step = b.step("test", "Run skar tests");
    test_step.dependOn(&run_tests.step);

    // `zig build install-test` produces `zig-out/bin/skar-test`, the
    // test binary built without running. Used by the kcov-based
    // coverage recipe in the justfile.
    const install_test = b.addInstallArtifact(tests, .{});
    const install_test_step = b.step("install-test", "Install the test binary at zig-out/bin/skar-test");
    install_test_step.dependOn(&install_test.step);

    // Examples. Single-file runnable programs. Step name matches the
    // example's filename (examples/<stem>.zig → `zig build ex-<stem>`).
    // `ex-cases` accepts pass-through args after `--`: `zig build
    // ex-cases -- hex` or `-- --all`. `ex-bench` is force-built in
    // ReleaseFast — timing numbers are meaningless in Debug.
    addExample(b, skar_mod, cases_mod, target, optimize, "basic", null, "Run examples/basic.zig (happy-path only)");
    addExample(b, skar_mod, cases_mod, target, optimize, "status", null, "Run examples/status.zig (full Outcome branching)");
    addExample(b, skar_mod, cases_mod, target, optimize, "cases", null, "Run examples/cases.zig (run a named case or --all)");
    addExample(b, skar_mod, cases_mod, target, optimize, "bench", .ReleaseFast, "Run examples/bench.zig (per-case timing, release-built)");
}

fn addExample(
    b: *std.Build,
    skar_mod: *std.Build.Module,
    cases_mod: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    stem: []const u8,
    /// Per-example optimize override; null inherits the project-wide
    /// flag. Used by `ex-bench` to force ReleaseFast regardless of
    /// the top-level build setting.
    optimize_override: ?std.builtin.OptimizeMode,
    description: []const u8,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(b.fmt("examples/{s}.zig", .{stem})),
        .target = target,
        .optimize = optimize_override orelse optimize,
    });
    mod.addImport("skar", skar_mod);
    mod.addImport("cases", cases_mod);
    const exe = b.addExecutable(.{
        .name = b.fmt("skar-ex-{s}", .{stem}),
        .root_module = mod,
    });
    const run = b.addRunArtifact(exe);
    // Pass through any args after `--` on the `zig build` command.
    // Only `ex-cases` uses them today; the others ignore the arg slice.
    if (b.args) |args| run.addArgs(args);
    const step = b.step(b.fmt("ex-{s}", .{stem}), description);
    step.dependOn(&run.step);
}
