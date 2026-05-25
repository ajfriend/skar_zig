const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const skar_mod = b.addModule("skar", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Test fixture loader. Lives outside `src/` so its module path
    // doesn't conflict with the skar module path. Wired into skar_mod
    // as well so tests inside `src/tests/` can `@import("cases")`.
    const cases_mod = b.addModule("cases", .{
        .root_source_file = b.path("tests/cases.zig"),
        .target = target,
        .optimize = optimize,
    });
    skar_mod.addImport("cases", cases_mod);

    const lib = b.addLibrary(.{
        .name = "skar",
        .root_module = skar_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    const cli_mod = b.createModule(.{
        .root_source_file = b.path("cli/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_mod.addImport("skar", skar_mod);
    cli_mod.addImport("cases", cases_mod);
    const cli = b.addExecutable(.{
        .name = "skar-cli",
        .root_module = cli_mod,
    });
    b.installArtifact(cli);

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_mod.addImport("skar", skar_mod);
    bench_mod.addImport("cases", cases_mod);
    const bench_exe = b.addExecutable(.{
        .name = "skar-bench",
        .root_module = bench_mod,
    });
    b.installArtifact(bench_exe);

    // Test runner roots at the library module — `root.zig`'s `test {}`
    // block pulls in `src/tests/all.zig` which aggregates every
    // `*_test.zig`. Tests live in the same module path as the
    // library sources, so they can reach internals via filesystem
    // `@import("../halfspace.zig")` etc. without the `_internal`
    // namespace dance.
    const tests = b.addTest(.{ .name = "skar-test", .root_module = skar_mod });
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

    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.setCwd(b.path(""));
    const bench_step = b.step("bench", "Run skar bench");
    bench_step.dependOn(&run_bench.step);
}
