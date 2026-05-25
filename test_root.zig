//! Root for the test target. Lives at the repo root so the test
//! module's filesystem-import scope covers both `src/` (for the
//! library under test) and `tests/` (for the test files
//! themselves). Nothing else lives here.

test {
    _ = @import("tests/all.zig");
}
