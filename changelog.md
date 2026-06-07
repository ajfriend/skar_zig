# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-06

### Fixed

- DGGS cells in the **H3 r7–r10 band** (~1–3% of cells) that stalled at
  `did_not_converge` under the strict default `gap_tol = 1e-6` now certify.
  The certificate active-set cutoff (`algo.ACTIVE_THRESH`) was lowered from
  `1e-6` to `1e-12`. At the old value it coincided with the default tolerance,
  so Newton polish dropped the small-weight (~1e-7) binding constraints of
  these near-circular degenerate D-optimal designs, flooring the duality gap
  at ~1.7e-6. Dropping a binding constraint of dual mass `m` inflates the gap
  by `O(m)`, so a cutoff six orders below `gap_tol` makes it numerically
  invisible. The fix is a single internal constant; the public API is
  unchanged.

### Changed

- Cells that previously returned `did_not_converge` at the strict default now
  certify; their aspect ratio refines in roughly the 7th significant digit.
  This is a behavior change (hence the minor bump) even though the API is the
  same.
- The genuine f64 duality-gap floors at the finest resolutions (S2 L30 /
  A5 r30) are unaffected — bit-identical before and after — and still
  correctly return `did_not_converge` at `gap_tol = 1e-6`, declining to
  certify a bound f64 cannot deliver.

### Tests

- Added H3 r8/r9/r10 band regression fixtures (must certify at 1e-6), a
  `gap_tol` scale-collision guard, and a convergence canary; the existing
  S2/A5 finest-resolution DNC guard is unchanged. Verified empirically that
  all H3 cells across all 16 resolutions — including pentagons and
  icosahedron-edge cells — converge at the strict default.

## [0.1.0] - 2026-05-31

Initial public release.

### Added

- Minimum-volume enclosing ellipsoidal-cone solver (`skar.solve`) for the
  spherical aspect-ratio problem: Farkas feasibility check, optional
  convex-hull preprocessing, and a hybrid Frank–Wolfe + Newton-polish outer
  loop with a constructed dual certificate.
- Public API: `Outcome` (`converged` / `infeasible` / `did_not_converge`),
  certificates, and `checkFeasibility`.
- Examples (`basic`, `status`, `cases`, `bench`) and standalone aspect-ratio
  studies (DGGS survey, US states, top-100 countries, interactive globe).
- Two-tier (fast/slow) test suite, CI, and MIT license.

[0.2.0]: https://github.com/ajfriend/skar_zig/releases/tag/v0.2.0
[0.1.0]: https://github.com/ajfriend/skar_zig/releases/tag/v0.1.0
