# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-06

H3 r7–r10 DGGS cells now converge at the strict default `gap_tol = 1e-6` —
lowered the certificate active-set cutoff `ACTIVE_THRESH` from 1e-6 to 1e-12.
Internal-only (public API unchanged); the S2/A5 finest-resolution f64 gap
floors are unaffected. Full write-up in commit
[a9c7207](https://github.com/ajfriend/skar_zig/commit/a9c7207).

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
