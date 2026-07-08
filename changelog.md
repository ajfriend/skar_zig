# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-08

Breaking: `SolveOptions.method` now defaults to `.trust`, which converges
on every input family constructed to date (wide caps, all surveyed
regions) at DGGS-speed parity; `.alternating` remains available and
bit-stable with the previous default. Near the f64 gap floor, which cells
certify at a strict tolerance shifts between paths (answers agree; see the
PR). The wide-cap fixtures join the bundled case manifest.
([#6](https://github.com/ajfriend/skar_zig/pull/6))

## [0.5.0] - 2026-07-08

New EXPERIMENTAL solver selection (`SolveOptions.method`): the `.trust`
path converges on the wide-angle and elongated inputs the default solver
structurally cannot (dense caps past ~82°, countries like France at the
default iteration budget), at DGGS success-speed parity; `.auto` runs the
default with `.trust` as fallback. Breaking: per-solve diagnostics
(`outer_iters`, `newton_polish_failures`) moved into a typed per-algorithm
`diag` union on `Converged`/`DidNotConverge`. Default solver behavior is
otherwise unchanged. ([#4](https://github.com/ajfriend/skar_zig/pull/4))

## [0.4.0] - 2026-06-07

Replace the inner-FW boost with a size-gated sparse FW initialization: same
A5 res-0 fix, ~56× faster there and ~3× faster on medium/large inputs. Internal
only; public API unchanged. ([#2](https://github.com/ajfriend/skar_zig/pull/2))

## [0.3.0] - 2026-06-07

All 12 A5 resolution-0 cells now converge at the strict default
`gap_tol = 1e-6`. Their dense `cell_to_boundary` polygons (~320 near-cocircular
points) made the outer-iteration count scale with the point count and overrun
`max_outer`; the inner Frank–Wolfe weight solve now gets a real per-cycle budget
when the working set is large (`algo.INNER_FW_BOOST_*`), draining the active set
in the first outer iteration (~145 → ~6 iters, ≈500× faster). Small inputs keep
the bit-identical 1-step path, so the genuine f64-floor finest-resolution cells
are unaffected. Internal-only (public API unchanged). Full write-up in
`docs/a5_res0_dnc_report.md` and commit
[7dfb376](https://github.com/ajfriend/skar_zig/commit/7dfb376).

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
