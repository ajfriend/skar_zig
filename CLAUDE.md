# skar — agent notes

Minimum-volume enclosing ellipsoidal-cone solver (the spherical aspect-ratio
problem). Core: `src/skar.zig` (`solve`, the outer loop, `mveeFw` inner MVEE),
`src/config.zig` (tuning knobs in `algo`/`tol`), `src/api.zig` (public surface).

## Build & test

- `zig build test` — fast unit suite (sub-second).
- `zig build test -Dslow=true` — full suite incl. randomized stress tests; this
  is the CI gate (`.github/workflows/ci.yml`). Run it before committing.
- `zig build ex-bench` — per-case timing (ReleaseFast, 100 reps).
- `zig build dggs-aspect` / `states-aspect` / `countries-aspect` — survey execs
  over `scripts/*/data/*.json` (per-cell aspect ratios + outcome counts).

## Performance & regression monitoring (read before "optimizing" the solver)

The hot/common path is **small DGGS cells** — 4–10 points (H3 hexagons, S2/A5
finest cells) — which solve in ~1–2 outer iterations and a few µs. Protect them:

- **Do NOT judge a solver change by `ex-bench`'s `TOTAL` line.** It sums
  wall-times and is dominated by the large synthetic cases (np400, ha_*), so a
  real small-cell regression hides in it. µs-scale wall-time on a 6-point cell is
  mostly noise anyway. Read the **per-case rows** (small vs large separately).
- **The deterministic small-cell guard is the CANARY exact-iteration-count
  tests** in `tests/dggs_dnc_test.zig` (e.g. "H3 r15 converges in 1 outer
  iteration"), plus the iteration ceilings in `tests/a5_res0_test.zig`.
- A CANARY iteration-count shift is a **regression signal**: understand what
  changed and flag it for human confirmation — do not silently bump the expected
  value. (The finest-resolution S2/A5 cells genuinely DNC at the strict 1e-6
  default — an f64 gap floor, not a bug; that's intended.)

When changing the solver, the full check is: `zig build test -Dslow=true` green
(watch CANARY shifts) + `ex-bench` per-case (small cells not slower) + a5_res0
still fast (`tests/a5_res0_test.zig`).

## Background / history

- `docs/a5_res0_dnc_report.md` — the A5 res-0 convergence work (boost → DGGS
  survey → sparse FW init) and the boost-vs-sparse measurements.
- `docs/away-step-fw.md` — proposed away-step FW to retire the sparse-init size
  gate (future work).
