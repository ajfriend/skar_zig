# DGGS aspect-ratio survey

Goal: for `N` random cells at the finest resolution in each of H3, S2,
and A5, compute the tightest enclosing-cone aspect ratio via `skar`,
build a per-system histogram, and visualize the worst-case cell in 2D
together with its enclosing ellipse.

`N = 10_000` to start, but the generator script must take `N` as a
configurable variable. Per the project Python convention: no CLI
parsing — variables at the top of each script, edited in place.

## Layout

All new artifacts live under `scripts/dggs/`:

```
scripts/dggs/
  README.md                # how to run each step end-to-end
  gen_cells.py             # step 1 — produce raw cell boundary data
  data/                    # generated artifacts (gitignored)
    h3.json
    s2.json
    a5.json
    aspect.json            # produced in step 2
  aspect.zig               # step 2 — run skar on every cell
  histogram.py             # step 3 — per-system histograms
  worst_plot.py            # step 4 — 2D projection of worst cell + ellipse
```

The Zig step is wired through `build.zig` as `zig build dggs-aspect`
so it picks up the existing `skar` module without ad-hoc paths.

## Step 1 — generate raw cell boundary data

**Deliverable:** `scripts/dggs/gen_cells.py` and three JSON files under
`scripts/dggs/data/`.

For each system, sample `N` random cell indexes at the finest
resolution and emit their vertex boundaries as unit 3-vectors (lon/lat
→ xyz on the unit sphere; the same convention `skar` already consumes
in `tests/cases/zon/h3_*.zon`).

- H3: resolution 15. Sample by drawing uniform-on-sphere lon/lat and
  calling `h3.latlng_to_cell(lat, lon, 15)`, deduplicating, then
  `h3.cell_to_boundary(cell)` for vertices.
- S2: level 30. Sample uniform points on the sphere, snap to leaf cell
  via `s2sphere.CellId.from_point(...).parent(30)`, dedupe, take the
  four corner vertices via `s2sphere.Cell(cell_id).get_vertex(i)`.
- A5: finest supported resolution. Use the `a5` Python package
  (`a5-py`); sample uniform points, `lonlat_to_cell(lon, lat, res)`,
  dedupe, `cell_to_boundary(cell)`.

Output schema per file (`data/<system>.json`):

```jsonc
{
  "system": "h3",
  "resolution": 15,
  "n_requested": 10000,
  "n_unique": 9998,
  "cells": [
    { "id": "8f...", "vertices": [[x,y,z], [x,y,z], ...] },
    ...
  ]
}
```

PEP 723 inline metadata at the top of the script declares
`h3`, `s2sphere`, `a5-py`, and `numpy` so `uv run gen_cells.py` is
self-contained. RNG seed lives in a top-of-file constant.

**Done when:** running the script produces all three JSON files with
roughly `N` unique cells each and every vertex within `1e-12` of the
unit sphere.

## Step 2 — compute aspect ratios via skar

**Deliverable:** `scripts/dggs/aspect.zig`, wired as a build step;
output is `data/aspect.json`.

**Solve tolerance:** the survey passes `gap_tol = 1e-3`, *not* skar's
strict 1e-6 default. At finest resolution the S2/A5 cells are sub-meter
scatters at an O(1) point on the sphere, so κ(A) ~ σ_max ~ 1e9 and the
duality gap has an f64 floor of ~1e-4–1e-3 (the optimal cone axis is a
sub-ulp rotation from the best representable `b`). At 1e-6 ~22% of S2 and
~47% of A5 cells correctly return `.did_not_converge` — but their aspect
ratios are accurate regardless of the gap (input-precision-limited, ~7
digits). Solving at 1e-3 lets every cell converge so the step-3 AR
distribution is *complete* rather than silently dropping those cells. See
`tests/dggs_dnc_test.zig` and `SolveOptions.gap_tol` for the floor.

For each system's cell list, call `skar.solve(allocator, vertices,
.{ .gap_tol = 1e-3 })` and capture:

- `outcome` tag (`converged` / `infeasible` / `did_not_converge`)
- on `converged`: `aspectRatio()`, `b()` (axis), and the `A` matrix
  (needed for the worst-case ellipse plot in step 4)
- cell id (echo through from input)

Output (`data/aspect.json`):

```jsonc
{
  "h3": {
    "n": 10000,
    "results": [
      { "id": "8f...", "outcome": "converged", "ar": 1.17, "b": [...], "A": [[...],[...],[...]] },
      { "id": "...",   "outcome": "did_not_converge" },
      ...
    ]
  },
  "s2": { ... },
  "a5": { ... }
}
```

**Done when:** the script runs to completion on all three inputs and
prints a one-line summary per system (`h3: 10000 converged, 0
infeasible, 0 did_not_converge`). A non-zero non-converged count is
not a failure of the script — it's a finding to surface.

## Step 3 — per-system histograms — DONE

**Deliverable:** `scripts/dggs/histogram.py`; writes `hist_h3.png`,
`hist_s2.png`, `hist_a5.png`, and a combined panel `hist_combined.png`
into `data/`.

Loads `data/aspect.json`, plots the AR distribution per system with shared
bins (log-count y-axis — AR clusters near the low end with a thin tail) for
easy comparison, and prints summary stats (min, median, p99, max,
converged/DNC counts) to stdout. Because step 2 solves at `gap_tol = 1e-3`,
every cell converges, so the histograms are the complete distribution
(non-converged count is 0); the `did_not_converge` count is still surfaced
in each title in case a future tolerance change reintroduces it.

Observed (N=10_000, seed 0xC0FFEE): H3 r15 median 1.05 / max 1.25; S2 L30
median 1.22 / max 1.72; A5 r30 is tightly clustered around 2.0–2.32
(median 2.13, min 1.99) — A5 cells are inherently ~2:1 elongated.

**Done when:** the four PNGs exist and the stats table is printed. ✅

## Step 4 — best/worst 2D plots with enclosing ellipse — DONE

**Deliverable:** `scripts/dggs/extremes_plot.py`; writes a single 3×2 grid
`data/extremes.png` — one row per system (H3, S2, A5), left column = the
best-AR (most circular) converged cell, right column = the worst-AR cell.
Step 2 records both extremes per system (`best`/`worst` in `aspect.json`,
each carrying axis `b`, cone matrix `A`, and boundary vertices).

Per panel:

1. Gnomonic-project the boundary vertices into the tangent plane at `b`:
   `y = Uᵀx / (b·x)`, with `U = [u1 u2]` orthonormal columns ⊥ `b`.
2. Overlay the enclosing-cone cross-section. **Correction to the original
   sketch:** skar's cone is the *second-order* cone `{x : ‖A x‖ ≤ b·x}`
   (`checkFeasibility` measures `‖A·x‖ − b·x`), so the relevant form is
   `A²`, not `A`. Since `A b = σ0 b`, `bᵀA U = 0` and
   `‖Ax‖²/(b·x)² = σ0² + yᵀ(UᵀA²U)y`; the cross-section is therefore
   `{ y : yᵀ (UᵀA² U) y = 1 − σ0² = 2/3 }` (the budget). Its geometric
   axis ratio is `σ2/σ1` — exactly the reported AR. The original
   `{yᵀ(UᵀAU)y = bᵀAb}` was the unconfirmed *linear* form and yields a
   √-rounded ellipse (axis ratio √AR), so it was wrong; use `A²`.
3. Rotate each panel into `M2`'s eigenframe so the ellipse's major axis is
   horizontal (minor vertical); `set_aspect("equal")`, axes in metres
   (×Earth radius). Annotate each panel with its AR + cell id.

**Done when:** `extremes.png` exists; every panel shows the cell enclosed by
its ellipse, farthest vertex on the boundary (`max yᵀM2y = 2/3`). ✅
best/worst AR — H3 1.000/1.250, S2 1.001/1.718, A5 1.990/2.319 (A5 is never
circular: even its best cell is ~2:1).

## Open questions to resolve as we go

- A5 finest resolution: confirm the integer; `a5-py` API names. May
  differ between published versions.
- Skar build integration: simplest path is a second executable in
  `build.zig` that imports `skar` and uses `std.json`. No new test
  surface needed.
- Whether step 2's output should be one combined file or three —
  combined is simpler to consume in step 3/4; revisit if it gets
  large.
