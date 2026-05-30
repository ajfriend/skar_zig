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

For each system's cell list, call `skar.solve(allocator, vertices,
.{})` and capture:

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

## Step 3 — per-system histograms

**Deliverable:** `scripts/dggs/histogram.py`; writes three PNGs into
`data/` (`hist_h3.png`, `hist_s2.png`, `hist_a5.png`) and optionally a
combined panel.

Load `data/aspect.json`, drop non-converged entries (count them in the
title), and plot the AR distribution per system with shared bins for
easy comparison. Print summary stats (min, median, p99, max,
non-converged count) to stdout.

**Done when:** the three PNGs exist and the stats table is printed.

## Step 4 — worst-case 2D plot with enclosing ellipse

**Deliverable:** `scripts/dggs/worst_plot.py`; writes
`data/worst_<system>.png` for each system.

For each system, pick the converged result with the largest aspect
ratio. Then:

1. Project the cell's boundary vertices into the tangent plane at the
   cone axis `b` (gnomonic projection: `u = (x - (b·x)b) / (b·x)` in a
   2D basis orthogonal to `b`).
2. Project the `A` quadratic form into the same 2D basis. With `U =
   [u1 u2]` (3×2 orthonormal columns spanning the tangent plane), the
   enclosing-cone boundary in tangent coords is the ellipse
   `{y : yᵀ (Uᵀ A U) y = (bᵀ A b)}` (drop the constant from the
   identity `xᵀAx = (b·x)² bᵀAb + …` — confirm derivation in
   implementation; small standalone proof goes in a comment).
3. Plot the projected vertices as a closed polygon and the ellipse as
   its boundary contour. Title with system, cell id, AR.

**Done when:** the three PNGs exist and visually show vertices
contained inside their ellipse.

## Open questions to resolve as we go

- A5 finest resolution: confirm the integer; `a5-py` API names. May
  differ between published versions.
- Skar build integration: simplest path is a second executable in
  `build.zig` that imports `skar` and uses `std.json`. No new test
  surface needed.
- Whether step 2's output should be one combined file or three —
  combined is simpler to consume in step 3/4; revisit if it gets
  large.
