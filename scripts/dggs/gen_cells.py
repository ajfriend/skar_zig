# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h3>=4.0",
#   "s2sphere",
#   "numpy",
#   "pya5",
# ]
# ///
"""
Step 1 of the DGGS aspect-ratio survey (see docs/dggs-aspect-survey-plan.md).

Samples N random cells at the finest resolution in H3, S2, and A5, and
writes per-system JSON with each cell's boundary vertices as unit
3-vectors (the format `skar.solve` consumes).

Edit the constants below in place — no CLI args by project convention.
Run with:  uv run scripts/dggs/gen_cells.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import h3
import s2sphere
import a5


# ---------------------------------------------------------------- config
N = 10_000
SEED = 0xC0FFEE

# Finest resolution / level per system.
H3_RES = 15           # h3-py supports 0..15
S2_LEVEL = 30         # s2sphere supports 0..30
A5_RES = a5.MAX_RESOLUTION  # currently 30

OUT_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------- helpers
def sample_uniform_lonlat(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-on-sphere samples as (lon_deg, lat_deg), shape (n, 2)."""
    u = rng.random(n)
    v = rng.random(n)
    lon = 360.0 * u - 180.0
    # lat = asin(2v - 1) gives uniform area on the sphere
    lat = np.degrees(np.arcsin(2.0 * v - 1.0))
    return np.column_stack([lon, lat])


def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    c = math.cos(lat)
    return (c * math.cos(lon), c * math.sin(lon), math.sin(lat))


def renormalize(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


# ---------------------------------------------------------------- per-system samplers
def gen_h3(samples: np.ndarray) -> list[dict]:
    """Each cell: 6 vertices (5 for the 12 pentagons). h3 returns (lat, lng)."""
    cells: list[str] = []
    seen: set[str] = set()
    for lon, lat in samples:
        cid = h3.latlng_to_cell(float(lat), float(lon), H3_RES)
        if cid in seen:
            continue
        seen.add(cid)
        cells.append(cid)

    out = []
    for cid in cells:
        boundary = h3.cell_to_boundary(cid)  # [(lat, lng), ...] open ring
        verts = [lonlat_to_xyz(lng, lat) for (lat, lng) in boundary]
        out.append({"id": cid, "vertices": verts})
    return out


def gen_s2(samples: np.ndarray) -> list[dict]:
    """S2 leaf cells (level 30) have 4 corner vertices."""
    cells: list[s2sphere.CellId] = []
    seen: set[int] = set()
    for lon, lat in samples:
        ll = s2sphere.LatLng.from_degrees(float(lat), float(lon))
        cid = s2sphere.CellId.from_lat_lng(ll)  # leaf == level 30
        if S2_LEVEL != 30:
            cid = cid.parent(S2_LEVEL)
        key = cid.id()
        if key in seen:
            continue
        seen.add(key)
        cells.append(cid)

    out = []
    for cid in cells:
        cell = s2sphere.Cell(cid)
        verts = []
        for i in range(4):
            p = cell.get_vertex(i)  # s2sphere.Point, indexable; not always unit
            verts.append(renormalize((p[0], p[1], p[2])))
        out.append({"id": format(cid.id(), "016x"), "vertices": verts})
    return out


def gen_a5(samples: np.ndarray) -> list[dict]:
    """A5 pentagonal cells; cell_to_boundary returns closed ring of (lon, lat)."""
    cells: list[int] = []
    seen: set[int] = set()
    for lon, lat in samples:
        cid = a5.lonlat_to_cell((float(lon), float(lat)), A5_RES)
        if cid in seen:
            continue
        seen.add(cid)
        cells.append(cid)

    out = []
    for cid in cells:
        ring = a5.cell_to_boundary(cid)  # closed ring of (lon, lat)
        # Drop the duplicated closing vertex.
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        verts = [lonlat_to_xyz(lon, lat) for (lon, lat) in ring]
        out.append({"id": a5.u64_to_hex(cid), "vertices": verts})
    return out


# ---------------------------------------------------------------- driver
def write_system(name: str, resolution: int, cells: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    payload = {
        "system": name,
        "resolution": resolution,
        "n_requested": N,
        "n_unique": len(cells),
        "cells": cells,
    }
    with path.open("w") as f:
        json.dump(payload, f)
    # Sanity: every vertex within 1e-9 of the unit sphere.
    worst = 0.0
    for c in cells:
        for x, y, z in c["vertices"]:
            worst = max(worst, abs(math.sqrt(x * x + y * y + z * z) - 1.0))
    print(f"  {name}: {len(cells)} unique cells, worst |‖v‖-1| = {worst:.2e}")
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    samples = sample_uniform_lonlat(N, rng)

    for name, res, fn in [
        ("h3", H3_RES, gen_h3),
        ("s2", S2_LEVEL, gen_s2),
        ("a5", A5_RES, gen_a5),
    ]:
        print(f"[{name}] res={res} sampling {N} points...")
        t0 = time.perf_counter()
        cells = fn(samples)
        dt = time.perf_counter() - t0
        print(f"  built in {dt:.2f}s")
        write_system(name, res, cells)


if __name__ == "__main__":
    main()
