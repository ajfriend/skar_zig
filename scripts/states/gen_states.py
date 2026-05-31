# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Step 1 of the US-states aspect-ratio example (mirrors scripts/dggs/gen_cells.py).

Fetches the canonical Leaflet us-states GeoJSON, drops DC, and writes
scripts/states/data/states.json with, per state:
  - `vertices`: the flat union of every ring's corners as unit 3-vectors
    (the format `skar.solve` consumes — for the aspect ratio the input is just
    a point set, so multipolygon/ring grouping is irrelevant);
  - `rings`: the per-ring lon/lat coordinates, kept so the plot (step 3) can
    draw each ring separately and avoid spurious connecting segments across
    multipolygon states (Alaska, Hawaii, Michigan).

Edit the constants below in place — no CLI args by project convention.
Run with:  uv run scripts/states/gen_states.py
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- config
SOURCE_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
    "master/data/geojson/us-states.json"
)
# The source has 52 features: 50 states + DC + Puerto Rico. Drop the two
# non-states to land on exactly 50.
EXCLUDE = {"District of Columbia", "Puerto Rico"}

OUT_DIR = Path(__file__).resolve().parent / "data"
GEOJSON_PATH = OUT_DIR / "us-states.geojson"
OUT_PATH = OUT_DIR / "states.json"


# ---------------------------------------------------------------- helpers
def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    c = math.cos(lat)
    return (c * math.cos(lon), c * math.sin(lon), math.sin(lat))


def fetch_geojson() -> dict:
    """Download the GeoJSON once, then read from the local cache."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not GEOJSON_PATH.exists():
        print(f"  fetching {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL) as resp:
            GEOJSON_PATH.write_bytes(resp.read())
        print(f"  cached {GEOJSON_PATH.relative_to(Path.cwd())}")
    else:
        print(f"  using cached {GEOJSON_PATH.relative_to(Path.cwd())}")
    with GEOJSON_PATH.open() as f:
        return json.load(f)


def feature_rings(geometry: dict) -> list[list[list[float]]]:
    """Flatten a Polygon/MultiPolygon geometry to a list of lon/lat rings.

    GeoJSON nesting: Polygon coords are [ring, ...]; MultiPolygon coords are
    [polygon, ...] where each polygon is [ring, ...]. Either way we want one
    flat list of rings (each ring a list of [lon, lat]). Per-ring grouping is
    preserved here only for the plot — the solver gets the flattened union.
    """
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    rings: list[list[list[float]]] = []
    if gtype == "Polygon":
        rings.extend(coords)
    elif gtype == "MultiPolygon":
        for poly in coords:
            rings.extend(poly)
    else:
        raise ValueError(f"unexpected geometry type: {gtype}")

    cleaned: list[list[list[float]]] = []
    for ring in rings:
        # GeoJSON rings are closed (first == last); drop the duplicate so we
        # don't feed skar a coincident point (cf. gen_cells.py A5 handling).
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        cleaned.append([[float(lon), float(lat)] for lon, lat in ring])
    return cleaned


# ---------------------------------------------------------------- driver
def main() -> None:
    t0 = time.perf_counter()
    gj = fetch_geojson()

    states = []
    for feat in gj["features"]:
        name = feat["properties"]["name"]
        if name in EXCLUDE:
            continue
        rings = feature_rings(feat["geometry"])
        vertices = [
            lonlat_to_xyz(lon, lat) for ring in rings for (lon, lat) in ring
        ]
        states.append({"name": name, "vertices": vertices, "rings": rings})

    states.sort(key=lambda s: s["name"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_url": SOURCE_URL,
        "n_states": len(states),
        "states": states,
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f)

    dt = time.perf_counter() - t0
    # Sanity: every vertex should sit on the unit sphere.
    worst = 0.0
    total_verts = 0
    for s in states:
        total_verts += len(s["vertices"])
        for x, y, z in s["vertices"]:
            worst = max(worst, abs(math.sqrt(x * x + y * y + z * z) - 1.0))
    print(f"  {len(states)} states, {total_verts} vertices, worst |‖v‖-1| = {worst:.2e}")
    print(f"  built in {dt:.2f}s")
    print(f"  wrote {OUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
