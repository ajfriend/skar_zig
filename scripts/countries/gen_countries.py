# /// script
# requires-python = ">=3.11"
# dependencies = ["geopandas", "pyogrio"]
# ///
"""
Step 1 of the top-100-countries aspect-ratio example (mirrors
scripts/states/gen_states.py).

Fetches the Natural Earth 110m admin-0 countries GeoJSON, ranks every feature by
equal-area area, keeps the largest N_TOP, and writes
scripts/countries/data/countries.json with, per country:
  - `vertices`: the flat union of every ring's corners as unit 3-vectors
    (the format `skar.solve` consumes — for the aspect ratio the input is just
    a point set, so multipolygon/ring grouping is irrelevant);
  - `rings`: the per-ring lon/lat coordinates, kept so the plot (step 3) can
    draw each ring separately and avoid spurious connecting segments across
    multipolygon countries (Russia, Indonesia, Canada, ...).

Area ranking is done here in Python with geopandas: reproject to an equal-area
CRS and take polygon areas (no spherical-area helper exists in the repo). All
features are ranked, with no exclusions (Antarctica included — it sits in a
polar cap well inside a hemisphere, so skar solves it fine).

Edit the constants below in place — no CLI args by project convention.
Run with:  uv run scripts/countries/gen_countries.py
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import geopandas as gpd


# ---------------------------------------------------------------- config
SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)
N_TOP = 100
# World Cylindrical Equal Area — any global equal-area CRS works; the exact
# choice doesn't change the top-N cut, only the absolute area numbers.
EQUAL_AREA_CRS = "EPSG:6933"
NAME_FIELD = "ADMIN"  # canonical, always-present country name

OUT_DIR = Path(__file__).resolve().parent / "data"
GEOJSON_PATH = OUT_DIR / "ne_110m_admin_0_countries.geojson"
OUT_PATH = OUT_DIR / "countries.json"


# ---------------------------------------------------------------- helpers
def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    c = math.cos(lat)
    return (c * math.cos(lon), c * math.sin(lon), math.sin(lat))


def fetch_geojson() -> Path:
    """Download the GeoJSON once, then reuse the local cache. Returns its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not GEOJSON_PATH.exists():
        print(f"  fetching {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL) as resp:
            GEOJSON_PATH.write_bytes(resp.read())
        print(f"  cached {GEOJSON_PATH.relative_to(Path.cwd())}")
    else:
        print(f"  using cached {GEOJSON_PATH.relative_to(Path.cwd())}")
    return GEOJSON_PATH


def polygon_rings(geom) -> list[list[list[float]]]:
    """Per-ring lon/lat lists for a shapely (Multi)Polygon.

    One flat list of rings (exterior + any interiors), each a list of [lon, lat].
    Per-ring grouping is preserved only for the plot — the solver gets the
    flattened union of all of them.
    """
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    rings: list[list[list[float]]] = []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            # shapely rings are closed (first == last); drop the duplicate so we
            # don't feed skar a coincident point (cf. gen_states.py).
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            rings.append([[float(lon), float(lat)] for lon, lat, *_ in coords])
    return rings


# ---------------------------------------------------------------- driver
def main() -> None:
    t0 = time.perf_counter()
    path = fetch_geojson()

    gdf = gpd.read_file(path)  # lon/lat, EPSG:4326
    # Rank by equal-area area; reproject a copy just to measure (keep the
    # original 4326 geometry for the vertex/ring extraction below).
    gdf["_area"] = gdf.to_crs(EQUAL_AREA_CRS).area
    top = gdf.sort_values("_area", ascending=False).head(N_TOP)

    countries = []
    for _, row in top.iterrows():
        rings = polygon_rings(row.geometry)
        vertices = [
            lonlat_to_xyz(lon, lat) for ring in rings for (lon, lat) in ring
        ]
        countries.append(
            {"name": str(row[NAME_FIELD]), "vertices": vertices, "rings": rings}
        )

    countries.sort(key=lambda c: c["name"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_url": SOURCE_URL,
        "n_countries": len(countries),
        "countries": countries,
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f)

    dt = time.perf_counter() - t0
    # Sanity: every vertex should sit on the unit sphere.
    worst = 0.0
    total_verts = 0
    for c in countries:
        total_verts += len(c["vertices"])
        for x, y, z in c["vertices"]:
            worst = max(worst, abs(math.sqrt(x * x + y * y + z * z) - 1.0))
    print(f"  {len(countries)} countries, {total_verts} vertices, worst |‖v‖-1| = {worst:.2e}")
    print(f"  built in {dt:.2f}s")
    print(f"  wrote {OUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
