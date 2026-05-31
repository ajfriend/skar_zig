# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Build a standalone interactive globe page with a Country/State toggle. Each mode
draws every (converged) feature of that dataset on a draggable orthographic
globe; a sortable table picks which one to highlight, and the page overlays the
spherical ellipse (skar's enclosing cone ∩ the unit sphere) that tightly
contains it.

This is just a packaging step: it embeds already-computed artifacts into a
self-contained HTML page. The enclosing-cone curve and each feature's area are
computed client-side in JS (area via d3.geoArea), so no per-feature precompute
happens here. Reuses, per dataset:
  - <gen>.json        (per-feature lon/lat rings; key "countries" / "states")
  - <gen>_aspect.json (per-feature cone axis b + A + AR; same key)
So run the countries and states pipelines first:
  just countries-gen && just countries-aspect
  just states-gen   && just states-aspect

Edit the constants below in place — no CLI args by project convention.
Run with:  uv run scripts/globe/gen_globe.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------- config
INITIAL_MODE = "country"  # "country" or "state"

HERE = Path(__file__).resolve().parent
COUNTRIES_DIR = HERE.parent / "countries" / "data"
STATES_DIR = HERE.parent / "states" / "data"
TEMPLATE = HERE / "template.html"
OUT_HTML = HERE / "index.html"

# (mode, geometry json, aspect json, top-level list key)
SPECS = [
    ("country", COUNTRIES_DIR / "countries.json", COUNTRIES_DIR / "countries_aspect.json", "countries"),
    ("state", STATES_DIR / "states.json", STATES_DIR / "states_aspect.json", "states"),
]


def load(geom_path: Path, aspect_path: Path, key: str) -> list[dict]:
    """Join per-feature rings (geometry file) with b/A/ar (aspect file) by name.

    Keeps only features present in both (i.e. that converged)."""
    for p in (geom_path, aspect_path):
        if not p.exists():
            raise SystemExit(
                f"missing {p} — run the countries and states pipelines first "
                "(see this script's docstring)"
            )
    rings_by = {g["name"]: g["rings"] for g in json.loads(geom_path.read_text())[key]}
    items = []
    for a in json.loads(aspect_path.read_text())[key]:
        if a["name"] in rings_by:
            items.append({
                "name": a["name"],
                "rings": rings_by[a["name"]],
                "b": a["b"],
                "A": a["A"],
                "ar": a["ar"],
            })
    items.sort(key=lambda x: x["name"])
    return items


def main() -> None:
    datasets = {mode: load(gp, ap, key) for mode, gp, ap, key in SPECS}

    html = (
        TEMPLATE.read_text()
        .replace("__DATASETS__", json.dumps(datasets))
        .replace("__INITIAL_MODE__", json.dumps(INITIAL_MODE))
    )
    OUT_HTML.write_text(html)
    for mode, items in datasets.items():
        print(f"  {mode}: {len(items)} selectable features")
    print(f"  initial mode = {INITIAL_MODE}")
    print(f"  wrote {OUT_HTML.relative_to(Path.cwd())}  — open it in a browser")


if __name__ == "__main__":
    main()
