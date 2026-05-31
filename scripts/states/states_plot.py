# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Step 3 of the US-states aspect-ratio example (mirrors scripts/dggs/extremes_plot.py).

Reads scripts/states/data/states.json (per-ring lon/lat outlines) and
states_aspect.json (per-state cone axis `b`, shape matrix `A`, aspect ratio),
joins them by name, and writes one PNG per state into data/<slug>.png. Each
panel shows the state boundary gnomonic-projected at its cone axis `b`, with the
tightest enclosing ellipse (the cone cross-section) overlaid, major axis along x.

Ellipse. skar's cone is the SECOND-ORDER cone {x : ‖A x‖ ≤ b·x}, so the relevant
form is A² (not A). With U = [u1 u2] ⊥ b and y = Uᵀx/(b·x), and A b = σ0 b (so
bᵀA U = 0), ‖Ax‖²/(b·x)² = σ0² + yᵀ(UᵀA²U)y; the cross-section is
{ y : yᵀ(UᵀA²U) y = 1 − σ0² = 2/3 }, axis ratio σ2/σ1 = the reported AR. Each
panel is rotated into M2's eigenframe with the MAJOR axis (smaller eigenvalue →
longer semi-axis) along x and the minor axis along y.

Caveat — Alaska: huge angular extent + antimeridian crossing. lon/lat→xyz
handles the wrap for the solve (AR is fine), but the gnomonic projection
stretches badly near 90° from the axis, so Alaska's panel looks distorted.
Per-ring drawing already prevents seam-crossing segments across its islands.

No CLI args (project convention): edit the constants below in place.
Run with:  uv run scripts/states/states_plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----- knobs -------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
STATES_JSON = DATA_DIR / "states.json"
ASPECT_JSON = DATA_DIR / "states_aspect.json"
EARTH_R_M = 6_371_008.8  # scale gnomonic (dimensionless) coords to metres
DPI = 200
# -------------------------------------------------------------------------


def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    c = math.cos(lat)
    return (c * math.cos(lon), c * math.sin(lon), math.sin(lat))


def tangent_basis(b: np.ndarray) -> np.ndarray:
    """Orthonormal U = [u1 u2] (3x2) spanning the plane ⊥ b."""
    cand = np.eye(3)[np.argmin(np.abs(b))]
    u1 = np.cross(cand, b); u1 /= np.linalg.norm(u1)
    u2 = np.cross(b, u1)
    return np.column_stack([u1, u2])


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def draw_state(state: dict, detail: dict) -> None:
    """Draw one state (each ring separately) + its enclosing ellipse."""
    b = np.asarray(detail["b"], dtype=float); b /= np.linalg.norm(b)
    A = np.asarray(detail["A"], dtype=float)
    U = tangent_basis(b)

    M2 = U.T @ (A @ A) @ U                              # tangent block of A²
    c = 1.0 - float(b @ A @ b) ** 2                     # budget level = 2/3
    evals, evecs = np.linalg.eigh(M2)                   # ascending; [:,0] = major dir
    # eigh's eigenvector signs are arbitrary, so `evecs` may be a reflection
    # (det -1) rather than a rotation — that would mirror the state. The
    # gnomonic frame (u1, u2) is already proper, so force det +1 to keep the
    # outline's true chirality (a 180° rotation is fine; a flip is not).
    if np.linalg.det(evecs) < 0:
        evecs[:, 1] = -evecs[:, 1]                       # negate minor dir; major still → x

    # Two proper rotations keep the major axis on x (they differ by 180°);
    # pick the one closer to a normal map view by making geographic north point
    # up. Negating both columns is still a rotation (det +1) and keeps the
    # major axis horizontal, so it only swings the panel around 180°.
    north_t = np.array([0.0, 0.0, 1.0]) - b[2] * b      # world north ⊥-projected to tangent
    north_y = float((north_t @ U) @ evecs @ np.array([0.0, 1.0]))
    if north_y < 0.0:
        evecs = -evecs                                   # 180° turn so north points up

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw each ring separately so multipolygon states (Alaska, Hawaii,
    # Michigan) don't get spurious segments connecting disjoint pieces.
    first = True
    for ring in state["rings"]:
        verts = np.array([lonlat_to_xyz(lon, lat) for lon, lat in ring])
        y = (verts @ U) / (verts @ b)[:, None]         # gnomonic projection
        y_aln = (y @ evecs) * EARTH_R_M                 # major-axis coord → x
        closed = np.vstack([y_aln, y_aln[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color="C0", lw=0.9,
                label="boundary" if first else None)
        first = False

    t = np.linspace(0.0, 2.0 * np.pi, 400)
    sx = np.sqrt(c / evals[0]) * EARTH_R_M              # major semi-axis → x
    sy = np.sqrt(c / evals[1]) * EARTH_R_M              # minor semi-axis → y
    ax.plot(sx * np.cos(t), sy * np.sin(t), "-", color="0.25", lw=1.5,
            label="enclosing ellipse")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("major axis (m)")
    ax.set_ylabel("minor axis (m)")
    ax.text(0.03, 0.97, f"{detail['name']}\nAR {detail['ar']:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{detail['name']}: tightest enclosing cone (AR {detail['ar']:.3f})")

    out = DATA_DIR / f"{slug(detail['name'])}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(Path.cwd())}")


def main() -> None:
    with open(STATES_JSON) as f:
        states = {s["name"]: s for s in json.load(f)["states"]}
    with open(ASPECT_JSON) as f:
        details = json.load(f)["states"]

    details.sort(key=lambda d: d["ar"], reverse=True)
    print(f"plotting {len(details)} states (most to least elongated)...")
    for d in details:
        draw_state(states[d["name"]], d)


if __name__ == "__main__":
    main()
