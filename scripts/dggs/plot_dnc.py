# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Orthographic-project the two DNC cells from tests/dggs_dnc_test.zig
into their tangent planes and plot the polygons side-by-side.

Centroid = unit-normalized mean of the input vertices.
Orthographic projection at b: u = e1·(x - (b·x)b),  v = e2·(x - (b·x)b),
where {e1, e2, b} is a right-handed orthonormal frame.

Output: scripts/dggs/data/dnc_polygons.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = {
    "A5 r30 (cell 2a08d74e8e79123c)": np.array(
        [
            [-8.76368008991394400e-1, 3.45295754150762360e-1, 3.35782600773052830e-1],
            [-8.76368008698072600e-1, 3.45295754812974860e-1, 3.35782600857627600e-1],
            [-8.76368008522131700e-1, 3.45295755483736640e-1, 3.35782600627055400e-1],
            [-8.76368008823817700e-1, 3.45295755231014470e-1, 3.35782600099559000e-1],
            [-8.76368009047065800e-1, 3.45295754541700400e-1, 3.35782600225741100e-1],
        ]
    ),
    "S2 L30 (cell 332c258c3f285f93)": np.array(
        [
            [-6.84434006983608300e-1, 7.11477104991097700e-1, 1.59218149586812550e-1],
            [-6.84434007909358400e-1, 7.11477104143007500e-1, 1.59218149397022360e-1],
            [-6.84434007784890200e-1, 7.11477104013621300e-1, 1.59218150510246930e-1],
            [-6.84434006859140100e-1, 7.11477104861711600e-1, 1.59218150700037110e-1],
        ]
    ),
}

OUT_PATH = Path(__file__).resolve().parent / "data" / "dnc_polygons.png"
EARTH_R_M = 6_371_008.8  # mean Earth radius — for human-readable scale


def tangent_frame(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (e1, e2) spanning the tangent plane at b. Pick whichever
    world axis is least parallel to b to avoid near-zero cross products."""
    candidate = np.eye(3)[np.argmin(np.abs(b))]
    e1 = np.cross(candidate, b)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    return e1, e2


def project(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Orthographic project verts (N, 3) at the unit-normalized mean.
    Returns (uv (N, 2), centroid_unit, span_in_meters)."""
    mean = verts.mean(axis=0)
    b = mean / np.linalg.norm(mean)
    e1, e2 = tangent_frame(b)
    tangential = verts - (verts @ b)[:, None] * b
    uv = np.column_stack([tangential @ e1, tangential @ e2])
    span_dimless = np.ptp(uv, axis=0).max()
    return uv, b, span_dimless * EARTH_R_M


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (label, verts) in zip(axes, CASES.items()):
        uv, b, span_m = project(verts)
        # Close the ring for plotting.
        ring = np.vstack([uv, uv[0:1]])
        ax.plot(ring[:, 0], ring[:, 1], "-o", lw=1.2, ms=5, color="C0")
        for i, (u, v) in enumerate(uv):
            ax.annotate(str(i), (u, v), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("u  (tangent units)")
        ax.set_ylabel("v  (tangent units)")
        # Equivalent Earth-surface scale, since these cells are at finest DGGS resolution.
        ax.set_title(
            f"{label}\nbbox span ≈ {span_m * 1000:.2f} mm on Earth"
        )
    fig.suptitle("DNC cells projected to tangent plane at centroid (orthographic)")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
