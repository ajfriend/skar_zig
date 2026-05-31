# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Step 4 of the DGGS aspect-ratio survey (see docs/dggs-aspect-survey-plan.md).

Reads the per-system `best` (smallest-AR / most circular) and `worst`
(largest-AR) converged cells from scripts/dggs/data/aspect.json and draws a
3x2 grid into data/extremes.png: one row per system (H3, S2, A5), left column
= best-AR example, right column = worst-AR example. Each panel shows the cell
boundary gnomonic-projected at its cone axis `b`, with the tightest enclosing
ellipse (the cone cross-section) overlaid.

Ellipse. skar's cone is the SECOND-ORDER cone {x : ‖A x‖ ≤ b·x}, so the
relevant form is A² (not A). With U = [u1 u2] ⊥ b and y = Uᵀx/(b·x), and
A b = σ0 b (so bᵀA U = 0), ‖Ax‖²/(b·x)² = σ0² + yᵀ(UᵀA²U)y; the cross-section
is { y : yᵀ(UᵀA²U) y = 1 − σ0² = 2/3 }, axis ratio σ2/σ1 = the reported AR.
Each panel is rotated into M2's eigenframe with the MAJOR axis (smaller
eigenvalue → longer semi-axis) along x and the minor axis along y.

No CLI args (project convention): edit the constants below in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----- knobs -------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
ASPECT_JSON = DATA_DIR / "aspect.json"
SYSTEMS = ["h3", "s2", "a5"]
SYS_LABEL = {"h3": "H3 r15", "s2": "S2 L30", "a5": "A5 r30"}
SYS_COLOR = {"h3": "C0", "s2": "C1", "a5": "C2"}
EARTH_R_M = 6_371_008.8  # scale gnomonic (dimensionless) coords to metres
# -------------------------------------------------------------------------


def tangent_basis(b: np.ndarray) -> np.ndarray:
    """Orthonormal U = [u1 u2] (3x2) spanning the plane ⊥ b."""
    cand = np.eye(3)[np.argmin(np.abs(b))]
    u1 = np.cross(cand, b); u1 /= np.linalg.norm(u1)
    u2 = np.cross(b, u1)
    return np.column_stack([u1, u2])


def draw(ax, detail: dict, color: str) -> None:
    """Draw one cell + its enclosing-cone ellipse, major axis along x."""
    b = np.asarray(detail["b"], dtype=float); b /= np.linalg.norm(b)
    A = np.asarray(detail["A"], dtype=float)
    verts = np.asarray(detail["vertices"], dtype=float)
    U = tangent_basis(b)

    y = (verts @ U) / (verts @ b)[:, None]              # gnomonic projection
    M2 = U.T @ (A @ A) @ U                              # tangent block of A²
    c = 1.0 - float(b @ A @ b) ** 2                     # budget level = 2/3

    evals, evecs = np.linalg.eigh(M2)                   # ascending; [:,0] = major dir
    y_aln = (y @ evecs) * EARTH_R_M                     # major-axis coord → x
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    sx = np.sqrt(c / evals[0]) * EARTH_R_M              # major semi-axis → x
    sy = np.sqrt(c / evals[1]) * EARTH_R_M              # minor semi-axis → y

    ring = np.vstack([y_aln, y_aln[0:1]])
    ax.plot(ring[:, 0], ring[:, 1], "-o", color=color, lw=1.3, ms=4, label="cell")
    ax.plot(sx * np.cos(t), sy * np.sin(t), "-", color="0.25", lw=1.5, label="enclosing ellipse")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main() -> None:
    with open(ASPECT_JSON) as f:
        data = json.load(f)
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    axes[0, 0].set_title("best AR (most circular)", fontsize=12, pad=10)
    axes[0, 1].set_title("worst AR", fontsize=12, pad=10)

    for row, s in enumerate(SYSTEMS):
        for col, kind in ((0, "best"), (1, "worst")):
            ax = axes[row, col]
            d = data[s][kind]
            draw(ax, d, SYS_COLOR[s])
            ax.set_xlabel("major axis (m)")
            if col == 0:
                ax.set_ylabel(f"{SYS_LABEL[s]}\nminor axis (m)")
            ax.text(0.03, 0.95, f"AR {d['ar']:.4f}\nid {d['id']}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("DGGS finest-resolution cells: best vs worst aspect ratio\n"
                 "(enclosing-cone cross-section ‖Ax‖ ≤ b·x; major axis horizontal)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = DATA_DIR / "extremes.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    for s in SYSTEMS:
        print(f"  {s}: best AR {data[s]['best']['ar']:.4f}  ·  worst AR {data[s]['worst']['ar']:.4f}")


if __name__ == "__main__":
    main()
