# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Step 3 of the DGGS aspect-ratio survey (see docs/dggs-aspect-survey-plan.md).

Loads scripts/dggs/data/aspect.json (produced by `zig build dggs-aspect`) and
plots the per-system aspect-ratio distribution for converged cells, with
shared bins so the systems are directly comparable. Writes one PNG per system
plus a combined panel, and prints a summary stats table.

The survey is solved at gap_tol = 1e-3 (not skar's strict 1e-6 default): at
finest resolution the S2/A5 cells hit an f64 duality-gap floor (~1e-4–1e-3)
and DNC at 1e-6, but their aspect ratios are accurate regardless. Running at
1e-3 lets every cell converge, so these histograms are the complete AR
distribution rather than dropping ~22% of S2 / ~47% of A5. See
scripts/dggs/aspect.zig and tests/dggs_dnc_test.zig.

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
N_BINS = 60
SYSTEMS = ["h3", "s2", "a5"]
SYS_LABEL = {"h3": "H3 r15", "s2": "S2 L30", "a5": "A5 r30"}
SYS_COLOR = {"h3": "C0", "s2": "C1", "a5": "C2"}
# -------------------------------------------------------------------------


def load() -> dict:
    with open(ASPECT_JSON) as f:
        return json.load(f)


def stats(ars: np.ndarray) -> dict:
    return {
        "n": ars.size,
        "min": float(ars.min()),
        "median": float(np.median(ars)),
        "p99": float(np.percentile(ars, 99)),
        "max": float(ars.max()),
    }


def main() -> None:
    data = load()
    ars = {s: np.asarray(data[s]["ars"], dtype=float) for s in SYSTEMS}
    dnc = {s: int(data[s]["counts"]["did_not_converge"]) for s in SYSTEMS}
    st_by_sys = {s: stats(ars[s]) for s in SYSTEMS}  # computed once, reused below

    # Shared bins across all systems for comparability. AR ≥ 1 by definition.
    global_max = max(a.max() for a in ars.values() if a.size)
    bins = np.linspace(1.0, global_max, N_BINS + 1)

    # Summary stats table.
    print(f"{'sys':5} {'n_conv':>8} {'n_dnc':>7} {'min':>10} {'median':>10} {'p99':>10} {'max':>10}")
    for s in SYSTEMS:
        st = st_by_sys[s]
        print(f"{s:5} {st['n']:>8} {dnc[s]:>7} {st['min']:>10.6f} "
              f"{st['median']:>10.6f} {st['p99']:>10.6f} {st['max']:>10.6f}")

    # Per-system PNGs.
    for s in SYSTEMS:
        st = st_by_sys[s]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ars[s], bins=bins, color=SYS_COLOR[s], edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")  # AR clusters near 1 with a long thin tail
        ax.set_xlabel("aspect ratio")
        ax.set_ylabel("cell count (log)")
        ax.set_title(
            f"{SYS_LABEL[s]} aspect ratio  (n={st['n']}, DNC={dnc[s]})\n"
            f"median {st['median']:.4f} · p99 {st['p99']:.4f} · max {st['max']:.4f}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = DATA_DIR / f"hist_{s}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")

    # Combined panel: shared x-axis for direct comparison.
    fig, axes = plt.subplots(len(SYSTEMS), 1, figsize=(8, 9), sharex=True)
    for ax, s in zip(axes, SYSTEMS):
        st = st_by_sys[s]
        ax.hist(ars[s], bins=bins, color=SYS_COLOR[s], edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_ylabel("count (log)")
        ax.set_title(f"{SYS_LABEL[s]}  (median {st['median']:.4f}, max {st['max']:.4f}, DNC {dnc[s]})",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("aspect ratio (shared bins, gap_tol = 1e-3)")
    fig.suptitle("DGGS finest-resolution aspect-ratio distributions", fontsize=12)
    fig.tight_layout()
    out = DATA_DIR / "hist_combined.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
