## analysis_utils.py
# Reusable helpers for post-simulation analysis.
# Relies on your existing distribution.py and measure.py modules.

from collections import namedtuple
import numpy as np
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from distribution import SphericalSpace, init_distance_matrix
from measure import (
    cluster_matrix,
    total_distance,
    average_nearest_neighbor_distance,
    nearest_neighbor_index_spherical,
)

# Use the same space object as your simulator
SPACE = SphericalSpace()

# A simple container for metric series
MetricSeries = namedtuple("MetricSeries", ["clusters", "total_dist", "avg_nnd", "nni"])


def compute_metrics_per_slot(slots, granularity: int = 1) -> MetricSeries:
    """
    Compute four spatial metrics for each slot.

    Args:
        slots: List of point-lists, one per slot.
        granularity: only recompute from scratch every `granularity` slots.

    Returns:
        MetricSeries of four numpy arrays, each of length = len(slots).
    """
    n = len(slots)
    c = np.empty(n, dtype=float)
    td = np.empty(n, dtype=float)
    nnd = np.empty(n, dtype=float)
    nni = np.empty(n, dtype=float)

    last_c = last_td = last_nnd = last_nni = 0.0
    for i, pts in enumerate(slots):
        if i % granularity == 0 and len(pts) > 1:
            dm = init_distance_matrix(pts, SPACE)
            last_c = cluster_matrix(dm)
            last_td = total_distance(dm)
            last_nnd = average_nearest_neighbor_distance(dm)
            last_nni = nearest_neighbor_index_spherical(dm, SPACE)[0]
        elif len(pts) <= 1:
            last_c = last_td = last_nnd = last_nni = 0.0

        c[i] = last_c
        td[i] = last_td
        nnd[i] = last_nnd
        nni[i] = last_nni

    return MetricSeries(c, td, nnd, nni)


def load_slots(path):
    with open(path) as f:
        return json.load(f)


def single_line(x, y, title, ylabel, outfile):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, lw=1.6)
    plt.title(title)
    plt.xlabel("Slot")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.savefig(outfile.with_suffix(".pdf"))
    print("✓", outfile)


def main(args):
    slots = load_slots(args.data)
    n = len(slots)
    x = list(range(1, n + 1))

    # Compute all four series in one call
    metrics = compute_metrics_per_slot(slots, granularity=10)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    single_line(
        x, metrics.clusters, "# Clusters", "clusters", outdir / "clusters_over_time.png"
    )
    single_line(
        x,
        metrics.total_dist,
        "Total Distance",
        "distance",
        outdir / "total_distance.png",
    )
    single_line(x, metrics.avg_nnd, "Avg. NND", "distance", outdir / "avg_nnd.png")
    single_line(x, metrics.nni, "NNI", "nni", outdir / "nni.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--data", required=True, help="Path to data.json (list of slots)"
    )
    p.add_argument(
        "-o", "--outdir", default="figures", help="Where to write PNG/PDF outputs"
    )
    # p.add_argument(
    #     "-g",
    #     "--granularity",
    #     type=int,
    #     default=1,
    #     help="Recompute metrics every g-th slot",
    # )
    args = p.parse_args()
    main(args)
