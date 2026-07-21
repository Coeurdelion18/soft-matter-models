"""
Grain-boundary analysis of particle configurations -- real or generated.

Renders the same four panels as the ground-truth analysis figure:
    density | |psi6| | arg(psi6) | Voronoi sides
and prints summary statistics. Optionally compares a generated configuration
against a reference (real) one: the figure gains a second row and Wasserstein
distances between the structural distributions are printed.

Works on finite open patches (no periodic box needed): neighbours come from a
Delaunay triangulation and particles within MARGIN of the convex hull are
excluded from statistics so edge artifacts don't pollute the numbers.

Usage:
    python evaluate.py generated_samples/sample_000.npz
    python evaluate.py generated_samples/sample_000.npz --ref data/patches/val/012_patch003.npz
    python evaluate.py data/patches/train/001_patch010.npz --out my_figure.png

Accepts either generated npz files (keys: pos, types, sizes) or patch files
from make_patches.py (keys: pos, node_scalars).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import Delaunay, Voronoi, ConvexHull, cKDTree
from scipy.stats import wasserstein_distance

MARGIN = 2.0            # exclude particles this close to the hull (in units
                        # of particle diameter) from all statistics
GB_PSI6_THRESHOLD = 0.7 # |psi6| below this counts as grain-boundary-like


# ── Structural analysis ───────────────────────────────────────────────────────

def load_configuration(path):
    """Return pos (N,2), types (N,), sizes (N,) from either file format."""
    d = np.load(path)
    pos = d["pos"].astype(np.float64)
    if "types" in d:
        types = d["types"].astype(np.float32)
        sizes = d["sizes"].astype(np.float32)
    elif "node_scalars" in d:
        ns = d["node_scalars"]
        sizes = ns[:, 7].astype(np.float32)
        types = ns[:, 8].astype(np.float32)
    else:
        raise KeyError(f"{path}: expected keys pos+types+sizes or pos+node_scalars")
    return pos, types, sizes


def delaunay_neighbors(pos):
    tri = Delaunay(pos)
    indptr, indices = tri.vertex_neighbor_vertices
    return [indices[indptr[i]:indptr[i + 1]] for i in range(len(pos))]


def psi_k(pos, neighbors, k=6):
    """Bond-orientational order parameter psi_k per particle (complex)."""
    psi = np.zeros(len(pos), dtype=complex)
    for i, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            continue
        d = pos[nbrs] - pos[i]
        ang = np.arctan2(d[:, 1], d[:, 0])
        psi[i] = np.exp(1j * k * ang).mean()
    return psi


def voronoi_cell_areas(pos):
    """Area of each bounded Voronoi cell; NaN for unbounded (edge) cells."""
    vor = Voronoi(pos)
    areas = np.full(len(pos), np.nan)
    for i, ridx in enumerate(vor.point_region):
        region = vor.regions[ridx]
        if len(region) == 0 or -1 in region:
            continue
        poly = vor.vertices[region]
        x, y = poly[:, 0], poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return areas


def interior_mask(pos, margin=MARGIN):
    """True for particles at least `margin` inside the convex hull."""
    hull = ConvexHull(pos)
    # hull.equations: normal . x + offset <= 0 for interior points
    dist = -(pos @ hull.equations[:, :2].T + hull.equations[:, 2])
    return dist.min(axis=1) >= margin


def nn_distances(pos):
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=2)
    return d[:, 1]


def analyse(pos, types, sizes):
    neighbors = delaunay_neighbors(pos)
    psi6 = psi_k(pos, neighbors, k=6)
    n_sides = np.array([len(n) for n in neighbors])
    interior = interior_mask(pos)
    return {
        "pos": pos, "types": types, "sizes": sizes,
        "psi6_mag": np.abs(psi6), "psi6_arg": np.angle(psi6),
        "n_sides": n_sides,
        "voronoi_area": voronoi_cell_areas(pos),
        "nn_dist": nn_distances(pos),
        "interior": interior,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def sides_cmap():
    """5 -> red, 6 -> orange, 7 -> white, <=4 -> dark red, >=8 -> green."""
    colors = ["#7b1a10", "#c0392b", "#e67e3c", "#f5f0e6", "#2a8c4a"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 4.5, 5.5, 6.5, 7.5, 20], cmap.N)
    return cmap, norm


def plot_row(axes, a, label, marker_size):
    pos, interior = a["pos"], a["interior"]

    ax = axes[0]
    ax.hist2d(pos[:, 0], pos[:, 1], bins=150, cmap="magma")
    # defect identity comes from size, not the type column: in this dataset
    # type 0 = large and type 1 = small (inverted vs the obvious convention)
    defect = a["sizes"] > 1.2
    ax.scatter(pos[defect, 0], pos[defect, 1], s=marker_size * 2,
               c="#ffd47f", edgecolors="none")
    ax.set_title(f"{label} — density", fontsize=10)

    ax = axes[1]
    ax.scatter(pos[interior, 0], pos[interior, 1], s=marker_size,
               c=a["psi6_mag"][interior], cmap="viridis", vmin=0, vmax=1,
               edgecolors="none")
    ax.set_title(f"{label} — |psi6|", fontsize=10)

    ax = axes[2]
    ax.scatter(pos[interior, 0], pos[interior, 1], s=marker_size,
               c=a["psi6_arg"][interior], cmap="twilight",
               vmin=-np.pi, vmax=np.pi, edgecolors="none")
    ax.set_title(f"{label} — arg(psi6)", fontsize=10)

    ax = axes[3]
    cmap, norm = sides_cmap()
    ax.scatter(pos[interior, 0], pos[interior, 1], s=marker_size,
               c=a["n_sides"][interior], cmap=cmap, norm=norm,
               edgecolors="none")
    ax.set_title(f"{label} — Voronoi sides", fontsize=10)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])


def print_stats(a, label):
    m = a["interior"]
    psi6 = a["psi6_mag"][m]
    sides = a["n_sides"][m]
    va = a["voronoi_area"][m]
    va = va[np.isfinite(va)]
    print(f"\n-- {label} ----------------------------------")
    print(f"  particles (interior/total):  {m.sum()} / {len(m)}")
    print(f"  defects (size>1.2):          {int((a['sizes'] > 1.2).sum())}")
    print(f"  mean |psi6|:                 {psi6.mean():.4f}")
    print(f"  GB fraction (|psi6|<{GB_PSI6_THRESHOLD}):    {(psi6 < GB_PSI6_THRESHOLD).mean():.4f}")
    print(f"  median NN distance:          {np.median(a['nn_dist'][m]):.4f}")
    print(f"  mean Voronoi area:           {va.mean():.4f}")
    frac6 = (sides == 6).mean()
    frac5 = (sides == 5).mean()
    frac7 = (sides == 7).mean()
    print(f"  coordination 5/6/7 fractions: {frac5:.3f} / {frac6:.3f} / {frac7:.3f}")


def compare_stats(a, b):
    """Wasserstein distances between structural distributions (a=gen, b=ref)."""
    ma, mb = a["interior"], b["interior"]
    pairs = [
        ("|psi6|",          a["psi6_mag"][ma],  b["psi6_mag"][mb]),
        ("NN distance",     a["nn_dist"][ma],   b["nn_dist"][mb]),
        ("Voronoi area",    a["voronoi_area"][ma], b["voronoi_area"][mb]),
        ("Voronoi sides",   a["n_sides"][ma].astype(float),
                            b["n_sides"][mb].astype(float)),
    ]
    print("\n-- generated vs reference (Wasserstein distances) --")
    for name, xa, xb in pairs:
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]
        print(f"  {name:15s}: {wasserstein_distance(xa, xb):.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sample", help="npz file to analyse (generated or patch)")
    parser.add_argument("--ref", default=None,
                        help="reference npz for side-by-side comparison")
    parser.add_argument("--out", default=None, help="output figure path")
    args = parser.parse_args()

    a = analyse(*load_configuration(args.sample))
    rows = [(a, Path(args.sample).stem)]
    if args.ref:
        b = analyse(*load_configuration(args.ref))
        rows.append((b, f"REF {Path(args.ref).stem}"))

    # marker size scaled so ~4000-particle patches render legibly
    marker_size = max(2.0, 4000.0 / len(a["pos"]) * 4.0)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.2 * n_rows), squeeze=False)
    for (data, label), row_axes in zip(rows, axes):
        plot_row(row_axes, data, label, marker_size)
        print_stats(data, label)

    if args.ref:
        compare_stats(a, b)

    out = args.out or ("outputs/results/" + Path(args.sample).stem + "_analysis.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
