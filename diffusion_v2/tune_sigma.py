"""
tune_sigma.py

Visual utility to verify sigma values in render_density_channel before
running generate_dataset.py on the full dataset.

Renders the density field at a sweep of alpha values, where
sigma_i = alpha * diameter_i (sigma scales with each particle's own
diameter — correct regardless of which type label is the larger species),
and prints a diagnostic checklist for each. Run this on one representative
LAMMPS file and pick the alpha that passes all three checks.

Usage:
    python tune_sigma.py --file data/raw/config/100
    python tune_sigma.py --file data/raw/config/100 --grid_size 384 --zoom 0.15
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from lammps_parser import read_lammps_data


# ---------------------------------------------------------------------------
# Core renderer (mirrors render_density_channel but accepts explicit sigmas)
# ---------------------------------------------------------------------------

def render(atoms, xlo, ylo, box_size_x, box_size_y,
           alpha, grid_size=384):
    """
    Single-channel amplitude-encoded density field.

    sigma is derived per-particle from its own diameter:
        sigma_i = alpha * diameter_i
    This is correct regardless of which integer type label corresponds
    to the larger particle — it reads diameter directly rather than
    assuming type==2 is "large".
    """
    image = np.zeros((grid_size, grid_size), dtype=np.float32)

    x_grid = np.linspace(xlo, xlo + box_size_x, grid_size)
    y_grid = np.linspace(ylo, ylo + box_size_y, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    max_diameter = atoms[:, 1].max()

    for particle_type, diameter, x, y in atoms:
        sigma     = alpha * diameter
        amplitude = diameter / max_diameter

        image += amplitude * np.exp(
            -((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2)
        )

    return image


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_vacancies(field, threshold=0.15):
    """
    A vacancy should appear as a local minimum clearly below the bulk crystal
    floor. We check whether the bottom `threshold` fraction of pixel values
    sits in a clearly separate mode from the bulk.
    Returns True if vacancies are likely visible.
    """
    flat = field.ravel()
    low  = np.percentile(flat, threshold * 100)
    bulk = np.percentile(flat, 50)           # median = typical crystal pixel
    return (bulk - low) / (bulk + 1e-8) > 0.3


def check_amplitude_contrast(atoms, field, xlo, ylo,
                              box_size_x, box_size_y, grid_size):
    """
    Particles with larger diameter should produce visibly higher peaks
    than particles with smaller diameter. Splits the population by
    diameter value directly (not by type label, since type labels do
    not necessarily correspond to small/large in a fixed way).
    Returns (mean_peak_small, mean_peak_large, contrast_ok).
    """
    diameters = atoms[:, 1]
    unique_diams = np.unique(diameters)

    if len(unique_diams) < 2:
        return None, None, True   # monodisperse system, skip check

    small_diam = unique_diams.min()
    large_diam = unique_diams.max()

    small = atoms[diameters == small_diam]
    large = atoms[diameters == large_diam]

    def sample_peaks(subset, n=30):
        peaks = []
        rng = np.random.default_rng(42)
        chosen = subset[rng.choice(len(subset), min(n, len(subset)), replace=False)]
        for _, _, x, y in chosen:
            ix = np.clip(int((x - xlo) / box_size_x * grid_size), 0, grid_size - 1)
            iy = np.clip(int((y - ylo) / box_size_y * grid_size), 0, grid_size - 1)
            peaks.append(field[iy, ix])   # exact centre pixel
        return np.mean(peaks) if peaks else 0.0

    p_small = sample_peaks(small) if len(small) > 0 else None
    p_large = sample_peaks(large) if len(large) > 0 else None

    if p_small is None or p_large is None:
        return p_small, p_large, True

    contrast_ok = p_large > p_small * 1.1
    return p_small, p_large, contrast_ok


def check_neighbour_separation(field, atoms, xlo, ylo,
                                box_size_x, box_size_y, grid_size):
    """
    Between adjacent particles of the bulk/matrix species (the most
    numerous diameter value — typically the dense crystalline lattice),
    the density should dip below half the particle peak value, i.e.
    particles are individually resolved rather than merged into a blob.
    Checks up to 60 random adjacent pairs from that sub-lattice.
    Returns fraction of pairs that show a clear dip.
    """
    diameters = atoms[:, 1]
    unique_diams, counts = np.unique(diameters, return_counts=True)
    bulk_diam = unique_diams[counts.argmax()]   # most numerous species
    bulk = atoms[diameters == bulk_diam]

    if len(bulk) < 2:
        return 1.0

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(bulk), min(60, len(bulk)), replace=False)
    sample = bulk[idxs]

    # Build simple KD-style nearest-neighbour search
    coords = sample[:, 2:]   # (N, 2)
    dists  = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=-1
    )
    np.fill_diagonal(dists, np.inf)
    nn_idx = dists.argmin(axis=1)

    clear_dip = 0
    n_pairs   = 0

    for i, j in zip(range(len(sample)), nn_idx):
        x1, y1 = sample[i, 2], sample[i, 3]
        x2, y2 = sample[j, 2], sample[j, 3]
        # Sample 10 points along the line between the two particles
        ts  = np.linspace(0.2, 0.8, 10)
        xs  = x1 + ts * (x2 - x1)
        ys  = y1 + ts * (y2 - y1)
        ixs = np.clip(
            ((xs - xlo) / box_size_x * grid_size).astype(int),
            0, grid_size - 1,
        )
        iys = np.clip(
            ((ys - ylo) / box_size_y * grid_size).astype(int),
            0, grid_size - 1,
        )
        mid_vals = field[iys, ixs]

        # Peak value at each endpoint
        ix1 = np.clip(int((x1 - xlo) / box_size_x * grid_size), 0, grid_size-1)
        iy1 = np.clip(int((y1 - ylo) / box_size_y * grid_size), 0, grid_size-1)
        peak = field[iy1, ix1]

        if mid_vals.min() < 0.6 * peak:
            clear_dip += 1
        n_pairs += 1

    return clear_dip / n_pairs if n_pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Sigma sweep plot
# ---------------------------------------------------------------------------

def alpha_sweep(
    atoms, xlo, ylo, box_size_x, box_size_y,
    alphas,
    grid_size=384,
    zoom=0.12,
):
    """
    Renders a 1D grid of images across alpha values (sigma_i = alpha * diameter_i).
    zoom: fraction of the box to show in the zoomed inset (0.12 = 12%)
    """
    n = len(alphas)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    gs  = GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.15)

    # Zoom window: centre of the box
    zw = zoom
    zx0 = 0.5 - zw / 2
    zx1 = 0.5 + zw / 2

    print("\nDiagnostic checklist")
    print("=" * 72)
    print(f"{'alpha':>7}  {'Vacancies':>10}  "
          f"{'Pk_small':>9}  {'Pk_large':>9}  {'Contrast':>9}  {'Sep%':>6}")
    print("-" * 72)

    for idx, a in enumerate(alphas):
        field = render(
            atoms, xlo, ylo, box_size_x, box_size_y,
            alpha=a, grid_size=grid_size,
        )

        # --- Zoom to centre region ---
        H, W  = field.shape
        r0, r1 = int(zx0 * H), int(zx1 * H)
        c0, c1 = int(zx0 * W), int(zx1 * W)
        zoomed = field[r0:r1, c0:c1]

        row, col = idx // ncols, idx % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(zoomed, origin="lower", cmap="inferno",
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Diagnostics ---
        vac_ok  = check_vacancies(field)
        p_s, p_l, con_ok = check_amplitude_contrast(
            atoms, field, xlo, ylo, box_size_x, box_size_y, grid_size
        )
        sep     = check_neighbour_separation(
            field, atoms, xlo, ylo, box_size_x, box_size_y, grid_size
        )
        sep_ok  = sep > 0.7

        all_ok  = vac_ok and con_ok and sep_ok

        tick    = "✓" if all_ok else "✗"
        color   = "limegreen" if all_ok else "tomato"

        title = (
            f"alpha={a:.3f}\n"
            f"vac={'✓' if vac_ok else '✗'}  "
            f"contrast={'✓' if con_ok else '✗'}  "
            f"sep={'✓' if sep_ok else '✗'}  "
            f"{tick}"
        )
        ax.set_title(title, fontsize=8,
                     color=color, fontweight="bold")

        ps_str = f"{p_s:.3f}" if p_s is not None else "  N/A"
        pl_str = f"{p_l:.3f}" if p_l is not None else "  N/A"
        print(
            f"{a:>7.3f}  "
            f"{'YES' if vac_ok else 'NO':>10}  "
            f"{ps_str:>9}  {pl_str:>9}  "
            f"{'YES' if con_ok else 'NO':>9}  "
            f"{sep*100:>5.0f}%"
        )

    print("=" * 72)
    print("Checks:  vac = vacancies visible  |  contrast = large-diameter peaks higher"
          "  |  sep = particles individually resolved")

    fig.suptitle(
        "Alpha sweep — density channel (zoomed to centre region)\n"
        "Green title = passes all checks",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Single-sigma deep inspection
# ---------------------------------------------------------------------------

def inspect_single(
    atoms, xlo, ylo, box_size_x, box_size_y,
    alpha,
    grid_size=384,
    zoom=0.12,
):
    """
    Detailed render for one chosen alpha (sigma_i = alpha * diameter_i).
    Shows: full field, zoomed centre, line profile, and amplitude
    histogram split by diameter (small vs large species).
    """
    field = render(
        atoms, xlo, ylo, box_size_x, box_size_y,
        alpha=alpha, grid_size=grid_size,
    )

    H, W  = field.shape
    zw    = zoom
    r0, r1 = int((0.5 - zw/2) * H), int((0.5 + zw/2) * H)
    c0, c1 = int((0.5 - zw/2) * W), int((0.5 + zw/2) * W)
    zoomed = field[r0:r1, c0:c1]

    # Line profile through the horizontal midline of the zoomed region
    mid_row   = zoomed[zoomed.shape[0] // 2, :]
    x_profile = np.linspace(0, 1, len(mid_row))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # 1. Full field
    axes[0].imshow(field, origin="lower", cmap="inferno")
    axes[0].set_title("Full field")
    rect = plt.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=1.5, edgecolor="cyan", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].axis("off")

    # 2. Zoomed centre
    axes[1].imshow(zoomed, origin="lower", cmap="inferno",
                   interpolation="nearest")
    axes[1].axhline(zoomed.shape[0] // 2, color="cyan", lw=1, ls="--")
    axes[1].set_title(f"Zoomed centre  (alpha={alpha:.3f})")
    axes[1].axis("off")

    # 3. Line profile
    axes[2].plot(x_profile, mid_row, color="steelblue", lw=1.2)
    axes[2].set_xlabel("Position (normalised)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Horizontal line profile")
    axes[2].axhline(mid_row.max() * 0.5, color="tomato",
                    lw=0.8, ls=":", label="50% peak")
    axes[2].legend(fontsize=8)

    # 4. Amplitude histogram — small vs large diameter species
    diameters = atoms[:, 1]
    unique_diams, counts = np.unique(diameters, return_counts=True)
    small_diam = unique_diams.min()
    large_diam = unique_diams.max()
    small = atoms[diameters == small_diam]
    large = atoms[diameters == large_diam]

    def particle_peaks(subset, n=200):
        peaks = []
        rng = np.random.default_rng(0)
        chosen = subset[rng.choice(len(subset), min(n, len(subset)), replace=False)]
        for _, _, x, y in chosen:
            ix = np.clip(int((x - xlo) / box_size_x * grid_size), 0, grid_size-1)
            iy = np.clip(int((y - ylo) / box_size_y * grid_size), 0, grid_size-1)
            peaks.append(field[iy, ix])
        return np.array(peaks)

    bins = np.linspace(0, field.max(), 60)
    if len(small) > 0:
        axes[3].hist(particle_peaks(small), bins=bins, alpha=0.6,
                     color="steelblue", label=f"diameter={small_diam:.2f} (n={len(small)})")
    if len(large) > 0 and large_diam != small_diam:
        axes[3].hist(particle_peaks(large), bins=bins, alpha=0.6,
                     color="tomato",    label=f"diameter={large_diam:.2f} (n={len(large)})")
    axes[3].set_xlabel("Peak density value")
    axes[3].set_ylabel("Count")
    axes[3].set_title("Peak amplitude by particle diameter")
    axes[3].legend(fontsize=8)

    # Print checklist
    vac_ok          = check_vacancies(field)
    p_s, p_l, con_ok = check_amplitude_contrast(
        atoms, field, xlo, ylo, box_size_x, box_size_y, grid_size
    )
    sep             = check_neighbour_separation(
        field, atoms, xlo, ylo, box_size_x, box_size_y, grid_size
    )
    sep_ok = sep > 0.7

    print(f"\nInspection: alpha={alpha:.3f}")
    print(f"  Vacancies visible       : {'✓ YES' if vac_ok  else '✗ NO  — increase alpha to smooth, or decrease to sharpen dips'}")
    if con_ok is True and p_s is None:
        print(f"  Amplitude contrast      : n/a (monodisperse system)")
    else:
        print(f"  Amplitude contrast      : {'✓ YES' if con_ok  else f'✗ NO  — large-diam peak ({p_l:.3f}) not > small-diam peak ({p_s:.3f}) × 1.1'}")
    print(f"  Particle separation     : {'✓ YES' if sep_ok  else f'✗ NO  — {sep*100:.0f}% of pairs resolved; decrease alpha to separate'}")

    fig.suptitle(
        f"Single-alpha inspection  alpha={alpha:.3f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visually verify sigma (alpha * diameter) values for the density channel"
    )
    parser.add_argument(
        "--file", type=str, required=True,
        help="Path to a single LAMMPS config file",
    )
    parser.add_argument(
        "--grid_size", type=int, default=384,
        help="Render resolution (default: 384, must match model sample_size)",
    )
    parser.add_argument(
        "--zoom", type=float, default=0.12,
        help="Fraction of box shown in zoomed region (default: 0.12)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="If set, skip the sweep and inspect this single alpha value "
             "(sigma_i = alpha * diameter_i)",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        help="Alpha values to sweep (default: 0.10 0.15 0.20 0.25 0.30 0.35)",
    )

    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    _, _, atoms, xlo, ylo, box_size_x, box_size_y = read_lammps_data(args.file)

    diameters = atoms[:, 1]
    unique_diams, counts = np.unique(diameters, return_counts=True)
    for d, c in zip(unique_diams, counts):
        print(f"  Diameter {d:.3f} : n={c}")
    print(f"  Box             : {box_size_x:.2f} × {box_size_y:.2f}")
    print(f"  Grid            : {args.grid_size} × {args.grid_size}")
    pixel_size = box_size_x / args.grid_size
    print(f"  Pixel size      : {pixel_size:.4f} (real units)")
    print(f"  Min diameter    : {unique_diams.min():.3f}  "
          f"({unique_diams.min()/pixel_size:.1f} pixels across)\n")

    if args.alpha is not None:
        # Single inspection mode
        inspect_single(
            atoms, xlo, ylo, box_size_x, box_size_y,
            alpha=args.alpha,
            grid_size=args.grid_size, zoom=args.zoom,
        )
    else:
        # Sweep mode
        alpha_sweep(
            atoms, xlo, ylo, box_size_x, box_size_y,
            alphas=args.alphas,
            grid_size=args.grid_size,
            zoom=args.zoom,
        )