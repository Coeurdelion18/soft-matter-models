"""
evaluate.py

Compares generated samples from the diffusion model against real
LAMMPS-derived samples across four metrics:

    1. g(r)              — radial distribution function from channel 0 (density)
    2. |psi6| statistics — mean, std, and histogram from channel 1
    3. arg(psi6) uniformity — phase distribution from channel 2
    4. Voronoi defect fraction — fraction of non-6 pixels from channel 3

Usage:
    # Compare a single generated .npy against the real dataset
    python evaluate.py \
        --generated  outputs/generated_sample.npy \
        --real_data  data/processed/dataset.npy \
        --output_dir outputs/eval

    # Evaluate multiple generated samples at once
    python evaluate.py \
        --generated  outputs/gen_0.npy outputs/gen_1.npy \
        --real_data  data/processed/dataset.npy \
        --output_dir outputs/eval
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Channel indices (must match compute_channels.py)
# ---------------------------------------------------------------------------
CH_DENSITY  = 0
CH_PSI6_MAG = 1
CH_PSI6_PHA = 2
CH_VORONOI  = 3

CHANNEL_NAMES = ["Density", "|ψ₆|", "arg(ψ₆)", "Voronoi sides"]


# ===========================================================================
# 1. g(r) from density channel
# ===========================================================================

def _density_to_gr(density_field, n_bins=200):
    """
    Estimate g(r) from a 2D density field (channel 0).

    Method: autocorrelate the density field via FFT, then radially average.
    This is equivalent to computing the pair correlation from the positions
    but works directly on the rasterised Gaussian field without needing
    particle coordinates.

    Returns (r, g_r) normalised so that g(r) → 1 at large r.
    """
    H, W = density_field.shape

    # Zero-mean the field before autocorrelation
    field = density_field - density_field.mean()

    # Autocorrelation via FFT
    F = np.fft.fft2(field)
    acf = np.fft.ifft2(F * np.conj(F)).real
    acf = np.fft.fftshift(acf)

    # Build radius map (in pixels)
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    r_max = min(cx, cy)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    gr = np.zeros(n_bins)
    norm = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i + 1])
        if mask.sum() > 0:
            gr[i]   = acf[mask].mean()
            norm[i] = mask.sum()

    # Normalise: divide by the value at large r (last 10% of bins)
    tail = gr[int(0.9 * n_bins):].mean()
    if tail != 0:
        gr /= tail

    return bin_centers, gr


def compute_mean_gr(images, channel=CH_DENSITY, n_bins=200):
    """
    Compute mean g(r) over a stack of images (N, 4, H, W).
    Returns (r, mean_gr, std_gr).
    """
    all_gr = []
    for img in images:
        r, gr = _density_to_gr(img[channel], n_bins=n_bins)
        all_gr.append(gr)
    all_gr = np.array(all_gr)
    return r, all_gr.mean(axis=0), all_gr.std(axis=0)


# ===========================================================================
# 2. |psi6| statistics
# ===========================================================================

def psi6_stats(images, channel=CH_PSI6_MAG):
    """
    Returns per-image mean and std of |psi6|, plus the flattened
    distribution across all pixels in all images.
    """
    flat = np.concatenate([img[channel].ravel() for img in images])
    means = np.array([img[channel].mean() for img in images])
    stds  = np.array([img[channel].std()  for img in images])
    return means, stds, flat


# ===========================================================================
# 3. arg(psi6) uniformity
# ===========================================================================

def phase_uniformity(images, channel=CH_PSI6_PHA, n_bins=72):
    """
    Histograms the arg(psi6) field across all images.
    A perfectly polycrystalline (orientationally disordered) sample
    produces a flat histogram. A single crystal would show 6 sharp peaks.

    Returns (bin_centers, histogram) normalised to sum to 1.
    """
    flat = np.concatenate([img[channel].ravel() for img in images])
    # channel is normalised to [0,1] by compute_all_channels;
    # rescale back to [-pi, pi] for interpretability
    flat_rescaled = flat * 2 * np.pi - np.pi
    counts, edges = np.histogram(flat_rescaled, bins=n_bins, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = counts / counts.sum()
    return centers, counts


# ===========================================================================
# 4. Voronoi defect fraction
# ===========================================================================

def defect_fraction(images, channel=CH_VORONOI):
    """
    Fraction of pixels with Voronoi side count != 6, normalised to [0,1].
    The channel was normalised to [0,1] during dataset generation so we
    need a threshold: values not at the '6-neighbour' peak are defects.

    Because the channel is per-sample normalised, we threshold on the
    mode of the distribution (most common value = 6 neighbours).
    Returns per-image defect fractions.
    """
    fracs = []
    for img in images:
        field = img[channel]
        # Mode bin of the histogram = 6-neighbour pixels
        counts, edges = np.histogram(field.ravel(), bins=100)
        mode_val = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
        tolerance = 0.05  # within 5% of the mode = hexagonal
        defect_mask = np.abs(field - mode_val) > tolerance
        fracs.append(defect_mask.mean())
    return np.array(fracs)


# ===========================================================================
# Wasserstein distances (scalar summary per metric)
# ===========================================================================

def wasserstein_gr(r_real, gr_real, r_gen, gr_gen):
    """Earth-mover distance between two g(r) curves treated as distributions."""
    # Interpolate generated onto real r grid if needed
    gr_gen_interp = np.interp(r_real, r_gen, gr_gen)
    # Clip to positive (distributions must be non-negative)
    p = np.clip(gr_real, 0, None)
    q = np.clip(gr_gen_interp, 0, None)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p /= p.sum()
    q /= q.sum()
    return wasserstein_distance(r_real, r_real, p, q)


# ===========================================================================
# Plotting
# ===========================================================================

def plot_gr_comparison(r_real, gr_real, std_real,
                       r_gen,  gr_gen,  std_gen,
                       output_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(r_real, gr_real, label="Real", color="steelblue", lw=2)
    ax.fill_between(r_real,
                    gr_real - std_real,
                    gr_real + std_real,
                    alpha=0.25, color="steelblue")

    ax.plot(r_gen, gr_gen, label="Generated", color="tomato", lw=2, ls="--")
    ax.fill_between(r_gen,
                    gr_gen - std_gen,
                    gr_gen + std_gen,
                    alpha=0.25, color="tomato")

    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("r  (pixels)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    ax.legend()
    ax.set_xlim(left=0)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved → {output_path}")
    plt.show()
    plt.close(fig)


def plot_psi6_comparison(flat_real, flat_gen, output_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 60)
    ax.hist(flat_real, bins=bins, density=True,
            alpha=0.6, label="Real",      color="steelblue")
    ax.hist(flat_gen,  bins=bins, density=True,
            alpha=0.6, label="Generated", color="tomato")
    ax.set_xlabel(r"$|\psi_6|$")
    ax.set_ylabel("Density")
    ax.set_title(r"Distribution of $|\psi_6|$")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved → {output_path}")
    plt.show()
    plt.close(fig)


def plot_phase_comparison(centers_real, hist_real,
                          centers_gen,  hist_gen,
                          output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             subplot_kw=dict(projection="polar"))

    for ax, centers, hist, label, color in zip(
        axes,
        [centers_real, centers_gen],
        [hist_real,    hist_gen],
        ["Real",       "Generated"],
        ["steelblue",  "tomato"],
    ):
        ax.bar(centers, hist,
               width=centers[1] - centers[0],
               color=color, alpha=0.7)
        ax.set_title(f"arg(ψ₆) — {label}", pad=12)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved → {output_path}")
    plt.show()
    plt.close(fig)


def plot_channel_fields(real_sample, gen_sample, output_path=None):
    """Side-by-side field maps for one real and one generated sample."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    cmaps = ["viridis", "plasma", "twilight", "RdYlGn"]

    for col, (name, cmap) in enumerate(zip(CHANNEL_NAMES, cmaps)):
        for row, (img, label) in enumerate(
            [(real_sample, "Real"), (gen_sample, "Generated")]
        ):
            ax = axes[row][col]
            ax.imshow(img[col], origin="lower", cmap=cmap)
            ax.set_title(f"{label} — {name}")
            ax.axis("off")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved → {output_path}")
    plt.show()
    plt.close(fig)


# ===========================================================================
# Main evaluation entry point
# ===========================================================================

def evaluate(
    generated_paths,
    real_data_path,
    output_dir="outputs/eval",
    n_real_samples=50,
    n_bins_gr=200,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading real data...")
    real_all = np.load(real_data_path)           # (N, 4, H, W)
    if real_all.ndim == 3:
        real_all = real_all[np.newaxis]          # handle single-sample .npy

    # Subsample real data for speed
    idx = np.random.choice(len(real_all),
                           min(n_real_samples, len(real_all)),
                           replace=False)
    real = real_all[idx]
    print(f"  Real samples : {len(real)} × {real.shape[1:]}")

    print("Loading generated samples...")
    gen_list = []
    for p in generated_paths:
        g = np.load(p)
        if g.ndim == 3:
            g = g[np.newaxis]
        elif g.ndim == 4 and g.shape[0] == 1:
            pass
        gen_list.append(g)
    gen = np.concatenate(gen_list, axis=0)       # (M, 4, H, W)
    print(f"  Generated samples : {len(gen)} × {gen.shape[1:]}")

    # ------------------------------------------------------------------
    # 1. g(r)
    # ------------------------------------------------------------------
    print("\n[1/4] Computing g(r)...")
    r_real, gr_real, std_real = compute_mean_gr(real, n_bins=n_bins_gr)
    r_gen,  gr_gen,  std_gen  = compute_mean_gr(gen,  n_bins=n_bins_gr)
    w_gr = wasserstein_gr(r_real, gr_real, r_gen, gr_gen)
    print(f"  Wasserstein distance (g(r)) : {w_gr:.4f}")
    plot_gr_comparison(r_real, gr_real, std_real,
                       r_gen,  gr_gen,  std_gen,
                       output_path=output_dir / "gr_comparison.png")

    # ------------------------------------------------------------------
    # 2. |psi6|
    # ------------------------------------------------------------------
    print("[2/4] Computing |ψ₆| statistics...")
    means_r, stds_r, flat_real = psi6_stats(real)
    means_g, stds_g, flat_gen  = psi6_stats(gen)
    w_psi6 = wasserstein_distance(flat_real, flat_gen)
    print(f"  Real      : mean |ψ₆| = {means_r.mean():.3f} ± {stds_r.mean():.3f}")
    print(f"  Generated : mean |ψ₆| = {means_g.mean():.3f} ± {stds_g.mean():.3f}")
    print(f"  Wasserstein distance (|ψ₆|) : {w_psi6:.4f}")
    plot_psi6_comparison(flat_real, flat_gen,
                         output_path=output_dir / "psi6_comparison.png")

    # ------------------------------------------------------------------
    # 3. arg(psi6) phase uniformity
    # ------------------------------------------------------------------
    print("[3/4] Computing arg(ψ₆) phase distributions...")
    c_real, h_real = phase_uniformity(real)
    c_gen,  h_gen  = phase_uniformity(gen)
    w_phase = wasserstein_distance(c_real, c_gen, h_real, h_gen)
    print(f"  Wasserstein distance (phase) : {w_phase:.4f}")
    plot_phase_comparison(c_real, h_real, c_gen, h_gen,
                          output_path=output_dir / "phase_comparison.png")

    # ------------------------------------------------------------------
    # 4. Voronoi defect fraction
    # ------------------------------------------------------------------
    print("[4/4] Computing defect fractions...")
    df_real = defect_fraction(real)
    df_gen  = defect_fraction(gen)
    print(f"  Real      : defect fraction = {df_real.mean():.3f} ± {df_real.std():.3f}")
    print(f"  Generated : defect fraction = {df_gen.mean():.3f}  ± {df_gen.std():.3f}")

    # ------------------------------------------------------------------
    # Field map comparison (first real vs first generated)
    # ------------------------------------------------------------------
    plot_channel_fields(real[0], gen[0],
                        output_path=output_dir / "field_comparison.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  g(r)  Wasserstein  : {w_gr:.4f}   (lower = better)")
    print(f"  |ψ₆| Wasserstein   : {w_psi6:.4f}   (lower = better)")
    print(f"  phase Wasserstein  : {w_phase:.4f}   (lower = better)")
    print(f"  defect Δ (abs)     : {abs(df_real.mean() - df_gen.mean()):.4f}   (lower = better)")
    print("=" * 50)

    return {
        "w_gr":          w_gr,
        "w_psi6":        w_psi6,
        "w_phase":       w_phase,
        "defect_real":   df_real.mean(),
        "defect_gen":    df_gen.mean(),
        "psi6_mean_real": means_r.mean(),
        "psi6_mean_gen":  means_g.mean(),
    }


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated crystal samples against real data"
    )
    parser.add_argument(
        "--generated", nargs="+", required=True,
        help="Path(s) to generated .npy files (shape: 1×4×H×W or 4×H×W)",
    )
    parser.add_argument(
        "--real_data", type=str, required=True,
        help="Path to stacked real dataset .npy (shape: N×4×H×W)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/eval",
        help="Directory to save evaluation plots",
    )
    parser.add_argument(
        "--n_real_samples", type=int, default=50,
        help="How many real samples to compare against (default: 50)",
    )

    args = parser.parse_args()

    import argparse  # already imported above, harmless
    evaluate(
        generated_paths=args.generated,
        real_data_path=args.real_data,
        output_dir=args.output_dir,
        n_real_samples=args.n_real_samples,
    )