"""
Generate a new particle configuration from pure noise using the trained
unconditional denoiser, and visualize it.

Usage:
    python sample.py

Requires checkpoints/model_best.pt (or model_final.pt) and
checkpoints/feature_stats.npz, both produced by train_patched.py.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion

# ── Config -- must match train_patched.py ─────────────────────────────────────
# ── Config -- must match train_patched.py ─────────────────────────────────────
CHECKPOINT     = "checkpoints/model_best.pt"
FEATURE_STATS  = "checkpoints/feature_stats.npz"

CUTOFF         = 1.5
HIDDEN_DIM     = 128
N_LAYERS       = 4
N_SCALAR_FEATS = 9
N_STEPS        = 1000

# full-box generation
N_PARTICLES     = 1000     # match your real simulation box size
DEFECT_FRACTION = 5/1000      # set this to your ACTUAL defect ratio if known --
                            # e.g. if your real boxes have exactly 200 defects,
                            # use 200/10050 instead of an approximate fraction
SIZE_SMALL      = 1.0
SIZE_LARGE      = 1.4

N_SAMPLES       = 1         # drop to 1 for full-box runs -- each sample is
                            # now expensive; generate more only once you've
                            # confirmed one looks right
# ───────────────────────────────────────────────────────────────────────────────


def build_sampling_node_scalars(n_particles, defect_fraction, size_small,
                                 size_large, feat_mean, feat_std):
    """
    Build the (N, 9) node_scalars tensor needed at sampling time.

    Structural features (voronoi_area, coordination, perimeter, neighbour
    distances, psi6, psi4 -- columns 0-6) are NOT knowable before positions
    exist, since they depend on the configuration itself. We set them to
    their normalised mean (0.0 after z-scoring) so the model receives
    "assume average local structure" rather than a misleading guess.

    size (col 7) and type (col 8) ARE knowable upfront -- they're an
    identity choice the caller makes, not something derived from geometry.
    These are set to real values and then normalised with the same stats
    used during training.
    """
    n_large = int(round(n_particles * defect_fraction))
    n_small = n_particles - n_large

    sizes = np.array([size_small] * n_small + [size_large] * n_large,
                     dtype=np.float32)
    types = np.array([0.0] * n_small + [1.0] * n_large, dtype=np.float32)

    # shuffle so small/large aren't spatially correlated by index order
    perm  = np.random.permutation(n_particles)
    sizes = sizes[perm]
    types = types[perm]

    # structural columns: raw value 0 (we don't know it), which becomes
    # -mean/std after normalisation -- but since we want "average" (i.e.
    # normalised 0), we directly set normalised value to 0 for those columns
    scalars_norm = np.zeros((n_particles, N_SCALAR_FEATS), dtype=np.float32)

    # size and type: normalise using the same training statistics
    scalars_norm[:, 7] = (sizes - feat_mean[7]) / feat_std[7]
    scalars_norm[:, 8] = (types - feat_mean[8]) / feat_std[8]

    return scalars_norm, types


def plot_configuration(pos, types, sizes, title, save_path=None, ax=None):
    """
    Scatter plot of a generated configuration.
    Small particles in green, large defects in red -- matching the
    two-channel overlay convention from the original LAMMPS visualizations.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    small_mask = types < 0.5
    large_mask = ~small_mask

    # marker size roughly proportional to particle diameter, scaled for visibility
    ax.scatter(pos[small_mask, 0], pos[small_mask, 1],
              s=(sizes[small_mask] * 30) ** 1.3, c="#2a8c4a",
              edgecolors="none", alpha=0.85, label="small")
    ax.scatter(pos[large_mask, 0], pos[large_mask, 1],
              s=(sizes[large_mask] * 30) ** 1.3, c="#c0392b",
              edgecolors="none", alpha=0.95, label="large (defect)")

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"saved {save_path}")
        return fig
    return ax


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = EGNNUnconditionalDenoiser(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_scalar_feats=N_SCALAR_FEATS,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"loaded {CHECKPOINT}")

    # ── Load feature normalisation stats ─────────────────────────────────────
    stats     = np.load(FEATURE_STATS)
    feat_mean = stats["mean"]
    feat_std  = stats["std"]
    print(f"loaded {FEATURE_STATS}")

    diffusion = PositionDiffusion(n_steps=N_STEPS, device=device)

    # ── Sample ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_SAMPLES, figsize=(7 * N_SAMPLES, 7))
    if N_SAMPLES == 1:
        axes = [axes]

    for i in tqdm(range(N_SAMPLES), desc="sampling", unit="config"):
        scalars_norm, types = build_sampling_node_scalars(
            N_PARTICLES, DEFECT_FRACTION, SIZE_SMALL, SIZE_LARGE,
            feat_mean, feat_std,
        )
        node_scalars = torch.from_numpy(scalars_norm)

        x_gen = diffusion.sample(
            model,
            n_particles=N_PARTICLES,
            node_scalars=node_scalars,
            cutoff=CUTOFF,
            feat_mean=feat_mean,
            feat_std=feat_std,
            device=device,
        )
        pos = x_gen.cpu().numpy()

        # recover unnormalised sizes for plotting marker scale
        sizes = scalars_norm[:, 7] * feat_std[7] + feat_mean[7]

        n_large = int(types.sum())
        plot_configuration(
            pos, types, sizes,
            title=f"sample {i}  |  N={N_PARTICLES}  defects={n_large}",
            ax=axes[i],
        )

    fig.suptitle("Generated configurations from pure noise", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("generated_samples.png", dpi=150, bbox_inches="tight")
    print("saved generated_samples.png")
    plt.show()


if __name__ == "__main__":
    main()