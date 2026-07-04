"""
Generate new particle configurations from pure noise with the trained
unconditional denoiser, and visualize them.

Usage:
    python sampling.py

Requires checkpoints/model_best.pt (or model_final.pt) produced by
train_patched.py. All architecture settings and the coordinate scale are
read from the checkpoint, so nothing here needs to match by hand.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion
from train_patched import identity_scalars

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT      = "checkpoints/model_best.pt"
OUT_DIR         = "generated_samples"   # coordinates saved here as npz for evaluate.py

N_PARTICLES     = 4000          # match the patch size the model was trained on
DEFECT_FRACTION = 50 / 10050    # composition of the real boxes
SIZE_SMALL      = 1.0
SIZE_LARGE      = 1.4

N_SAMPLES       = 1
USE_EMA         = True          # EMA weights sample noticeably better
CORRECTOR_STEPS = 1             # Langevin corrector iterations per reverse step
                                # (0 = plain ancestral DDPM; 1 doubles sampling
                                # cost but lets the configuration anneal)
# ──────────────────────────────────────────────────────────────────────────────


def build_sampling_identity(n_particles, defect_fraction, size_small, size_large):
    """
    Choose a composition (how many of each type) and encode it.

    NOTE the dataset's type convention is inverted vs the obvious one:
    type 0 = large defect (diameter 1.4), type 1 = small (diameter 1.0).
    The conditioning must match what the model saw during training, so we
    reproduce that convention here and use size (not type) for plotting.
    """
    n_large = int(round(n_particles * defect_fraction))
    n_small = n_particles - n_large

    sizes = np.array([size_small] * n_small + [size_large] * n_large,
                     dtype=np.float32)
    types = np.array([1.0] * n_small + [0.0] * n_large, dtype=np.float32)

    perm = np.random.permutation(n_particles)
    sizes, types = sizes[perm], types[perm]
    return identity_scalars(sizes, types), sizes, types


def plot_configuration(pos, types, sizes, title, ax):
    small_mask = sizes < 1.2
    large_mask = ~small_mask
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg = ckpt["config"]
    coord_scale = ckpt["coord_scale"]
    print(f"loaded {CHECKPOINT}  (coord_scale={coord_scale:.4f}, config={cfg})")

    model = EGNNUnconditionalDenoiser(
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_scalar_feats=cfg["n_scalar_feats"],
    ).to(device)
    model.load_state_dict(ckpt["ema"] if USE_EMA else ckpt["model"])
    model.eval()

    diffusion = PositionDiffusion(
        n_steps=cfg["n_steps"], device=device, k_neighbors=cfg["k_neighbors"],
        schedule=cfg.get("schedule", "cosine"),
    )

    Path(OUT_DIR).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, N_SAMPLES, figsize=(7 * N_SAMPLES, 7))
    if N_SAMPLES == 1:
        axes = [axes]

    for i in tqdm(range(N_SAMPLES), desc="sampling", unit="config"):
        cond, sizes, types = build_sampling_identity(
            N_PARTICLES, DEFECT_FRACTION, SIZE_SMALL, SIZE_LARGE,
        )
        x_gen = diffusion.sample(
            model,
            n_particles=N_PARTICLES,
            node_scalars=torch.from_numpy(cond),
            device=device,
            corrector_steps=CORRECTOR_STEPS,
            verbose=True,
        )
        pos = x_gen.cpu().numpy() * coord_scale   # back to physical units

        out_path = Path(OUT_DIR) / f"sample_{i:03d}.npz"
        np.savez(out_path, pos=pos.astype(np.float32), types=types, sizes=sizes)
        print(f"saved {out_path}  (run: python evaluate.py {out_path})")

        plot_configuration(
            pos, types, sizes,
            title=f"sample {i}  |  N={N_PARTICLES}  defects={int((sizes > 1.2).sum())}",
            ax=axes[i],
        )

    fig.suptitle("Generated configurations from pure noise", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("generated_samples.png", dpi=150, bbox_inches="tight")
    print("saved generated_samples.png")
    plt.show()


if __name__ == "__main__":
    main()
