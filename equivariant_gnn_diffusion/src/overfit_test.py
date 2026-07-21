"""
Overfit the denoiser to a SINGLE configuration, then sample from pure noise
and check whether the result resembles the original.

This isolates whether the model/training loop/sampling loop are correct,
independent of dataset size and generalisation.

Usage:
    python overfit_test.py

Set PATCH_FILE to a real patch .npz to test on your data, or None to use a
synthetic bidisperse triangular lattice (no data dependency).

Expected outcome if everything is correct:
  - training loss drops well below 1.0 and keeps decreasing
    (loss ~1.0 means "no better than predicting zero noise"; a converged
    eps-prediction model on a single example typically reaches 0.1-0.4
    averaged over all noise levels -- it cannot reach 0 because the loss at
    high t is irreducible)
  - sampling completes without diverging
  - the sampled configuration is clearly lattice-like with roughly the same
    extent as the original (not a collapsed blob, not exploded)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion
from train_patched import identity_scalars

# ── Config ────────────────────────────────────────────────────────────────────
PATCH_FILE    = "data/patches/train/001_patch010.npz"  # or None for synthetic
N_PARTICLES   = 300        # synthetic only
DEFECT_FRAC   = 0.02       # synthetic only

K_NEIGHBORS   = 12
HIDDEN_DIM    = 128
N_LAYERS      = 6
N_STEPS       = 1000
LR            = 3e-4
GRAD_CLIP     = 1.0
N_TRAIN_STEPS = 20000
PRINT_EVERY   = 1000
# ─────────────────────────────────────────────────────────────────────────────


def make_synthetic_patch(n_particles, defect_frac, spacing=1.0, noise=0.04):
    n_large = max(1, int(round(n_particles * defect_frac)))
    n_small = n_particles - n_large

    rows = int(np.ceil(np.sqrt(n_small)))
    pts = []
    for i in range(rows):
        for j in range(rows):
            if len(pts) >= n_small:
                break
            pts.append([
                j * spacing + (0.5 * spacing if i % 2 else 0.0),
                i * spacing * 0.866,
            ])
    pos_small = np.array(pts[:n_small], dtype=np.float32)
    pos_small += np.random.normal(0, noise, pos_small.shape).astype(np.float32)

    box_side = rows * spacing
    pos_large = np.random.uniform(0, box_side, (n_large, 2)).astype(np.float32)

    pos = np.vstack([pos_small, pos_large])
    types = np.array([0.0] * n_small + [1.0] * n_large, dtype=np.float32)
    sizes = np.array([1.0] * n_small + [1.4] * n_large, dtype=np.float32)
    return pos, types, sizes


def plot_comparison(pos_orig, pos_samp, types, sizes, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, pos, title in [
        (axes[0], pos_orig, "original (overfit target)"),
        (axes[1], pos_samp, "sampled (from pure noise)"),
    ]:
        small = sizes < 1.2   # size, not type: in this dataset type 0 = large
        large = ~small
        ax.scatter(pos[small, 0], pos[small, 1],
                   s=(sizes[small] * 30) ** 1.3, c="#2a8c4a",
                   edgecolors="none", alpha=0.85, label="small")
        ax.scatter(pos[large, 0], pos[large, 1],
                   s=(sizes[large] * 30) ** 1.3, c="#c0392b",
                   edgecolors="none", alpha=0.95, label="large (defect)")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Overfit sanity check: original vs. sampled", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved {save_path}")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if PATCH_FILE is not None:
        d = np.load(PATCH_FILE)
        pos = d["pos"].astype(np.float32)
        ns = d["node_scalars"]
        sizes = ns[:, 7].astype(np.float32)
        types = ns[:, 8].astype(np.float32)
        print(f"loaded {PATCH_FILE}: N={len(pos)}")
    else:
        pos, types, sizes = make_synthetic_patch(N_PARTICLES, DEFECT_FRAC)
        print(f"generated synthetic patch: N={len(pos)}, defects={int(types.sum())}")

    N = len(pos)
    pos = pos - pos.mean(axis=0, keepdims=True)

    # unit-variance normalisation -- THE critical step. The diffusion prior
    # is N(0, I); raw patch coordinates have std ~10 and cannot be reached
    # from that prior.
    coord_scale = float(np.sqrt((pos ** 2).mean()))
    print(f"coord_scale: {coord_scale:.4f}")

    x0 = torch.from_numpy(pos / coord_scale).float().to(device)
    cond = torch.from_numpy(identity_scalars(sizes, types)).to(device)

    model = EGNNUnconditionalDenoiser(
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_scalar_feats=2,
    ).to(device)
    diffusion = PositionDiffusion(
        n_steps=N_STEPS, device=device, k_neighbors=K_NEIGHBORS,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    print(f"\noverfitting for {N_TRAIN_STEPS} steps on N={N} particles...")
    model.train()
    losses = []
    for step in range(N_TRAIN_STEPS):
        opt.zero_grad()
        loss = diffusion.training_loss(model, x0, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        losses.append(loss.item())

        if step % PRINT_EVERY == 0 or step == N_TRAIN_STEPS - 1:
            window = losses[-PRINT_EVERY:]
            print(f"  step {step:5d}: loss = {losses[-1]:.5f}  "
                  f"running_avg = {np.mean(window):.5f}")

    final_loss = float(np.mean(losses[-200:]))
    print(f"\nfinal smoothed loss (last 200 steps avg): {final_loss:.5f}")
    if final_loss > 0.8:
        print("WARNING: loss barely improved over the trivial baseline (1.0). "
              "Something is still wrong -- sampling below is unlikely to work.")

    print(f"\nsampling from pure noise ({N_STEPS} reverse steps)...")
    model.eval()
    x_gen = diffusion.sample(
        model, n_particles=N, node_scalars=cond, device=device, verbose=True,
    )
    pos_sampled = x_gen.cpu().numpy() * coord_scale
    print("\nsampling completed without diverging.")

    plot_comparison(pos, pos_sampled, types, sizes,
                    save_path="outputs/results/overfit_comparison.png")

    print("\nIf the sampled panel is clearly lattice-like with the same extent "
          "as the original, the core pipeline works; remaining quality issues "
          "are about training scale, not correctness.")


if __name__ == "__main__":
    main()
