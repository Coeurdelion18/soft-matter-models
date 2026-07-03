"""
Overfit the unconditional EGNN denoiser to a SINGLE configuration, then
sample from pure noise and check whether the result resembles the original.

This isolates whether the model/training loop/sampling loop are correct,
independent of dataset size, feature normalisation across many boxes, or
N-extrapolation effects -- all confounds in the full pipeline.

Usage:
    python overfit_test.py

By default this generates a synthetic single configuration (bidisperse
noisy triangular lattice) so the script runs standalone with no data
dependency. To use one of your real patches instead, set PATCH_FILE below
to a path like "data/patches/train/box001_patch003.npz".

Expected outcome if everything is correct:
  - training loss drops to near zero (< 0.05, ideally < 0.01)
  - sampling completes without diverging
  - the sampled configuration visually resembles the original lattice
    (small particles in a roughly hexagonal arrangement, large defects
    placed somewhere reasonable -- it will NOT be pixel-identical to the
    original since diffusion sampling is stochastic, but the *structure*
    should be clearly lattice-like, not noise)

If sampling still diverges or produces garbage here, the bug is in the
model/sampling code itself, not in feature scaling or N-extrapolation.
If this succeeds cleanly, the issues you were chasing are specific to
multi-box training and/or large-N sampling, not the core pipeline.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion
from node_features import normalise_scalars, N_SCALAR_FEATS

# ── Config ─────────────────────────────────────────────────────────────────────
PATCH_FILE   = "data/patches/train/001_patch010.npz"     # set to a real patch .npz path to use real data instead
N_PARTICLES  = 300      # only used if PATCH_FILE is None (synthetic data)
DEFECT_FRAC  = 0.02      # only used if PATCH_FILE is None

CUTOFF       = 1.5
HIDDEN_DIM   = 128
N_LAYERS     = 2
N_STEPS      = 1000
LR           = 2e-4     # higher than full-dataset training (2e-4) since
                         # overfitting one example needs faster convergence
GRAD_CLIP    = 1.0       # tighter than the 1.0 used elsewhere -- see note below
N_TRAIN_STEPS = 100000     # overfitting needs many more gradient steps than
                         # a single epoch on a full dataset would give
PRINT_EVERY  = 500

# NOTE ON TRAINING STABILITY:
# This architecture has no normalisation layers (LayerNorm/BatchNorm) on the
# scalar feature channel. During overfit testing, this caused occasional
# single-step loss spikes of several orders of magnitude (observed: one
# spike to ~56,000 against a typical loss of ~0.7-0.9) at apparently random
# points during training, likely triggered by a high-noise-level training
# step producing an extreme raw prediction. Gradient clipping (GRAD_CLIP)
# protects the parameters from being corrupted by these spikes -- training
# recovers and continues improving afterward -- but the spikes themselves
# are a symptom of real architectural fragility, not something this script
# fully fixes. If you see loss spike to a huge value and then continue
# decreasing afterward, that's expected, not a bug in this script. If
# training diverges and does NOT recover, GRAD_CLIP may need to be even
# tighter (try 0.1-0.3).
# ─────────────────────────────────────────────────────────────────────────────


def make_synthetic_patch(n_particles, defect_frac, spacing=1.0, noise=0.04):
    """
    Synthetic bidisperse noisy triangular lattice -- gives the overfit
    test real structure to learn without needing your actual data files.
    """
    n_large = max(1, int(round(n_particles * defect_frac)))
    n_small = n_particles - n_large

    rows = int(np.ceil(np.sqrt(n_small)))
    pts  = []
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

    box_side  = rows * spacing
    pos_large = np.random.uniform(0, box_side, (n_large, 2)).astype(np.float32)

    pos   = np.vstack([pos_small, pos_large])
    types = np.array([0] * n_small + [1] * n_large, dtype=np.int32)
    sizes = np.array([1.0] * n_small + [1.8] * n_large, dtype=np.float32)
    box_size = np.array([box_side, box_side], dtype=np.float32)

    return pos, types, sizes, box_size


def get_node_scalars(pos, types, sizes, box_size):
    """Compute structural node features via freud, matching the real pipeline."""
    from node_features import build_node_scalars
    return build_node_scalars(pos, types, sizes, box_size)


def plot_comparison(pos_original, types_original, sizes_original,
                    pos_sampled, types_sampled, sizes_sampled, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    for ax, pos, types, sizes, title in [
        (axes[0], pos_original, types_original, sizes_original, "original (overfit target)"),
        (axes[1], pos_sampled,  types_sampled,  sizes_sampled,  "sampled (from pure noise)"),
    ]:
        small = types < 0.5
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

    # ── Load or generate the single configuration ────────────────────────────
    if PATCH_FILE is not None:
        d = np.load(PATCH_FILE)
        pos          = d["pos"]
        node_scalars_raw = d["node_scalars"]
        # if your patch files don't store type/size separately, adapt this
        types = (node_scalars_raw[:, 8]).round().astype(int)
        sizes = node_scalars_raw[:, 7]
        print(f"loaded {PATCH_FILE}: N={len(pos)}")
    else:
        pos, types, sizes, box_size = make_synthetic_patch(N_PARTICLES, DEFECT_FRAC)
        node_scalars_raw = get_node_scalars(pos, types, sizes, box_size)
        print(f"generated synthetic patch: N={len(pos)}, "
              f"defects={types.sum()}")

    N = len(pos)

    # ── Normalise features (skip size/type columns -- see earlier discussion) ─
    norm_list, feat_mean, feat_std = normalise_scalars([node_scalars_raw],
                                                        skip_cols=(7, 8))
    node_scalars = torch.from_numpy(norm_list[0]).float()
    print(f"feat_mean: {feat_mean.round(3)}")
    print(f"feat_std:  {feat_std.round(3)}")

    x0 = torch.from_numpy(pos).float().to(device)
    node_scalars = node_scalars.to(device)

    # ── Model + diffusion ─────────────────────────────────────────────────────
    model = EGNNUnconditionalDenoiser(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_scalar_feats=N_SCALAR_FEATS,
    ).to(device)
    diffusion = PositionDiffusion(n_steps=N_STEPS, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    # ── Overfit ───────────────────────────────────────────────────────────────
    print(f"\noverfitting for {N_TRAIN_STEPS} steps on N={N} particles...")
    model.train()
    losses = []
    max_loss_seen = 0.0
    for step in range(N_TRAIN_STEPS):
        opt.zero_grad()
        loss = diffusion.training_loss(model, x0, node_scalars, cutoff=CUTOFF)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        loss_val = loss.item()
        losses.append(loss_val)
        max_loss_seen = max(max_loss_seen, loss_val)

        if step % PRINT_EVERY == 0 or step == N_TRAIN_STEPS - 1:
            window = losses[-PRINT_EVERY:]
            print(f"  step {step:5d}: loss = {loss_val:.5f}  "
                 f"running_avg = {np.mean(window):.5f}")

    final_loss = float(np.mean(losses[-200:]))  # smoothed, not single-step
    print(f"\nfinal smoothed loss (last 200 steps avg): {final_loss:.5f}")
    print(f"max single-step loss seen during training: {max_loss_seen:.2f}")
    if max_loss_seen > 50 * final_loss:
        print("NOTE: large transient loss spike(s) occurred during training. "
              "This is a known artefact of the architecture's lack of "
              "normalisation layers (see comment near GRAD_CLIP above) -- "
              "expected to recover via gradient clipping, not necessarily a bug.")
    if final_loss > 0.3:
        print("WARNING: loss did not converge well. Either the model/training "
              "loop has a bug, or N_TRAIN_STEPS needs to be higher. "
              "Sampling results below may not be meaningful.")

    # ── Sample ────────────────────────────────────────────────────────────────
    print(f"\nsampling from pure noise ({N_STEPS} reverse steps)...")
    model.eval()
    try:
        x_gen = diffusion.sample(
            model,
            n_particles=N,
            node_scalars=node_scalars,
            cutoff=CUTOFF,
            device=device,
            verbose=True,
        )
        pos_sampled = x_gen.cpu().numpy()
        print("\nsampling completed without diverging.")
    except RuntimeError as e:
        print(f"\nSAMPLING FAILED: {e}")
        print("This means the bug is in the model or sampling loop itself, "
              "not in dataset-scale feature normalisation or N-extrapolation.")
        return

    # ── Compare ───────────────────────────────────────────────────────────────
    pos_centered = pos - pos.mean(axis=0)  # match CoM-free convention for fair comparison
    plot_comparison(
        pos_centered, types, sizes,
        pos_sampled, types, sizes,   # same types/sizes since identity is fixed
        save_path="overfit_comparison.png",
    )

    print("\nIf the sampled panel looks roughly lattice-like (not pure noise, "
          "not exploded to huge coordinates), the core pipeline works and the "
          "issues from before are specific to multi-box training and/or "
          "large-N sampling extrapolation.")


if __name__ == "__main__":
    main()