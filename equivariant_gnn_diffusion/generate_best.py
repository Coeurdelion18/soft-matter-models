"""
Generate samples from the BEST model so far: the 120-epoch unconditional
circular-patch model (mean sample |psi6| 0.82), saved as
checkpoints/model_uncond_logsigma_backup.pt.

That checkpoint predates FiLM conditioning; the current EGNNLayer's FiLM
blocks are zero-initialised (exact identity), so loading with the FiLM keys
left at init reproduces the original network's behaviour exactly.
It was also trained with sigma_min=0.002 (pre-square-patch schedule), which
must be passed explicitly.

Usage:
    python generate_best.py [--n-samples 5] [--out generated_samples/best_model]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion
from train_patched import identity_scalars

CHECKPOINT   = "checkpoints/model_uncond_logsigma_backup.pt"
SIGMA_MIN    = 0.002        # schedule this checkpoint was trained with
N_PARTICLES  = 4000         # its training patch size
DEFECT_FRAC  = 50 / 10050
SIZE_SMALL, SIZE_LARGE = 1.0, 1.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--out", default="generated_samples/best_model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg = ckpt["config"]
    coord_scale = ckpt["coord_scale"]
    print(f"loaded {CHECKPOINT} (coord_scale={coord_scale:.4f}, config={cfg})")

    model = EGNNUnconditionalDenoiser(
        hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
        n_scalar_feats=cfg["n_scalar_feats"],
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["ema"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("film" in k for k in missing), f"non-FiLM missing keys: {missing}"
    print(f"loaded EMA weights ({len(missing)} FiLM keys left at identity init)")
    model.eval()

    diffusion = PositionDiffusion(
        n_steps=cfg["n_steps"], device=device,
        k_neighbors=cfg["k_neighbors"], schedule=cfg["schedule"],
        sigma_min=SIGMA_MIN,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_samples
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

    for i in range(n):
        n_large = int(round(N_PARTICLES * DEFECT_FRAC))
        sizes = np.array([SIZE_SMALL] * (N_PARTICLES - n_large)
                         + [SIZE_LARGE] * n_large, dtype=np.float32)
        types = np.where(sizes > 1.2, 0.0, 1.0).astype(np.float32)  # inverted convention
        perm = np.random.permutation(N_PARTICLES)
        sizes, types = sizes[perm], types[perm]

        x = diffusion.sample(
            model, N_PARTICLES,
            torch.from_numpy(identity_scalars(sizes, types)),
            device=device,
            repulsion_radii=torch.from_numpy(sizes / 2.0 / coord_scale),
            corrector_steps=1,
        )
        pos = (x.cpu().numpy() * coord_scale).astype(np.float32)

        out = out_dir / f"best_{i:03d}.npz"
        np.savez(out, pos=pos, types=types, sizes=sizes)
        print(f"saved {out}")

        ax = axes[0][i]
        small = sizes < 1.2
        ax.scatter(pos[small, 0], pos[small, 1], s=2.5, c="#2a8c4a", alpha=0.8)
        ax.scatter(pos[~small, 0], pos[~small, 1], s=10, c="#c0392b")
        ax.set_aspect("equal")
        ax.set_title(f"best_{i:03d}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=140, bbox_inches="tight")
    print(f"saved {out_dir / 'overview.png'}")
    print(f"\nEvaluate any sample with:")
    print(f"  python evaluate.py {out_dir}/best_000.npz --ref data/patches/val/<patch>.npz")


if __name__ == "__main__":
    main()
