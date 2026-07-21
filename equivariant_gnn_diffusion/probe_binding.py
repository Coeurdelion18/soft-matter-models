"""
Conditioning-binding acceptance test.

Conditions the trained model on an extreme synthetic psi6 map -- left half
"grain boundary" (0.75), right half "perfect crystal" (1.0) -- and measures
the |psi6| difference between the two halves of the generated sample.

Interpretation:
    split >= ~0.15   conditioning binds (real maps should then be followable)
    split ~ 0.03     model effectively ignores the field (observed before FiLM)

Usage:
    python probe_binding.py [--w 2.0] [--n 4900]
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion
from field_conditioning import FieldSampler
from train_patched import identity_scalars
from evaluate import delaunay_neighbors, psi_k, interior_mask

CHECKPOINT = "checkpoints/model_best.pt"
DENSITY = 1.005          # particles per unit area (from the real boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, nargs="+", default=[2.0])
    parser.add_argument("--n", type=int, default=4900)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg = ckpt["config"]
    scale = ckpt["coord_scale"]
    model = EGNNUnconditionalDenoiser(
        cfg["hidden_dim"], cfg["n_layers"], cfg["n_scalar_feats"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diff = PositionDiffusion(cfg["n_steps"], device=device,
                             k_neighbors=cfg["k_neighbors"],
                             schedule=cfg["schedule"])

    # extreme map: left half boundary-like, right half crystal
    G = 64
    grid = np.ones((G, G), dtype=np.float32)
    grid[: G // 2, :] = 0.75
    half_phys = float(np.sqrt(args.n / DENSITY)) / 2
    h = half_phys / scale
    field_fn = FieldSampler(grid, (-h, h, -h, h), device)

    N = args.n
    sizes = np.full(N, 1.0, dtype=np.float32)
    n_def = max(1, int(round(N * 50 / 10050)))
    sizes[np.random.choice(N, n_def, replace=False)] = 1.4
    types = np.where(sizes > 1.2, 0.0, 1.0).astype(np.float32)
    cond = torch.from_numpy(identity_scalars(sizes, types))
    rr = torch.from_numpy(sizes / 2.0 / scale)

    for w in args.w:
        x = diff.sample(model, N, cond, device=device, field_fn=field_fn,
                        field_guidance=w, repulsion_radii=rr,
                        corrector_steps=1)
        pos = x.cpu().numpy() * scale
        nb = delaunay_neighbors(pos)
        p6 = np.abs(psi_k(pos, nb, 6))
        m = interior_mask(pos)
        left = m & (pos[:, 0] < 0)
        right = m & (pos[:, 0] >= 0)
        split = p6[right].mean() - p6[left].mean()
        print(f"w={w}:  |psi6| left (target 0.75) = {p6[left].mean():.4f}   "
              f"right (target 1.0) = {p6[right].mean():.4f}   "
              f"split = {split:+.4f}   "
              f"{'BINDS' if split >= 0.15 else 'weak/no binding'}")

        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(pos[m, 0], pos[m, 1], s=4, c=p6[m], cmap="viridis",
                        vmin=0, vmax=1)
        ax.axvline(0, color="red", lw=1, ls="--")
        ax.set_aspect("equal")
        ax.set_title(f"half-map binding probe, w={w}  "
                     f"(left target 0.75 / right target 1.0)")
        fig.colorbar(sc)
        out = f"binding_probe_w{w:g}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
