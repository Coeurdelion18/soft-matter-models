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
from field_conditioning import make_psi6_map, FieldSampler, NullField
from train_patched import identity_scalars

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT      = "checkpoints/model_best.pt"
OUT_DIR         = "outputs/generated_samples"   # coordinates saved here as npz for evaluate.py

N_PARTICLES     = 4000          # match the patch size the model was trained on
DEFECT_FRACTION = 50 / 10050    # composition of the real boxes
SIZE_SMALL      = 1.0
SIZE_LARGE      = 1.4

N_SAMPLES       = 1
USE_EMA         = True          # EMA weights sample noticeably better
CORRECTOR_STEPS = 1             # Langevin corrector iterations per reverse step
                                # (0 = plain ancestral DDPM; 1 doubles sampling
                                # cost but lets the configuration anneal)

# For a field-conditioned model, three ways to choose the conditioning:
#   TARGET_MAP   = "outputs/generated_maps/map_000.npy": condition on a NEW layout
#                  sampled by map_generator.py (two-stage generation --
#                  brand-new pattern, nothing user-provided)
#   TARGET_PATCH = "data/patches/...npz": replicate a real patch's psi6 map
#                  and composition (controlled regeneration)
#   both None:    unconditional (classifier-free null token)
# TARGET_MAP takes precedence over TARGET_PATCH.
TARGET_MAP      = None
TARGET_PATCH    = "data/patches/val/468_patch000.npz"  # most boundary-rich val patch
FIELD_GUIDANCE  = 2.0           # classifier-free guidance weight (1.0 = plain
                                # conditional; >1 binds the sample more strongly
                                # to the target psi6 map, 2x model cost)

REPULSION       = True          # soft-disk overlap guidance in the last
                                # 25% of reverse steps -- removes overlapping
                                # pairs and carves cavities around defects
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=None,
                        help="override TARGET_MAP (path to a generated .npy map)")
    parser.add_argument("--prefix", default="sample",
                        help="output filename prefix in generated_samples/")
    args = parser.parse_args()

    global TARGET_MAP
    if args.map is not None:
        TARGET_MAP = args.map

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

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, N_SAMPLES, figsize=(7 * N_SAMPLES, 7))
    if N_SAMPLES == 1:
        axes = [axes]

    # ── Conditioning setup ────────────────────────────────────────────────────
    field_conditioned = cfg.get("field_conditioned", False)
    field_fn = None
    if field_conditioned:
        if TARGET_MAP is not None:
            grid = np.load(TARGET_MAP)
            map_ckpt = torch.load("checkpoints/map_generator.pt",
                                  map_location="cpu")
            h = map_ckpt["half_width"] / coord_scale
            field_fn = FieldSampler(grid, (-h, h, -h, h), device)
            print(f"conditioning on GENERATED map {TARGET_MAP} "
                  f"(half_width={map_ckpt['half_width']:.2f})")
        elif TARGET_PATCH is not None:
            d = np.load(TARGET_PATCH)
            tpos = d["pos"].astype(np.float64)
            tpos -= tpos.mean(axis=0)
            grid, extent = make_psi6_map(tpos)
            extent = tuple(e / coord_scale for e in extent)
            field_fn = FieldSampler(grid, extent, device)
            # replicate the target's composition too
            tns = d["node_scalars"]
            target_sizes = tns[:, 7].astype(np.float32)
            target_types = tns[:, 8].astype(np.float32)
            print(f"conditioning on psi6 map + composition of {TARGET_PATCH} "
                  f"(N={len(tpos)})")
        else:
            field_fn = NullField()
            print("field-conditioned model, unconditional (null-token) sampling")

    for i in tqdm(range(N_SAMPLES), desc="sampling", unit="config"):
        if field_conditioned and TARGET_MAP is None and TARGET_PATCH is not None:
            sizes, types = target_sizes, target_types
            cond = identity_scalars(sizes, types)
            n_gen = len(sizes)
        else:
            cond, sizes, types = build_sampling_identity(
                N_PARTICLES, DEFECT_FRACTION, SIZE_SMALL, SIZE_LARGE,
            )
            n_gen = N_PARTICLES

        repulsion_radii = None
        if REPULSION:
            repulsion_radii = torch.from_numpy(sizes / 2.0 / coord_scale)

        x_gen = diffusion.sample(
            model,
            n_particles=n_gen,
            node_scalars=torch.from_numpy(cond),
            device=device,
            corrector_steps=CORRECTOR_STEPS,
            field_fn=field_fn,
            field_guidance=FIELD_GUIDANCE,
            repulsion_radii=repulsion_radii,
            verbose=True,
        )
        pos = x_gen.cpu().numpy() * coord_scale   # back to physical units

        out_path = Path(OUT_DIR) / f"{args.prefix}_{i:03d}.npz"
        np.savez(out_path, pos=pos.astype(np.float32), types=types, sizes=sizes)
        print(f"saved {out_path}  (run: python evaluate.py {out_path})")

        plot_configuration(
            pos, types, sizes,
            title=f"sample {i}  |  N={n_gen}  defects={int((sizes > 1.2).sum())}",
            ax=axes[i],
        )

    fig.suptitle("Generated configurations from pure noise", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"outputs/results/{args.prefix}s_overview.png", dpi=150, bbox_inches="tight")
    print(f"saved outputs/results/{args.prefix}s_overview.png")
    plt.show()


if __name__ == "__main__":
    main()
