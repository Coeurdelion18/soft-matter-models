"""
Train the unconditional generator on precomputed patches.

Run make_patches.py first to generate data/patches/train/ and data/patches/val/.
Then run this script.

Usage:
    python train.py
"""

import glob
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion
from node_features import normalise_scalars

# ── Hyperparameters ───────────────────────────────────────────────────────────
TRAIN_DIR      = "data/patches/train"
VAL_DIR        = "data/patches/val"

CUTOFF         = 1.5
HIDDEN_DIM     = 128
N_LAYERS       = 2
N_SCALAR_FEATS = 9
N_STEPS        = 1000
LR             = 2e-4
N_EPOCHS       = 40
# ─────────────────────────────────────────────────────────────────────────────


def load_patches(patch_dir):
    """
    Load all patch files from a directory.
    Returns list of dicts with pos and node_scalars as numpy arrays.
    """
    paths = sorted(glob.glob(f"{patch_dir}/*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No patch files found in {patch_dir}. "
            f"Run make_patches.py first."
        )
    patches = []
    for p in tqdm(paths, desc=f"loading {Path(patch_dir).name}", unit="patch"):
        d = np.load(p)
        patches.append({
            "pos":          d["pos"],
            "node_scalars": d["node_scalars"],
        })
    return patches


def main():
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load patches ──────────────────────────────────────────────────────────
    train_patches = load_patches(TRAIN_DIR)
    val_patches   = load_patches(VAL_DIR)
    print(f"train patches: {len(train_patches)}")
    print(f"val   patches: {len(val_patches)}")
    print(f"particles per patch: ~{len(train_patches[0]['pos'])}")

    # ── Normalise features ────────────────────────────────────────────────────
    # Fit on training set only, apply same stats to val
    train_scalars = [p["node_scalars"] for p in train_patches]
    norm_train, feat_mean, feat_std = normalise_scalars(train_scalars)
    for patch, ns in zip(train_patches, norm_train):
        patch["node_scalars"] = ns

    for patch in val_patches:
        patch["node_scalars"] = (patch["node_scalars"] - feat_mean) / feat_std

    os.makedirs("checkpoints", exist_ok=True)
    np.savez("checkpoints/feature_stats.npz", mean=feat_mean, std=feat_std)
    print("saved checkpoints/feature_stats.npz")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = EGNNUnconditionalDenoiser(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_scalar_feats=N_SCALAR_FEATS,
    ).to(device)
    diffusion = PositionDiffusion(n_steps=N_STEPS, device=device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epoch_bar     = tqdm(range(N_EPOCHS), desc="training", unit="epoch")

    for epoch in epoch_bar:

        # --- train ---
        model.train()
        np.random.shuffle(train_patches)
        train_loss, n = 0.0, 0
        train_bar = tqdm(train_patches, desc=f"  train", unit="patch", leave=False)

        for patch in train_bar:
            x0      = torch.from_numpy(patch["pos"]).to(device)
            scalars = torch.from_numpy(patch["node_scalars"]).to(device)

            opt.zero_grad()
            loss = diffusion.training_loss(model, x0, scalars, cutoff=CUTOFF)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()
            n          += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- val ---
        model.eval()
        val_loss, nv = 0.0, 0
        val_bar = tqdm(val_patches, desc=f"  val  ", unit="patch", leave=False)

        with torch.no_grad():
            for patch in val_bar:
                x0      = torch.from_numpy(patch["pos"]).to(device)
                scalars = torch.from_numpy(patch["node_scalars"]).to(device)
                loss    = diffusion.training_loss(model, x0, scalars, cutoff=CUTOFF)
                val_loss += loss.item()
                nv       += 1
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        tl = train_loss / max(n, 1)
        vl = val_loss   / max(nv, 1)
        epoch_bar.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}",
                              best=f"{best_val_loss:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), "checkpoints/model_best.pt")

    torch.save(model.state_dict(), "checkpoints/model_final.pt")
    print(f"\ndone. best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()