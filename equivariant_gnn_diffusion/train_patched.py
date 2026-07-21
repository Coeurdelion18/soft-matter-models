"""
Train the unconditional generator on precomputed patches.

Run make_patches.py first to generate data/patches/train/ and data/patches/val/.
Existing patch files work unchanged: only columns 7 (size) and 8 (type) of the
stored node_scalars are used for conditioning. The structural descriptors
(Voronoi, psi6, ...) are deliberately NOT fed to the model -- they are computed
from the clean positions, so during training they leak the answer, and at
sampling time they don't exist. Keep them for evaluating generated samples.

Usage:
    python train_patched.py
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion, remove_com
from field_conditioning import make_psi6_map, FieldSampler, NullField

# ── Hyperparameters ───────────────────────────────────────────────────────────
TRAIN_DIR    = "data/patches/train"
VAL_DIR      = "data/patches/val"

K_NEIGHBORS  = 12
HIDDEN_DIM   = 128
N_LAYERS     = 6
N_STEPS      = 1000
LR           = 2e-4
N_EPOCHS     = 60
EMA_DECAY    = 0.999
GRAD_CLIP    = 1.0
SCHEDULE     = "logsigma"   # log-spaced noise ladder matched to the data's
                            # length scales; see logsigma_beta_schedule in
                            # diffusion.py for why cosine failed here

# Continue from an existing checkpoint (None = train from scratch).
# NOTE: only resume into the SAME schedule AND conditioning setup the
# checkpoint was trained with; both change the input semantics.
RESUME_FROM  = None   # input dim changed (2-channel field): fresh start required

# Fraction of training steps forced into the lowest 15% of noise levels
# (0 = uniform). With the logsigma schedule the step allocation already
# covers fine scales, so this hack is no longer needed.
LOW_T_FRAC   = 0.0

# psi6-field conditioning: the model additionally sees a coarse |psi6| map
# (computed from the clean patch during training, supplied by the user at
# sampling time). FIELD_DROPOUT is the classifier-free fraction of steps
# where the field is replaced by the null token so the model also learns
# unconditional generation.
FIELD_COND    = True
FIELD_DROPOUT = 0.15

CHECKPOINT_DIR = "checkpoints"
# ─────────────────────────────────────────────────────────────────────────────

# identity feature encoding: fixed, data-independent, so sampling never needs
# training-set statistics for these
SIZE_CENTER, SIZE_SCALE = 1.2, 0.2   # small=1.0 -> -1, large=1.4 -> +1


def identity_scalars(sizes, types):
    """(N,), (N,) -> (N, 2) conditioning features in O(1) range."""
    return np.stack([
        (sizes - SIZE_CENTER) / SIZE_SCALE,
        types * 2.0 - 1.0,
    ], axis=-1).astype(np.float32)


def load_patches(patch_dir):
    paths = sorted(glob.glob(f"{patch_dir}/*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No patch files found in {patch_dir}. Run make_patches.py first."
        )
    patches = []
    for p in tqdm(paths, desc=f"loading {Path(patch_dir).name}", unit="patch"):
        d = np.load(p)
        pos = d["pos"].astype(np.float32)
        ns = d["node_scalars"]
        sizes = ns[:, 7].astype(np.float32)
        types = ns[:, 8].astype(np.float32)
        pos = pos - pos.mean(axis=0, keepdims=True)
        patch = {
            "pos": pos,
            "cond": identity_scalars(sizes, types),
        }
        if FIELD_COND:
            grid, extent = make_psi6_map(pos)   # physical units
            patch["field_grid"] = grid
            patch["field_extent"] = extent
        patches.append(patch)
    return patches


def compute_coord_scale(patches):
    """
    Global RMS coordinate value over the (CoM-free) training set. Dividing by
    this brings positions to unit variance, which the diffusion prior N(0, I)
    requires. One global scalar (not per-patch) so all patches share units.
    """
    sq_sum, count = 0.0, 0
    for p in patches:
        sq_sum += float((p["pos"] ** 2).sum())
        count += p["pos"].size
    return float(np.sqrt(sq_sum / count))


class EMA:
    """Exponential moving average of model weights; use these for sampling."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def save_checkpoint(path, model_state, ema_state, coord_scale):
    torch.save({
        "model": model_state,
        "ema": ema_state,
        "coord_scale": coord_scale,
        "config": {
            "hidden_dim": HIDDEN_DIM,
            "n_layers": N_LAYERS,
            "n_steps": N_STEPS,
            "k_neighbors": K_NEIGHBORS,
            "n_scalar_feats": 4 if FIELD_COND else 2,
            "schedule": SCHEDULE,
            "field_conditioned": FIELD_COND,
        },
    }, path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_patches = load_patches(TRAIN_DIR)
    val_patches = load_patches(VAL_DIR)
    print(f"train patches: {len(train_patches)}")
    print(f"val   patches: {len(val_patches)}")
    print(f"particles per patch: ~{len(train_patches[0]['pos'])}")

    coord_scale = compute_coord_scale(train_patches)
    print(f"coord_scale (global RMS coordinate): {coord_scale:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    n_scalar_feats = 4 if FIELD_COND else 2   # identity (2) + field (value, flag)
    model = EGNNUnconditionalDenoiser(
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        n_scalar_feats=n_scalar_feats,
    ).to(device)
    diffusion = PositionDiffusion(
        n_steps=N_STEPS, device=device, k_neighbors=K_NEIGHBORS,
        schedule=SCHEDULE,
    )
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"resumed weights from {RESUME_FROM}")
        # keep the coord_scale the model was trained with -- it must not
        # change across resumes or the learned length scales break
        coord_scale = ckpt["coord_scale"]
        print(f"using checkpoint coord_scale: {coord_scale:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ema = EMA(model, EMA_DECAY)
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        ema.shadow = {k: v.clone().to(device) for k, v in ckpt["ema"].items()}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    best_val_loss = float("inf")
    epoch_bar = tqdm(range(N_EPOCHS), desc="training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        np.random.shuffle(train_patches)
        train_loss, n = 0.0, 0
        train_bar = tqdm(train_patches, desc="  train", unit="patch", leave=False)

        for patch in train_bar:
            x0 = torch.from_numpy(patch["pos"]).to(device) / coord_scale
            cond = torch.from_numpy(patch["cond"]).to(device)

            field_fn = None
            if FIELD_COND:
                if np.random.rand() < FIELD_DROPOUT:
                    field_fn = NullField()
                else:
                    ext = tuple(e / coord_scale for e in patch["field_extent"])
                    field_fn = FieldSampler(patch["field_grid"], ext, device)

            opt.zero_grad()
            loss = diffusion.training_loss(model, x0, cond,
                                           low_t_frac=LOW_T_FRAC,
                                           field_fn=field_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ema.update(model)

            train_loss += loss.item()
            n += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss, nv = 0.0, 0
        val_bar = tqdm(val_patches, desc="  val  ", unit="patch", leave=False)
        with torch.no_grad():
            for patch in val_bar:
                x0 = torch.from_numpy(patch["pos"]).to(device) / coord_scale
                cond = torch.from_numpy(patch["cond"]).to(device)
                field_fn = None
                if FIELD_COND:
                    ext = tuple(e / coord_scale for e in patch["field_extent"])
                    field_fn = FieldSampler(patch["field_grid"], ext, device)
                loss = diffusion.training_loss(model, x0, cond,
                                               field_fn=field_fn)
                val_loss += loss.item()
                nv += 1
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        tl = train_loss / max(n, 1)
        vl = val_loss / max(nv, 1)
        epoch_bar.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}",
                              best=f"{best_val_loss:.4f}")

        # written every epoch so an interrupted run never loses more than one
        save_checkpoint(f"{CHECKPOINT_DIR}/model_last.pt",
                        model.state_dict(), ema.shadow, coord_scale)

        if vl < best_val_loss:
            best_val_loss = vl
            save_checkpoint(f"{CHECKPOINT_DIR}/model_best.pt",
                            model.state_dict(), ema.shadow, coord_scale)

    save_checkpoint(f"{CHECKPOINT_DIR}/model_final.pt",
                    model.state_dict(), ema.shadow, coord_scale)
    print(f"\ndone. best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
