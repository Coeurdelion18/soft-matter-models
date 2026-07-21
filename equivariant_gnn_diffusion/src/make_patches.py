"""
Extract fixed-size spatial patches from full simulation boxes using a
regular grid of patch centers, then save each patch as its own .npz file.

Run once before training:
    python make_patches.py

Output files (in PATCH_DIR) have keys:
    pos:          (M, 2)  float32   particle positions, CoM-centered
    node_scalars: (M, 9)  float32   precomputed structural features
    parent:       str               source npz filename (for debugging)

Training then loops over patch files directly -- no runtime graph building,
no feature computation, no cropping. The training loop just does:
    d = np.load(patch_file)
    x0      = torch.from_numpy(d["pos"])
    scalars = torch.from_numpy(d["node_scalars"])
"""

import glob
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from node_features import build_node_scalars

# ── Config ────────────────────────────────────────────────────────────────────
NPY_DIR        = "data/npy"         # input: full simulation boxes
PATCH_DIR      = "data/patches"     # output: individual patch files

N_TARGET       = 4900               # target particles per SQUARE patch (~70 diameters
                                    # wide = 2-3 grain diameters, so several grains and
                                    # boundary junctions per patch; ~2.6 GB VRAM/step
                                    # on RTX 4050 by measured scaling)

N_CROPS_PER_BOX = 5                 # random fully-inside square crops per box
                                    # (a fixed grid would give only ~1 full 70-wide
                                    # square per 100-wide box; random offsets give
                                    # diversity while keeping every patch a full square)

MIN_PARTICLES  = int(N_TARGET * 0.85)   # sanity floor; full-square crops should
                                        # always be near N_TARGET

# train/val split — done at box level here so patches from the same box
# never appear in both sets
VAL_FRACTION   = 0.1
# ──────────────────────────────────────────────────────────────────────────────


def compute_half_side(box_size, n_total, n_target):
    """
    Half side-length of a square patch containing approximately n_target
    particles, assuming uniform particle density.
    """
    Lx, Ly  = box_size[0], box_size[1]
    density = n_total / (Lx * Ly)
    side    = float(np.sqrt(n_target / density))
    return side / 2.0


def random_square_centers(pos, half_side, n_crops, rng):
    """
    Random patch centers such that the full square [c-h, c+h]^2 lies inside
    the particle extent -- every patch is a complete square (no clipped
    shapes, which would give the model inconsistent envelopes to learn).
    """
    lo = pos.min(axis=0) + half_side
    hi = pos.max(axis=0) - half_side
    if np.any(hi <= lo):
        raise ValueError(
            f"patch side {2*half_side:.1f} does not fit inside the box; "
            f"reduce N_TARGET"
        )
    return rng.uniform(lo, hi, size=(n_crops, 2)).astype(np.float32)


def extract_patches_from_box(box_data, half_side, min_particles, rng):
    """
    Extract N_CROPS_PER_BOX random square patches from one box.

    Returns list of dicts, each with:
        pos:          (M, 2)  positions shifted so patch CoM = 0
        node_scalars: (M, 9)  structural features (already computed)
    """
    pos      = box_data["pos"]            # (N, 2)
    scalars  = box_data["node_scalars"]   # (N, 9)

    centers = random_square_centers(pos, half_side, N_CROPS_PER_BOX, rng)

    patches = []
    for center in centers:
        diffs = np.abs(pos - center)
        mask  = (diffs[:, 0] <= half_side) & (diffs[:, 1] <= half_side)
        idx   = np.where(mask)[0]

        if len(idx) < min_particles:
            continue

        patch_pos     = pos[idx].copy()
        patch_scalars = scalars[idx].copy()

        # center the patch positions (CoM removal at save time so training
        # doesn't have to do it -- diffusion.remove_com still applies during
        # the forward pass, this just keeps file values near zero for sanity)
        patch_pos -= patch_pos.mean(axis=0, keepdims=True)

        patches.append({
            "pos":          patch_pos.astype(np.float32),
            "node_scalars": patch_scalars.astype(np.float32),
        })

    return patches


def main():
    paths = sorted(glob.glob(f"{NPY_DIR}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {NPY_DIR}")
    print(f"Found {len(paths)} boxes in {NPY_DIR}")

    # ── train / val split at box level ───────────────────────────────────────
    np.random.seed(42)   # fixed seed so split is reproducible across reruns
    rng = np.random.default_rng(123)   # crop centers, reproducible
    shuffled  = np.random.permutation(len(paths))
    n_val     = max(1, int(len(paths) * VAL_FRACTION))
    val_idxs  = set(shuffled[:n_val].tolist())
    train_idxs = set(shuffled[n_val:].tolist())

    train_dir = Path(PATCH_DIR) / "train"
    val_dir   = Path(PATCH_DIR) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # remove stale patches from previous runs -- a different N_TARGET yields
    # fewer files per box, so leftovers would silently mix patch sizes
    removed = 0
    for d in (train_dir, val_dir):
        for old in d.glob("*.npz"):
            old.unlink()
            removed += 1
    if removed:
        print(f"removed {removed} stale patch files")

    train_count = val_count = 0

    for file_idx, fpath in enumerate(tqdm(paths, desc="extracting patches", unit="box")):
        d        = np.load(fpath)
        pos      = d["pos"]
        types    = d["type"]
        sizes    = d["size"]
        box_size = d["box_size"]

        # compute node scalars for this box
        # in make_patches.py, before calling build_node_scalars:
        pos_centered = pos - pos.mean(axis=0)   # rough centering
        scalars = build_node_scalars(pos_centered, types, sizes, box_size)

        box_data = {
            "pos":          pos,
            "node_scalars": scalars,
            "box_size":     box_size,
        }

        half_side = compute_half_side(box_size, len(pos), N_TARGET)

        patches = extract_patches_from_box(box_data, half_side, MIN_PARTICLES, rng)

        is_val  = file_idx in val_idxs
        out_dir = val_dir if is_val else train_dir
        stem    = Path(fpath).stem

        for i, patch in enumerate(patches):
            out_path = out_dir / f"{stem}_patch{i:03d}.npz"
            np.savez(
                out_path,
                pos=patch["pos"],
                node_scalars=patch["node_scalars"],
                parent=str(fpath),
            )
            if is_val:
                val_count += 1
            else:
                train_count += 1

    print(f"\ndone.")
    print(f"  train patches: {train_count}  -> {train_dir}")
    print(f"  val   patches: {val_count}    -> {val_dir}")
    print(f"\nExpected particles per patch: ~{N_TARGET}")
    print(f"Square patch side: computed per box from N_TARGET={N_TARGET}")
    print(f"\nUpdate train.py:")
    print(f"  TRAIN_DIR = '{train_dir}'")
    print(f"  VAL_DIR   = '{val_dir}'")


if __name__ == "__main__":
    main()