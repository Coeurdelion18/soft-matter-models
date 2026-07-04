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

N_TARGET       = 4000               # target particles per patch (~110 diameters wide,
                                    # several grains + boundary junctions per patch;
                                    # measured 2.1 GB VRAM per training step on RTX 4050)

OVERLAP        = 0.5                # fraction of patch diameter to overlap
                                    # 0.0 = no overlap (fewest patches)
                                    # 0.5 = centers spaced by one radius (more patches)
                                    # 0.5 recommended: ensures boundary segments
                                    # near grid lines aren't split across patches

MIN_PARTICLES  = 2000               # discard patches with fewer than this
                                    # (can happen at box edges)

# train/val split — done at box level here so patches from the same box
# never appear in both sets
VAL_FRACTION   = 0.1
# ──────────────────────────────────────────────────────────────────────────────


def compute_patch_radius(box_size, n_total, n_target):
    """
    Radius that gives approximately n_target particles per circular patch,
    assuming uniform particle density.
    """
    Lx, Ly     = box_size[0], box_size[1]
    box_area   = Lx * Ly
    density    = n_total / box_area
    patch_area = n_target / density
    return float(np.sqrt(patch_area / np.pi))


def grid_centers(pos, patch_radius, overlap):
    """Use actual position extents rather than assuming origin at 0."""
    spacing  = 2.0 * patch_radius * (1.0 - overlap)
    pos_min  = pos.min(axis=0)
    pos_max  = pos.max(axis=0)
    xs = np.arange(pos_min[0] + patch_radius, pos_max[0], spacing)
    ys = np.arange(pos_min[1] + patch_radius, pos_max[1], spacing)
    centers = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    return centers


def extract_patches_from_box(box_data, patch_radius, min_particles):
    """
    Extract all grid patches from one box.

    box_data: dict with keys pos, type, size, box_size, node_scalars

    Returns list of dicts, each with:
        pos:          (M, 2)  positions shifted so patch CoM = 0
        node_scalars: (M, 9)  structural features (already computed)
    """
    pos      = box_data["pos"]            # (N, 2)
    scalars  = box_data["node_scalars"]   # (N, 9)
    box_size = box_data["box_size"]       # (2,)
    N        = len(pos)

    centers = grid_centers(pos, patch_radius, OVERLAP)

    patches = []
    for center in centers:
        diffs = pos - center
        dists = np.linalg.norm(diffs, axis=1)
        mask  = dists <= patch_radius
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

        # compute patch radius from actual box dimensions and particle count
        patch_radius = compute_patch_radius(box_size, len(pos), N_TARGET)

        patches = extract_patches_from_box(box_data, patch_radius, MIN_PARTICLES)

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
    print(f"Patch radius used: computed per box from N_TARGET={N_TARGET}")
    print(f"\nUpdate train.py:")
    print(f"  TRAIN_DIR = '{train_dir}'")
    print(f"  VAL_DIR   = '{val_dir}'")


if __name__ == "__main__":
    main()