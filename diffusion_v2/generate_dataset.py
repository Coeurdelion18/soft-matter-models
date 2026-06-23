"""
generate_dataset.py

Converts a directory of LAMMPS config files into a .npy dataset of
5-channel images ready for the diffusion model.

Output shape per sample: (4, grid_size, grid_size)

Usage:
    python generate_dataset.py \
        --input_dir  data/raw/config \
        --output_dir data/processed \
        --grid_size  384 \
        --workers    4
"""

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lammps_parser import read_lammps_data
from compute_channels import compute_all_channels


# ---------------------------------------------------------------------------
# Per-file worker  (must be a top-level function for multiprocessing)
# ---------------------------------------------------------------------------

def process_file(args):
    """
    Load one LAMMPS file and return the 5-channel image.
    Returns (filename, image) on success, (filename, None) on failure.
    """
    filepath, grid_size, normalize = args

    try:
        (
            _count1, _count2,
            atoms,
            xlo, ylo,
            box_size_x, box_size_y,
        ) = read_lammps_data(filepath)

        image = compute_all_channels(
            atoms=atoms,
            xlo=xlo,
            ylo=ylo,
            box_size_x=box_size_x,
            box_size_y=box_size_y,
            grid_size=grid_size,
            normalize=normalize,
        )
        return str(filepath), image

    except Exception:
        print(f"\n[WARN] Failed on {filepath}:\n{traceback.format_exc()}")
        return str(filepath), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_dataset(
    input_dir: str,
    output_dir: str,
    grid_size: int = 384,
    normalize: bool = True,
    workers: int = 4,
    save_individual: bool = True,
    save_stacked: bool = True,
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files (no extension filter — LAMMPS files are plain numbered)
    filepaths = sorted(
        p for p in input_dir.iterdir() if p.is_file()
    )
    print(f"Found {len(filepaths)} files in {input_dir}")

    worker_args = [
        (fp, grid_size, normalize) for fp in filepaths
    ]

    images   = []
    names    = []
    failed   = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_file, a): a[0] for a in worker_args
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Rendering",
        ):
            filepath, image = future.result()
            if image is None:
                failed.append(filepath)
                continue

            name = Path(filepath).stem  # e.g. "100"

            if save_individual:
                out_path = output_dir / f"{name}.npy"
                np.save(out_path, image)

            images.append(image)
            names.append(name)

    print(f"\nProcessed {len(images)} / {len(filepaths)} files successfully.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")

    if save_stacked and images:
        # Stack into (N, 4, H, W) dataset array
        dataset = np.stack(images, axis=0).astype(np.float32)
        out_path = output_dir / "dataset.npy"
        np.save(out_path, dataset)
        print(f"\nStacked dataset saved → {out_path}")
        print(f"  Shape : {dataset.shape}")
        print(f"  dtype : {dataset.dtype}")
        print(f"  Size  : {dataset.nbytes / 1e9:.2f} GB")

        # Also save the file ordering so you can recover metadata later
        names_path = output_dir / "dataset_names.txt"
        names_path.write_text("\n".join(names))
        print(f"  Names → {names_path}")

    return images, names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 5-channel image dataset from LAMMPS configs"
    )
    parser.add_argument(
        "--input_dir",  type=str, default="../data/raw/config",
        help="Directory containing LAMMPS data files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Directory to write .npy output files",
    )
    parser.add_argument(
        "--grid_size",  type=int, default=384,
        help="Pixel resolution of each output image (default: 384)",
    )
    parser.add_argument(
        "--workers",    type=int, default=6,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--no_normalize", action="store_true",
        help="Skip per-channel normalisation to [0,1]",
    )
    parser.add_argument(
        "--no_individual", action="store_true",
        help="Skip saving individual .npy files per config",
    )
    parser.add_argument(
        "--no_stack", action="store_true",
        help="Skip saving the stacked dataset.npy",
    )

    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        normalize=not args.no_normalize,
        workers=args.workers,
        save_individual=not args.no_individual,
        save_stacked=not args.no_stack,
    )