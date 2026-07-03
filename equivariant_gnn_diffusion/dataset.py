"""
Fixed LAMMPS reader (adds box_size) + standalone g(r) analysis script.

Two entry points:

1. Fix your npz files (run once):
       python data/run_gr_analysis.py --convert

   This re-reads data/config/* and saves data/npy/*.npz with box_size included.

2. Compute g(r) and estimate cutoff:
       python data/run_gr_analysis.py

   Loads up to N_FILES_FOR_GR files from data/npy/, computes and plots g(r),
   prints the recommended CUTOFF value to paste into train_unconditional.py.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from g_r_analysis import average_gr, estimate_cutoff, plot_gr
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_DIR      = "data/config"   # raw LAMMPS files
NPY_DIR         = "data/npy"      # converted npz files
N_FILES_FOR_GR  = 10              # number of files to average g(r) over
R_MAX           = None            # None = half the box; set explicitly if needed
N_BINS          = 300
SAVE_PLOT       = "gr_analysis.png"
# ──────────────────────────────────────────────────────────────────────────────


# ── Fixed LAMMPS reader ───────────────────────────────────────────────────────

def read_lammps_file(filepath):
    """
    Read a LAMMPS sphere-style data file.

    Expected atom line format (sphere style, no mol column):
        atom-ID  atom-type  diameter  density  x  y  z  [ix iy iz]

    Returns dict with keys:
        pos:      (N, 2) float32   x, y positions
        type:     (N,)   int32     0-indexed particle type (0=small, 1=large)
        size:     (N,)   float32   particle diameter
        box_size: (2,)   float32   [Lx, Ly]
    """
    positions      = []
    particle_types = []
    particle_sizes = []
    xlo = xhi = ylo = yhi = None
    reading_atoms = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # ── box dimensions ──────────────────────────────────────────────
            if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
                xlo, xhi = float(parts[0]), float(parts[1])
            elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
                ylo, yhi = float(parts[0]), float(parts[1])

            # ── section markers ─────────────────────────────────────────────
            if line.startswith("Atoms"):
                reading_atoms = True
                continue

            if reading_atoms:
                if line.startswith((
                    "Velocities", "Masses", "Bonds",
                    "Angles", "Dihedrals", "Impropers",
                )):
                    break

                if len(parts) < 7:
                    continue

                try:
                    particle_type = int(parts[1])
                    diameter      = float(parts[2])
                    x             = float(parts[4])
                    y             = float(parts[5])
                except ValueError:
                    continue

                particle_types.append(particle_type - 1)  # 0-index
                particle_sizes.append(diameter)
                positions.append([x, y])

    if xlo is None or ylo is None:
        raise ValueError(f"Could not parse box dimensions from {filepath}")

    return {
        "pos":      np.asarray(positions,      dtype=np.float32),
        "type":     np.asarray(particle_types, dtype=np.int32),
        "size":     np.asarray(particle_sizes, dtype=np.float32),
        "box_size": np.array([xhi - xlo, yhi - ylo], dtype=np.float32),
    }


def convert_directory(input_dir=CONFIG_DIR, output_dir=NPY_DIR):
    """Re-convert all LAMMPS files, now including box_size in the npz."""
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.iterdir())
    print(f"Converting {len(files)} files from {input_dir} -> {output_dir}")

    for filepath in files:
        if not filepath.is_file():
            continue
        try:
            data = read_lammps_file(filepath)
            out  = output_dir / f"{filepath.stem}.npz"
            np.savez(
                out,
                pos=data["pos"],
                type=data["type"],
                size=data["size"],
                box_size=data["box_size"],
            )
            print(f"  saved {out.name}  "
                  f"N={len(data['pos'])}  "
                  f"box={data['box_size']}")
        except Exception as e:
            print(f"  FAILED {filepath.name}: {e}")


# ── npz loader ────────────────────────────────────────────────────────────────

def load_npz_files(npy_dir=NPY_DIR, n_files=None):
    """
    Load npz files from npy_dir. Returns a list of dicts:
        {"pos": (N,2), "type": (N,), "size": (N,), "box_size": (2,)}

    If box_size is missing from a file (old npz without the fix above),
    raises a clear error telling you to re-run --convert.
    """
    paths = sorted(glob.glob(f"{npy_dir}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {npy_dir}")

    if n_files is not None:
        paths = paths[:n_files]

    configs = []
    for p in paths:
        d = np.load(p)
        if "box_size" not in d:
            raise KeyError(
                f"{p} is missing 'box_size'. "
                f"Re-run: python data/run_gr_analysis.py --convert"
            )
        configs.append({
            "pos":      d["pos"],
            "type":     d["type"],
            "size":     d["size"],
            "box_size": d["box_size"],
        })

    print(f"Loaded {len(configs)} configurations from {npy_dir}")
    return configs


# ── g(r) analysis ─────────────────────────────────────────────────────────────

def run_gr_analysis(configs, r_max=R_MAX, n_bins=N_BINS, save_path=SAVE_PLOT):
    """
    Average g(r) over configs, estimate cutoff, plot, and print the result.
    """
    pos_list   = [c["pos"]      for c in configs]
    box_list   = [c["box_size"] for c in configs]
    types_list = [c["type"]     for c in configs]

    print(f"Computing g(r) averaged over {len(configs)} configurations...")
    gr_result = average_gr(
        pos_list, box_list,
        r_max=r_max, n_bins=n_bins,
        types_list=types_list,
    )

    cutoff, first_peak, diag = estimate_cutoff(gr_result)

    print(f"\n  first peak:              r = {first_peak:.4f}")
    print(f"  recommended cutoff:      r = {cutoff:.4f}   <-- use this as CUTOFF")
    print(f"  all minima after peak:   {[f'{x:.4f}' for x in diag['all_mins']]}")
    print(f"\nPaste into train_unconditional.py:")
    print(f"  CUTOFF = {cutoff:.4f}")

    fig = plot_gr(
        gr_result,
        cutoff=cutoff,
        title=f"g(r) averaged over {len(configs)} configs  |  cutoff = {cutoff:.4f}",
        type_labels={0: "small", 1: "large"},
        save_path=save_path,
    )
    plt.show()
    return cutoff


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--convert", action="store_true",
        help="Re-convert LAMMPS files to npz (adds box_size). Run this first.",
    )
    args = parser.parse_args()

    if args.convert:
        convert_directory()
    else:
        configs = load_npz_files(n_files=N_FILES_FOR_GR)
        run_gr_analysis(configs)