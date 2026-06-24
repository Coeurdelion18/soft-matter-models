from pathlib import Path

import numpy as np


def read_lammps_file(filepath):
    """
    Read a LAMMPS data file and return particle information.

    Returns
    -------
    dict
        {
            "pos":  (N, 2) array of x,y positions,
            "type": (N,) array of particle types,
            "size": (N,) array of particle diameters
        }
    """

    positions = []
    particle_types = []
    particle_sizes = []

    reading_atoms = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("Atoms"):
                reading_atoms = True
                continue

            if reading_atoms:

                # reached next section
                if line.startswith(
                    (
                        "Velocities",
                        "Masses",
                        "Bonds",
                        "Angles",
                        "Dihedrals",
                        "Impropers",
                    )
                ):
                    break

                parts = line.split()

                if len(parts) < 7:
                    continue

                try:
                    particle_type = int(parts[1])
                    diameter = float(parts[2])

                    x = float(parts[4])
                    y = float(parts[5])

                except ValueError:
                    continue

                particle_types.append(particle_type)
                particle_sizes.append(diameter)
                positions.append([x, y])

    return {
        "pos": np.asarray(positions, dtype=np.float32),
        "type": np.asarray(particle_types, dtype=np.int32),
        "size": np.asarray(particle_sizes, dtype=np.float32),
    }


def convert_directory(input_dir, output_dir):
    """
    Convert every LAMMPS file in a directory to .npy format.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(input_dir.iterdir()):

        if not filepath.is_file():
            continue

        try:
            data = read_lammps_file(filepath)

            output_file = output_dir / f"{filepath.stem}.npy"
            np.save(output_file, data)

            print(f"Saved {output_file}")

        except Exception as e:
            print(f"Failed on {filepath.name}: {e}")


if __name__ == "__main__":
    convert_directory(
        "data/config",
        "data/npy",
    )