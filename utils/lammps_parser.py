import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "data/raw/config/100"
OUTPUT_PATH = "data/raw/npy100.npy"

def read_lammps_data(filepath):
    """
    Read a LAMMPS sphere-style data file.

    Extracts:
    - particle type
    - diameter
    - x, y position
    - simulation box size

    Expected atom format:
    id mol type diameter x y z ix iy iz
    """

    atoms = []
    count_1 = 0
    count_2 = 0

    xlo, xhi = None, None
    ylo, yhi = None, None

    reading_atoms = False

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        # -------------------------
        # Parse box size
        # -------------------------
        if len(parts) >= 4:
            if parts[-2:] == ["xlo", "xhi"]:
                xlo = float(parts[0])
                xhi = float(parts[1])

            elif parts[-2:] == ["ylo", "yhi"]:
                ylo = float(parts[0])
                yhi = float(parts[1])

        # -------------------------
        # Find Atoms section
        # -------------------------
        if stripped.startswith("Atoms"):
            reading_atoms = True
            continue

        # -------------------------
        # Parse atoms
        # -------------------------
        if reading_atoms:

            # stop at next section
            if stripped.startswith((
                "Velocities",
                "Masses",
                "Bonds",
                "Angles",
                "Dihedrals",
                "Impropers"
            )):
                break

            if len(parts) < 7:
                continue

            try:
                # id type diameter x y z ix iy iz

                particle_type = int(parts[1])
                if particle_type == 1:
                    count_1 += 1

                elif particle_type == 2:
                    count_2 += 1

                diameter = float(parts[2])

                x = float(parts[4])
                y = float(parts[5])

                atoms.append(
                    (particle_type, diameter, x, y)
                )

            except ValueError:
                continue

    if xlo is None or ylo is None:
        raise ValueError("Could not parse simulation box.")

    atoms = np.array(atoms)

    box_size_x = xhi - xlo
    box_size_y = yhi - ylo

    return count_1, count_2, atoms, xlo, ylo, box_size_x, box_size_y