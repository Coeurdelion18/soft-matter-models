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


def render_gaussian_channels(
    atoms,
    xlo,
    ylo,
    box_size_x,
    box_size_y,
    grid_size=192,
    sigma_scale=0.22,
):
    """
    Render atoms into two Gaussian channels.

    sigma = sigma_scale * diameter / 2
    """

    image = np.zeros(
        (2, grid_size, grid_size),
        dtype=np.float32
    )

    x_grid = np.linspace(
        xlo,
        xlo + box_size_x,
        grid_size
    )

    y_grid = np.linspace(
        ylo,
        ylo + box_size_y,
        grid_size
    )

    X, Y = np.meshgrid(x_grid, y_grid)

    for particle_type, diameter, x, y in atoms:

        particle_type = int(particle_type)

        # if particle_type not in [1, 2]:
        #     continue

        channel = particle_type - 1

        sigma = 0.20 if particle_type == 2 else 0.28 #sigma_scale * (diameter / 2)

        gaussian = np.exp(
            -(
                (X - x) ** 2 +
                (Y - y) ** 2
            ) / (2 * sigma**2)
        )

        image[channel] += gaussian

    return image


def lammps_to_numpy(
    input_file,
    output_file="rendered.npy",
    grid_size=192,
    sigma_scale=1.0,
    normalize=False,
):
    """
    Convert LAMMPS file → 2-channel numpy array.
    """

    (   count_1,
        count_2,
        atoms,
        xlo,
        ylo,
        box_size_x,
        box_size_y
    ) = read_lammps_data(input_file)

    # print(f"Loaded {len(atoms)} atoms")
    # print(
    #     f"Box size: "
    #     f"{box_size_x:.3f} × "
    #     f"{box_size_y:.3f}"
    # )
    # print(f"Total count: {len(atoms)}")
    # print(f"Type 1 particle count: {count_1}")
    # print(f"Type 2 particle count: {count_2}")

    image = render_gaussian_channels(
        atoms=atoms,
        xlo=xlo,
        ylo=ylo,
        box_size_x=box_size_x,
        box_size_y=box_size_y,
        grid_size=grid_size,
        sigma_scale=sigma_scale,
    )

    if normalize:
        image_min = image.min()
        image_max = image.max()

        image = (
            image - image_min
        ) / (
            image_max - image_min + 1e-8
        )

    np.save(output_file, image)

    # print(f"Saved to {output_file}")
    # print("Shape:", image.shape)

    return image


# ==================================
# Example usage
# ==================================
if __name__ == "__main__":
    image = lammps_to_numpy(
        input_file=FILE_PATH,
        output_file=OUTPUT_PATH,
        grid_size=384,
        sigma_scale=1.0,   # sigma = diameter/2
        normalize=False
    )