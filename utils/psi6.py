import numpy as np
import freud
import matplotlib.pyplot as plt

from lammps_parser import read_lammps_data
from config import DATA_PATH

def psi6(filepath):
    _, _, atoms, _, _, box_size_x, box_size_y = read_lammps_data(filepath)
    coords = atoms[:, 2:]
    coords = np.column_stack((coords, np.zeros(coords.shape[0])))
    box = freud.box.Box(box_size_x, box_size_y)

    # Build neighbor list via Voronoi tessellation (natural choice for hexatic order)
    voronoi = freud.locality.Voronoi()
    voronoi.compute(system=(box, coords))
    nlist = voronoi.nlist

    # Compute hexatic order parameter
    hexatic = freud.order.Hexatic(k=6)
    hexatic.compute(system=(box, coords), neighbors=nlist)

    psi6_values = hexatic.particle_order  # Complex ψ₆ per particle
    return psi6_values

if __name__ == "__main__":
    psi6_values = psi6(DATA_PATH)

    # Global order parameter (magnitude of mean)
    global_psi6 = np.abs(np.mean(psi6_values))
    print(f"Global |ψ₆|: {global_psi6:.4f}")

    # Distribution of per-particle magnitudes
    magnitudes = np.abs(psi6_values)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(magnitudes, bins=50, edgecolor='black')
    plt.xlabel(r"$|\psi_6|$")
    plt.ylabel("Count")
    plt.title(r"Distribution of $|\psi_6|$")

    plt.subplot(1, 2, 2)
    plt.scatter(np.real(psi6_values), np.imag(psi6_values), alpha=0.3, s=5)
    plt.xlabel(r"Re($\psi_6$)")
    plt.ylabel(r"Im($\psi_6$)")
    plt.title(r"$\psi_6$ in the complex plane")
    plt.axis('equal')

    plt.tight_layout()
    plt.show()