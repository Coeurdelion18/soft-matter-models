import numpy as np
import freud
import matplotlib.pyplot as plt

from lammps_parser import read_lammps_data
from config import DATA_PATH

#Read the data directly from the LAMMPS files

#The form of atoms is [particle_type, diameter, x_pos, y_pos]

def g_r(filepath):
    _, _, atoms, _, _, box_size_x, box_size_y = read_lammps_data(filepath)
    coords = atoms[:,2:]
    coords = np.column_stack((coords, np.zeros(coords.shape[0])))
    box = freud.box.Box(box_size_x, box_size_y)
    L = 0.5 * min(box_size_x, box_size_y)
    rdf = freud.density.RDF(bins=200, r_max=L/2)
    rdf.compute(system=(box, coords))
    return rdf.bin_centers, rdf.rdf #x, y on the plot

if __name__ == "__main__":
    r, g = g_r(DATA_PATH)
    plt.plot(r, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.show()