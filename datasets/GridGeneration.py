import numpy as np
import matplotlib.pyplot as plt

def generateHexagonalGrid(nx = 64, ny = 64, a = 1.0, defect_rate = 0.10):

    coords = []

    for j in range(ny):
        y = j * (np.sqrt(3) / 2) * a

        for i in range(nx):
            x = i * a

            if j % 2 == 1:
                x += 0.5 * a

            coords.append([x, y])

    coords = np.array(coords, dtype = float)

    mask = np.random.random(len(coords)) > defect_rate
    coords = coords[mask]

    #We add a random global translation and random displacement noise to prevent overfitting the model
    shift = np.array([
            np.random.uniform(0, a),
            np.random.uniform(0, (np.sqrt(3) / 2) * a)
        ])

    coords += shift
    coords += np.random.normal(loc=0.0, scale=0.03 * a, size=coords.shape)
    
    Lx = (nx - 0.5) * a
    Ly = (ny - 1) * (np.sqrt(3) / 2) * a

    return coords, (Lx, Ly)



def coords_to_density_grid(coords, box_size, grid_size = 256, sigma = 1.0):
    
    Lx, Ly = box_size
    grid = np.zeros((grid_size, grid_size), dtype = float)

    xpix = coords[:, 0] / Lx * (grid_size - 1)
    ypix = coords[:, 1] / Ly * (grid_size - 1)

    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing = 'xy')

    for x, y in zip(xpix, ypix):

        dx = X - x
        dy = Y - y
        grid += np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    return grid



def plotGrid(
    grid,
    figsize=(8, 8),
    cmap='viridis',
    interpolation='nearest',
    title='Grid'
):
    """
    Plot grid/density field.
    """

    plt.figure(figsize=figsize)

    plt.imshow(
        grid,
        origin='lower',
        cmap=cmap,
        interpolation=interpolation
    )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    coords, box_size = generateHexagonalGrid(nx=48, ny=48, a=1.0, defect_rate=0.10)
    density_grid = coords_to_density_grid(
        coords,
        box_size,
        grid_size=128,
        sigma=0.4
    )

    plotGrid(
        density_grid,
        cmap='viridis',
        interpolation='bilinear',
        title='Smooth Density Field'
    )