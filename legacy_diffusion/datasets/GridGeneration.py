import numpy as np
import matplotlib.pyplot as plt
import random

GRID_SIZE = 192

def generateHexagonalGrid(nx = 64, ny = 64, a = 1.0, defect_rate = 0.10, pbc=False):

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

    if pbc:
        coords[:,0] %= Lx
        coords[:,1] %= Ly

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

def generate_diatomic_density_grid(coords, box_size, grid_size=GRID_SIZE, sigma_a=0.3, sigma_b=0.42):
    Lx, Ly = box_size
    grid = np.zeros((grid_size, grid_size), dtype=float)

    xpix = coords[:, 0] / Lx * (grid_size - 1)
    ypix = coords[:, 1] / Ly * (grid_size - 1)

    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing = 'xy')

    for x, y in zip(xpix, ypix):
        dx = X - x
        dy = Y - y
        if random.random() < 0.20:
            grid += np.exp(-(dx**2 + dy**2) / (2 * sigma_b**2))
        else:
            grid += np.exp(-(dx**2 + dy**2) / (2 * sigma_a**2))
        
    return grid

def generate_two_channel_density_grid(coords, box_size, grid_size=GRID_SIZE, sigma_a=0.3, sigma_b=0.42):
    Lx, Ly = box_size
    grid1 = np.zeros((grid_size, grid_size), dtype=float)
    grid2 = np.zeros((grid_size, grid_size), dtype=float)

    xpix = coords[:, 0] / Lx * (grid_size - 1)
    ypix = coords[:, 1] / Ly * (grid_size - 1)
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='xy')

    for x, y in zip(xpix, ypix):
        dx = X - x
        dy = Y - y
        if random.random() < 0.20:
            grid2 += np.exp(-(dx**2 + dy**2) / (2 * sigma_b**2))
        else:
            grid1 += np.exp(-(dx**2 + dy**2) / (2 * sigma_a**2))
        
    merged_image = np.stack([grid1, grid2], axis=0)
    return grid1, grid2, merged_image


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

import numpy as np
import matplotlib.pyplot as plt


def plot_two_channel_image(img, titles=None, cmap="viridis"):
    """
    Plot a two-channel image.

    Parameters
    ----------
    img : np.ndarray
        Shape can be:
        - (H, W, 2)  -> channel last
        - (2, H, W)  -> channel first

    titles : list[str] or None
        Optional titles for channels

    cmap : str
        Matplotlib colormap
    """

    img = np.asarray(img)

    # Detect format
    if img.ndim != 3:
        raise ValueError("Input must be a 3D array")

    if img.shape[-1] == 2:  # (H, W, 2)
        ch1 = img[:, :, 0]
        ch2 = img[:, :, 1]

    elif img.shape[0] == 2:  # (2, H, W)
        ch1 = img[0]
        ch2 = img[1]

    else:
        raise ValueError(
            "Image must have shape (H, W, 2) or (2, H, W)"
        )

    if titles is None:
        titles = ["Channel 1", "Channel 2"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(ch1, cmap=cmap)
    axes[0].set_title(titles[0])
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ch2, cmap=cmap)
    axes[1].set_title(titles[1])
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def show_two_channel_overlay(
    img,
    normalize=True,
    interpolation="bilinear",
    gamma=0.7,
    figsize=(8, 8)
):
    """
    Show a 2-channel image as one RGB overlay.

    Parameters
    ----------
    img : np.ndarray
        Shape:
        - (2, H, W)
        - (H, W, 2)

    normalize : bool
        Whether to normalize intensities

    interpolation : str
        Matplotlib interpolation mode
        ('nearest', 'bilinear', 'bicubic')

    gamma : float
        Gamma correction (<1 brightens faint Gaussian tails)

    figsize : tuple
        Figure size
    """

    import numpy as np
    import matplotlib.pyplot as plt

    img = np.asarray(img)

    # Handle channel-first vs channel-last
    if img.ndim != 3:
        raise ValueError("Input must be 3D")

    if img.shape[0] == 2:  # (2, H, W)
        ch1, ch2 = img[0], img[1]

    elif img.shape[-1] == 2:  # (H, W, 2)
        ch1, ch2 = img[:, :, 0], img[:, :, 1]

    else:
        raise ValueError(
            "Expected shape (2,H,W) or (H,W,2)"
        )

    ch1 = ch1.astype(float).copy()
    ch2 = ch2.astype(float).copy()

    # Global normalization (important!)
    # if normalize:
    #     global_max = max(ch1.max(), ch2.max())

    #     if global_max > 0:
    #         ch1 /= global_max
    #         ch2 /= global_max

    # # Gamma correction to reveal Gaussian tails
    # if gamma is not None:
    #     ch1 = np.power(ch1, gamma)
    #     ch2 = np.power(ch2, gamma)

    # Convert from [-1,1] -> [0,1]
    ch1 = (ch1 + 1.0) / 2.0
    ch2 = (ch2 + 1.0) / 2.0

    # Safety clip
    ch1 = np.clip(ch1, 0.0, 1.0)
    ch2 = np.clip(ch2, 0.0, 1.0)

    if normalize:
        global_max = max(ch1.max(), ch2.max())
        if global_max > 0:
            ch1 /= global_max
            ch2 /= global_max

    if gamma is not None:
        ch1 = np.power(ch1, gamma)
        ch2 = np.power(ch2, gamma)

    # RGB image
    rgb = np.zeros((*ch1.shape, 3), dtype=float)

    rgb[:, :, 0] = ch1  # red
    rgb[:, :, 1] = ch2  # green

    plt.figure(figsize=figsize)

    plt.imshow(
        rgb,
        origin="lower",
        interpolation=interpolation
    )

    plt.title("Two-channel overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    coords, box_size = generateHexagonalGrid(nx=48, ny=48, a=1.0, defect_rate=0.10, pbc=False)
    # density_grid = coords_to_density_grid(
    #     coords,
    #     box_size,
    #     grid_size=128,
    #     sigma=0.4
    # )

    density_grid = generate_diatomic_density_grid(coords, box_size, GRID_SIZE, sigma_a=0.5, sigma_b=0.70)

    plotGrid(
        density_grid,
        cmap='viridis',
        interpolation='bilinear',
        title='Smooth Density Field'
    )
    d1, d2, img = generate_two_channel_density_grid(coords, box_size, GRID_SIZE, sigma_a=0.5, sigma_b=0.70)
    show_two_channel_overlay(img)