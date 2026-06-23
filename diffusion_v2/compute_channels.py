import numpy as np
import freud


def render_density_channel(
    atoms,
    xlo,
    ylo,
    box_size_x,
    box_size_y,
    grid_size=384,
):
    """
    Channel 0: single Gaussian density field.

    sigma is derived per-particle from its own diameter:
        sigma_i = ALPHA * diameter_i
    This is correct regardless of which integer type label corresponds
    to the larger particle (validated via tune_sigma.py: ALPHA = 0.20).

    Amplitude is scaled by diameter / max_diameter so that large
    particles produce higher peaks than small ones. Together, sigma
    and amplitude encode both position and size in one channel.
    """
    ALPHA = 0.20  # validated via tune_sigma.py sweep — see tuning notes

    image = np.zeros((1, grid_size, grid_size), dtype=np.float32)

    x_grid = np.linspace(xlo, xlo + box_size_x, grid_size)
    y_grid = np.linspace(ylo, ylo + box_size_y, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    max_diameter = atoms[:, 1].max()

    for particle_type, diameter, x, y in atoms:
        sigma     = ALPHA * diameter
        amplitude = diameter / max_diameter  # in (0, 1]

        gaussian = amplitude * np.exp(
            -((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2)
        )
        image[0] += gaussian

    return image  # shape (1, H, W)


def compute_freud_channels(
    atoms,
    box_size_x,
    box_size_y,
    grid_size=384,
):
    """
    Channels 2, 3, 4: |psi6|, arg(psi6), Voronoi side count.
    All three are computed per-particle and then splatted onto a grid
    using nearest-neighbour assignment (no Gaussian blur needed —
    these are slowly-varying fields).

    Returns array of shape (3, grid_size, grid_size).
    """
    coords_2d = atoms[:, 2:]  # (N, 2)
    coords_3d = np.column_stack(
        (coords_2d, np.zeros(len(atoms)))
    ).astype(np.float64)

    box = freud.box.Box(Lx=box_size_x, Ly=box_size_y, is2D=True)

    # ------------------------------------------------------------------
    # Voronoi — used for both the neighbor list and side-count channel
    # ------------------------------------------------------------------
    voronoi = freud.locality.Voronoi()
    voronoi.compute(system=(box, coords_3d))
    nlist = voronoi.nlist

    # Number of Voronoi neighbors per particle (6 = perfect hexagon)
    n_neighbors = np.array(
        [nlist.neighbor_counts[i] for i in range(len(atoms))],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Hexatic order  psi6
    # ------------------------------------------------------------------
    hexatic = freud.order.Hexatic(k=6)
    hexatic.compute(system=(box, coords_3d), neighbors=nlist)
    psi6 = hexatic.particle_order  # complex, shape (N,)

    psi6_magnitude = np.abs(psi6).astype(np.float32)
    psi6_phase     = np.angle(psi6).astype(np.float32)  # in [-pi, pi]

    # ------------------------------------------------------------------
    # Splat per-particle scalars onto pixel grid (nearest neighbour)
    # ------------------------------------------------------------------
    # Particle positions are in real-space; map to pixel indices.
    # atoms[:, 2] = x,  atoms[:, 3] = y
    # Box origin may not be 0 — use relative coords.
    x_min = coords_2d[:, 0].min()
    y_min = coords_2d[:, 1].min()

    ix = np.clip(
        ((coords_2d[:, 0] - x_min) / box_size_x * grid_size).astype(int),
        0, grid_size - 1,
    )
    iy = np.clip(
        ((coords_2d[:, 1] - y_min) / box_size_y * grid_size).astype(int),
        0, grid_size - 1,
    )

    ch_psi6_mag   = np.zeros((grid_size, grid_size), dtype=np.float32)
    ch_psi6_phase = np.zeros((grid_size, grid_size), dtype=np.float32)
    ch_voronoi    = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Accumulate (multiple particles may land in the same pixel)
    count = np.zeros((grid_size, grid_size), dtype=np.float32)

    np.add.at(ch_psi6_mag,   (iy, ix), psi6_magnitude)
    np.add.at(ch_psi6_phase, (iy, ix), psi6_phase)
    np.add.at(ch_voronoi,    (iy, ix), n_neighbors)
    np.add.at(count,         (iy, ix), 1.0)

    # Average where multiple particles share a pixel
    mask = count > 0
    ch_psi6_mag[mask]   /= count[mask]
    ch_psi6_phase[mask] /= count[mask]
    ch_voronoi[mask]    /= count[mask]

    # Fill empty pixels by nearest occupied pixel (scipy fallback)
    try:
        from scipy.ndimage import distance_transform_edt
        empty = ~mask
        if empty.any():
            _, idx = distance_transform_edt(empty, return_indices=True)
            ch_psi6_mag[empty]   = ch_psi6_mag[  idx[0][empty], idx[1][empty]]
            ch_psi6_phase[empty] = ch_psi6_phase[idx[0][empty], idx[1][empty]]
            ch_voronoi[empty]    = ch_voronoi[   idx[0][empty], idx[1][empty]]
    except ImportError:
        pass  # leave empty pixels as zero if scipy unavailable

    freud_channels = np.stack(
        [ch_psi6_mag, ch_psi6_phase, ch_voronoi], axis=0
    )  # (3, H, W)

    return freud_channels


def compute_all_channels(
    atoms,
    xlo,
    ylo,
    box_size_x,
    box_size_y,
    grid_size=384,
    normalize=True,
):
    """
    Returns a (4, H, W) float32 array:

        ch 0  — density (amplitude = diameter/d_max); encodes position + size
        ch 1  — |psi6|        in [0, 1]
        ch 2  — arg(psi6)     in [-pi, pi]
        ch 3  — Voronoi sides (typically 4–8; 6 = perfect hexagon)
    """
    density  = render_density_channel(
        atoms, xlo, ylo, box_size_x, box_size_y, grid_size
    )
    freud_ch = compute_freud_channels(
        atoms, box_size_x, box_size_y, grid_size
    )

    image = np.concatenate([density, freud_ch], axis=0)  # (4, H, W)

    if normalize:
        # Normalize each channel independently to [0, 1]
        for c in range(image.shape[0]):
            lo, hi = image[c].min(), image[c].max()
            image[c] = (image[c] - lo) / (hi - lo + 1e-8)

    return image