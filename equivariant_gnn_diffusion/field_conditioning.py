"""
Coarse psi6-field conditioning for the diffusion model.

The idea: condition generation on a SMOOTH SPATIAL MAP of |psi6| -- the same
kind of picture as the ground-truth psi6 plots -- rather than on exact
per-particle descriptors. Unlike per-particle structural features (which
leak the answer during training and don't exist at sampling time), a coarse
field is something the user can SUPPLY at generation time: extracted from a
real configuration, or synthesised/drawn. The model then generates particles
whose local order realises that map, which is precisely "replicate this psi6
plot with new particles".

The field is queried at each particle's CURRENT position at every denoising
step, so the conditioning stays spatially attached as particles move.

Conventions:
    - grids are (G, G) arrays indexed [ix, iy] (x = first axis)
    - extent = (xmin, xmax, ymin, ymax) in whatever coordinate units the
      field will be queried with (the training/sampling code uses
      NORMALISED coordinates, i.e. physical / coord_scale)
    - the null (unconditional) token is a field value of -1.0, distinct
      from real |psi6| values which lie in [0, 1]
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, Delaunay

# field channel encoding: the model receives TWO field features per particle,
#   [ (F - FIELD_CENTER) / FIELD_SCALE ,  has_field ]
# Contrast normalisation matters: raw |psi6| maps sit at 0.95 +/- 0.05 while the
# neighbouring input channels span +/-1, so an unnormalised field is nearly
# constant and the network learns to ignore it (observed: CFG guidance had no
# effect). After normalisation grains ~ 0 and boundaries ~ -4: salient.
# The has_field flag (1 = real map, 0 = null/unconditional) replaces the old
# magic -1 value, which would collide with the rescaled range.
FIELD_CENTER = 0.95
FIELD_SCALE = 0.05
N_FIELD_FEATS = 2

HULL_MARGIN = 2.0   # particles this close to the patch hull are excluded from
                    # map construction: their Delaunay psi6 is artificially low
                    # (neighbours missing outside the patch), which otherwise
                    # paints a fake "boundary ring" around every training map --
                    # both the map generator and the particle model then learn
                    # the ring artifact instead of real interior boundaries


def per_particle_psi6(pos):
    """|psi6| per particle from a Delaunay triangulation. pos: (N, 2)."""
    pos = np.asarray(pos, dtype=np.float64)
    tri = Delaunay(pos)
    indptr, indices = tri.vertex_neighbor_vertices
    psi = np.zeros(len(pos), dtype=complex)
    for i in range(len(pos)):
        nbrs = indices[indptr[i]:indptr[i + 1]]
        if len(nbrs) == 0:
            continue
        d = pos[nbrs] - pos[i]
        ang = np.arctan2(d[:, 1], d[:, 0])
        psi[i] = np.exp(1j * 6 * ang).mean()
    return np.abs(psi).astype(np.float32)


def rasterize_field(pos, values, grid_n=64, smooth_px=2.0, extent=None):
    """
    Average per-particle values onto a square grid and smooth.

    Returns (grid (G, G) float32 indexed [ix, iy], extent tuple) in the same
    units as pos. Empty cells (outside the particle cloud) get the mean value.
    If `extent` is given, the grid is binned over that frame instead of the
    bounding square of `pos`.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if extent is None:
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        center = (lo + hi) / 2
        half = float((hi - lo).max()) / 2 + 1e-6
        extent = (center[0] - half, center[0] + half,
                  center[1] - half, center[1] + half)

    rng = [[extent[0], extent[1]], [extent[2], extent[3]]]
    wsum, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=grid_n,
                                range=rng, weights=values)
    count, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=grid_n, range=rng)

    wsum = gaussian_filter(wsum, smooth_px)
    count = gaussian_filter(count, smooth_px)
    field = np.where(count > 1e-3, wsum / np.maximum(count, 1e-6),
                     float(np.mean(values)))
    return field.astype(np.float32), extent


def interior_hull_mask(pos, margin=HULL_MARGIN):
    """True for particles at least `margin` inside the convex hull."""
    hull = ConvexHull(pos)
    dist = -(pos @ hull.equations[:, :2].T + hull.equations[:, 2])
    return dist.min(axis=1) >= margin


def make_psi6_map(pos, grid_n=64, smooth_px=2.0):
    """
    Per-particle |psi6| -> smoothed grid + extent, computed from INTERIOR
    particles only (see HULL_MARGIN). The extent still covers all particles,
    so lookups at rim positions read border-clamped interior values.
    """
    pos_all = np.asarray(pos, dtype=np.float64)
    values = per_particle_psi6(pos_all)
    interior = interior_hull_mask(pos_all)

    # frame over ALL particles, so lookups anywhere in the patch are valid
    lo, hi = pos_all.min(axis=0), pos_all.max(axis=0)
    center = (lo + hi) / 2
    half = float((hi - lo).max()) / 2 + 1e-6
    extent = (center[0] - half, center[0] + half,
              center[1] - half, center[1] + half)

    grid, _ = rasterize_field(pos_all[interior], values[interior],
                              grid_n=grid_n, smooth_px=smooth_px,
                              extent=extent)
    return grid, extent


class FieldSampler:
    """
    Bilinear lookup of a scalar grid at arbitrary (torch) positions.
    Positions outside the extent are clamped to the border.
    """

    def __init__(self, grid, extent, device):
        self.g = torch.as_tensor(grid, dtype=torch.float32, device=device)
        self.extent = extent
        self.G = self.g.shape[0]

    def __call__(self, x):
        """x: (N, 2) tensor -> (N, 2) [normalised field value, has_field=1]."""
        xmin, xmax, ymin, ymax = self.extent
        G = self.G
        fx = ((x[:, 0] - xmin) / (xmax - xmin) * (G - 1)).clamp(0, G - 1)
        fy = ((x[:, 1] - ymin) / (ymax - ymin) * (G - 1)).clamp(0, G - 1)
        x0 = fx.floor().long().clamp(0, G - 2)
        y0 = fy.floor().long().clamp(0, G - 2)
        tx = (fx - x0.float()).unsqueeze(-1)
        ty = (fy - y0.float()).unsqueeze(-1)
        g = self.g
        v00 = g[x0, y0].unsqueeze(-1)
        v10 = g[x0 + 1, y0].unsqueeze(-1)
        v01 = g[x0, y0 + 1].unsqueeze(-1)
        v11 = g[x0 + 1, y0 + 1].unsqueeze(-1)
        val = ((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 +
               (1 - tx) * ty * v01 + tx * ty * v11)
        val = (val - FIELD_CENTER) / FIELD_SCALE
        return torch.cat([val, torch.ones_like(val)], dim=-1)


class NullField:
    """Unconditional token: [0, has_field=0] everywhere."""

    def __call__(self, x):
        return torch.zeros(x.shape[0], N_FIELD_FEATS,
                           device=x.device, dtype=x.dtype)
