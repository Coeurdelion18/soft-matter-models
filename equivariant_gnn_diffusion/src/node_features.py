"""
Compute per-particle structural features for use as node_scalars in the
EGNN unconditional denoiser.

All nine features are rotation/translation invariant scalars:
    0  voronoi_area          Voronoi cell area
    1  coordination          number of Voronoi neighbours
    2  voronoi_perimeter     total Voronoi cell perimeter (sum of shared edge lengths)
    3  mean_neighbour_dist   mean distance to Voronoi neighbours
    4  std_neighbour_dist    standard deviation of distances to Voronoi neighbours
    5  psi6_magnitude        |psi_6|, hexatic bond-orientational order
    6  psi4_magnitude        |psi_4|, square bond-orientational order
    7  size                  particle diameter (passed through directly)
    8  type_float            particle type as 0.0 / 1.0

Dependencies: freud-analysis, numpy
    pip install freud-analysis

Usage:
    from data.node_features import build_node_scalars

    scalars = build_node_scalars(pos, types, sizes, box_size)
    # scalars: (N, 9) float32 array, one row per particle

Integration with load_boxes():

    def load_boxes():
        boxes = []
        for f in sorted(glob.glob("data/npy/*.npz")):
            d = np.load(f)
            pos      = d["pos"]
            types    = d["type"]
            sizes    = d["size"]
            box_size = d["box_size"]
            scalars  = build_node_scalars(pos, types, sizes, box_size)
            boxes.append({
                "pos":          pos,
                "type":         types,
                "size":         sizes,
                "box_size":     box_size,
                "node_scalars": scalars,
            })
        return boxes

N_SCALAR_FEATS = 9  -- set this in train_unconditional.py
"""

import numpy as np
import freud


# ── Feature column index constants (import these in train_unconditional.py) ──
FEAT_VORONOI_AREA     = 0
FEAT_COORDINATION     = 1
FEAT_VORONOI_PERIM    = 2
FEAT_MEAN_NEIGH_DIST  = 3
FEAT_STD_NEIGH_DIST   = 4
FEAT_PSI6             = 5
FEAT_PSI4             = 6
FEAT_SIZE             = 7
FEAT_TYPE             = 8
N_SCALAR_FEATS        = 9


def build_node_scalars(pos, types, sizes, box_size):
    """
    Compute all structural node features for one configuration.

    Parameters
    ----------
    pos:      (N, 2) float array   particle positions (in box coordinates)
    types:    (N,)   int array     particle types, 0-indexed (0=small, 1=large)
    sizes:    (N,)   float array   particle diameters
    box_size: (2,) or float        simulation box side lengths [Lx, Ly]

    Returns
    -------
    scalars: (N, 9) float32 array
        Columns in order: voronoi_area, coordination, voronoi_perimeter,
        mean_neighbour_dist, std_neighbour_dist, psi6, psi4, size, type_float
    """
    pos      = np.asarray(pos,   dtype=np.float32)
    types    = np.asarray(types, dtype=np.int32)
    sizes    = np.asarray(sizes, dtype=np.float32)
    box_size = np.atleast_1d(np.asarray(box_size, dtype=np.float32))
    Lx       = float(box_size[0])
    Ly       = float(box_size[1] if len(box_size) > 1 else box_size[0])
    N        = len(pos)

    # freud requires 3D positions with z=0 for 2D systems
    pos3d = np.column_stack([pos, np.zeros(N, dtype=np.float32)])

    box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)

    # ── Voronoi ───────────────────────────────────────────────────────────────
    voro = freud.locality.Voronoi()
    voro.compute((box, pos3d))

    # voro.volumes gives the 2D cell area (freud uses "volume" even in 2D)
    voronoi_area = voro.volumes.astype(np.float32)           # (N,)

    # nlist.query_point_indices tells us which particle each edge belongs to
    qpi       = voro.nlist.query_point_indices               # (E,) int
    weights   = voro.nlist.weights.astype(np.float64)        # (E,) shared edge length
    distances = voro.nlist.distances.astype(np.float64)      # (E,) centre-to-centre

    # aggregate per particle
    coordination = np.zeros(N, dtype=np.float32)
    np.add.at(coordination, qpi, 1)

    voronoi_perimeter = np.zeros(N, dtype=np.float64)
    np.add.at(voronoi_perimeter, qpi, weights)

    sum_d = np.zeros(N, dtype=np.float64)
    np.add.at(sum_d, qpi, distances)

    sum_d2 = np.zeros(N, dtype=np.float64)
    np.add.at(sum_d2, qpi, distances ** 2)

    safe_coord       = np.clip(coordination, 1, None)
    mean_neigh_dist  = (sum_d  / safe_coord).astype(np.float32)
    var_neigh_dist   = (sum_d2 / safe_coord) - (sum_d / safe_coord) ** 2
    std_neigh_dist   = np.sqrt(np.clip(var_neigh_dist, 0, None)).astype(np.float32)

    # ── Bond-orientational order ──────────────────────────────────────────────
    # psi6: hexatic order -- near 1 in a triangular lattice, near 0 at boundaries
    # psi4: square order  -- useful for detecting square-lattice defect arrangements
    # We use the Voronoi neighbour list for both, consistent with the Voronoi features
    hex6 = freud.order.Hexatic(k=6)
    hex6.compute((box, pos3d), neighbors=voro.nlist)
    psi6 = np.abs(hex6.particle_order).astype(np.float32)   # (N,) in [0,1]

    hex4 = freud.order.Hexatic(k=4)
    hex4.compute((box, pos3d), neighbors=voro.nlist)
    psi4 = np.abs(hex4.particle_order).astype(np.float32)   # (N,) in [0,1]

    # ── Assemble feature matrix ───────────────────────────────────────────────
    scalars = np.stack([
        voronoi_area,
        coordination,
        voronoi_perimeter.astype(np.float32),
        mean_neigh_dist,
        std_neigh_dist,
        psi6,
        psi4,
        sizes,
        types.astype(np.float32),
    ], axis=-1)                                              # (N, 9)

    return scalars.astype(np.float32)


def normalise_scalars(scalars_list, skip_cols=(7, 8)):
    """
    Compute mean and std across all configurations and return normalised
    versions. Call this on the training set and apply the same stats to
    validation and test sets.

    Parameters
    ----------
    scalars_list: list of (N_i, 9) arrays, one per configuration
    skip_cols:    column indices to leave unnormalised (mean=0, std=1
                  passthrough). Default skips size (7) and type (8) --
                  z-scoring type is dangerous when defects are rare
                  (e.g. 0.5% prevalence inflates the minority class's
                  z-score to extreme magnitudes, destabilising sampling).

    Returns
    -------
    normalised_list: list of (N_i, 9) arrays, z-scored per feature column
                     (except skip_cols, passed through unchanged)
    mean:           (9,) float32
    std:            (9,) float32
    """
    all_scalars = np.concatenate(scalars_list, axis=0)     # (total_N, 9)
    mean = all_scalars.mean(axis=0)
    std  = all_scalars.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)                  # avoid div-by-zero
                                                            # for constant features
    for c in skip_cols:
        mean[c] = 0.0
        std[c]  = 1.0

    normalised_list = [(s - mean) / std for s in scalars_list]
    return normalised_list, mean.astype(np.float32), std.astype(np.float32)


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    # synthetic bidisperse triangular lattice
    N_small, N_large = 200, 4
    spacing = 1.0
    rows    = int(np.ceil(np.sqrt(N_small)))
    pts     = []
    for i in range(rows):
        for j in range(rows):
            if len(pts) >= N_small:
                break
            pts.append([
                j * spacing + (0.5 * spacing if i % 2 else 0.0),
                i * spacing * 0.866,
            ])
    box_side = rows * spacing
    pos_s = np.array(pts[:N_small], dtype=np.float32) + np.random.normal(0, 0.03, (N_small, 2))
    pos_l = np.random.uniform(0, box_side, (N_large, 2)).astype(np.float32)
    pos   = np.vstack([pos_s, pos_l])
    types = np.array([0] * N_small + [1] * N_large, dtype=np.int32)
    sizes = np.array([1.0] * N_small + [1.8] * N_large, dtype=np.float32)

    scalars = build_node_scalars(pos, types, sizes, box_size=[box_side, box_side])

    print(f"scalars shape: {scalars.shape}")
    print(f"\nFeature stats across all {len(pos)} particles:")
    names = [
        "voronoi_area", "coordination", "voronoi_perimeter",
        "mean_neigh_dist", "std_neigh_dist", "psi6", "psi4",
        "size", "type",
    ]
    for i, name in enumerate(names):
        col = scalars[:, i]
        print(f"  {name:22s}  mean={col.mean():.4f}  std={col.std():.4f}"
              f"  min={col.min():.4f}  max={col.max():.4f}")

    print("\nSmall vs large particle psi6 (should differ):")
    print(f"  small: {scalars[:N_small, FEAT_PSI6].mean():.4f}")
    print(f"  large: {scalars[N_small:,  FEAT_PSI6].mean():.4f}")

    # test normalisation
    norm, mean, std = normalise_scalars([scalars, scalars])
    print(f"\nnormalise_scalars: mean~0 check: {norm[0].mean(0).round(3)}")
    print(f"normalise_scalars: std~1  check: {norm[0].std(0).round(3)}")