"""
Radial distribution function g(r), partial g(r) by species pair, and
automatic cutoff estimation for bidisperse particle systems.

Usage (standalone):
    python data/gr_analysis.py

Or from another script:
    from data.gr_analysis import compute_gr, estimate_cutoff, plot_gr

The cutoff returned by estimate_cutoff() is the first minimum of g(r)
after the first peak -- the standard physical choice for a neighbour
cutoff in liquid/glass systems. Pass it directly as CUTOFF in
train_unconditional.py and build_edges_from_tensor().
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import argrelmin, argrelmax


# ── Core g(r) computation ─────────────────────────────────────────────────────

def compute_gr(positions, box_size, r_max=None, n_bins=200, types=None):
    """
    Compute the radial distribution function g(r), optionally broken down
    into partial g_ab(r) for each species pair.

    positions: (N, 2) float array
    box_size:  float or (2,) array  -- simulation box side length(s)
    r_max:     maximum r to compute; defaults to half the box diagonal
    n_bins:    number of histogram bins
    types:     (N,) int array of particle types (e.g. 0 and 1); if None,
               only the total g(r) is returned

    Returns dict with keys:
        r          (n_bins,)   bin centres
        gr         (n_bins,)   total g(r)
        partials   dict mapping (a, b) -> (n_bins,) partial g_ab(r)
                   only present if types is not None
    """
    box = np.atleast_1d(np.array(box_size, dtype=float))
    if box.ndim == 0 or len(box) == 1:
        box = np.array([box.item(), box.item()])
    Lx, Ly = box

    N = len(positions)
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly)

    bins   = np.linspace(0, r_max, n_bins + 1)
    r_mid  = 0.5 * (bins[:-1] + bins[1:])
    dr     = bins[1] - bins[0]
    volume = Lx * Ly
    rho    = N / volume

    # -- total g(r) -----------------------------------------------------------
    # cKDTree with boxsize requires all coordinates in [0, L); wrap first
    pos_wrapped = positions.copy()
    pos_wrapped[:, 0] = pos_wrapped[:, 0] % Lx
    pos_wrapped[:, 1] = pos_wrapped[:, 1] % Ly
    tree  = cKDTree(pos_wrapped, boxsize=[Lx, Ly])
    total = np.zeros(n_bins)
    for i, pos in enumerate(pos_wrapped):
        idxs = tree.query_ball_point(pos, r=r_max)
        idxs = [j for j in idxs if j != i]
        if not idxs:
            continue
        diffs = positions[idxs] - pos
        # minimum image convention
        diffs -= Lx * np.round(diffs[:, 0:1] / Lx) * np.array([1, 0])
        diffs -= Ly * np.round(diffs[:, 1:2] / Ly) * np.array([0, 1])
        dists = np.linalg.norm(diffs, axis=1)
        counts, _ = np.histogram(dists, bins=bins)
        total += counts

    # normalise: divide by ideal-gas expectation for each shell
    shell_area = 2 * np.pi * r_mid * dr
    gr = total / (N * rho * shell_area)

    result = {"r": r_mid, "gr": gr}

    # -- partial g_ab(r) -------------------------------------------------------
    if types is not None:
        species   = np.unique(types)
        partials  = {}
        for a in species:
            for b in species:
                if b < a:
                    continue  # compute a<=b only, copy for b<a at the end
                idx_a = np.where(types == a)[0]
                idx_b = np.where(types == b)[0]
                N_a, N_b = len(idx_a), len(idx_b)
                rho_b = N_b / volume

                partial = np.zeros(n_bins)
                for i in idx_a:
                    pos_i = pos_wrapped[i]
                    if a == b:
                        candidates = [j for j in idx_b if j != i]
                    else:
                        candidates = list(idx_b)
                    if not candidates:
                        continue
                    diffs = pos_wrapped[candidates] - pos_i
                    diffs -= Lx * np.round(diffs[:, 0:1] / Lx) * np.array([1, 0])
                    diffs -= Ly * np.round(diffs[:, 1:2] / Ly) * np.array([0, 1])
                    dists = np.linalg.norm(diffs, axis=1)
                    counts, _ = np.histogram(dists, bins=bins)
                    partial += counts

                partial /= (N_a * rho_b * shell_area)
                partials[(a, b)] = partial
                if a != b:
                    partials[(b, a)] = partial   # symmetric

        result["partials"] = partials

    return result


# ── Cutoff estimation ─────────────────────────────────────────────────────────

def estimate_cutoff(gr_result, min_peak_height=0.5):
    """
    Find the first minimum of g(r) after the first peak -- the standard
    cutoff choice for nearest-neighbour graphs in glass/liquid systems.

    gr_result: dict returned by compute_gr()

    Returns:
        cutoff      float   the recommended cutoff radius
        first_peak  float   position of the first peak (for reference)
        diagnostics dict    peak/min positions for debugging

    If multiple configurations were averaged (see average_gr), pass the
    averaged result -- the peak/minimum positions will be more reliable.
    """
    r  = gr_result["r"]
    gr = gr_result["gr"]

    peaks = argrelmax(gr, order=3)[0]
    mins  = argrelmin(gr, order=3)[0]

    # filter out noise peaks below threshold
    peaks = [p for p in peaks if gr[p] > min_peak_height]
    if not peaks:
        raise ValueError(
            "No peak found in g(r). Check that r_max and n_bins are "
            "appropriate for your system, and that box_size is correct."
        )

    first_peak_idx = peaks[0]
    first_peak_r   = r[first_peak_idx]

    # first minimum that comes AFTER the first peak
    mins_after = [m for m in mins if m > first_peak_idx]
    if not mins_after:
        raise ValueError(
            "No minimum found after first peak in g(r). "
            "Try increasing r_max or n_bins."
        )

    cutoff_idx = mins_after[0]
    cutoff     = float(r[cutoff_idx])

    return cutoff, first_peak_r, {
        "all_peaks": r[peaks].tolist(),
        "all_mins":  r[[m for m in mins if m > first_peak_idx]].tolist(),
    }


def average_gr(positions_list, box_sizes, r_max=None, n_bins=200, types_list=None):
    """
    Compute g(r) averaged over multiple configurations -- gives a cleaner
    curve for cutoff estimation than a single configuration.

    positions_list: list of (N, 2) arrays
    box_sizes:      list of floats or (2,) arrays, one per configuration
    types_list:     list of (N,) type arrays, or None

    Returns averaged gr_result dict (same structure as compute_gr()).
    """
    has_types = types_list is not None
    gr_sum    = None
    partial_sums = {}

    for i, (pos, box) in enumerate(zip(positions_list, box_sizes)):
        types = types_list[i] if has_types else None
        result = compute_gr(pos, box, r_max=r_max, n_bins=n_bins, types=types)

        if gr_sum is None:
            gr_sum = result["gr"].copy()
            r      = result["r"]
        else:
            gr_sum += result["gr"]

        if has_types and "partials" in result:
            for key, val in result["partials"].items():
                partial_sums[key] = partial_sums.get(key, 0) + val

    n = len(positions_list)
    out = {"r": r, "gr": gr_sum / n}
    if partial_sums:
        out["partials"] = {k: v / n for k, v in partial_sums.items()}
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_gr(gr_result, cutoff=None, save_path=None, title=None,
            type_labels=None):
    """
    Plot total g(r) and, if available, partial g_ab(r) on the same figure.

    gr_result:   dict returned by compute_gr() or average_gr()
    cutoff:      if provided, draws a vertical line at the cutoff radius
    save_path:   if provided, saves the figure to this path
    title:       optional figure title
    type_labels: dict mapping type int to display name,
                 e.g. {0: "small (green)", 1: "large defect (red)"}
    """
    has_partials = "partials" in gr_result
    n_panels     = 2 if has_partials else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4),
                             squeeze=False)
    r  = gr_result["r"]
    gr = gr_result["gr"]

    # ── panel 1: total g(r) ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(r, gr, color="#2a6ebe", lw=1.8, label="total g(r)")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="ideal gas (g=1)")

    if cutoff is not None:
        ax.axvline(cutoff, color="#e05c2a", lw=1.5, ls="--",
                   label=f"cutoff = {cutoff:.3f}")

    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Total radial distribution function")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # ── panel 2: partials ────────────────────────────────────────────────────
    if has_partials:
        ax2 = axes[0, 1]
        colors = ["#3B8BD4", "#EF9F27", "#5cb85c", "#d9534f"]
        labels_ = type_labels or {}
        done    = set()
        ci      = 0
        for (a, b), partial in gr_result["partials"].items():
            key = (min(a, b), max(a, b))
            if key in done:
                continue
            done.add(key)
            la = labels_.get(a, f"type {a}")
            lb = labels_.get(b, f"type {b}")
            label = f"g_{la}-{lb}(r)"
            ax2.plot(r, partial, color=colors[ci % len(colors)],
                     lw=1.5, label=label)
            ci += 1

        if cutoff is not None:
            ax2.axvline(cutoff, color="#e05c2a", lw=1.5, ls="--",
                        label=f"cutoff = {cutoff:.3f}")

        ax2.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax2.set_xlabel("r")
        ax2.set_ylabel("g(r)")
        ax2.set_title("Partial radial distribution functions")
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")

    return fig


# ── Convenience: run everything from a list of position dicts ─────────────────

def analyse_cutoff(configs, n_configs_to_avg=10, r_max=None, n_bins=300,
                   type_labels=None, plot=True, save_path=None):
    """
    High-level entry point: given a list of configuration dicts (each with
    keys "pos", "types", "box_size"), compute and plot g(r) averaged over
    up to n_configs_to_avg configurations, estimate the cutoff, and return it.

    configs: list of dicts, each:
        {
            "pos":      (N, 2) float array,
            "types":    (N,)   int array (0 / 1),
            "box_size": float or (2,) array,
        }

    Returns: cutoff (float)
    """
    subset       = configs[:n_configs_to_avg]
    pos_list     = [c["pos"]   for c in subset]
    box_list     = [c["box_size"] for c in subset]
    types_list   = [c["types"] for c in subset]

    print(f"Computing g(r) averaged over {len(subset)} configurations...")
    gr_result = average_gr(pos_list, box_list,
                           r_max=r_max, n_bins=n_bins,
                           types_list=types_list)

    cutoff, first_peak, diag = estimate_cutoff(gr_result)
    print(f"  first peak at r = {first_peak:.4f}")
    print(f"  first minimum (recommended cutoff) at r = {cutoff:.4f}")
    print(f"  all minima after first peak: {[f'{x:.4f}' for x in diag['all_mins']]}")

    if plot:
        plot_gr(gr_result, cutoff=cutoff,
                title=f"g(r) averaged over {len(subset)} configs  "
                      f"|  cutoff = {cutoff:.4f}",
                type_labels=type_labels,
                save_path=save_path)
        plt.show()

    return cutoff


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Demo on a synthetic bidisperse system.
    Replace with your real data by calling analyse_cutoff(configs) where
    configs is a list of dicts loaded from your LAMMPS files.
    """
    np.random.seed(0)



    configs = []

    cutoff = analyse_cutoff(
        configs,
        n_configs_to_avg=5,
        r_max=4.0,
        n_bins=300,
        type_labels={1: "small", 0: "large"},
        plot=True,
        save_path="outputs/results/gr_analysis.png",
    )

    print(f"\nUse CUTOFF = {cutoff:.4f} in train_unconditional.py")