"""
1. CENTER-OF-MASS-FREE SUBSPACE.
   An exactly translation-equivariant denoiser implies the *data*
   distribution it's trained to match is only defined up to translation
   (shift every particle by the same vector -> equally valid sample).
   A standard isotropic Gaussian over absolute (x, y) is NOT invariant
   under translation (translating a Gaussian changes its mean), so naively
   running DDPM on absolute coordinates is mathematically inconsistent
   with an equivariant model -- the prior and the model disagree about
   what symmetry the distribution has.
   Fix (standard in equivariant generative modeling, e.g. Hoogeboom et al.
   EDM for molecules): always remove the center of mass, both from data
   during training and from the noise prior during sampling. This makes
   the working distribution genuinely translation-invariant (it's a
   distribution over *shapes*, not *placements*), consistent with what
   the network can represent.

2. DYNAMIC GRAPH RECONSTRUCTION
   Edges must be rebuilt from the current noisy
   x_t at every denoising step (more expensive, but necessary; this is
   also standard practice in equivariant point-cloud diffusion).
"""

import torch
from scipy.spatial import cKDTree
import numpy as np

def remove_com(x):
    #x is of shape (N, 2)
    return x - x.mean(dim=0, keepdim=True)

def build_edges_from_tensor(x, cutoff, max_neighbors=32):
    x_np = x.detach().cpu().numpy()
    N = x_np.shape[0]
    tree = cKDTree(x_np)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

    if len(pairs) < N:
        k = min(max_neighbors, N-1)
        dists, idxs = tree.query(x_np, k=k+1)
        src = np.repeat(np.arange(N), k)
        dst = idxs[:, 1:].reshape(-1)
    
    else:
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    
    delta = x_np[src] - x_np[dst]
    dist = np.linalg.norm(delta, axis=-1, keepdims=True).astype(np.float32)

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long().to(x.device)
    edge_attr = torch.from_numpy(dist).to(x.device)
    return edge_index, edge_attr