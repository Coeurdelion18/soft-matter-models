"""
E(2)-equivariant denoiser for position diffusion.

Architecture follows the EGNN of Satorras et al. with the stability
modifications used in "Equivariant Diffusion for Molecule Generation in 3D"
(Hoogeboom et al., 2022):

  - coordinate messages are direction-normalised: rel / (dist + 1), so an
    update can never blow up when two noisy points happen to be far apart
  - the final linear layer of the coordinate MLP is initialised near zero,
    so each layer moves points only slightly at the start of training
  - sigmoid attention on messages lets a layer down-weight the irrelevant
    far neighbours that kNN inevitably introduces at high noise levels
  - the predicted noise is projected back into the CoM-free subspace,
    matching the target noise distribution exactly

IMPORTANT: this model assumes coordinates have been normalised to roughly
unit variance (see COORD_SCALE in train_patched.py). Feeding raw simulation
coordinates (std ~10) will not work -- the diffusion prior is N(0, I).
"""

import math

import torch
import torch.nn as nn


def mlp(in_dim, hidden_dim, out_dim, n_hidden=1, act=nn.SiLU):
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_hidden):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class EGNNLayer(nn.Module):
    """
    NOTE ON SYMMETRY: this layer is rotation-equivariant but NOT translation-
    equivariant -- it uses the distance from the origin |x_i| as an invariant
    feature and the radial direction x_i/|x_i| as an equivariant update
    direction. That is intentional and valid because the whole pipeline runs
    in the CoM-centred frame (data and every reverse-diffusion step have
    their centre of mass removed), where the origin is physically meaningful.
    Without radial information a purely local kNN model has no way to learn
    the global extent of a patch: every interior particle looks identical,
    so samples collapse into drifting clumps instead of a bounded disk.
    """

    N_RBF = 24
    RBF_MAX = 0.4   # normalised units; nearest-neighbour spacing is ~0.04
                    # for 4000-particle patches (~0.1 for 1000-particle ones)

    def __init__(self, hidden_dim, act=nn.SiLU):
        super().__init__()
        # RBF expansion of edge length: raw dist/dist^2 scalars are a weak
        # signal for discriminating the fine scales that matter here (the
        # lattice spacing is only ~0.1 in normalised units); a Gaussian
        # basis ladder makes those distinctions linearly separable
        centers = torch.linspace(0.0, self.RBF_MAX, self.N_RBF)
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = self.RBF_MAX / (self.N_RBF - 1)

        # message inputs: h_src, h_dst, dist, dist^2, RBF(dist)
        self.edge_mlp = mlp(2 * hidden_dim + 2 + self.N_RBF,
                            hidden_dim, hidden_dim, act=act)
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.coord_mlp = mlp(hidden_dim, hidden_dim, 1, act=act)
        # radial gate: lets the layer push particles toward/away from the CoM
        self.radial_mlp = mlp(hidden_dim + 1, hidden_dim, 1, act=act)
        # node update sees the radial norm as an extra invariant scalar
        self.node_mlp = mlp(2 * hidden_dim + 1, hidden_dim, hidden_dim, act=act)
        self.norm = nn.LayerNorm(hidden_dim)

        # near-zero init keeps early coordinate updates small; without this
        # the stacked layers amplify each other before training finds a
        # sensible scale (the loss-spike behaviour seen previously)
        for m in (self.coord_mlp, self.radial_mlp):
            nn.init.xavier_uniform_(m[-1].weight, gain=1e-3)
            nn.init.zeros_(m[-1].bias)

    def forward(self, h, x, edge_index):
        """
        h: (N, hidden_dim) invariant node features
        x: (N, 2)          coordinates in the CoM-centred frame
        edge_index: (2, E) src/dst node indices
        """
        src, dst = edge_index
        rel = x[src] - x[dst]                       # (E, 2)
        dist2 = (rel ** 2).sum(-1, keepdim=True)    # (E, 1)
        dist = dist2.clamp(min=1e-12).sqrt()

        rbf = torch.exp(-((dist - self.rbf_centers[None, :]) / self.rbf_width) ** 2)
        m_ij = self.edge_mlp(torch.cat([h[src], h[dst], dist, dist2, rbf], dim=-1))
        m_ij = m_ij * self.att_mlp(m_ij)

        # direction-normalised coordinate message
        x_msg = (rel / (dist + 1.0)) * self.coord_mlp(m_ij)   # (E, 2)

        N = h.shape[0]
        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones_like(dist))
        deg = deg.clamp(min=1.0)

        agg_m = torch.zeros_like(h)
        agg_m.index_add_(0, dst, m_ij)
        agg_m = agg_m / deg

        agg_x = torch.zeros_like(x)
        agg_x.index_add_(0, dst, x_msg)
        agg_x = agg_x / deg

        r = x.norm(dim=-1, keepdim=True)                       # (N, 1) invariant
        h_new = self.norm(h + self.node_mlp(torch.cat([h, agg_m, r], dim=-1)))

        radial = (x / (r + 1.0)) * self.radial_mlp(torch.cat([h_new, r], dim=-1))
        x_new = x + agg_x + radial
        return h_new, x_new


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard transformer/DDPM sinusoidal embedding. Input is the normalised
    noise level t/T in [0, 1]; internally rescaled to [0, 1000] so the
    frequency ladder (10000^{-k/half}) actually resolves different timesteps.
    """

    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t):
        # t: (N, 1) in [0, 1]
        args = t * 1000.0 * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class EGNNUnconditionalDenoiser(nn.Module):
    """
    Predicts the diffusion noise eps from noisy positions x_t.

    Conditioning is ONLY on per-particle identity (size, type) and the noise
    level. Structural descriptors (Voronoi, psi6, ...) must NOT be used as
    conditioning: they are computed from the clean positions, so during
    training they leak the answer, and at sampling time they don't exist --
    the model would face an input distribution it has never seen.
    Use those descriptors to *evaluate* generated samples instead.
    """

    def __init__(self, hidden_dim=128, n_layers=6, n_scalar_feats=2):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.scalar_proj = nn.Linear(n_scalar_feats, hidden_dim)
        self.input_mlp = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x_t, edge_index, noise_level, node_scalars):
        """
        x_t:          (N, 2) noisy positions, CoM-free, unit-scale
        edge_index:   (2, E) kNN graph built on x_t
        noise_level:  (N, 1) or (1, 1) value of t/T
        node_scalars: (N, n_scalar_feats) identity features (size, type)
        """
        N = x_t.shape[0]
        if noise_level.shape[0] != N:
            noise_level = noise_level.expand(N, 1)

        t_emb = self.time_embed(noise_level)
        s_emb = self.scalar_proj(node_scalars.float())
        h = self.input_mlp(torch.cat([t_emb, s_emb], dim=-1))

        x = x_t
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        eps = x - x_t
        # project into the CoM-free subspace: the target noise is CoM-free,
        # so the prediction must be too
        eps = eps - eps.mean(dim=0, keepdim=True)
        return eps
