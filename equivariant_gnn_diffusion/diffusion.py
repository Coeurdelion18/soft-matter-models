"""
Diffusion over absolute particle positions for unconditional generation.

Two required changes vs. displacement diffusion (model/diffusion.py):

1. CENTER-OF-MASS-FREE SUBSPACE.
   An exactly translation-equivariant denoiser implies the data
   distribution it's trained to match is only defined up to translation
   (shift every particle by the same vector -> equally valid sample).
   A standard isotropic Gaussian over absolute (x, y) is NOT invariant
   under translation, so naively running DDPM on absolute coordinates is
   mathematically inconsistent with an equivariant model.
   Fix: always remove CoM from data during training and from the noise
   prior during sampling, and re-center after every reverse step.

2. DYNAMIC GRAPH RECONSTRUCTION.
   No fixed t0 graph exists -- edges are rebuilt from the current noisy
   x_t at every denoising step.
"""
import torch
from scipy.spatial import cKDTree
import numpy as np


def remove_com(x):
    """x: (N, 2) -> (N, 2) with center of mass subtracted."""
    return x - x.mean(dim=0, keepdim=True)


def build_edges_from_tensor(x, cutoff, max_neighbors=12):
    """
    Radius graph from current (possibly noisy) positions.
    Falls back to kNN when the radius graph is too sparse -- this happens
    early in the reverse process when x_t is still mostly noise.

    cutoff:        set to first minimum of g(r) for your system
    max_neighbors: kNN fallback k; set >= max physical coordination number
                   (12 is safe for bidisperse 2D systems)
    """
    x_np = x.detach().cpu().numpy()
    N    = x_np.shape[0]
    tree = cKDTree(x_np)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

    # if len(pairs) < N:
    #     # fallback: kNN to keep graph connected during noisy early steps
    k = min(max_neighbors, N - 1)
    _, idxs = tree.query(x_np, k=k + 1)
    src = np.repeat(np.arange(N), k)
    dst = idxs[:, 1:].reshape(-1)
    # else:
    #     src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    #     dst = np.concatenate([pairs[:, 1], pairs[:, 0]])

    delta = x_np[src] - x_np[dst]
    dist  = np.linalg.norm(delta, axis=-1, keepdims=True).astype(np.float32)

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long().to(x.device)
    edge_attr  = torch.from_numpy(dist).to(x.device)
    return edge_index, edge_attr


class PositionDiffusion:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.n_steps = n_steps
        self.device  = device
        betas        = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas       = 1.0 - betas
        alpha_bars   = torch.cumprod(alphas, dim=0)
        self.betas, self.alphas, self.alpha_bars = betas, alphas, alpha_bars

    def q_sample(self, x0, t, noise=None):
        """
        x0 must already be CoM-free.
        Noise is also drawn CoM-free so the noised sample stays CoM-free.
        """
        if noise is None:
            noise = remove_com(torch.randn_like(x0))
        ab  = self.alpha_bars[t].unsqueeze(-1)  # (N, 1)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return x_t, noise

    def training_loss(self, model, x0, node_scalars, cutoff, t=None,
                      global_cond=None, clamp_pred=False):
        """
        x0:           (N, 2)               real particle configuration
        node_scalars: (N, n_scalar_feats)  per-particle features (size, type, ...)
        cutoff:       radius for graph construction at the noised positions
        clamp_pred:   if True, clamp pred_noise to [-10, 10] before computing
                      loss. WARNING: tested and found to be harmful during
                      training -- once predictions saturate the clamp,
                      gradients through the clamped region are zero, so the
                      model gets permanently stuck predicting near the clamp
                      boundary (observed: loss flatlines at ~100 instead of
                      decreasing). Left here disabled by default; use a
                      smaller learning rate or a robust loss (e.g. Huber)
                      instead if occasional large-loss steps are a problem.
        """
        x0 = remove_com(x0)
        N  = x0.shape[0]
        if t is None:
            t_scalar = torch.randint(0, self.n_steps, (1,), device=x0.device).item()
            t = torch.full((N,), t_scalar, device=x0.device, dtype=torch.long)

        x_t, noise   = self.q_sample(x0, t)
        edge_index, edge_attr = build_edges_from_tensor(x_t, cutoff)
        noise_level  = t.float().unsqueeze(-1) / self.n_steps

        pred_noise = model(x_t, edge_index, edge_attr, noise_level,
                           node_scalars, global_cond)
        if clamp_pred:
            pred_noise = torch.clamp(pred_noise, min=-10.0, max=10.0)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, n_particles, node_scalars, cutoff,
               device=None, global_cond=None, verbose=False):
        """
        Generate one configuration of n_particles from pure noise.

        node_scalars: (n_particles, n_scalar_feats)
            Must be provided at sampling time -- particle identities (type,
            size) are fixed even as positions are denoised. The caller is
            responsible for setting the composition (how many of each type).
        """
        device       = device or self.device
        x_t          = remove_com(torch.randn(n_particles, 2, device=device))
        node_scalars = node_scalars.to(device)

        for step in reversed(range(self.n_steps)):
            t           = torch.full((n_particles,), step, device=device, dtype=torch.long)
            noise_level = t.float().unsqueeze(-1) / self.n_steps

            edge_index, edge_attr = build_edges_from_tensor(x_t, cutoff)
            pred_noise = model(x_t, edge_index, edge_attr, noise_level,
                               node_scalars, global_cond)
            pred_noise = torch.clamp(pred_noise, min=-10.0, max=10.0)

            alpha     = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta      = self.betas[step]
            coef1     = 1.0 / alpha.sqrt()
            coef2     = beta / (1 - alpha_bar).sqrt()
            mean      = coef1 * (x_t - coef2 * pred_noise)

            if step > 0:
                noise = remove_com(torch.randn_like(x_t))
                x_t   = mean + beta.sqrt() * noise
            else:
                x_t = mean
            x_t = remove_com(x_t)  # re-center after every update to prevent drift

            if verbose and step % 100 == 0:
                print(f"  step {step}: x_t max abs = {x_t.abs().max().item():.4f}  "
                     f"pred_noise max abs = {pred_noise.abs().max().item():.4f}")

            if not torch.isfinite(x_t).all():
                raise RuntimeError(f"x_t became non-finite at reverse step {step}")

        return x_t