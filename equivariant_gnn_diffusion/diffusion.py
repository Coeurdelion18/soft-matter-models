"""
DDPM over absolute particle positions, restricted to the CoM-free subspace.

Key conventions (all enforced here or in the training script):

1. UNIT COORDINATE SCALE. The diffusion prior is N(0, I), so training data
   must arrive with roughly unit variance. train_patched.py computes a single
   global COORD_SCALE from the training set; positions are divided by it
   before entering this module and samples are multiplied by it afterwards.

2. CoM-FREE SUBSPACE. A translation-equivariant denoiser only defines the
   data distribution up to translation, so data, noise, and every reverse
   step live in the subspace with zero centre of mass.

3. DYNAMIC GRAPH. Edges are rebuilt from the current noisy x_t with a kNN
   query at every training step and every reverse step.

4. COSINE SCHEDULE. Less noise near t=0 than the linear schedule, which
   matters here because the structure to resolve (lattice spacing) is only
   ~0.1 in normalised units.
"""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree


def remove_com(x):
    """x: (N, 2) -> (N, 2) with centre of mass subtracted."""
    return x - x.mean(dim=0, keepdim=True)


def build_knn_edges(x, k=12):
    """
    kNN graph from current (possibly noisy) positions.
    Returns edge_index (2, E) with messages flowing src -> dst.
    """
    x_np = x.detach().cpu().numpy()
    N = x_np.shape[0]
    k = min(k, N - 1)
    tree = cKDTree(x_np)
    _, idxs = tree.query(x_np, k=k + 1)
    dst = np.repeat(np.arange(N), k)
    src = idxs[:, 1:].reshape(-1)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long().to(x.device)
    return edge_index


def cosine_beta_schedule(n_steps, s=0.008):
    """Nichol & Dhariwal (2021) cosine alpha_bar schedule."""
    steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(1e-8, 0.999).float()


def logsigma_beta_schedule(n_steps, sigma_min=0.002, sigma_max=60.0):
    """
    Log-spaced noise-to-signal ladder (the Karras/EDM insight, expressed as
    a VP alpha_bar schedule): sigma_eff = sqrt(1-ab)/sqrt(ab) is geometric
    from sigma_min to sigma_max.

    WHY: schedules designed for images (linear, cosine) allocate steps by
    absolute noise power, which for these point patterns burns ~90% of the
    steps at noise levels coarser than 3 lattice spacings and reaches a
    minimum noise ~8x LARGER than the thermal jitter of the real patterns.
    Diagnostics showed the reverse process destroying a real crystal noised
    to t=25 -- the fine-structure regime was simply absent from training and
    sampling. Log-spacing gives every length scale (field -> grain ->
    lattice -> jitter) a proportional share of the steps.

    sigma_min: below the lattice-site thermal jitter in normalised units
               (jitter ~0.003 for 4000-particle patches at coord_scale ~17)
    sigma_max: large enough that x_T is indistinguishable from pure noise
    """
    sig = torch.logspace(math.log10(sigma_min), math.log10(sigma_max),
                         n_steps, dtype=torch.float64)
    alpha_bars = 1.0 / (1.0 + sig ** 2)          # decreasing in t
    ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
    betas = 1.0 - alpha_bars / ab_prev
    return betas.clamp(1e-8, 0.999).float()


class PositionDiffusion:
    def __init__(self, n_steps=1000, device="cpu", k_neighbors=12,
                 schedule="logsigma"):
        self.n_steps = n_steps
        self.device = device
        self.k_neighbors = k_neighbors
        self.schedule = schedule

        if schedule == "logsigma":
            betas = logsigma_beta_schedule(n_steps).to(device)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(n_steps).to(device)
        else:
            raise ValueError(f"unknown schedule: {schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas, self.alphas, self.alpha_bars = betas, alphas, alpha_bars

    def q_sample(self, x0, t, noise=None):
        """x0 must already be CoM-free and unit-scale."""
        if noise is None:
            noise = remove_com(torch.randn_like(x0))
        ab = self.alpha_bars[t].unsqueeze(-1)  # (N, 1)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return x_t, noise

    def training_loss(self, model, x0, node_scalars, t=None,
                      low_t_frac=0.0, low_t_cut=0.15):
        """
        x0:           (N, 2) clean positions, ALREADY divided by COORD_SCALE
        node_scalars: (N, F) identity features (size, type)

        low_t_frac: probability of drawing t from the lowest low_t_cut
            fraction of the schedule instead of uniformly. Crystalline order
            lives at the finest length scale, which only low-noise steps
            train; uniform t-sampling visits that regime so rarely that
            samples come out under-ordered (liquid-like). 0.0 = uniform.
        """
        x0 = remove_com(x0)
        N = x0.shape[0]
        if t is None:
            if low_t_frac > 0 and torch.rand(1).item() < low_t_frac:
                hi = max(1, int(self.n_steps * low_t_cut))
            else:
                hi = self.n_steps
            t_scalar = torch.randint(0, hi, (1,), device=x0.device).item()
            t = torch.full((N,), t_scalar, device=x0.device, dtype=torch.long)

        x_t, noise = self.q_sample(x0, t)
        edge_index = build_knn_edges(x_t, k=self.k_neighbors)
        noise_level = t.float().unsqueeze(-1) / self.n_steps

        pred_noise = model(x_t, edge_index, noise_level, node_scalars)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, n_particles, node_scalars, device=None,
               x0_clip=4.0, corrector_steps=0, corrector_snr=0.15,
               verbose=False):
        """
        Generate one configuration of n_particles from pure noise.
        Returns positions in NORMALISED units -- multiply by COORD_SCALE.

        x0_clip: static thresholding on the per-particle radius. At every
            step the implied clean sample x0_hat is rescaled so no particle
            lies further than x0_clip from the CoM (data lives within ~2-3
            of zero after normalisation, so 4.0 is generous). Norm-based
            rather than per-coordinate clamping so no square geometry is
            injected into the samples. This caps how far a bad prediction
            can throw the trajectory without distorting the reverse-process
            mean when predictions are sane. Set to None to disable.

        corrector_steps: Langevin corrector iterations after each reverse
            step (predictor-corrector sampling, Song et al. 2021). Ancestral
            DDPM alone suffers exposure bias here: by the time the noise is
            low the configuration is a disordered liquid, and the low-noise
            denoiser -- trained only on "crystal + small jitter" -- cannot
            crystallise it. Corrector steps let the system relax WITHIN each
            noise level before the noise is reduced (annealing), which is
            how the physical system orders too. Each corrector step costs
            one extra model evaluation per reverse step.
        corrector_snr: Langevin signal-to-noise step-size parameter.
        """
        device = device or self.device
        x_t = remove_com(torch.randn(n_particles, 2, device=device))
        node_scalars = node_scalars.to(device)

        for step in reversed(range(self.n_steps)):
            noise_level = torch.full((1, 1), step / self.n_steps, device=device)
            edge_index = build_knn_edges(x_t, k=self.k_neighbors)
            pred_noise = model(x_t, edge_index, noise_level, node_scalars)

            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            alpha_bar_prev = (
                self.alpha_bars[step - 1] if step > 0
                else torch.tensor(1.0, device=device)
            )
            beta = self.betas[step]

            # implied clean sample, optionally thresholded (radially, so the
            # clip is rotation-invariant and adds no square bias)
            x0_hat = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            if x0_clip is not None:
                radii = x0_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                x0_hat = x0_hat * (x0_clip / radii).clamp(max=1.0)

            # DDPM posterior q(x_{t-1} | x_t, x0_hat)
            coef_x0 = alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)
            coef_xt = alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)
            mean = coef_x0 * x0_hat + coef_xt * x_t

            if step > 0:
                sigma = (beta * (1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
                noise = remove_com(torch.randn_like(x_t))
                x_t = mean + sigma * noise
            else:
                x_t = mean
            x_t = remove_com(x_t)

            # Langevin corrector at the new noise level
            if corrector_steps > 0 and step > 0:
                lvl = step - 1
                sigma_lvl = (1 - self.alpha_bars[lvl]).sqrt()
                lvl_t = torch.full((1, 1), lvl / self.n_steps, device=device)
                for _ in range(corrector_steps):
                    edge_index = build_knn_edges(x_t, k=self.k_neighbors)
                    eps_hat = model(x_t, edge_index, lvl_t, node_scalars)
                    score = -eps_hat / sigma_lvl
                    noise = remove_com(torch.randn_like(x_t))
                    # step size from the SNR rule (Song et al., alg. 4/5)
                    g = score.norm()
                    n = noise.norm()
                    eps_step = 2 * (corrector_snr * n / g.clamp(min=1e-12)) ** 2
                    x_t = x_t + eps_step * score + (2 * eps_step).sqrt() * noise
                    x_t = remove_com(x_t)

            if verbose and step % 100 == 0:
                print(f"  step {step:4d}: |x_t|_max = {x_t.abs().max().item():.3f}  "
                      f"|eps_hat|_max = {pred_noise.abs().max().item():.3f}")

            if not torch.isfinite(x_t).all():
                raise RuntimeError(f"x_t became non-finite at reverse step {step}")

        return x_t
