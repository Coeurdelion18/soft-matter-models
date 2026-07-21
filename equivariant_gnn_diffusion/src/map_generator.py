"""
Stage-1 generator: diffusion over coarse psi6 maps (64x64 images).

Two-stage generation of grain boundary patterns:
    stage 1 (this file):  sample a NEW plausible psi6 map -- the mesoscale
                          grain/boundary LAYOUT -- as a small image
    stage 2 (sampling.py): the field-conditioned EGNN turns any map into a
                          physically correct particle configuration

Why two stages: the particle-level GNN is local (kNN messages) and
structurally cannot invent global grain layouts -- its unconditional samples
have too many small grains. A 64x64 image model sees the whole layout at
once, so mesoscale statistics (grain size, boundary curvature, junctions)
are easy for it. Division of labour by length scale.

Usage:
    python map_generator.py train              # ~minutes on GPU
    python map_generator.py sample --n 4       # writes generated_maps/map_XXX.npy + preview png

Then in sampling.py set TARGET_MAP = "outputs/generated_maps/map_000.npy".
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from field_conditioning import make_psi6_map

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR   = "data/patches/train"
CKPT        = "checkpoints/map_generator.pt"
OUT_DIR     = "outputs/generated_maps"

GRID_N      = 64
N_STEPS     = 400
BATCH       = 32
N_EPOCHS    = 1500   # ~47 batches/epoch -> ~70k steps; thin sparse boundary
                     # lines need far more optimisation than the handful of
                     # steps 60 epochs gave (maps came out blank)
LR          = 2e-4
BASE_CH     = 32
# ──────────────────────────────────────────────────────────────────────────────


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.temb = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.temb(t)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64, channels 32/64/128/128."""

    def __init__(self, base=BASE_CH, t_dim=128):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.time = TimeEmbed(t_dim)
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = ConvBlock(c1, c1, t_dim)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.d2 = ConvBlock(c2, c2, t_dim)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.d3 = ConvBlock(c3, c3, t_dim)
        self.down3 = nn.Conv2d(c3, c3, 3, stride=2, padding=1)
        self.mid = ConvBlock(c3, c3, t_dim)
        self.up3 = nn.ConvTranspose2d(c3, c3, 4, stride=2, padding=1)
        self.u3 = ConvBlock(c3 + c3, c3, t_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.u2 = ConvBlock(c2 + c2, c2, t_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.u1 = ConvBlock(c1 + c1, c1, t_dim)
        self.out = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x, t):
        te = self.time(t)
        h1 = self.d1(self.inp(x), te)
        h2 = self.d2(self.down1(h1), te)
        h3 = self.d3(self.down2(h2), te)
        m = self.mid(self.down3(h3), te)
        u = self.u3(torch.cat([self.up3(m), h3], dim=1), te)
        u = self.u2(torch.cat([self.up2(u), h2], dim=1), te)
        u = self.u1(torch.cat([self.up1(u), h1], dim=1), te)
        return self.out(u)


def cosine_alpha_bars(n_steps, s=0.008):
    steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    return (f / f[0]).float()


class MapDiffusion:
    def __init__(self, n_steps=N_STEPS, device="cpu"):
        self.n_steps = n_steps
        self.device = device
        ab = cosine_alpha_bars(n_steps).to(device)
        self.alpha_bars = ab[1:]
        self.alphas = ab[1:] / ab[:-1]
        self.betas = 1 - self.alphas

    def loss(self, model, x0):
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alpha_bars[t][:, None, None, None]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = model(x_t, t)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, n, shape=(1, GRID_N, GRID_N)):
        x = torch.randn(n, *shape, device=self.device)
        for step in reversed(range(self.n_steps)):
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            eps = model(x, t)
            alpha, ab, beta = self.alphas[step], self.alpha_bars[step], self.betas[step]
            ab_prev = self.alpha_bars[step - 1] if step > 0 else torch.tensor(1.0, device=self.device)
            x0_hat = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-3, 3)
            mean = ab_prev.sqrt() * beta / (1 - ab) * x0_hat + alpha.sqrt() * (1 - ab_prev) / (1 - ab) * x
            if step > 0:
                sigma = (beta * (1 - ab_prev) / (1 - ab)).sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean
        return x


def load_maps():
    paths = sorted(glob.glob(f"{TRAIN_DIR}/*.npz"))
    maps, half_widths = [], []
    for p in tqdm(paths, desc="computing psi6 maps", unit="patch"):
        d = np.load(p)
        pos = d["pos"].astype(np.float64)
        pos -= pos.mean(axis=0)
        grid, extent = make_psi6_map(pos, grid_n=GRID_N)
        maps.append(grid)
        half_widths.append((extent[1] - extent[0]) / 2)
    return np.stack(maps), float(np.mean(half_widths))


def train(device):
    maps, half_width = load_maps()
    mean, std = float(maps.mean()), float(maps.std())
    data = torch.from_numpy((maps - mean) / std).unsqueeze(1).float()
    print(f"{len(data)} maps, value mean={mean:.4f} std={std:.4f}, "
          f"half_width={half_width:.2f}")

    model = SmallUNet().to(device)
    diff = MapDiffusion(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"map generator parameters: {n_params:,}")

    for epoch in tqdm(range(N_EPOCHS), desc="training map generator"):
        perm = torch.randperm(len(data))
        losses = []
        for i in range(0, len(data), BATCH):
            x0 = data[perm[i:i + BATCH]].to(device)
            opt.zero_grad()
            loss = diff.loss(model, x0)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            tqdm.write(f"  epoch {epoch}: loss = {np.mean(losses):.4f}")

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "half_width": half_width, "grid_n": GRID_N,
                "n_steps": N_STEPS}, CKPT)
    print(f"saved {CKPT}")


def sample(device, n):
    ckpt = torch.load(CKPT, map_location=device)
    model = SmallUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = MapDiffusion(n_steps=ckpt["n_steps"], device=device)

    x = diff.sample(model, n).cpu().numpy()[:, 0]
    maps = np.clip(x * ckpt["std"] + ckpt["mean"], 0.0, 1.0)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i in range(n):
        np.save(f"{OUT_DIR}/map_{i:03d}.npy", maps[i])
        axes[0][i].imshow(maps[i].T, origin="lower", cmap="viridis",
                          vmin=0, vmax=1)
        axes[0][i].set_title(f"map_{i:03d}", fontsize=9)
        axes[0][i].axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/preview.png", dpi=130, bbox_inches="tight")
    print(f"saved {n} maps + preview to {OUT_DIR}/ "
          f"(half_width={ckpt['half_width']:.2f} physical units)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    if args.mode == "train":
        train(dev)
    else:
        sample(dev, args.n)
