"""
overfit_test.py

Sanity check before committing to full training: can the model learn
to denoise a SINGLE sample? If this fails, full training will fail too
— this isolates pipeline/data/model bugs from "needs more data/epochs".

What a healthy run looks like:
    - Loss drops from ~1.0 (random noise variance) toward ~0 within
      a few hundred steps (not epochs — steps, since there's only 1 sample)
    - Generated sample visually resembles the single training image
      by the end

What a FAILING run looks like (and what it tells you):
    - Loss never drops below ~0.5-1.0          -> model/data bug, not a
                                                   training-time problem
    - Loss drops then NaNs                      -> likely a normalization
                                                   or mixed-precision issue
    - Loss drops but generated image is noise    -> sampling loop bug,
                                                   not a training bug

Usage:
    python overfit_test.py --sample_idx 0 --steps 800
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler

from model import build_model
from config import DEVICE, IMAGE_SIZE


CHANNEL_NAMES = ["Density", "|psi6|", "arg(psi6)", "Voronoi sides"]
CHANNEL_CMAPS = ["inferno", "viridis", "twilight", "RdYlGn"]


def load_single_sample(dataset_path, idx=0):
    data = np.load(dataset_path)          # (N, 4, H, W)
    sample = data[idx]                    # (4, H, W)
    return torch.tensor(sample, dtype=torch.float32)


def overfit(
    dataset_path="data/processed/dataset.npy",
    sample_idx=0,
    steps=800,
    lr=1e-4,
    log_every=50,
    output_dir="outputs/overfit_test",
):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load the single training image
    # ------------------------------------------------------------------
    image = load_single_sample(dataset_path, sample_idx).to(DEVICE)
    image = image.unsqueeze(0)   # (1, 4, H, W) — batch of 1

    print(f"Loaded sample {sample_idx} from {dataset_path}")
    print(f"  Shape : {image.shape}")
    print(f"  Range : [{image.min().item():.3f}, {image.max().item():.3f}]")

    # Save the ground-truth image for visual comparison later
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for c in range(4):
        axes[c].imshow(image[0, c].cpu().numpy(), origin="lower",
                       cmap=CHANNEL_CMAPS[c])
        axes[c].set_title(f"GT — {CHANNEL_NAMES[c]}")
        axes[c].axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ground_truth.png", dpi=120)
    plt.close()
    print(f"  Saved ground truth -> {output_dir}/ground_truth.png")

    # ------------------------------------------------------------------
    # Model, scheduler, optimizer — no EMA, no AMP, kept simple
    # so failures can't hide behind those mechanisms
    # ------------------------------------------------------------------
    model = build_model().to(DEVICE)
    model.train()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    best_loss = float("inf")
    best_state = None
    GRAD_CLIP_NORM = 1.0  # standard for diffusion training; prevents single-step blowups

    print(f"\nOverfitting on 1 sample for {steps} steps...")
    for step in range(steps):
        noise = torch.randn_like(image)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (1,), device=DEVICE
        ).long()

        noisy_image = noise_scheduler.add_noise(image, noise, timesteps)

        optimizer.zero_grad()
        noise_pred = model(noisy_image, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        # Clip gradients to prevent a single bad step from destabilising
        # the weights permanently (this is what caused the divergence at
        # step ~1600 in the previous run — the loss jumped to baseline
        # and never recovered because nothing capped the gradient norm).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRAD_CLIP_NORM
        )

        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Keep a copy of the best-performing weights so a late divergence
        # can never erase earlier progress.
        if loss_val < best_loss and not torch.isnan(loss):
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  loss = {loss_val:.6f}  "
                  f"grad_norm = {grad_norm:.3f}")

        if torch.isnan(loss):
            print("\n[FAIL] Loss became NaN. Likely cause: learning rate "
                  "too high, or a normalization/scale issue in the data. "
                  "Stopping early.")
            break

    # ------------------------------------------------------------------
    # Loss curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Overfit test — loss curve (1 sample)")
    ax.axhline(1.0, color="grey", ls=":", lw=0.8, label="random-noise baseline (~1.0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=120)
    plt.close()
    print(f"\nSaved loss curve -> {output_dir}/loss_curve.png")

    final_loss = losses[-1] if losses else float("nan")
    print(f"\nFinal step loss : {final_loss:.6f}")
    print(f"Best loss seen  : {best_loss:.6f}  (this is what gets used for generation)")

    if final_loss > 0.3 and final_loss > best_loss * 3:
        print("[WARNING] Loss diverged late in training (final >> best). "
              "Restoring best-checkpoint weights before generating, since "
              "the live weights are no longer usable. Consider lowering "
              "the learning rate or reducing --steps to roughly where "
              "the best loss occurred.")
    elif best_loss > 0.3:
        print("[WARNING] Even the best loss seen did not drop substantially. "
              "This points to a pipeline bug, not a 'needs more training' "
              "issue. Do not proceed to full training yet.")
    else:
        print("[OK] Loss dropped as expected. Proceeding to generate "
              "a sample from pure noise to visually confirm.")

    # ------------------------------------------------------------------
    # Generate from pure noise — always use the BEST weights seen during
    # training, not whatever the model currently holds. This protects
    # against late-training divergence destroying an otherwise-successful
    # run (see: step ~1600 divergence in the previous attempt).
    # ------------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    noise_scheduler_infer = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    )

    with torch.no_grad():
        sample = torch.randn((1, 4, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
        for t in noise_scheduler_infer.timesteps:
            noise_pred = model(sample, t).sample
            sample = noise_scheduler_infer.step(noise_pred, t, sample).prev_sample

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for c in range(4):
        axes[c].imshow(sample[0, c].cpu().numpy(), origin="lower",
                       cmap=CHANNEL_CMAPS[c])
        axes[c].set_title(f"Generated — {CHANNEL_NAMES[c]}")
        axes[c].axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/generated.png", dpi=120)
    plt.close()
    print(f"Saved generated sample -> {output_dir}/generated.png")

    print(f"\nCompare {output_dir}/ground_truth.png against "
          f"{output_dir}/generated.png — they should look visually similar "
          f"if the pipeline is healthy.")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overfit the model on a single sample as a pipeline sanity check"
    )
    parser.add_argument("--dataset_path", type=str, default="data/processed/dataset.npy")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs/overfit_test")

    args = parser.parse_args()

    overfit(
        dataset_path=args.dataset_path,
        sample_idx=args.sample_idx,
        steps=args.steps,
        lr=args.lr,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )