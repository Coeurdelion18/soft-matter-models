from model import build_model

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import (
    DEVICE,
    CHECKPOINT_PATH,
    OUTPUT_PATH,
    IMAGE_SIZE
)
from diffusers import DDPMScheduler

# =========================================================
# CHANNEL METADATA (must match compute_channels.py ordering)
# =========================================================

CHANNEL_NAMES = ["Density", "|psi6|", "arg(psi6)", "Voronoi sides"]
CHANNEL_CMAPS = ["inferno", "viridis", "twilight", "RdYlGn"]
NUM_CHANNELS = len(CHANNEL_NAMES)

# =========================================================
# MODEL
# =========================================================

model = build_model().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model.eval()

# =========================================================
# SCHEDULER
# =========================================================

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2"
)

# =========================================================
# GENERATION
# =========================================================

@torch.no_grad()
def generate():

    sample = torch.randn(
        (1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        device=DEVICE
    )

    timesteps = noise_scheduler.timesteps

    for i, t in enumerate(timesteps):

        noise_pred = model(
            sample,
            t
        ).sample

        sample = noise_scheduler.step(
            noise_pred,
            t,
            sample
        ).prev_sample

        if i % 100 == 0:
            print(
                f"Step {i}/{len(timesteps)}"
            )

    return sample


# =========================================================
# VISUALIZATION
# =========================================================

def show_channels(img):

    img = img.squeeze(0)  # (4, H, W)

    fig, axes = plt.subplots(
        1,
        NUM_CHANNELS,
        figsize=(4 * NUM_CHANNELS, 4)
    )

    for c in range(NUM_CHANNELS):
        channel = img[c].cpu().numpy()
        axes[c].imshow(
            channel,
            origin="lower",
            cmap=CHANNEL_CMAPS[c]
        )
        axes[c].set_title(CHANNEL_NAMES[c])
        axes[c].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    sample = generate()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    np.save(
        OUTPUT_PATH,
        sample.cpu().numpy()
    )

    print(
        f"Saved sample to {OUTPUT_PATH}"
    )

    show_channels(sample)