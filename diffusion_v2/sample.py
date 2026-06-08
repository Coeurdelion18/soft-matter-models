from model import build_model

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
        (1, 2, IMAGE_SIZE, IMAGE_SIZE),
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

    img = img.squeeze(0)

    channel0 = img[0].cpu().numpy()
    channel1 = img[1].cpu().numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    axes[0].imshow(
        channel0,
        origin="lower"
    )
    axes[0].set_title("Channel 0")

    axes[1].imshow(
        channel1,
        origin="lower"
    )
    axes[1].set_title("Channel 1")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    sample = generate()

    np.save(
        OUTPUT_PATH,
        sample.cpu().numpy()
    )

    print(
        f"Saved sample to {OUTPUT_PATH}"
    )

    show_channels(sample)