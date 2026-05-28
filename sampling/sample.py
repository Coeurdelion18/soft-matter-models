import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.model import SimpleUnet, device
from models.forward_diffusion import (
    T,
    linear_beta_schedule,
    get_index_from_list,
    show_tensor_image
)

# =========================================================
# Parameters
# =========================================================

IMG_SIZE = 192

MODEL_PATH = "checkpoints/diffusion_model.pth"

# =========================================================
# Load model
# =========================================================

model = SimpleUnet().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

# =========================================================
# Diffusion constants
# =========================================================

betas = linear_beta_schedule(timesteps=T)

alphas = 1. - betas

alphas_cumprod = torch.cumprod(alphas, axis=0)

alphas_cumprod_prev = F.pad(
    alphas_cumprod[:-1],
    (1, 0),
    value=1.0
)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1. - alphas_cumprod
)

posterior_variance = (
    betas
    * (1. - alphas_cumprod_prev)
    / (1. - alphas_cumprod)
)

# =========================================================
# Reverse diffusion step
# =========================================================

@torch.no_grad()
def sample_timestep(x, t):

    betas_t = get_index_from_list(
        betas,
        t,
        x.shape
    )

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x.shape
    )

    sqrt_recip_alphas_t = get_index_from_list(
        sqrt_recip_alphas,
        t,
        x.shape
    )

    # Predict noise
    noise_pred = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * noise_pred
        / sqrt_one_minus_alphas_cumprod_t
    )

    posterior_variance_t = get_index_from_list(
        posterior_variance,
        t,
        x.shape
    )

    if t[0] == 0:

        return model_mean

    else:

        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(
            posterior_variance_t
        ) * noise

# =========================================================
# Sampling loop
# =========================================================

@torch.no_grad()
def sample_plot_image():

    # Start from pure Gaussian noise
    img = torch.randn(
        (1, 1, IMG_SIZE, IMG_SIZE),
        device=device
    )

    num_images = 10
    stepsize = T // num_images

    # Bigger figure
    fig = plt.figure(figsize=(18, 10))

    # -----------------------------
    # Top row: intermediate timesteps
    # -----------------------------

    plot_idx = 1

    for i in range(T - 1, -1, -1):

        t = torch.full(
            (1,),
            i,
            device=device,
            dtype=torch.long
        )

        img = sample_timestep(img, t)

        img = torch.clamp(
            img,
            -1.0,
            1.0
        )

        # Show selected timesteps
        if i % stepsize == 0:

            ax = fig.add_subplot(
                2,
                num_images,
                plot_idx
            )

            plt.title(f"t = {i}")

            show_tensor_image(
                img.detach().cpu()
            )

            plot_idx += 1

    # -----------------------------
    # Bottom row: final image
    # -----------------------------

    ax = plt.subplot2grid(
        (2, num_images),
        (1, 0),
        colspan=num_images
    )

    plt.title(
        "Final Generated Lattice",
        fontsize=16
    )

    show_tensor_image(
        img.detach().cpu()
    )

    #Also save the image
    torch.save(img, "new_gen.pt")

    plt.tight_layout()
    plt.show()

# =========================================================
# Run sampling
# =========================================================

if __name__ == "__main__":
    sample_plot_image()