import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from models.model import SimpleUnet, device
from models.forward_diffusion import (
    T,
    forward_diffusion_sample,
    get_index_from_list,
    linear_beta_schedule
)

# =========================================================
# SETTINGS
# =========================================================

CHECKPOINT_PATH = "checkpoints/single_image_grain_boundary_T50_strong_schedule.pth"
DATA_PATH = "data/raw/single_grain_boundary.npy"

# =========================================================
# LOAD MODEL
# =========================================================

model = SimpleUnet().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model.eval()

# =========================================================
# DIFFUSION CONSTANTS
# =========================================================

betas = torch.linspace(0.0001, 0.1, T)

alphas = 1.0 - betas

alphas_cumprod = torch.cumprod(
    alphas,
    dim=0
)

alphas_cumprod_prev = F.pad(
    alphas_cumprod[:-1],
    (1, 0),
    value=1.0
)

sqrt_recip_alphas = torch.sqrt(
    1.0 / alphas
)

sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1.0 - alphas_cumprod
)

posterior_variance = (
    betas
    * (1.0 - alphas_cumprod_prev)
    / (1.0 - alphas_cumprod)
)

print()
print("T =", T)
print("alpha_bar(T-1) =", alphas_cumprod[-1].item())
print("sqrt(alpha_bar(T-1)) =", torch.sqrt(alphas_cumprod[-1]).item())
print()

# =========================================================
# REVERSE STEP
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

    noise_pred = model(
        x,
        t
    )

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

    noise = torch.randn_like(x)

    return (
        model_mean
        + torch.sqrt(posterior_variance_t)
        * noise
    )

# =========================================================
# LOAD IMAGE
# =========================================================

dataset = torch.from_numpy(
    np.load(DATA_PATH)
).float()

x0 = dataset[0].unsqueeze(0).to(device)

print("x0 shape:", x0.shape)

# =========================================================
# CREATE TRUE x_(T-1)
# =========================================================

start_t = T - 1

t = torch.tensor(
    [start_t],
    device=device,
    dtype=torch.long
)

xt, _ = forward_diffusion_sample(
    x0,
    t
)

# =========================================================
# RUN FULL REVERSE CHAIN
# =========================================================

img = xt.clone()

for i in range(start_t - 1, -1, -1):

    t = torch.tensor(
        [i],
        device=device,
        dtype=torch.long
    )

    img = sample_timestep(
        img,
        t
    )

# =========================================================
# VISUALIZATION
# =========================================================

real_img = x0[0].detach().cpu()
recovered_img = img[0].detach().cpu()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10)
)

axes[0, 0].imshow(
    real_img[0],
    origin="lower"
)
axes[0, 0].set_title("Real Channel 0")

axes[0, 1].imshow(
    recovered_img[0],
    origin="lower"
)
axes[0, 1].set_title("Recovered Channel 0")

axes[1, 0].imshow(
    real_img[1],
    origin="lower"
)
axes[1, 0].set_title("Real Channel 1")

axes[1, 1].imshow(
    recovered_img[1],
    origin="lower"
)
axes[1, 1].set_title("Recovered Channel 1")

for ax in axes.flatten():
    ax.axis("off")

plt.tight_layout()
plt.show()

print(alphas_cumprod[-1])