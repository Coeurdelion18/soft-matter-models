import torch
import matplotlib.pyplot as plt

from models.model import SimpleUnet, device
from models.forward_diffusion import (
    forward_diffusion_sample,
    get_index_from_list,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod
)

# =========================================================
# SETTINGS
# =========================================================

CHECKPOINT_PATH = "checkpoints/grain_boundary_diffusion_model.pth"

# timestep to test
TEST_TIMESTEP = 49 

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
# LOAD TRAINING IMAGE
# =========================================================

dataset = torch.from_numpy(
    __import__("numpy").load(
        "data/raw/single_grain_boundary.npy"
    )
).float()

# shape should be:
# (1, 2, H, W)

x0 = dataset[0].unsqueeze(0).to(device)

print("x0 shape:", x0.shape)

# =========================================================
# FORWARD DIFFUSION
# =========================================================

t = torch.tensor(
    [TEST_TIMESTEP],
    device=device,
    dtype=torch.long
)

xt, true_noise = forward_diffusion_sample(
    x0,
    t
)

# =========================================================
# PREDICT NOISE
# =========================================================

with torch.no_grad():

    pred_noise = model(
        xt,
        t
    )

# =========================================================
# RECONSTRUCT x0
# =========================================================

sqrt_alpha_bar_t = get_index_from_list(
    sqrt_alphas_cumprod,
    t,
    xt.shape
)

sqrt_one_minus_alpha_bar_t = get_index_from_list(
    sqrt_one_minus_alphas_cumprod,
    t,
    xt.shape
)

x0_pred = (
    xt
    - sqrt_one_minus_alpha_bar_t * pred_noise
) / sqrt_alpha_bar_t

# =========================================================
# VISUALIZATION
# =========================================================

real_img = x0[0].detach().cpu()
recon_img = x0_pred[0].detach().cpu()

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
    recon_img[0],
    origin="lower"
)
axes[0, 1].set_title("Reconstructed Channel 0")

axes[1, 0].imshow(
    real_img[1],
    origin="lower"
)
axes[1, 0].set_title("Real Channel 1")

axes[1, 1].imshow(
    recon_img[1],
    origin="lower"
)
axes[1, 1].set_title("Reconstructed Channel 1")

for ax in axes.flatten():
    ax.axis("off")

plt.tight_layout()
#plt.show()
print(sqrt_alphas_cumprod[-1])