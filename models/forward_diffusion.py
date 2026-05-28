import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets.GenerateDataset import HexLatticeDataset

# =========================================================
# Parameters
# =========================================================

T = 300

BATCH_SIZE = 32

DATASET_PATH = "hex_lattice_dataset.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Beta Schedule
# =========================================================

def linear_beta_schedule(
    timesteps,
    start=0.0001,
    end=0.02
):

    return torch.linspace(start, end, timesteps)

# =========================================================
# Diffusion Constants
# =========================================================

betas = linear_beta_schedule(timesteps=T)

alphas = 1.0 - betas

alphas_cumprod = torch.cumprod(alphas, axis=0)

alphas_cumprod_prev = F.pad(
    alphas_cumprod[:-1],
    (1, 0),
    value=1.0
)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1.0 - alphas_cumprod
)

posterior_variance = (
    betas
    * (1.0 - alphas_cumprod_prev)
    / (1.0 - alphas_cumprod)
)

# =========================================================
# Helper Function
# =========================================================

def get_index_from_list(vals, t, x_shape):

    """
    Returns values indexed at timestep t,
    reshaped for broadcasting.
    """

    batch_size = t.shape[0]

    out = vals.gather(-1, t.cpu())

    return out.reshape(
        batch_size,
        *((1,) * (len(x_shape) - 1))
    ).to(t.device)

# =========================================================
# Forward Diffusion
# =========================================================

def forward_diffusion_sample(x_0, t):

    """
    Adds Gaussian noise to image x_0
    at timestep t.
    """

    device = x_0.device

    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod,
        t,
        x_0.shape
    )

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x_0.shape
    )

    x_t = (
        sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        +
        sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    )

    return x_t, noise

# =========================================================
# Dataset Loader
# =========================================================

def load_dataset():

    dataset = HexLatticeDataset(DATASET_PATH)

    return dataset

# =========================================================
# Visualization Helper
# =========================================================

def show_tensor_image(image):

    """
    Displays tensor image.
    """

    # Remove batch dimension if present
    if len(image.shape) == 4:

        image = image[0]

    # Remove channel dimension
    image = image.squeeze(0)

    # Clamp values
    image = torch.clamp(image, -1.0, 1.0)

    # Convert from [-1,1] -> [0,1]
    image = (image + 1) / 2

    # Convert to numpy
    image = image.detach().cpu().numpy()

    plt.imshow(
        image,
        cmap='viridis',
        origin='lower'
    )

    plt.axis('off')

# =========================================================
# Demo / Visualization
# =========================================================

if __name__ == "__main__":

    dataset = load_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Take first batch
    image = next(iter(dataloader)).to(device)

    plt.figure(figsize=(20, 4))

    num_images = 10

    stepsize = int(T / num_images)

    for idx in range(0, T, stepsize):

        t = torch.full(
            (image.shape[0],),
            idx,
            device=device,
            dtype=torch.long
        )

        plt.subplot(
            1,
            num_images,
            int(idx / stepsize) + 1
        )

        noisy_image, noise = forward_diffusion_sample(
            image,
            t
        )

        show_tensor_image(noisy_image)

        plt.title(f"t = {idx}")

    plt.tight_layout()

    plt.show()