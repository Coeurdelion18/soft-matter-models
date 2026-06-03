from torch import nn
import torch
import math
import torch.nn.functional as F

# =========================================================
# Device
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Sinusoidal Time Embeddings
# =========================================================

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):

        device = time.device

        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )

        return embeddings

# =========================================================
# U-Net Block
# =========================================================

class Block(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        time_emb_dim,
        up=False
    ):

        super().__init__()

        self.up = up

        self.time_mlp = nn.Linear(
            time_emb_dim,
            out_ch
        )

        if up:

            # UPSAMPLE FIRST
            self.transform = nn.ConvTranspose2d(
                in_ch,
                in_ch,
                kernel_size=4,
                stride=2,
                padding=1
            )

            self.conv1 = nn.Conv2d(
                in_ch * 2,
                out_ch,
                kernel_size=3,
                padding=1
            )

        else:

            self.conv1 = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            )

            self.transform = nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=4,
                stride=2,
                padding=1
            )

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )

        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

    def forward(self, x, t, residual=None):

        # -------------------------
        # Decoder path
        # -------------------------
        if self.up:

            # FIRST UPSAMPLE
            x = self.transform(x)

            # NOW spatial sizes match
            x = torch.cat((x, residual), dim=1) #dim=1 because that's the channel dimension

        # -------------------------
        # First conv
        # -------------------------

        h = self.conv1(x)
        h = self.relu(h)
        h = self.bnorm1(h)

        # -------------------------
        # Add timestep embedding
        # -------------------------

        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        # -------------------------
        # Second conv
        # -------------------------

        h = self.conv2(h)
        h = self.relu(h)
        h = self.bnorm2(h)

        residual_out = h

        # -------------------------
        # Encoder downsample
        # -------------------------

        if not self.up:
            h = self.transform(h)

        return h, residual_out

# =========================================================
# Simple U-Net
# =========================================================

class SimpleUnet(nn.Module):

    def __init__(self):

        super().__init__()

        image_channels = 2

        down_channels = (32, 64, 128, 256)

        up_channels = (256, 128, 64, 32)

        out_dim = 2

        time_emb_dim = 32

        # Time embedding

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection

        self.conv0 = nn.Conv2d(
            image_channels,
            down_channels[0],
            kernel_size=3,
            padding=1
        )

        # Downsampling path

        self.downs = nn.ModuleList([

            Block(
                down_channels[i],
                down_channels[i + 1],
                time_emb_dim,
                up=False
            )

            for i in range(len(down_channels) - 1)

        ])

        # Upsampling path

        self.ups = nn.ModuleList([

            Block(
                up_channels[i],
                up_channels[i + 1],
                time_emb_dim,
                up=True
            )

            for i in range(len(up_channels) - 1)

        ])

        # Final layer

        self.output = nn.Conv2d(
            up_channels[-1],
            out_dim,
            kernel_size=1
        )

    def forward(self, x, timestep):

        timestep = timestep.float()

        t = self.time_mlp(timestep)

        x = self.conv0(x)

        residual_inputs = []

        # Encoder

        for down in self.downs:

            x, residual = down(x, t)

            residual_inputs.append(residual)

        # Decoder

        for up in self.ups:

            residual_x = residual_inputs.pop()

            x, _ = up(
                x,
                t,
                residual_x
            )

        return self.output(x)

from models.forward_diffusion import forward_diffusion_sample


def get_loss(model, x_0, t):

    # Create noisy image x_t and true noise ε
    x_noisy, noise = forward_diffusion_sample(
        x_0,
        t
    )

    # Predict noise using U-Net
    noise_pred = model(
        x_noisy,
        t
    )

    # Compare predicted noise to true noise
    loss = F.mse_loss(
        noise_pred,
        noise
    )

    return loss