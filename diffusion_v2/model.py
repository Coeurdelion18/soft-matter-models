from diffusers import UNet2DModel

def build_model():
    model = UNet2DModel(
        sample_size=384,
        in_channels=4,   # density, |psi6|, arg(psi6), voronoi sides
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model