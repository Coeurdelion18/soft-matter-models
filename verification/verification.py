import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from models.forward_diffusion import T
from sampling.sample import sample_timestep

# image_path = "new_gen.pt"
# img = torch.load(image_path)
IMG_SIZE = 192
# MODEL_PATH = "diffusion_model.pth"
# NX = 48
# NY = 48
# A = 1.0
# LX = (NX - 0.5) * A
# LY = (NY - 1) * (np.sqrt(3) / 2) * A


device = "cuda" if torch.cuda.is_available() else "cpu"

coord_lens = []

for _ in range(2):
    img = torch.randn(
        (1, 1, IMG_SIZE, IMG_SIZE),
        device=device
    )

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

    img = img.squeeze()
    img = torch.clamp(img, -1.0, 1.0)
    img = (img + 1)/2
    img = img.detach().cpu().numpy()
    coords = peak_local_max(img, min_distance=2, threshold_abs=0.7)
    coord_lens.append(len(coords))
    
coord_lens = np.array(coord_lens, dtype=float)
print(f"Mean number of peaks: {np.mean(coord_lens)}")
print(f"Standard deviation of number of peaks: {np.std(coord_lens)}")

#We get 1900 +/- 25 peaks with the same dimensions as the sample images
