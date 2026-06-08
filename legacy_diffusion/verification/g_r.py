#Given the density field representation, we need to extract the coordinates of the particles.
#Shouldn't be too difficult, we'll use a peak filter from skimages
#For the binary mixture, we use two channels, so we will apply this filter on both channels individually and then merge the coordinate lists
from skimage.feature import peak_local_max
import torch
import numpy as np

def extract_coordinates(img: torch.Tensor):
    ch1 = img[0].squeeze()
    ch2 = img[1].squeeze()

    ch1 = ch1.detach().cpu().numpy()

    ch2 = ch2.detach().cpu().numpy()

    coords_ch1 = peak_local_max(
        ch1,
        min_distance=1,
        threshold_rel=0.05,
        exclude_border=False
    )
    coords_ch2 = peak_local_max(
        ch2,
        min_distance=1,
        threshold_rel=0.05,
        exclude_border=False
    )

    return coords_ch1, coords_ch2

path = "data/raw/npy100.npy"
img = np.load(path)
torch_img = torch.tensor(img)
c1, c2 = extract_coordinates(torch_img)
print(len(c1))
print(len(c2))
print(len(c1) + len(c2))
