# generate_dataset.py

import numpy as np
from tqdm import tqdm
from pathlib import Path

from datasets.GridGeneration import (
    generateHexagonalGrid,
    coords_to_density_grid,
    generate_diatomic_density_grid,
    generate_two_channel_density_grid
)

import torch
from torch.utils.data import Dataset

# =========================================================
# Dataset generation parameters
# =========================================================

NUM_SAMPLES = 500

NX = 48
NY = 48

LATTICE_PARAMETER = 1.0
DEFECT_RATE = 0.10

GRID_SIZE = 192
SIGMA = 0.4

OUTPUT_FILE = "hex_lattice_dataset.npy"

# =========================================================
# Generate dataset
# =========================================================

# if __name__ == "__main__":
#     dataset = []

#     for _ in tqdm(range(NUM_SAMPLES)):

#         coords, box_size = generateHexagonalGrid(
#             nx=NX,
#             ny=NY,
#             a=LATTICE_PARAMETER,
#             defect_rate=DEFECT_RATE
#         )

#         density_grid = coords_to_density_grid(
#             coords,
#             box_size,
#             grid_size=GRID_SIZE,
#             sigma=SIGMA
#         )

#         # Normalize to [0, 1]
#         density_grid = density_grid / density_grid.max()

#         # Add channel dimension
#         density_grid = density_grid.astype(np.float32)
#         density_grid = density_grid[None, :, :]

#         dataset.append(density_grid)

#     # =========================================================
#     # Convert to numpy array
#     # =========================================================

#     dataset = np.stack(dataset)

#     print("Dataset shape:", dataset.shape)

#     # Expected:
#     # (NUM_SAMPLES, 1, GRID_SIZE, GRID_SIZE)

#     # =========================================================
#     # Save dataset
#     # =========================================================

#     np.save(OUTPUT_FILE, dataset)

#     print(f"Dataset saved to: {OUTPUT_FILE}")

# if __name__ == "__main__":

#     dataset = []
#     for _ in tqdm(range(NUM_SAMPLES)):

#         coords, box_size = generateHexagonalGrid(nx=48, ny=48, a=1.0, defect_rate=0.10, pbc=False)
#         _, _, img = generate_two_channel_density_grid(coords, box_size, GRID_SIZE, sigma_a=0.5, sigma_b=0.7)

#         # Normalize to [0, 1]
#         global_max = img.max()

#         if global_max > 0:
#             img /= global_max

#         dataset.append(img)
    
#     dataset = np.stack(dataset)
#     print("Dataset shape:", dataset.shape)

    # Expected:
    # (NUM_SAMPLES, 1, GRID_SIZE, GRID_SIZE)

    # =========================================================
    # Save dataset
    # =========================================================

    # np.save("data/raw/binary_two_channel_hex.npy", dataset)

    # print("Dataset saved to: binary_two_channel_hex.npy")


class HexLatticeDataset(Dataset):
    
    def __init__(self, npy_file):
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x
    
class GrainBoundaryDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float)
        return x
    
def process_directory_dataset(directory_path, output_path):
    folder = Path(directory_path)
    images = []

    for file in tqdm(sorted(folder.iterdir())):
        if not file.is_file():
            continue

        if file.suffix != ".npy":
            continue
        
        img = np.load(file)
        max_val = img.max()
        if max_val > 0:
            img /= max_val

        images.append(img)

    if len(images) == 0:
        raise ValueError("No .npy files found")
    
    dataset = np.stack(images)
    np.save(output_path, dataset)

if __name__ == "__main__":
    dir_path = "data/raw/config_npy"
    out_path = "data/raw/grain_boundary_dataset.npy"
    process_directory_dataset(dir_path, out_path)