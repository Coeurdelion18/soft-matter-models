# generate_dataset.py

import numpy as np
from tqdm import tqdm

from datasets.GridGeneration import (
    generateHexagonalGrid,
    coords_to_density_grid
)

import torch
from torch.utils.data import Dataset

# =========================================================
# Dataset generation parameters
# =========================================================

NUM_SAMPLES = 1500

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

if __name__ == "__main__":
    dataset = []

    for _ in tqdm(range(NUM_SAMPLES)):

        coords, box_size = generateHexagonalGrid(
            nx=NX,
            ny=NY,
            a=LATTICE_PARAMETER,
            defect_rate=DEFECT_RATE
        )

        density_grid = coords_to_density_grid(
            coords,
            box_size,
            grid_size=GRID_SIZE,
            sigma=SIGMA
        )

        # Normalize to [0, 1]
        density_grid = density_grid / density_grid.max()

        # Add channel dimension
        density_grid = density_grid.astype(np.float32)
        density_grid = density_grid[None, :, :]

        dataset.append(density_grid)

    # =========================================================
    # Convert to numpy array
    # =========================================================

    dataset = np.stack(dataset)

    print("Dataset shape:", dataset.shape)

    # Expected:
    # (NUM_SAMPLES, 1, GRID_SIZE, GRID_SIZE)

    # =========================================================
    # Save dataset
    # =========================================================

    np.save(OUTPUT_FILE, dataset)

    print(f"Dataset saved to: {OUTPUT_FILE}")

class HexLatticeDataset(Dataset):
    
    def __init__(self, npy_file):
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x