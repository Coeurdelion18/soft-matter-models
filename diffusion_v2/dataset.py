import numpy as np
import torch
from torch.utils.data import Dataset

class GrainBoundaryDataset(Dataset):

    def __init__(self, path):
        self.data = np.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)