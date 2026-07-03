import numpy as np
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"device: {device}")
# if device == "cuda":
#     print(f"GPU: {torch.cuda.get_device_name(0)}")

arr = np.load("data/npy/001.npz")
pos = arr["pos"]
print(pos.min(axis=0))
print(pos.max(axis=0))