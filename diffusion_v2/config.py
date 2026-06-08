import torch

IMAGE_SIZE = 384
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/grain_boundary_diffusion_ema.pth"
OUTPUT_PATH = "outputs/generated_sample.npy"