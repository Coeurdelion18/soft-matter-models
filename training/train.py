import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import SimpleUnet, device
from models.forward_diffusion import T
from datasets.GenerateDataset import HexLatticeDataset, GrainBoundaryDataset
from models.model import get_loss

# =========================================================
# Parameters
# =========================================================

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50

DATASET_PATH = "data/raw/grain_boundary_dataset.npy"

MODEL_SAVE_PATH = "checkpoints/grain_boundary_diffusion_model.pth"

# =========================================================
# Load Dataset
# =========================================================

dataset = GrainBoundaryDataset(DATASET_PATH)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================================================
# Model
# =========================================================

model = SimpleUnet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================================================
# Training Loop
# =========================================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    progress_bar = tqdm(dataloader)

    for step, batch in enumerate(progress_bar):

        optimizer.zero_grad()

        batch = batch.to(device)

        # Random timestep for each image
        t = torch.randint(
            0,
            T,
            (batch.shape[0],),
            device=device
        ).long()

        scaler = torch.amp.GradScaler("cuda")
        
        with torch.amp.autocast("cuda"):
            loss = get_loss(model, batch, t)

        # loss = get_loss(model, batch, t)

        # loss.backward()

        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_description(
            f"Epoch {epoch+1} | Loss: {loss.item():.6f}"
        )

    avg_loss = epoch_loss / len(dataloader)

    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.6f}")

# =========================================================
# Save Model
# =========================================================

torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")