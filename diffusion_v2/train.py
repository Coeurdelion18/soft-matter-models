from dataset import GrainBoundaryDataset
from model import build_model
from utils import build_ema
from config import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    CHECKPOINT_PATH
)
from tqdm import tqdm
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2"
                ) #Note that this noise_scheduler takes care of timestep embedding by itself

dataset = GrainBoundaryDataset("data/processed/dataset.npy")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Training loop

model = build_model().to(DEVICE)
ema = build_ema(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scaler = torch.amp.GradScaler("cuda")

losses = []

# =========================================================
# CHECKPOINTING CONFIG
# =========================================================
CHECKPOINT_EVERY = 15          # save a periodic checkpoint every N epochs
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOSSES_PATH = CHECKPOINT_PATH.replace(".pth", "_losses.npy")

start_epoch = 0
best_loss = float("inf")

# Resume from the most recent periodic checkpoint, if one exists
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pth")
if os.path.exists(LATEST_PATH):
    print(f"Found existing checkpoint at {LATEST_PATH} — resuming.")
    state = torch.load(LATEST_PATH, map_location=DEVICE)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    start_epoch = state["epoch"] + 1
    best_loss = state.get("best_loss", float("inf"))
    if os.path.exists(LOSSES_PATH):
        losses = list(np.load(LOSSES_PATH))
    print(f"  Resuming from epoch {start_epoch}, best_loss={best_loss:.6f}")


def save_checkpoint(path, epoch, best_loss):
    """Full training state — model, EMA, optimizer, scaler, epoch, losses."""
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }, path)


for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images in progress_bar:

        images = images.to(DEVICE)

        batch_size = images.shape[0]

        # Generate sample noise

        noise = torch.randn_like(images)

        # Choose a random timestep

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=DEVICE
        ).long()

        # Add noise

        noisy_images = noise_scheduler.add_noise(
            images,
            noise,
            timesteps
        )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            noise_pred = model(
                noisy_images,
                timesteps
            ).sample

            loss = F.mse_loss(
                noise_pred,
                noise
            )


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = running_loss / len(dataloader)

    losses.append(epoch_loss)
    np.save(LOSSES_PATH, np.array(losses))   # persist every epoch, not just at the end

    print(
        f"\nEpoch {epoch+1}: "
        f"avg loss = {epoch_loss:.6f}"
    )

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------
    is_best = epoch_loss < best_loss
    if is_best:
        best_loss = epoch_loss
        save_checkpoint(
            os.path.join(CHECKPOINT_DIR, "best.pth"),
            epoch, best_loss
        )
        print(f"  New best loss ({best_loss:.6f}) — saved best.pth")

    # Always overwrite "latest" so a crash never loses more than one epoch
    save_checkpoint(LATEST_PATH, epoch, best_loss)

    # Periodic timestamped snapshot, kept permanently (not overwritten)
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        periodic_path = os.path.join(
            CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"
        )
        save_checkpoint(periodic_path, epoch, best_loss)
        print(f"  Periodic checkpoint saved -> {periodic_path}")


# =========================================================
# Final save — EMA weights only, in the format sample.py expects
# =========================================================
torch.save(
    ema.ema_model.state_dict(),
    CHECKPOINT_PATH
)

print(
    f"\nSaved final EMA model to:\n{CHECKPOINT_PATH}"
)

np.save(LOSSES_PATH, np.array(losses))
print(f"Saved loss history to:\n{LOSSES_PATH}")