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
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2"
                ) #Note that this noise_scheduler takes care of timestep embedding by itself

dataset = GrainBoundaryDataset("../data/raw/grain_boundary_dataset.npy")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Training loop

model = build_model().to(DEVICE)
ema = build_ema(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scaler = torch.amp.GradScaler("cuda")

losses = []

for epoch in range(EPOCHS):
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

    print(
        f"\nEpoch {epoch+1}: "
        f"avg loss = {epoch_loss:.6f}"
    )


torch.save(
    ema.ema_model.state_dict(),
    CHECKPOINT_PATH
)

print(
    f"\nSaved EMA model to:\n{CHECKPOINT_PATH}"
)