"""
Train the unconditional generator for bidisperse grain boundary systems.

Each training example is a boundary patch extracted from a full simulation
box. Patches are centered on grain boundary particles (identified by low
hexatic order) and contain both particle positions and scalar features
(size, type, and any other structural descriptors).

Usage:
    python scripts/train_unconditional.py

Replace load_boxes() with your real data loader. Each box should be a dict:
    {
        "pos":    np.ndarray (N, 2),   particle positions
        "types":  np.ndarray (N,),     0 = small, 1 = large defect
        "sizes":  np.ndarray (N,),     continuous particle diameter
    }
Additional scalar features (hexatic order, local density, etc.) can be
appended to node_scalars in build_node_scalars() below.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial import cKDTree
import glob
from egnn import EGNNUnconditionalDenoiser
from diffusion import PositionDiffusion

# ── Hyperparameters ────────────────────────────────────────────────────────────
CUTOFF          = 1.5    # set to first minimum of g(r) for your system
HIDDEN_DIM      = 128
N_LAYERS        = 4      # 2 = information from second-nearest graph neighbours
N_SCALAR_FEATS  = 2      # size + type; add more if you extend build_node_scalars()
N_STEPS         = 1000
LR              = 2e-4
N_EPOCHS        = 50
VAL_FRACTION    = 0.1    # fraction of boxes held out; split at box level, not patch level
# ──────────────────────────────────────────────────────────────────────────────

def load_boxes():
    boxes = []
    for f in sorted(glob.glob("data/npy/*.npz")):
        d = np.load(f)
        boxes.append({"pos": d["pos"], "type": d["type"], "size": d["size"]})
    return boxes


def build_node_scalars(pos, type, size):
    """
    Build (N, N_SCALAR_FEATS) float array of per-particle features.
    Extend this function to add hexatic order, local density, etc.
    All features should be rotation/translation invariant scalars.
    """
    


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    all_boxes = load_boxes()
    np.random.shuffle(all_boxes)
    n_val   = max(1, int(len(all_boxes) * VAL_FRACTION))
    val_set = all_boxes[:n_val]
    train_set = all_boxes[n_val:]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EGNNUnconditionalDenoiser(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_scalar_feats=N_SCALAR_FEATS,
    ).to(device)
    diffusion = PositionDiffusion(n_steps=N_STEPS, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(N_EPOCHS):
        # --- train ---
        model.train()
        train_loss, n = 0.0, 0
        for box in train_set:
            x0     = torch.from_numpy(box["pos"]).to(device)
            scalars = torch.from_numpy(box["node_scalars"]).to(device)
            opt.zero_grad()
            loss = diffusion.training_loss(model, x0, scalars, cutoff=CUTOFF)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n += 1

        # --- val ---
        model.eval()
        val_loss, nv = 0.0, 0
        with torch.no_grad():
            for box in val_set:
                x0      = torch.from_numpy(box["pos"]).to(device)
                scalars = torch.from_numpy(box["node_scalars"]).to(device)
                loss = diffusion.training_loss(model, x0, scalars, cutoff=CUTOFF)
                val_loss += loss.item()
                nv += 1

        tl = train_loss / max(n, 1)
        vl = val_loss  / max(nv, 1)
        print(f"epoch {epoch:3d}: train loss = {tl:.4f}  val loss = {vl:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), "checkpoints/model_unconditional_best.pt")

    # also save final epoch
    torch.save(model.state_dict(), "checkpoints/model_unconditional_final.pt")
    print(f"done. best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()