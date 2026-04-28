# ============================================================
# predictor.py  |  Track B  |  Owner: Syeda
# JEPA-lite World Model — predicts next latent state from
# current DINOv3 CLS embedding + action one-hot.
#
# Paper 1 insight (Mirowski et al. 2017):
#   Auxiliary predictive tasks force better spatial representations.
#   JEPA is the 2024 implementation of that insight — predicting
#   in latent embedding space, not pixel space.
#
# Surprise = MSE(predicted_z_t1, actual_z_t1)
#   High surprise -> unexpected transition -> interesting frontier
#   -> triggers YOLOE (Riya) + biases frontier selection
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import yaml
import os
import matplotlib.pyplot as plt


# ■■ Action encoding ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
ACT_MAP = {
    "MoveAhead":   0,
    "RotateLeft":  1,
    "RotateRight": 2,
    "LookUp":      3,
    "LookDown":    4,
}


def action_onehot(action_str):
    """Convert action string to one-hot tensor (5-dim)."""
    v = torch.zeros(5)
    if action_str in ACT_MAP:
        v[ACT_MAP[action_str]] = 1.0
    return v


# ■■ Dataset ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class TrajectoryDataset(Dataset):
    """
    Loads (z_t, action, z_t1) triples from the shared JSON
    exported by Rayyan's collect_trajectories.py.

    Each sample is a dict with keys:
        z_t    : [384 floats]   — DINOv3 CLS token at step t
        action : str            — one of 5 valid actions
        z_t1   : [384 floats]   — DINOv3 CLS token at step t+1
        pos    : [x, z]         — agent position (not used here)
        rot    : float          — agent rotation  (not used here)
        scene  : str            — AI2-THOR scene  (not used here)
    """

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []
        skipped = 0
        for d in data:
            if d["action"] not in ACT_MAP:
                skipped += 1
                continue
            z_t  = torch.tensor(d["z_t"],  dtype=torch.float32)
            z_t1 = torch.tensor(d["z_t1"], dtype=torch.float32)
            act  = action_onehot(d["action"])
            self.samples.append((z_t, act, z_t1))

        print(f"[TrajectoryDataset] Loaded {len(self.samples)} samples "
              f"from {json_path}" +
              (f" (skipped {skipped})" if skipped else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ■■ JEPA Model ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class JEPAPredictor(nn.Module):
    """
    Joint Embedding Predictive Architecture (lite).

    Input:  z_t (384) + action one-hot (5) = 389 dim
    Output: predicted z_t1 (384 dim)

    Architecture:
        Linear(389 -> hidden) -> LayerNorm -> GELU
        -> Linear(hidden -> hidden) -> GELU
        -> Linear(hidden -> 384)

    Paper 1 insight: predicting next latent state forces
    the system to build good spatial representations.
    JEPA predicts in meaning space, not pixel space.
    """

    def __init__(self, z_dim=384, act_dim=5, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, a):
        """z: (B, 384), a: (B, 5) -> predicted_z: (B, 384)"""
        inp = torch.cat([z, a], dim=-1)
        return self.net(inp)


# ■■ Training Function ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def train_jepa(data_path,
               model_out="models/jepa_predictor.pt",
               z_dim=384, act_dim=5, hidden=512,
               lr=1e-3, batch_size=64, epochs=60):
    """
    Train the JEPA predictor on trajectory data.

    Healthy training behaviour:
        Epoch  1: loss ~0.28–0.35
        Epoch 20: loss ~0.15–0.20
        Epoch 60: loss ~0.05–0.12  (target: < 0.15)

    If stuck above 0.20 after 30 epochs:
        -> Try lr=1e-4
        -> Try hidden=768
        -> Verify tensor shapes: z_t must be (B,384), action must be (B,5)
    """
    os.makedirs(os.path.dirname(model_out) if os.path.dirname(model_out) else "models",
                exist_ok=True)

    # Ensure numeric types (YAML may parse scientific notation as string)
    lr = float(lr)
    batch_size = int(batch_size)
    epochs = int(epochs)

    dataset = TrajectoryDataset(data_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=0)

    model   = JEPAPredictor(z_dim=z_dim, act_dim=act_dim, hidden=hidden)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )   
    loss_fn = nn.MSELoss()

    history = []

    print(f"\n{'='*50}")
    print(f"JEPA Training: {epochs} epochs, lr={lr}, "
          f"batch={batch_size}, hidden={hidden}")
    print(f"Dataset: {len(dataset)} samples -> {len(loader)} batches")
    print(f"{'='*50}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for z_t, a, z_t1 in loader:
            opt.zero_grad()
            pred = model(z_t, a)
            loss = loss_fn(pred, z_t1)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        history.append(avg)
        scheduler.step(avg)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  loss = {avg:.6f}")

    # Save model
    torch.save(model.state_dict(), model_out)
    print(f"\n[SAVED] Model -> {model_out}  (final loss = {history[-1]:.6f})")

    # Save loss curve
    curve_path = os.path.join(os.path.dirname(model_out), "loss_curve_jepa.png")
    plt.figure(figsize=(10, 5))
    plt.plot(history, color="#6366F1", linewidth=2, label="Training MSE")
    plt.axhline(0.15, color="#EF4444", linestyle="--", linewidth=1.5,
                label="Target (< 0.15)")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("JEPA Training Loss - Paper 1 World Model", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[SAVED] Loss curve -> {curve_path}")

    if history[-1] > 0.15:
        print("\n[!] WARNING: Final loss above 0.15!")
        print("  -> Try: lr=1e-4, hidden=768, or check data quality")
    else:
        print(f"\n[OK] Target met: final loss {history[-1]:.6f} < 0.15")

    return model, history


# ■■ Inference / Surprise ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def load_jepa(model_path="models/jepa_predictor.pt",
              z_dim=384, act_dim=5, hidden=512):
    """Load a trained JEPA model for inference."""
    model = JEPAPredictor(z_dim=z_dim, act_dim=act_dim, hidden=hidden)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"[JEPA] Loaded from {model_path}")
    return model


def prediction_error(predictor, z_t, action_str, z_t1_actual):
    """
    Compute surprise score: MSE between JEPA-predicted and
    actual next DINOv3 embedding.

    High error = unexpected transition = interesting frontier
               = trigger YOLOE + bias frontier selection.

    Args:
        predictor   : trained JEPAPredictor (eval mode)
        z_t         : (384,) current DINOv3 CLS embedding
        action_str  : action string, e.g. "MoveAhead"
        z_t1_actual : (384,) actual next DINOv3 CLS embedding

    Returns:
        float — surprise score (MSE)
    """
    a  = action_onehot(action_str).unsqueeze(0)          # (1, 5)
    zt = z_t.unsqueeze(0) if z_t.dim() == 1 else z_t    # (1, 384)

    with torch.no_grad():
        z_pred = predictor(zt, a)

    return F.mse_loss(z_pred.squeeze(), z_t1_actual).item()


# ■■ Quick self-test ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
if __name__ == "__main__":
    # Load config
    cfg = yaml.safe_load(open("config/config.yaml"))
    jepa_cfg = cfg["jepa"]
    traj_cfg = cfg["trajectory"]

    print("=" * 60)
    print("predictor.py -- Day 2 Quick Test (mock data)")
    print("=" * 60)

    # Use mock data for quick test
    data_path = traj_cfg["mock_path"]

    model, history = train_jepa(
        data_path,
        model_out=jepa_cfg["model_path"],
        z_dim=cfg["dino"]["cls_dim"],
        hidden=jepa_cfg["hidden_dim"],
        lr=jepa_cfg["lr"],
        batch_size=jepa_cfg["batch_size"],
        epochs=10,  # quick test — use 10 epochs
    )

    # Test prediction_error
    dummy_z  = torch.randn(384)
    dummy_z1 = torch.randn(384)
    err = prediction_error(model, dummy_z, "MoveAhead", dummy_z1)
    print(f"\n[TEST] prediction_error on random input: {err:.4f}")

    # Test load_jepa
    loaded = load_jepa(jepa_cfg["model_path"],
                       z_dim=cfg["dino"]["cls_dim"],
                       hidden=jepa_cfg["hidden_dim"])
    err2 = prediction_error(loaded, dummy_z, "MoveAhead", dummy_z1)
    print(f"[TEST] prediction_error after reload:    {err2:.4f}")
    assert abs(err - err2) < 1e-5, "Model reload produced different results!"

    print("\n" + "=" * 60)
    print("predictor.py READY [OK]")
    print("=" * 60)
