"""
Day 3 -- Full JEPA Training on 5000-sample dataset
Train JEPA, validate, save model + loss curve, run sanity checks.
"""
import torch
import torch.nn as nn
import yaml
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import (
    train_jepa, load_jepa, prediction_error,
    TrajectoryDataset, JEPAPredictor
)
from torch.utils.data import DataLoader

# Load config
cfg = yaml.safe_load(open("config/config.yaml"))
jepa_cfg = cfg["jepa"]
traj_cfg = cfg["trajectory"]
dino_cfg = cfg["dino"]

print("=" * 60)
print("DAY 3: Full JEPA Training (5000 samples, 100 epochs)")
print("=" * 60)

# ============================================================
# STEP 1: Full training on all_trajectories.json
# ============================================================
data_path = traj_cfg["full_path"]
model_path = jepa_cfg["model_path"]

model, history = train_jepa(
    data_path,
    model_out=model_path,
    z_dim=dino_cfg["cls_dim"],
    hidden=jepa_cfg["hidden_dim"],
    lr=jepa_cfg["lr"],
    batch_size=jepa_cfg["batch_size"],
    epochs=jepa_cfg["epochs"],
)

# ============================================================
# STEP 2: Validation on hold-out 10%
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Validation on hold-out 10%")
print("=" * 60)

dataset = TrajectoryDataset(data_path)
n_val = len(dataset) // 10
val_indices = range(len(dataset) - n_val, len(dataset))

loaded_model = load_jepa(
    model_path,
    z_dim=dino_cfg["cls_dim"],
    hidden=jepa_cfg["hidden_dim"],
)
loaded_model.eval()

loss_fn = nn.MSELoss()
val_loss = 0.0
for i in val_indices:
    z_t, a, z_t1 = dataset[i]
    with torch.no_grad():
        pred = loaded_model(z_t.unsqueeze(0), a.unsqueeze(0))
        val_loss += loss_fn(pred.squeeze(), z_t1).item()

val_loss /= n_val
print(f"  Validation MSE: {val_loss:.6f}  (target: < 0.20)")
if val_loss < 0.20:
    print("  [OK] Validation target met!")
else:
    print("  [!] Validation loss above 0.20 -- may need more epochs or lr tuning")

# ============================================================
# STEP 3: Sanity check -- surprise on random vs training data
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Surprise sanity check")
print("=" * 60)

# High surprise sample (random noise -- model has never seen this)
z_random = torch.randn(384)
z_random_next = torch.randn(384)
err_high = prediction_error(loaded_model, z_random, "MoveAhead", z_random_next)

# Low surprise sample (from dataset -- model has trained on similar data)
z_t, a_oh, z_t1 = dataset[0]
# Find the action string for the first sample
data_raw = json.load(open(data_path))
action_str = data_raw[0]["action"]
err_low = prediction_error(loaded_model, z_t, action_str, z_t1)

print(f"  Surprise on RANDOM input:     {err_high:.6f}  (should be HIGH)")
print(f"  Surprise on TRAINING sample:  {err_low:.6f}  (should be LOW)")

if err_high > err_low:
    print("  [OK] Sanity check passed! Random > Training surprise")
else:
    print("  [!] WARNING: surprise not calibrated -- investigate data/model")

# ============================================================
# STEP 4: Test all 5 actions produce different predictions
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Action differentiation check")
print("=" * 60)

z_test = z_t.clone()
actions = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]
predictions = []
for act in actions:
    err = prediction_error(loaded_model, z_test, act, z_t1)
    predictions.append(err)
    print(f"  {act:15s}  surprise = {err:.6f}")

unique_preds = len(set(round(p, 6) for p in predictions))
if unique_preds > 1:
    print(f"  [OK] {unique_preds}/5 unique predictions -- actions are differentiated")
else:
    print("  [!] All actions produce same prediction -- model may not use action input")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DAY 3 SUMMARY")
print("=" * 60)
print(f"  Final training loss:  {history[-1]:.6f}  {'[OK]' if history[-1] < 0.15 else '[!] above target'}")
print(f"  Validation MSE:       {val_loss:.6f}  {'[OK]' if val_loss < 0.20 else '[!] above target'}")
print(f"  Surprise calibrated:  {'YES' if err_high > err_low else 'NO'}")
print(f"  Model saved:          {model_path}")
print(f"  Loss curve saved:     models/loss_curve_jepa.png")
print("=" * 60)
