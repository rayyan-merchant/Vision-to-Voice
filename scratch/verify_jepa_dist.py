import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import numpy as np
from src.predictor import load_jepa, prediction_error, TrajectoryDataset

def test_jepa_quality():
    cfg = yaml.safe_load(open("config/config.yaml"))
    jepa_cfg = cfg["jepa"]
    dino_cfg = cfg["dino"]
    
    print(f"Loading JEPA from {jepa_cfg['model_path']}...")
    model = load_jepa(
        jepa_cfg["model_path"],
        z_dim=dino_cfg["cls_dim"],
        hidden=jepa_cfg["hidden_dim"]
    )
    model.eval()
    
    data_path = cfg["trajectory"]["full_path"]
    dataset = TrajectoryDataset(data_path)
    
    surprises = []
    # Test first 100 samples
    for i in range(min(100, len(dataset))):
        z_t, a_oh, z_t1 = dataset[i]
        # We need the action string. TrajectoryDataset doesn't store it, 
        # it converts to one-hot. But we can recover it or just use the one-hot directly
        # if we modify prediction_error or just call model directly.
        with torch.no_grad():
            pred = model(z_t.unsqueeze(0), a_oh.unsqueeze(0))
            err = F.mse_loss(pred.squeeze(), z_t1).item()
            surprises.append(err)
            
    print(f"Mean surprise on training data: {np.mean(surprises):.6f}")
    print(f"Std surprise on training data:  {np.std(surprises):.6f}")
    print(f"Max surprise on training data:  {np.max(surprises):.6f}")
    print(f"Min surprise on training data:  {np.min(surprises):.6f}")

    # Test random noise
    random_surprises = []
    for _ in range(100):
        z1 = torch.randn(384)
        z2 = torch.randn(384)
        a = torch.zeros(1, 5)
        a[0, 0] = 1.0
        with torch.no_grad():
            pred = model(z1.unsqueeze(0), a)
            err = F.mse_loss(pred.squeeze(), z2).item()
            random_surprises.append(err)
            
    print(f"Mean surprise on RANDOM noise: {np.mean(random_surprises):.6f}")

    # Test state sensitivity: same action, different z_t
    z_t, a_oh, z_t1 = dataset[0]
    z_other, _, _ = dataset[100]
    with torch.no_grad():
        pred1 = model(z_t.unsqueeze(0), a_oh.unsqueeze(0))
        pred2 = model(z_other.unsqueeze(0), a_oh.unsqueeze(0))
        diff = torch.norm(pred1 - pred2).item()
    print(f"Prediction difference for different z_t (same action): {diff:.6f}")
    if diff < 1e-4:
        print("[!] VULNERABILITY: JEPA ignores starting state z_t!")
    else:
        print("[OK] JEPA is sensitive to starting state.")

if __name__ == "__main__":
    test_jepa_quality()
