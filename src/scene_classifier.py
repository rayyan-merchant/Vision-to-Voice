# ============================================================
# scene_classifier.py  |  Track B  |  Owner: Syeda
# Scene Context MLP — Paper 3 (Epstein et al. 2019)
# Action-place compatibility is LEARNED from DINOv3 features.
#
# "Why can't I dance in the mall?" — because a learned model
# knows that some actions are inappropriate in certain spaces.
#
# This module:
#   1. Loads labelled scene data (DINOv3 CLS + scores)
#   2. Trains a small MLP to predict action appropriateness
#   3. Provides filter() to override bad action choices
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import yaml
import os
import copy
import numpy as np
import matplotlib.pyplot as plt


# ■■ Action index mapping ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
ACTION_IDX = {"move_fast": 0, "stop_wait": 1, "navigate": 2}
IDX_ACTION = {v: k for k, v in ACTION_IDX.items()}


# ■■ Dataset ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class SceneLabelDataset(Dataset):
    """
    Loads labelled frames from labels.json for training
    the Scene Context MLP.

    Each entry has:
        z      : 384-dim DINOv3 CLS token embedding
        scores : {"move_fast": float, "stop_wait": float, "navigate": float}

    Returns:
        (z_tensor, target_tensor) where target is [move_fast, stop_wait, navigate]
    """

    def __init__(self, labels_path="data/scene_labels/labels.json",
                 split="train", val_ratio=0.2, seed=42):
        with open(labels_path, "r") as f:
            data = json.load(f)

        # Deterministic shuffle for reproducible train/val split
        rng = np.random.RandomState(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)

        n_val = int(len(data) * val_ratio)

        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.samples = []
        for i in selected:
            d = data[i]
            z = torch.tensor(d["z"], dtype=torch.float32)
            scores = d["scores"]
            target = torch.tensor(
                [scores["move_fast"], scores["stop_wait"], scores["navigate"]],
                dtype=torch.float32
            )
            self.samples.append((z, target))

        print(f"[SceneLabelDataset] {split}: {len(self.samples)} samples "
              f"(from {len(data)} total, val_ratio={val_ratio})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ■■ Model ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class SceneContextMLP(nn.Module):
    """
    Scene Context MLP — Paper 3 implementation.

    Architecture:
        Linear(384 → 256) → ReLU → Dropout(0.3)
        → Linear(256 → 64) → ReLU
        → Linear(64 → 3) → Sigmoid

    Output: [move_fast, stop_wait, navigate] scores in [0, 1]

    The filter() method checks if a proposed action is appropriate
    for the current scene. If not, it suggests the best alternative.
    """

    def __init__(self, input_dim=384, hidden1=256, hidden2=64,
                 dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 3),
            nn.Sigmoid(),
        )

    def forward(self, cls_token):
        """
        Forward pass. Handles both 1D (384,) and 2D (B, 384) input.
        """
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)
        return self.net(cls_token)

    def filter(self, proposed_action, cls_token, threshold=0.4):
        """
        Paper 3 action-place compatibility filter.

        If the proposed action's score is below threshold,
        return the highest-scoring action instead.
        Otherwise return the proposed action unchanged.

        Args:
            proposed_action: str — one of "move_fast", "stop_wait", "navigate"
            cls_token: torch.Tensor — (384,) or (B, 384) DINOv3 CLS token
            threshold: float — minimum acceptable score

        Returns:
            str — either proposed_action or the highest-scoring alternative
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(cls_token).squeeze(0)  # (3,)

        action_idx = ACTION_IDX[proposed_action]
        action_score = scores[action_idx].item()

        if action_score < threshold:
            best_idx = scores.argmax().item()
            return IDX_ACTION[best_idx]

        return proposed_action


# ■■ Training ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def train_scene_mlp(labels_path="data/scene_labels/labels.json",
                    model_out="models/scene_mlp.pt",
                    input_dim=384, hidden1=256, hidden2=64,
                    dropout=0.3, lr=1e-3, batch_size=32, epochs=60):
    """
    Train the Scene Context MLP.

    - Computes class weights: weight = 1 / class_frequency
    - Uses nn.BCELoss(weight=weights)
    - Saves BEST model (lowest val loss), not last epoch
    - Prints val loss every 10 epochs
    """
    os.makedirs(os.path.dirname(model_out) if os.path.dirname(model_out)
                else "models", exist_ok=True)

    # Ensure numeric types (YAML may parse scientific notation as str)
    lr = float(lr)
    batch_size = int(batch_size)
    epochs = int(epochs)

    # Load datasets
    train_ds = SceneLabelDataset(labels_path, split="train")
    val_ds   = SceneLabelDataset(labels_path, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Compute class weights: weight = 1 / mean_target per output ──
    all_targets = torch.stack([t for _, t in train_ds.samples])  # (N, 3)
    class_freq = all_targets.mean(dim=0).clamp(min=0.05)         # (3,)
    weights = 1.0 / class_freq
    weights = weights / weights.sum() * 3.0   # normalize so sum = 3

    print(f"\n[Class weights] move_fast={weights[0]:.3f}, "
          f"stop_wait={weights[1]:.3f}, navigate={weights[2]:.3f}")

    # ── Model, optimizer, loss ──
    model   = SceneContextMLP(input_dim=input_dim, hidden1=hidden1,
                              hidden2=hidden2, dropout=dropout)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(weight=weights.unsqueeze(0))  # (1,3) broadcasts

    best_val_loss = float("inf")
    train_history = []
    val_history   = []

    print(f"\n{'='*50}")
    print(f"Scene MLP Training: {epochs} epochs, lr={lr}, "
          f"batch={batch_size}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*50}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_train, n_train = 0.0, 0
        for z, target in train_loader:
            opt.zero_grad()
            pred = model(z)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            total_train += loss.item()
            n_train += 1

        avg_train = total_train / max(n_train, 1)
        train_history.append(avg_train)

        # ── Validate ──
        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for z, target in val_loader:
                pred = model(z)
                loss = loss_fn(pred, target)
                total_val += loss.item()
                n_val += 1

        avg_val = total_val / max(n_val, 1)
        val_history.append(avg_val)

        # Save best model (to disk immediately)
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_out)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            tag = " *best*" if is_best else ""
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}{tag}")

    print(f"\n[SAVED] Best model -> {model_out} "
          f"(best val_loss={best_val_loss:.6f})")

    # ── Save loss curve ──
    curve_path = os.path.join(
        os.path.dirname(model_out) or "models",
        "loss_curve_scene_mlp.png"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, color="#6366F1", linewidth=2, label="Train Loss")
    plt.plot(val_history,   color="#F59E0B", linewidth=2, label="Val Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.title("Scene MLP Training — Paper 3 Action-Place Compatibility",
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[SAVED] Loss curve -> {curve_path}")

    return model, train_history, val_history


# ■■ Evaluation ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def evaluate_scene_mlp(labels_path="data/scene_labels/labels.json",
                       model_path="models/scene_mlp.pt",
                       input_dim=384, hidden1=256, hidden2=64,
                       dropout=0.3):
    """
    Evaluate the trained Scene MLP on the validation split.

    - Prints accuracy and full classification report
    - Saves report to models/scene_mlp_report.txt
    - Returns (accuracy, report_string)
    """
    from sklearn.metrics import classification_report, accuracy_score

    val_ds = SceneLabelDataset(labels_path, split="val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=0)

    # Load trained model
    model = SceneContextMLP(input_dim=input_dim, hidden1=hidden1,
                            hidden2=hidden2, dropout=dropout)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"\n[Eval] Loaded model from {model_path}")

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for z, target in val_loader:
            pred = model(z)
            all_preds.append(pred)
            all_targets.append(target)

    preds   = torch.cat(all_preds).numpy()     # (N, 3)
    targets = torch.cat(all_targets).numpy()   # (N, 3)

    # Binarize at 0.5 for classification metrics
    pred_bin   = (preds > 0.5).astype(int)
    target_bin = (targets > 0.5).astype(int)

    action_names = ["move_fast", "stop_wait", "navigate"]
    accuracy = accuracy_score(target_bin.flatten(), pred_bin.flatten())

    report = classification_report(
        target_bin, pred_bin,
        target_names=action_names,
        zero_division=0
    )

    # Regression quality (continuous)
    mse = float(np.mean((preds - targets) ** 2))

    result_text = (
        f"Scene MLP Evaluation Report\n"
        f"{'='*50}\n"
        f"Model: {model_path}\n"
        f"Val samples: {len(val_ds)}\n"
        f"{'='*50}\n\n"
        f"Overall Accuracy (binarized @ 0.5): {accuracy:.4f}\n"
        f"Mean Squared Error (continuous):    {mse:.6f}\n\n"
        f"Classification Report:\n{report}\n"
    )

    print(result_text)

    # Save report
    report_path = os.path.join(
        os.path.dirname(model_path) or "models",
        "scene_mlp_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(result_text)
    print(f"[SAVED] Report -> {report_path}")

    return accuracy, result_text


# ■■ Entry point ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
if __name__ == "__main__":
    cfg     = yaml.safe_load(open("config/config.yaml"))
    mlp_cfg = cfg["scene_mlp"]

    # ── Train ──
    model, train_hist, val_hist = train_scene_mlp(
        labels_path=mlp_cfg["label_path"],
        model_out=mlp_cfg["model_path"],
        input_dim=cfg["dino"]["cls_dim"],
        hidden1=mlp_cfg["hidden1"],
        hidden2=mlp_cfg["hidden2"],
        dropout=mlp_cfg["dropout"],
        lr=mlp_cfg["lr"],
        batch_size=mlp_cfg["batch_size"],
        epochs=mlp_cfg["epochs"],
    )

    # ── Evaluate ──
    accuracy, report = evaluate_scene_mlp(
        labels_path=mlp_cfg["label_path"],
        model_path=mlp_cfg["model_path"],
        input_dim=cfg["dino"]["cls_dim"],
        hidden1=mlp_cfg["hidden1"],
        hidden2=mlp_cfg["hidden2"],
        dropout=mlp_cfg["dropout"],
    )

    target = float(mlp_cfg["accuracy_target"])
    if accuracy >= target:
        print(f"\n[OK] Target met: accuracy {accuracy:.4f} >= {target}")
    else:
        print(f"\n[!] WARNING: Accuracy {accuracy:.4f} < {target}")
        print("  -> Try: more labelled data, lr=5e-4, or more epochs")

    print("\n" + "=" * 50)
    print("scene_classifier.py READY")
    print("=" * 50)
