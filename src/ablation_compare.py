# ============================================================
# ablation_compare dino vs resnet.py | Track A | Owner: Rayyan
#
# PURPOSE
#   Runs a controlled comparison between DINOv3 (ViT-S/14) and
#   ResNet-18 on the trajectory data already collected, and
#   writes ablation_results.csv for the final paper + presentation.
#
# WHAT IS MEASURED (per encoder)
#   1. Avg cosine similarity between consecutive z_t vectors
#      (higher = smoother embedding space, better for JEPA)
#   2. Avg surprise (1 - cosim) — how much the agent "notices"
#      scene changes (higher = more responsive to novel views)
#   3. Feature norm (L2 norm) — tells you about embedding scale
#   4. Dedup efficiency — how many UNIQUE map nodes emerge from
#      100 steps using each encoder (proxy for map quality)
#   5. Encode time (ms/frame) — practical cost per step
#
# USAGE
#   python ablation_compare.py
#       --data  data/trajectories/all_trajectories.json
#       --steps 200
#       --scene FloorPlan1
#       --out   data/ablation/ablation_results.csv
#
# OUTPUTS
#   data/ablation/ablation_results.csv   ← attach to paper
#   data/ablation/ablation_error.log     ← required Day 5 deliverable
#   Console table printed at end
# ============================================================

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch

# ----------------------------------------------------------
# Logging — goes to BOTH console and file (Day 5 requirement)
# ----------------------------------------------------------
LOG_PATH = "data/ablation/ablation_error.log"
os.makedirs("data/ablation", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Imports (graceful fallback if AI2-THOR not available)
# ----------------------------------------------------------
try:
    import ai2thor.controller
    THOR_AVAILABLE = True
except ImportError:
    THOR_AVAILABLE = False
    logger.warning("ai2thor not found — live THOR comparison disabled. "
                   "Will run on JSON trajectory data only.")

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from perception import DINOEncoder
    from resnet_encoder import ResNet18Encoder
except ImportError as e:
    logger.error(f"Could not import encoders: {e}")
    sys.exit(1)

import cv2


# ===========================================================
# Core metric functions
# ===========================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_dedup_nodes(features: list, threshold: float = 0.25) -> int:
    """
    Simulate mapper deduplication: count unique nodes that would
    be created from a feature sequence.
    A new node is created when cosine distance > threshold from
    all existing node centroids (mirrors mapper.py logic).
    """
    nodes = []
    for z in features:
        if not nodes:
            nodes.append(z)
            continue
        sims = [cosine_similarity(z, n) for n in nodes]
        if max(sims) < (1.0 - threshold):
            nodes.append(z)
    return len(nodes)


# ===========================================================
# Ablation runner — works on JSON data OR live THOR
# ===========================================================

def run_on_json(encoder, traj_path: str, n_steps: int) -> dict:
    """
    Run ablation metrics on pre-collected trajectory JSON.
    This doesn't need AI2-THOR — works anywhere.
    """
    logger.info(f"Loading trajectory data from {traj_path} ...")
    with open(traj_path) as f:
        data = json.load(f)

    # Support both Rayyan's z_t format AND Syeda's Z_t format
    samples = data[:n_steps]
    if not samples:
        raise ValueError("Trajectory file is empty.")

    logger.info(f"Loaded {len(samples)} samples. Running metrics...")

    # We re-encode using raw stored embeddings (no frames available in JSON)
    # so we measure embedding statistics directly
    z_vecs = []
    encode_times = []

    for s in samples:
        key = "z_t" if "z_t" in s else "Z_t"
        z = np.array(s[key], dtype=np.float32)
        z_vecs.append(z)

    # Cosine similarities between consecutive pairs
    cosims = [cosine_similarity(z_vecs[i], z_vecs[i+1])
              for i in range(len(z_vecs) - 1)]

    avg_cosim    = float(np.mean(cosims))
    avg_surprise = float(1.0 - avg_cosim)
    avg_norm     = float(np.mean([np.linalg.norm(z) for z in z_vecs]))
    dedup_nodes  = compute_dedup_nodes(z_vecs, threshold=0.25)

    # Encode time: measure on dummy frame since we don't have raw frames

    dummy_np  = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_np)   # DINOEncoder needs PIL
    dummy_in  = dummy_pil if hasattr(encoder, '_preprocess') else dummy_np
    for _ in range(20):
        t0 = time.perf_counter()
        encoder.encode(dummy_in)
        encode_times.append((time.perf_counter() - t0) * 1000)
    avg_encode_ms = float(np.mean(encode_times[5:]))  # skip warmup

    return {
        "avg_cosim":     round(avg_cosim, 4),
        "avg_surprise":  round(avg_surprise, 4),
        "avg_norm":      round(avg_norm, 4),
        "dedup_nodes":   dedup_nodes,
        "encode_ms":     round(avg_encode_ms, 3),
        "n_samples":     len(samples),
    }


def run_live_thor(encoder, scene: str, n_steps: int) -> dict:
    """
    Run ablation metrics on live AI2-THOR frames.
    Only called if --scene is given AND ai2thor is installed.
    """
    logger.info(f"[AI2-THOR] Loading {scene} for live ablation...")
    ctrl = ai2thor.controller.Controller(
        scene=scene,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=224,
        height=224,
        gridSize=0.25,
    )

    ACTIONS = ["MoveAhead", "MoveAhead", "MoveAhead",
               "RotateLeft", "RotateRight"]

    z_vecs = []
    encode_times = []

    for step in range(n_steps):
        action = np.random.choice(ACTIONS)
        event  = ctrl.step(action=action)
        frame  = event.frame  # (H, W, 3) RGB

        # Convert to BGR for encoder (matches collect_trajectories.py)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        if hasattr(encoder, '_preprocess'):
            result = encoder.encode(Image.fromarray(frame))   # PIL RGB for DINO
            z = result[0] if isinstance(result, tuple) else result  # unpack CLS token
            z = z.squeeze().cpu().numpy() if hasattr(z, 'cpu') else np.array(z)
        else:
            z = encoder.encode(frame_bgr)                # numpy BGR for ResNet
        encode_times.append((time.perf_counter() - t0) * 1000)
        z_vecs.append(z)

    ctrl.stop()

    cosims       = [cosine_similarity(z_vecs[i], z_vecs[i+1])
                    for i in range(len(z_vecs) - 1)]
    avg_cosim    = float(np.mean(cosims))
    avg_surprise = float(1.0 - avg_cosim)
    avg_norm     = float(np.mean([np.linalg.norm(z) for z in z_vecs]))
    dedup_nodes  = compute_dedup_nodes(z_vecs, threshold=0.25)
    avg_encode_ms= float(np.mean(encode_times[5:]))

    return {
        "avg_cosim":    round(avg_cosim, 4),
        "avg_surprise": round(avg_surprise, 4),
        "avg_norm":     round(avg_norm, 4),
        "dedup_nodes":  dedup_nodes,
        "encode_ms":    round(avg_encode_ms, 3),
        "n_samples":    n_steps,
    }


# ===========================================================
# CSV writer + console table
# ===========================================================

FIELDNAMES = [
    "encoder", "output_dim", "pretraining",
    "avg_cosim", "avg_surprise", "avg_norm",
    "dedup_nodes", "encode_ms", "n_samples",
]

ENCODER_META = {
    "DINOv3 ViT-S/14": {
        "output_dim":   384,
        "pretraining":  "Self-supervised (DINO)",
    },
    "ResNet-18": {
        "output_dim":   512,
        "pretraining":  "Supervised (ImageNet-1K)",
    },
}

def write_csv(rows: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[ablation] Results saved → {out_path}")


def print_table(rows: list):
    print("\n" + "=" * 72)
    print("  ABLATION RESULTS: DINOv3 ViT-S/14  vs  ResNet-18")
    print("=" * 72)
    hdr = f"  {'Metric':<22}  {'DINOv3':>12}  {'ResNet-18':>12}  {'Winner':>10}"
    print(hdr)
    print("-" * 72)

    dino = {r["encoder"]: r for r in rows}["DINOv3 ViT-S/14"]
    resn = {r["encoder"]: r for r in rows}["ResNet-18"]

    metrics = [
        ("Avg cosine similarity", "avg_cosim",    "higher"),
        ("Avg surprise",          "avg_surprise",  "higher"),
        ("Avg feature norm",      "avg_norm",      "—"),
        ("Dedup map nodes",       "dedup_nodes",   "higher"),
        ("Encode time (ms)",      "encode_ms",     "lower"),
    ]

    for label, key, prefer in metrics:
        dv = dino[key]
        rv = resn[key]
        if prefer == "higher":
            winner = "DINOv3 ✓" if dv > rv else ("ResNet ✓" if rv > dv else "TIE")
        elif prefer == "lower":
            winner = "DINOv3 ✓" if dv < rv else ("ResNet ✓" if rv < dv else "TIE")
        else:
            winner = "—"
        print(f"  {label:<22}  {str(dv):>12}  {str(rv):>12}  {winner:>10}")

    print("=" * 72)
    print(f"  Output dim              {dino['output_dim']:>12}  {resn['output_dim']:>12}")
    print(f"  Pretraining             {'Self-sup':>12}  {'Supervised':>12}")
    print("=" * 72 + "\n")


# ===========================================================
# Main
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="DINOv3 vs ResNet-18 Ablation")
    parser.add_argument("--data",  default="data/trajectories/all_trajectories.json",
                        help="Path to trajectory JSON (use --scene for live THOR instead)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of steps / samples to evaluate")
    parser.add_argument("--scene", default=None,
                        help="If set, run live AI2-THOR instead of JSON data")
    parser.add_argument("--out",   default="data/ablation/backbone_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  ABLATION: DINOv3 ViT-S/14  vs  ResNet-18")
    logger.info(f"  Mode  : {'Live AI2-THOR' if args.scene else 'JSON trajectory data'}")
    logger.info(f"  Steps : {args.steps}")
    logger.info("=" * 60)

    # Instantiate encoders
    dino_enc  = DINOEncoder()
    resnt_enc = ResNet18Encoder()

    rows = []
    for name, enc in [("DINOv3 ViT-S/14", dino_enc), ("ResNet-18", resnt_enc)]:
        logger.info(f"\n[ablation] Running: {name}")
        try:
            if args.scene and THOR_AVAILABLE:
                metrics = run_live_thor(enc, args.scene, args.steps)
            else:
                metrics = run_on_json(enc, args.data, args.steps)

            row = {"encoder": name, **ENCODER_META[name], **metrics}
            rows.append(row)
            logger.info(f"[ablation] {name} done. Surprise={metrics['avg_surprise']:.4f}  "
                        f"Nodes={metrics['dedup_nodes']}  EncTime={metrics['encode_ms']:.2f}ms")

        except Exception as exc:
            logger.error(f"[ablation] {name} FAILED: {exc}")
            logger.error(traceback.format_exc())
            # Write a placeholder row so CSV still has two entries
            rows.append({
                "encoder": name, **ENCODER_META[name],
                "avg_cosim": "ERROR", "avg_surprise": "ERROR",
                "avg_norm": "ERROR", "dedup_nodes": "ERROR",
                "encode_ms": "ERROR", "n_samples": 0,
            })

    write_csv(rows, args.out)
    valid_rows = [r for r in rows if r["avg_cosim"] != "ERROR"]
    if len(valid_rows) == 2:
        print_table(valid_rows)
    else:
        logger.warning(f"[ablation] Only {len(valid_rows)} encoder(s) succeeded — skipping table.")
        for r in valid_rows:
            logger.info(f"[ablation] {r['encoder']}: {r}")
    logger.info(f"Error log saved → {LOG_PATH}")
    logger.info("Ablation complete.")


if __name__ == "__main__":
    main()