# ============================================================
# ablation_paper3.py | Track A | Owner: Rayyan
# Day 6 — Ablation 2: Paper 3 Filter On vs Off
#
# WHAT THIS MEASURES
#   Condition A (filter ON):  scene_clf.filter() applied each step
#   Condition B (filter OFF): raw action always used, no filtering
#
#   Per step: if chosen action has appropriateness score < 0.3
#             → count as "inappropriate action"
#
#   Report: inappropriate_actions per 10 steps for each condition
#
# OUTPUT
#   Appends to data/ablation/ablation_results.csv:
#     ablation, value
#     paper3_on,  X.XX
#     paper3_off, Y.YY
#
# USAGE
#   python src/ablation_paper3.py --scene FloorPlan1
# ============================================================

import argparse
import csv
import os
import random
import sys
import logging
import numpy as np
from PIL import Image

import torch
import yaml

# ── logging ──────────────────────────────────────────────────
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/ablation", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/logs/ablation_paper3.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")
from src.perception       import DINOEncoder
from src.scene_classifier import SceneContextMLP

SCENE_ACTIONS    = ["move_fast", "stop_wait", "navigate"]
INAPPROP_THRESH  = 0.3   # score below this = inappropriate
EPISODES         = 20
STEPS_PER_EP     = 10


# ── runner ───────────────────────────────────────────────────

def run_paper3_ablation(controller, encoder, scene_clf, use_filter: bool):
    """
    Run EPISODES × STEPS_PER_EP steps.
    Returns total inappropriate action count across all steps.
    """
    total_steps        = EPISODES * STEPS_PER_EP
    inappropriate_count = 0
    label = "ON" if use_filter else "OFF"

    logger.info(f"[paper3-{label}] Starting {EPISODES} episodes × "
                f"{STEPS_PER_EP} steps (filter={use_filter})")

    for ep in range(EPISODES):
        controller.reset()

        for step in range(STEPS_PER_EP):
            frame      = Image.fromarray(controller.last_event.frame)
            cls, _     = encoder.encode(frame)

            # Raw proposed scene action (random for isolation)
            proposed   = random.choice(SCENE_ACTIONS)

            if use_filter:
                filtered = scene_clf.filter(proposed, cls, threshold=0.4)
                final    = filtered
            else:
                final    = proposed

            # Score the FINAL chosen action for appropriateness
            with torch.no_grad():
                scores = scene_clf(cls).squeeze()   # (3,) probabilities

            action_idx = SCENE_ACTIONS.index(final)
            score      = float(scores[action_idx])

            is_inappropriate = score < INAPPROP_THRESH
            if is_inappropriate:
                inappropriate_count += 1
                logger.debug(f"[paper3-{label}] ep={ep} step={step} "
                             f"action={final} score={score:.3f} → INAPPROPRIATE")

            # Map scene action to THOR action and execute
            thor_action = "MoveAhead" if final == "move_fast" else "RotateRight"
            controller.step(thor_action)

        if ep % 5 == 0:
            logger.info(f"[paper3-{label}] Episode {ep}/{EPISODES} complete. "
                        f"Inappropriate so far: {inappropriate_count}")

    per_10 = (inappropriate_count / total_steps) * 10
    logger.info(f"[paper3-{label}] Total inappropriate: {inappropriate_count}/{total_steps} "
                f"= {per_10:.2f} per 10 steps")
    return per_10


# ── csv append ───────────────────────────────────────────────

def append_to_csv(rows, out_path):
    """Append paper3 results to existing ablation_results.csv."""
    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "value"])
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"[csv] Appended paper3 results → {out_path}")


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  default="FloorPlan1")
    parser.add_argument("--csv",    default="data/ablation/paper3_results.csv")
    args = parser.parse_args()

    config_path = os.path.join("config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("  Ablation 2: Paper 3 Filter ON vs OFF")
    logger.info(f"  Scene={args.scene}  Episodes={EPISODES}  "
                f"StepsPerEp={STEPS_PER_EP}")
    logger.info("=" * 60)

    encoder = DINOEncoder()

    scene_clf = SceneContextMLP(input_dim=cfg["dino"]["cls_dim"])
    scene_clf.load_state_dict(
        torch.load(cfg["scene_mlp"]["model_path"],
                   map_location="cpu", weights_only=True)
    )
    scene_clf.eval()

    from ai2thor.controller import Controller

    # ── Condition A: Filter ON ───────────────────────────────
    ctrl = Controller(
        scene=args.scene,
        width=cfg["ai2thor"]["width"],
        height=cfg["ai2thor"]["height"],
        fieldOfView=cfg["ai2thor"]["fov"],
    )
    on_per_10  = run_paper3_ablation(ctrl, encoder, scene_clf, use_filter=True)
    ctrl.stop()

    # ── Condition B: Filter OFF ──────────────────────────────
    ctrl = Controller(
        scene=args.scene,
        width=cfg["ai2thor"]["width"],
        height=cfg["ai2thor"]["height"],
        fieldOfView=cfg["ai2thor"]["fov"],
    )
    off_per_10 = run_paper3_ablation(ctrl, encoder, scene_clf, use_filter=False)
    ctrl.stop()

    # ── Results ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ABLATION 2 RESULTS — Paper 3 Filter")
    print("=" * 60)
    print(f"  Filter ON  → {on_per_10:.2f} inappropriate actions per 10 steps")
    print(f"  Filter OFF → {off_per_10:.2f} inappropriate actions per 10 steps")
    improvement = off_per_10 - on_per_10
    if improvement > 0:
        print(f"  Filter reduced inappropriate actions by {improvement:.2f} per 10 steps ✓")
    else:
        print(f"  ⚠  Filter did not reduce inappropriate actions.")
        print(f"     Report guidance: the MLP may need more training data,")
        print(f"     or the threshold (0.4) may need tuning.")
    print("=" * 60)

    append_to_csv(
        [{"ablation": "paper3_on",  "value": round(on_per_10, 3)},
         {"ablation": "paper3_off", "value": round(off_per_10, 3)}],
        args.csv
    )


if __name__ == "__main__":
    main()