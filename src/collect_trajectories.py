# ============================================================
# collect_trajectories.py  |  Track A  |  Owner: Rayyan
#
# PURPOSE
#   Runs an agent through AI2-THOR scenes, encodes every frame
#   with DINOv3, and saves (z_t, action, z_t1, pos, rot, scene)
#   tuples as JSON.
#
# USAGE
#   # Day 3 — mock dataset (50 samples, fast, for Syeda to start JEPA):
#   python collect_trajectories.py --mode mock
#
#   # Day 7 — full dataset (5000 samples):
#   python collect_trajectories.py --mode full
#
# OUTPUT FORMAT (shared data contract — NEVER change this format)
#   [
#     {
#       "z_t":    [384 floats],   <- DINOv3 CLS token at step t
#       "action": "MoveAhead",    <- action taken (one of 5 strings)
#       "z_t1":   [384 floats],   <- DINOv3 CLS token at step t+1
#       "pos":    [x, z],         <- agent position (x, z plane)
#       "rot":    float,           <- agent rotation in degrees
#       "scene":  "FloorPlan1"    <- which AI2-THOR scene
#     },
#     ...
#   ]
# ============================================================

import argparse
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

# Add src/ to path so we can import perception.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.perception import DINOEncoder


# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]

# Diverse scene list — mix of apartments (1xx), living rooms (2xx), bedrooms (3xx)
SCENES = [
    "FloorPlan1",   "FloorPlan2",   "FloorPlan3",   "FloorPlan4",   "FloorPlan5",
    "FloorPlan201", "FloorPlan202", "FloorPlan203",
    "FloorPlan301", "FloorPlan302",
]

MOCK_SAMPLES        = 50
FULL_SAMPLES        = 5000
SCENE_SWITCH_EVERY  = 500   # rotate to next scene every N samples
MOCK_PATH           = "data/trajectories/mock_trajectories.json"
FULL_PATH           = "data/trajectories/all_trajectories.json"


# ── Main collection function ──────────────────────────────────────────────────
def collect(n_samples: int, output_path: str):
    """
    Collect n_samples trajectory steps and write to output_path.
    Each step: encode frame → take random action → encode next frame → save.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Import AI2-THOR here so the file can still be read without it ────────
    try:
        from ai2thor.controller import Controller
    except ImportError:
        print("[ERROR] ai2thor not installed. Run: pip install ai2thor==5.0.0")
        sys.exit(1)

    encoder   = DINOEncoder()
    samples   = []
    scene_idx = 0

    print(f"\n{'='*55}")
    print(f"  Collecting {n_samples} trajectory samples")
    print(f"  Output → {output_path}")
    print(f"{'='*55}\n")

    # Start controller on first scene
    print(f"[AI2-THOR] Loading scene {SCENES[scene_idx]}...")
    ctrl = Controller(
        scene=SCENES[scene_idx],
        width=224,
        height=224,
        fieldOfView=90,
        renderDepthImage=False,    # we don't need depth
        renderObjectImage=False,   # we don't need instance seg
    )
    print("[AI2-THOR] Ready.\n")

    t_start = time.time()

    while len(samples) < n_samples:
        step = len(samples)

        # ── Switch scene every SCENE_SWITCH_EVERY samples for variety ────────
        if step > 0 and step % SCENE_SWITCH_EVERY == 0:
            scene_idx = (scene_idx + 1) % len(SCENES)
            new_scene = SCENES[scene_idx]
            print(f"  [Scene switch] → {new_scene}  (step {step})")
            ctrl.reset(scene=new_scene)

        # ── Encode current frame (z_t) ────────────────────────────────────────
        frame_t = Image.fromarray(ctrl.last_event.frame)
        z_t, _  = encoder.encode(frame_t)

        pos_meta = ctrl.last_event.metadata["agent"]["position"]
        rot_meta = ctrl.last_event.metadata["agent"]["rotation"]["y"]

        # ── Take a random action ──────────────────────────────────────────────
        action = random.choice(ACTIONS)
        event  = ctrl.step(action)

        # ── Encode next frame (z_t1) ──────────────────────────────────────────
        frame_t1 = Image.fromarray(event.frame)
        z_t1, _  = encoder.encode(frame_t1)

        # ── Save sample ───────────────────────────────────────────────────────
        samples.append({
            "z_t":    z_t.tolist(),             # list of 384 floats
            "action": action,                    # string
            "z_t1":   z_t1.tolist(),            # list of 384 floats
            "pos":    [pos_meta["x"], pos_meta["z"]],   # [x, z]
            "rot":    rot_meta,                  # float degrees
            "scene":  SCENES[scene_idx],         # string
        })

        # ── Progress reporting ────────────────────────────────────────────────
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t_start
            rate    = (step + 1) / elapsed if elapsed > 0 else 0
            eta     = (n_samples - step - 1) / rate if rate > 0 else 0
            print(
                f"  [{step+1:>5}/{n_samples}]  "
                f"scene={SCENES[scene_idx]:<14}  "
                f"action={action:<12}  "
                f"rate={rate:.1f}/s  ETA={eta:.0f}s"
            )

    ctrl.stop()

    # ── Write JSON ────────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(samples, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Done! Saved {len(samples)} samples → {output_path}")
    print(f"  Time: {elapsed:.1f}s  |  File size: {os.path.getsize(output_path)/1024:.0f} KB")
    print(f"{'='*55}\n")

    # ── Quick validation ──────────────────────────────────────────────────────
    _validate(output_path)


def _validate(path: str):
    """Load the saved file and check the format is correct."""
    with open(path) as f:
        data = json.load(f)

    sample = data[0]
    errors = []

    if len(sample["z_t"])  != 384: errors.append(f"z_t length is {len(sample['z_t'])}, expected 384")
    if len(sample["z_t1"]) != 384: errors.append(f"z_t1 length is {len(sample['z_t1'])}, expected 384")
    if sample["action"] not in ACTIONS: errors.append(f"Unknown action: {sample['action']}")
    if len(sample["pos"]) != 2: errors.append(f"pos length is {len(sample['pos'])}, expected 2")
    if not isinstance(sample["rot"], float): errors.append(f"rot is not float: {type(sample['rot'])}")

    if errors:
        print("[VALIDATION FAILED]")
        for e in errors: print(f"  ✗ {e}")
    else:
        print("[VALIDATION PASSED]")
        print(f"  ✓ {len(data)} samples, all fields correct")
        print(f"  ✓ Sample 0: action={sample['action']}, scene={sample['scene']}")
        print(f"  ✓ z_t[:4]  = {sample['z_t'][:4]}")
        print(f"  ✓ z_t1[:4] = {sample['z_t1'][:4]}")

    return len(errors) == 0


# ── Verify existing file (without re-collecting) ──────────────────────────────
def verify(path: str):
    """Just validate an existing trajectory file. Useful for debugging."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return
    print(f"Validating {path}...")
    _validate(path)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-to-Voice trajectory collector")
    parser.add_argument(
        "--mode",
        choices=["mock", "full", "verify_mock", "verify_full"],
        default="mock",
        help=(
            "mock         → collect 50 samples (Day 3, for Syeda)\n"
            "full         → collect 5000 samples (Day 7)\n"
            "verify_mock  → validate existing mock JSON\n"
            "verify_full  → validate existing full JSON"
        )
    )
    args = parser.parse_args()

    if args.mode == "mock":
        print(">>> MODE: MOCK (50 samples — for Syeda to start JEPA coding)")
        collect(MOCK_SAMPLES, MOCK_PATH)
        print(f"\nNEXT STEP: Send '{MOCK_PATH}' to Syeda NOW so she can start predictor.py")

    elif args.mode == "full":
        print(">>> MODE: FULL (5000 samples — replaces mock)")
        collect(FULL_SAMPLES, FULL_PATH)
        print(f"\nNEXT STEP: Tell Syeda to switch to '{FULL_PATH}' for JEPA training")

    elif args.mode == "verify_mock":
        verify(MOCK_PATH)

    elif args.mode == "verify_full":
        verify(FULL_PATH)