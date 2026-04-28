# label_frames.py
# ==========================================================
# Universal Scene Labeling Tool
#
# Usage:
#   python3 label_frames.py --folder "data/library(livingroom)"
#   python3 label_frames.py --folder "data/cafeteria(kitchen)"
#   python3 label_frames.py --folder "data/bathroom"
#
# Keys inside image window:
#   C = corridor
#   S = study_area
#   L = library
#   F = cafeteria
#   B = bathroom
#   X = skip
#   Q = quit safely
# ==========================================================

import os
import json
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from src.perception import DINOEncoder


# ==========================================================
# LABEL CONFIG
# ==========================================================
SCORE_MAP = {
    "corridor":   {"move_fast": 0.9, "stop_wait": 0.2, "navigate": 0.9},
    "study_area": {"move_fast": 0.3, "stop_wait": 0.8, "navigate": 0.6},
    "library":    {"move_fast": 0.1, "stop_wait": 0.9, "navigate": 0.4},
    "cafeteria":  {"move_fast": 0.5, "stop_wait": 0.9, "navigate": 0.7},
    "bathroom":   {"move_fast": 0.2, "stop_wait": 0.8, "navigate": 0.4},
}

KEY_MAP = {
    "c": "corridor",
    "s": "study_area",
    "l": "library",
    "f": "cafeteria",
    "b": "bathroom",
    "x": None,      # skip
    "q": "QUIT"
}

MASTER_LABEL_FILE = "data/scene_labels/all_labels.json"


# ==========================================================
# HELPERS
# ==========================================================
def load_existing_labels():
    os.makedirs("data/scene_labels", exist_ok=True)

    if os.path.exists(MASTER_LABEL_FILE):
        with open(MASTER_LABEL_FILE, "r") as f:
            return json.load(f)
    return []


def save_labels(labels):
    with open(MASTER_LABEL_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def get_frame_list(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])


def print_dataset_stats(labels):
    if not labels:
        print("No labels saved yet.")
        return

    room_counts = Counter([x["room_type"] for x in labels])
    scene_counts = Counter([x["scene"] for x in labels])

    print("\n================ DATASET SUMMARY ================")
    print("Total labeled samples:", len(labels))

    print("\nBy class:")
    for k, v in room_counts.items():
        print(f"  {k:<12} : {v}")

    print("\nBy scene:")
    for k, v in scene_counts.items():
        print(f"  {k:<22} : {v}")

    print("================================================")


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        required=True,
        help='Example: "data/library(livingroom)"'
    )
    args = parser.parse_args()

    frames_dir = args.folder

    if not os.path.exists(frames_dir):
        print(f"[ERROR] Folder not found: {frames_dir}")
        return

    scene_name = os.path.basename(frames_dir)

    # ------------------------------------------------------
    # Load frames
    # ------------------------------------------------------
    frame_files = get_frame_list(frames_dir)

    if len(frame_files) == 0:
        print("No PNG files found.")
        return

    # ------------------------------------------------------
    # Load existing labels
    # ------------------------------------------------------
    labels = load_existing_labels()

    done_paths = {
        item["image_path"]
        for item in labels
        if "image_path" in item
    }

    frame_files = [
        f for f in frame_files
        if os.path.join(frames_dir, f) not in done_paths
    ]

    print(f"\nScene: {scene_name}")
    print(f"Already labeled globally: {len(done_paths)}")
    print(f"Remaining in this folder: {len(frame_files)}")

    if len(frame_files) == 0:
        print("Nothing left to label in this folder.")
        print_dataset_stats(labels)
        return

    # ------------------------------------------------------
    # Load encoder once
    # ------------------------------------------------------
    enc = DINOEncoder()

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    current_key = {"value": None}

    def on_key(event):
        if event.key:
            current_key["value"] = event.key.lower()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ------------------------------------------------------
    # Label loop
    # ------------------------------------------------------
    for idx, fname in enumerate(frame_files):

        fpath = os.path.join(frames_dir, fname)
        img = mpimg.imread(fpath)

        current_key["value"] = None

        ax.clear()
        ax.imshow(img)
        ax.set_title(
            f"[{idx+1}/{len(frame_files)}] {fname}\n"
            f"Scene: {scene_name}\n"
            "C=corridor | S=study | L=library | "
            "F=cafeteria | B=bathroom | X=skip | Q=quit",
            fontsize=10
        )
        ax.axis("off")
        plt.draw()

        while current_key["value"] is None:
            plt.pause(0.1)

        choice = current_key["value"]

        if choice not in KEY_MAP:
            print("Invalid key. Skipped.")
            continue

        room_type = KEY_MAP[choice]

        if room_type == "QUIT":
            print("\nStopped safely.")
            break

        if room_type is None:
            print(f"Skipped: {fname}")
            continue

        # --------------------------------------------------
        # Encode image
        # --------------------------------------------------
        pil_img = Image.open(fpath).convert("RGB")
        cls, _ = enc.encode(pil_img)

        sample = {
            "frame_id": fname,
            "scene": scene_name,
            "image_path": fpath,
            "room_type": room_type,
            "z": cls.tolist(),
            "scores": SCORE_MAP[room_type]
        }

        labels.append(sample)
        save_labels(labels)

        print(
            f"Labeled {fname:<20} -> {room_type:<10} | "
            f"Total dataset: {len(labels)}"
        )

    plt.close()

    # ------------------------------------------------------
    # Final Stats
    # ------------------------------------------------------
    print_dataset_stats(labels)
    print(f"\nSaved to: {MASTER_LABEL_FILE}")


if __name__ == "__main__":
    main()