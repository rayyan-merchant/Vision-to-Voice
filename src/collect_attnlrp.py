import json
import os
import glob
import numpy as np
from PIL import Image


class AttnLRPCollector:
    """
    Collects AttnLRP attribution maps, source frames, and metadata
    for the Week 5 Clever Hans audit.

    Features:
        - Resumes from existing data (finds highest frame_id on disk)
        - Auto-flushes metadata every `flush_every` records
        - Squeeze-guards .npy shape to (224, 224)
    """

    def __init__(self, base_dir="data/saved_attnlrp_maps", flush_every=10):
        self.base_dir = base_dir
        self.frames_dir = os.path.join(base_dir, "frames")
        self.attnlrp_dir = os.path.join(base_dir, "attnlrp")
        self.metadata_path = os.path.join(base_dir, "metadata.json")
        self.flush_every = flush_every

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.attnlrp_dir, exist_ok=True)

        # Resume: load existing metadata if present, otherwise recover from files
        self.metadata_list = []
        self.collector_count = 0
        self._resume()

    def _resume(self):
        """Pick up where we left off if prior data exists."""
        # Try loading existing metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata_list = json.load(f)
            self.collector_count = len(self.metadata_list)
            print(f"[AttnLRPCollector] Resumed from metadata.json — "
                  f"{self.collector_count} existing records")
            return

        # No metadata but maybe orphaned files from a crashed run
        existing_frames = glob.glob(os.path.join(self.frames_dir, "frame_*.png"))
        existing_maps = glob.glob(os.path.join(self.attnlrp_dir, "attnlrp_*.npy"))

        if existing_frames or existing_maps:
            # Find the highest ID to continue numbering from there
            ids = []
            for p in existing_frames:
                base = os.path.basename(p)
                try:
                    ids.append(int(base.replace("frame_", "").replace(".png", "")))
                except ValueError:
                    pass
            for p in existing_maps:
                base = os.path.basename(p)
                try:
                    ids.append(int(base.replace("attnlrp_", "").replace(".npy", "")))
                except ValueError:
                    pass

            if ids:
                # Orphaned files without metadata — delete and start fresh
                print(f"[AttnLRPCollector] Found {len(existing_frames)} orphaned frames "
                      f"and {len(existing_maps)} orphaned .npy files without metadata.json")
                print(f"[AttnLRPCollector] Clearing orphaned files for clean start...")
                for p in existing_frames:
                    os.remove(p)
                for p in existing_maps:
                    os.remove(p)
                self.collector_count = 0
                print(f"[AttnLRPCollector] Clean start from 0")
        else:
            print(f"[AttnLRPCollector] Starting fresh collection in {self.base_dir}")

    def record(self, frame, attn_map, action, surprise, scene, node_id):
        """Save one frame + attribution map + metadata entry."""
        fid = str(self.collector_count).zfill(3)

        # Squeeze attn_map to (224, 224) if needed
        if isinstance(attn_map, np.ndarray):
            attn_map = np.squeeze(attn_map)
        else:
            # torch tensor — convert
            attn_map = attn_map.detach().cpu().numpy().squeeze()

        if attn_map.shape != (224, 224):
            print(f"[AttnLRPCollector] Warning: Expected (224,224), got {attn_map.shape}")

        frame_path = os.path.join(self.frames_dir, f"frame_{fid}.png")
        map_path = os.path.join(self.attnlrp_dir, f"attnlrp_{fid}.npy")

        # Save image and numpy array
        frame.save(frame_path)
        np.save(map_path, attn_map.astype(np.float32))

        self.metadata_list.append({
            "frame_id": fid,
            "action": action,
            "surprise": round(float(surprise), 4),
            "scene": scene,
            "node_id": int(node_id)
        })

        self.collector_count += 1

        # Auto-flush periodically to protect against crashes
        if self.collector_count % self.flush_every == 0:
            self.flush()
            print(f"[AttnLRPCollector] Auto-flushed at {self.collector_count} records")

    def flush(self):
        """Write metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata_list, f, indent=2)
        print(f"[AttnLRPCollector] Flushed {len(self.metadata_list)} records to {self.metadata_path}")
