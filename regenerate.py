import json
import os
import torch
import numpy as np
from PIL import Image
import sys

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.saliency import SaliencyEngine
from src.perception import DINOEncoder

def main():
    metadata_path = "data/saved_attnlrp_maps/metadata.json"
    frames_dir = "data/saved_attnlrp_maps/frames"
    out_dir = "data/saved_attnlrp_maps/attnlrp"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("[1/3] Loading models...")
    encoder = DINOEncoder()
    engine = SaliencyEngine(encoder.model)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    print(f"[2/3] Regenerating {len(metadata)} attribution maps offline...")
    
    for i, entry in enumerate(metadata):
        fid = entry["frame_id"]
        frame_path = os.path.join(frames_dir, f"frame_{fid}.png")
        out_path = os.path.join(out_dir, f"attnlrp_{fid}.npy")
        
        if not os.path.exists(frame_path):
            print(f"Warning: missing {frame_path}")
            continue
            
        # Load and preprocess
        img = Image.open(frame_path).convert("RGB")
        tensor = encoder._preprocess(img)
        
        # Get high-quality map (SmoothGrad with percentile clip + blur)
        smap = engine.get_map(tensor, method="attnlrp")
        
        # Save
        np.save(out_path, smap)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(metadata)}")
            
    print("[3/3] Done! All maps regenerated successfully.")

if __name__ == "__main__":
    main()
