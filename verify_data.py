import os
import json
import numpy as np

def verify_dataset(base_dir="data/saved_attnlrp_maps"):
    print("=" * 60)
    print("  AttnLRP Dataset Verification")
    print("=" * 60)
    
    metadata_path = os.path.join(base_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"[FAIL] metadata.json not found at {metadata_path}")
        return
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    n_samples = len(metadata)
    print(f"Total samples recorded: {n_samples}")
    if n_samples < 100:
        print(f"[FAIL] Need 100+ samples, got {n_samples}")
    else:
        print("[PASS] 100+ samples recorded")
        
    scenes = set(m["scene"] for m in metadata)
    print(f"Unique scenes: {len(scenes)} ({', '.join(scenes)})")
    if len(scenes) < 2:
        print("[FAIL] Need >= 2 scenes")
    else:
        print("[PASS] 2+ unique scenes")
        
    actions = set(m["action"] for m in metadata)
    print(f"Unique actions: {len(actions)} ({', '.join(actions)})")
    if len(actions) < 2:
        print("[FAIL] Action diversity is too low (all same action)")
    else:
        print("[PASS] Action diversity looks good")
        
    surprises = [m["surprise"] for m in metadata]
    var = np.var(surprises)
    print(f"Surprise score variance: {var:.4f}")
    if var == 0.0:
        print("[FAIL] Surprise scores are identical, predictor might be broken")
    else:
        print("[PASS] Surprise score varies")
        
    missing_files = 0
    bad_shapes = 0
    for m in metadata:
        fid = m["frame_id"]
        png_path = os.path.join(base_dir, "frames", f"frame_{fid}.png")
        npy_path = os.path.join(base_dir, "attnlrp", f"attnlrp_{fid}.npy")
        
        if not os.path.exists(png_path): missing_files += 1
        if not os.path.exists(npy_path): missing_files += 1
        
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            if arr.shape != (224, 224): bad_shapes += 1
            
    if missing_files > 0:
        print(f"[FAIL] {missing_files} files (PNG or NPY) missing")
    else:
        print("[PASS] All files present")
        
    if bad_shapes > 0:
        print(f"[FAIL] {bad_shapes} NPY files have wrong shape (not 224x224)")
    else:
        print("[PASS] All NPY files are (224, 224)")
        
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/saved_attnlrp_maps")
    args = parser.parse_args()
    verify_dataset(args.dir)
