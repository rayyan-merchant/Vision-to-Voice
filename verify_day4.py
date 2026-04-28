import json
import torch
import yaml
from collections import Counter
from src.scene_classifier import SceneContextMLP

def check_class_balance(labels_path):
    print("\n" + "="*50)
    print("TASK 3: Class Balance Check")
    print("="*50)
    
    with open(labels_path, "r") as f:
        data = json.load(f)
        
    total = len(data)
    print(f"Total labeled frames: {total}/300")
    
    counts = Counter([d.get("room_type", "unknown") for d in data])
    
    print("\nCounts per room type:")
    imbalanced = False
    for room, count in counts.most_common():
        pct = count / total * 100
        print(f"  - {room:<15}: {count:3d} ({pct:.1f}%)")
        if pct < 15.0:
            imbalanced = True
            
    print("\nStatus:")
    if imbalanced:
        print("[!] WARNING: Significant class imbalance detected (class < 15%).")
        print("    -> During today's labelling session, focus on collecting more of the underrepresented classes.")
    else:
        print("[OK] Class balance looks reasonable.")
        
    if "study_area" not in counts:
        print("\n[!] CRITICAL WARNING: 'study_area' class is completely missing!")
        print("    -> The prompt requires: Corridor, Open Study Area, Library, Cafeteria.")
        print("    -> You currently have 'bathroom' instead of 'study_area'.")

def test_filter(model_path, cfg):
    print("\n" + "="*50)
    print("TASK 4: filter() Manual Test")
    print("="*50)
    
    mlp_cfg = cfg["scene_mlp"]
    
    model = SceneContextMLP(
        input_dim=cfg["dino"]["cls_dim"], 
        hidden1=mlp_cfg["hidden1"],
        hidden2=mlp_cfg["hidden2"], 
        dropout=mlp_cfg["dropout"]
    )
    
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    print("Model loaded. Testing filter() behavior...\n")
    
    # Let's get a real token from the dataset to make it realistic
    with open(mlp_cfg["label_path"], "r") as f:
        data = json.load(f)
    
    # Test 1: Real token, likely to override
    print("--- Test 1: Real token (library) ---")
    lib_entry = next((d for d in data if d["room_type"] == "library"), data[0])
    lib_z = torch.tensor(lib_entry["z"], dtype=torch.float32)
    
    with torch.no_grad():
        scores = model(lib_z).squeeze()
        print(f"Model raw scores for library frame:")
        print(f"  move_fast : {scores[0]:.3f}")
        print(f"  stop_wait : {scores[1]:.3f}")
        print(f"  navigate  : {scores[2]:.3f}")
    
    # Try an inappropriate action (move_fast in library)
    res1 = model.filter("move_fast", lib_z, threshold=0.4)
    print(f"\nProposed: 'move_fast' -> filter() returned: '{res1}'")
    assert res1 != "move_fast", "Filter failed to override inappropriate action!"
    
    # Try an appropriate action (stop_wait in library)
    res2 = model.filter("stop_wait", lib_z, threshold=0.4)
    print(f"Proposed: 'stop_wait' -> filter() returned: '{res2}'")
    assert res2 == "stop_wait", "Filter incorrectly overrode appropriate action!"
    
    # Test 2: Random token (2D input) to ensure no crashes
    print("\n--- Test 2: Random token (2D input test) ---")
    dummy_2d = torch.randn(1, 384)
    res3 = model.filter("navigate", dummy_2d, threshold=0.4)
    print(f"2D input, Proposed: 'navigate' -> filter() returned: '{res3}'")
    
    print("\n[OK] filter() tested successfully! Overrides and pass-throughs work.")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config/config.yaml"))
    
    check_class_balance(cfg["scene_mlp"]["label_path"])
    test_filter(cfg["scene_mlp"]["model_path"], cfg)
