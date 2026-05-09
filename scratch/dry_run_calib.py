import torch
import numpy as np
import yaml
from PIL import Image
from src.predictor import load_jepa, prediction_error
from ai2thor.controller import Controller
from src.perception import DINOEncoder

def dry_run_calibration():
    cfg = yaml.safe_load(open("config/config.yaml"))
    
    print("Initializing AI2-THOR...")
    controller = Controller(
        scene="FloorPlan210",
        width=224,
        height=224,
        fieldOfView=90
    )
    
    print("Loading models...")
    encoder = DINOEncoder()
    predictor = load_jepa(
        cfg["jepa"]["model_path"],
        z_dim=cfg["dino"]["cls_dim"],
        hidden=cfg["jepa"]["hidden_dim"]
    )
    predictor.eval()
    
    ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]
    import random
    
    surprises = []
    prev_cls = None
    prev_act = None
    
    n_warmup = 50
    print(f"Running {n_warmup} warmup steps...")
    for i in range(n_warmup):
        frame = Image.fromarray(controller.last_event.frame)
        cls, _ = encoder.encode(frame)
        if prev_cls is not None:
            err = prediction_error(predictor, prev_cls, prev_act, cls)
            surprises.append(err)
            if i % 10 == 0:
                print(f"  Step {i}: surprise={err:.4f}")
        
        prev_act = random.choice(ACTIONS)
        prev_cls = cls
        controller.step(prev_act)
        
    mean_s = np.mean(surprises)
    std_s = np.std(surprises)
    threshold = np.clip(mean_s + 1.0 * std_s, 0.5, 20.0)
    
    print("\nCalibration Results:")
    print(f"  Mean: {mean_s:.4f}")
    print(f"  Std:  {std_s:.4f}")
    print(f"  Threshold (raw): {mean_s + std_s:.4f}")
    print(f"  Threshold (clip): {threshold:.4f}")
    
    controller.stop()

if __name__ == "__main__":
    dry_run_calibration()
