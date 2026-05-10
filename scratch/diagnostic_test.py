import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

print("--- DIAGNOSTIC ---")

# Check a saved map
map_path = "data/saved_attnlrp_maps/attnlrp/attnlrp_000.npy"
if os.path.exists(map_path):
    m = np.load(map_path)
    print("saved map — min:", m.min(), "max:", m.max(), "mean:", m.mean())
else:
    print(f"File not found: {map_path}")

# Check live grad mimicking Fix A scenario
from src.saliency import SaliencyEngine

print("\nTesting Live Grad (without torch.enable_grad) inside a no_grad block:")
engine = SaliencyEngine(None) # we don't need dino for attnlrp

with torch.no_grad():
    input_tensor = torch.randn(1, 3, 224, 224)
    # what happens in attnlrp_map
    x = input_tensor.clone().requires_grad_(True)
    out = engine.attnlrp_model(x)
    try:
        out[0].sum().backward()
        print("x.grad is None:", x.grad is None)
        if x.grad is not None:
            print("x.grad stats — min:", x.grad.min().item(), "max:", x.grad.max().item())
            raw = x.grad.abs().mean(dim=1).squeeze()
            print("raw relevance — min:", raw.min().item(), "max:", raw.max().item())
    except Exception as e:
        print("Backward failed:", e)

print("\nTesting Live Grad (Normal):")
input_tensor = torch.randn(1, 3, 224, 224)
x = input_tensor.clone().requires_grad_(True)
out = engine.attnlrp_model(x)
out[0].sum().backward()
print("x.grad is None:", x.grad is None)
if x.grad is not None:
    print("x.grad stats — min:", x.grad.min().item(), "max:", x.grad.max().item())
    raw = x.grad.abs().mean(dim=1).squeeze()
    print("raw relevance — min:", raw.min().item(), "max:", raw.max().item())
