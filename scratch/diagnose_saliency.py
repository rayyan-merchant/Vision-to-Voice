import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.saliency import SaliencyEngine

print("=== DIAGNOSTIC 1: Saved Map ===")
try:
    m = np.load("data/saved_attnlrp_maps/attnlrp/attnlrp_000.npy")
    print("saved map — min:", m.min(), "max:", m.max(), "mean:", m.mean())
except Exception as e:
    print("Failed to load saved map:", e)

print("\n=== DIAGNOSTIC 2: Live Computation ===")
try:
    # Use dummy tensor
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Simulate a no_grad environment upstream (which is Fix A)
    with torch.no_grad():
        x_no_grad = input_tensor.clone()
        
    print("x_no_grad.requires_grad:", x_no_grad.requires_grad)

    # Load engine
    print("Loading DINO...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino.eval()
    engine = SaliencyEngine(dino)
    
    print("AttnLRP model type:", type(engine.attnlrp_model))

    x = x_no_grad.clone().requires_grad_(True)
    out = engine.attnlrp_model(x)
    out[0].sum().backward()
    
    print("x.grad is None:", x.grad is None)
    if x.grad is not None:
        print("x.grad stats — min:", x.grad.min().item(), "max:", x.grad.max().item())
        
        raw = x.grad.abs().mean(dim=1).squeeze()
        print("raw relevance — min:", raw.min().item(), "max:", raw.max().item())
except Exception as e:
    print("Live computation failed:", e)
