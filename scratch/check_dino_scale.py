import torch
from src.perception import DINOEncoder
from PIL import Image
import numpy as np

def check_dino_scale():
    enc = DINOEncoder()
    dummy = Image.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )
    cls, _ = enc.encode(dummy)
    
    print(f"DINO CLS shape: {cls.shape}")
    print(f"DINO CLS Mean:  {cls.mean().item():.6f}")
    print(f"DINO CLS Std:   {cls.std().item():.6f}")
    print(f"DINO CLS Norm:  {torch.norm(cls).item():.6f}")
    print(f"DINO CLS Max:   {cls.max().item():.6f}")
    print(f"DINO CLS Min:   {cls.min().item():.6f}")

if __name__ == "__main__":
    check_dino_scale()
