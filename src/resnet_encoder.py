# ============================================================
# resnet_encoder.py | Track A | Owner: Rayyan
#
# PURPOSE
#   Provides a ResNet18-based visual encoder as the baseline
#   for ablation against DINOv3. Mirrors the DINOEncoder API
#   so ablation_compare.py can swap them with zero friction.
#
# API (identical to DINOEncoder in perception.py)
#   enc = ResNet18Encoder()
#   z   = enc.encode(frame_bgr)          → np.ndarray (512,)
#   attn = enc.attention_map(frame_bgr)  → np.ndarray (H, W) or None
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Normalisation — ImageNet stats (same as DINO pre-training)
# ----------------------------------------------------------
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


class ResNet18Encoder:
    """
    Wraps torchvision ResNet-18 (pretrained on ImageNet) as a
    drop-in replacement for DINOEncoder.

    Key differences from DINOv3:
      - Output dim: 512  (vs 384 for DINOv3 ViT-S/14)
      - No native attention map → returns None from attention_map()
      - Supervised ImageNet pretraining (vs self-supervised DINO)
    """

    DIM = 512   # feature dimensionality

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print("[ResNet18Encoder] Loading ResNet-18 pretrained weights...")
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Strip the classification head — keep everything up to avgpool
        self.model = nn.Sequential(*list(base.children())[:-1])
        self.model.eval().to(self.device)
        # Freeze all parameters (inference only)
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"[ResNet18Encoder] Ready. Output dim = {self.DIM}. Device = {self.device}.")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @torch.no_grad()
    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Encode a single BGR frame → 512-d feature vector.

        Parameters
        ----------
        frame_bgr : np.ndarray, shape (H, W, 3), dtype uint8
            Raw BGR frame from AI2-THOR or cv2.

        Returns
        -------
        np.ndarray, shape (512,), dtype float32
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = _transform(frame_rgb).unsqueeze(0).to(self.device)   # (1, 3, 224, 224)
        feat = self.model(x)                                       # (1, 512, 1, 1)
        return feat.squeeze().cpu().numpy().astype(np.float32)     # (512,)

    @torch.no_grad()
    def encode_batch(self, frames: list) -> np.ndarray:
        """
        Encode a list of BGR frames in one forward pass.

        Parameters
        ----------
        frames : list of np.ndarray (H, W, 3)

        Returns
        -------
        np.ndarray, shape (N, 512), dtype float32
        """
        tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(_transform(rgb))
        batch = torch.stack(tensors).to(self.device)   # (N, 3, 224, 224)
        feats = self.model(batch)                       # (N, 512, 1, 1)
        return feats.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)  # (N, 512)

    def attention_map(self, frame_bgr: np.ndarray):
        """
        ResNet-18 has no built-in attention mechanism.
        Returns None — ablation_compare.py handles this gracefully.
        """
        return None


# ----------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------
if __name__ == "__main__":
    enc = ResNet18Encoder()

    # Synthetic random frame (mimics AI2-THOR output)
    dummy = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    z = enc.encode(dummy)
    assert z.shape == (512,), f"Expected (512,) got {z.shape}"
    assert z.dtype == np.float32

    batch_z = enc.encode_batch([dummy, dummy])
    assert batch_z.shape == (2, 512)

    attn = enc.attention_map(dummy)
    assert attn is None, "attention_map should return None for ResNet18"

    print("[PASS] encode()        → shape (512,)")
    print("[PASS] encode_batch()  → shape (2, 512)")
    print("[PASS] attention_map() → None (expected)")
    print("ResNet18Encoder self-test complete.")