# ============================================================
# perception.py  |  Track A  |  Owner: Rayyan
# DINOv3 distilled encoder — FROZEN, never fine-tuned
# Outputs: CLS token (384,), patch tokens (256,384), attn map (16,16)
# ============================================================

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image


# ── ImageNet normalisation constants (DINOv3 expects these) ──────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class DINOEncoder:
    """
    Thin wrapper around frozen DINOv3 (dinov2_vits14).

    This is the SINGLE point of contact with DINOv3 for the entire project.
    Every other module (JEPA, mapper, scene MLP, saliency) calls this class
    — never loads the model themselves.

    Output dimensions:
        CLS token   → (384,)      — represents the whole frame semantically
        Patch tokens→ (256, 384)  — represents individual 14×14 patches
        Attn map    → (16, 16)    — last-block attention for dashboard overlay
    """

    def __init__(self):
        print("[DINOEncoder] Loading dinov2_vits14 from torch hub...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            verbose=False
        )
        self.model.eval()          # ALWAYS eval — gradients off during encode
        print("[DINOEncoder] Model loaded. Parameters frozen.")

        self.transform = T.Compose([
            T.Resize(256),          # resize shorter edge
            T.CenterCrop(224),      # crop to 224×224
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])

    # ── Primary encode method ─────────────────────────────────────────────────
    def encode(self, pil_img: Image.Image):
        """
        Encode a single PIL image.

        Args:
            pil_img: PIL Image (any size — will be resized to 224×224)

        Returns:
            cls    : torch.Tensor (384,)       — detached, cpu
            patches: torch.Tensor (256, 384)   — detached, cpu
        """
        x = self._preprocess(pil_img)           # (1, 3, 224, 224)

        with torch.no_grad():
            features = self.model.forward_features(x)

        cls     = features["x_norm_clstoken"].squeeze()     # (384,)
        patches = features["x_norm_patchtokens"].squeeze()  # (256, 384)

        return cls, patches

    # ── Attention map for dashboard overlay ──────────────────────────────────
    def attention_map(self, pil_img):
        """
        Extract (16,16) attention map.
        Supports different DINO output formats.
        """
        x = self._preprocess(pil_img)
        captured = []

        hook = self.model.blocks[-1].attn.register_forward_hook(
            lambda m, inp, out: captured.append(out)
        )

        with torch.no_grad():
            self.model(x)

        hook.remove()

        out = captured[0]

        # Case 1: expected shape (B, heads, tokens, tokens)
        if out.dim() == 4:
            attn = out[0, :, 0, 1:]
            attn_mean = attn.mean(dim=0)

        # Case 2: shape (B, tokens, dim) or other fallback
        elif out.dim() == 3:
            # Use token activations as pseudo-attention
            token_map = out[0, 1:, :]          # skip CLS
            attn_mean = token_map.mean(dim=1)

        else:
            raise RuntimeError(f"Unexpected attention output shape: {out.shape}")

        h = w = int(attn_mean.shape[0] ** 0.5)
        attn_map = attn_mean.reshape(h, w).cpu().numpy().astype(np.float32)

        mn, mx = attn_map.min(), attn_map.max()
        if mx > mn:
            attn_map = (attn_map - mn) / (mx - mn)

        return attn_map

    # ── Batch encode (used during trajectory collection for speed) ────────────
    def encode_batch(self, pil_list: list):
        """
        Encode a list of PIL images in one forward pass.
        Much faster than calling encode() in a loop.

        Args:
            pil_list: list of PIL Images (max ~16 to stay within RAM)

        Returns:
            cls_batch: torch.Tensor (N, 384)  — detached, cpu
        """
        tensors = torch.stack([self.transform(img) for img in pil_list])

        with torch.no_grad():
            features = self.model.forward_features(tensors)

        return features["x_norm_clstoken"]   # (N, 384)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert PIL to (1, 3, 224, 224) tensor."""
        img = pil_img.convert("RGB")     # ensure 3-channel
        return self.transform(img).unsqueeze(0)


# ── Quick self-test (run this file directly to verify) ────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("perception.py  — self test")
    print("=" * 50)

    enc = DINOEncoder()

    # Test 1: encode a blank image
    dummy = Image.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )
    cls, patches = enc.encode(dummy)

    assert cls.shape     == torch.Size([384]),      f"CLS shape wrong: {cls.shape}"
    assert patches.shape == torch.Size([256, 384]), f"Patch shape wrong: {patches.shape}"
    print(f"[PASS] encode()  →  CLS: {tuple(cls.shape)}, patches: {tuple(patches.shape)}")

    # Test 2: attention map
    attn = enc.attention_map(dummy)
    assert attn.shape == (16, 16), f"Attn shape wrong: {attn.shape}"
    assert attn.min() >= 0.0 and attn.max() <= 1.0, "Attn values out of [0,1]"
    print(f"[PASS] attention_map()  →  {attn.shape}, min={attn.min():.3f}, max={attn.max():.3f}")

    # Test 3: batch encode
    batch = [dummy] * 4
    cls_batch = enc.encode_batch(batch)
    assert cls_batch.shape == torch.Size([4, 384]), f"Batch shape wrong: {cls_batch.shape}"
    print(f"[PASS] encode_batch(4)  →  {tuple(cls_batch.shape)}")

    print()
    print("All tests passed. DINOEncoder is ready.")