# ============================================================
# saliency.py  |  Track C  |  Owner: Riya Bhart
# SmoothGrad (Paper 5) + AttnLRP (Paper 4) attribution engine
#
# Unified entry point for the dashboard and clever_hans.py.
# Tries AttnLRP first (lxt library); falls back to SmoothGrad
# silently if lxt is unavailable or model loading fails.
#
# Week 2: Build & test on static campus photos
# Week 5: Feed attribution maps into KMeans clustering
# ============================================================

import torch
import torch.nn as nn
import numpy as np

# ── AttnLRP import (graceful fallback) ────────────────────────────
HAS_LXT = False
try:
    from lxt.models.vit import dino_vits14_lrp
    HAS_LXT = True
except (ImportError, ModuleNotFoundError):
    pass  # silently fall back to SmoothGrad

from captum.attr import NoiseTunnel, Saliency


# ── DINOWrapper for Captum compatibility ──────────────────────────
class DINOWrapper(nn.Module):
    """Wraps a DINO backbone so Captum can target a scalar output.

    DINO outputs a (B, 384) feature vector.  Captum's Saliency
    requires indexing into a specific output neuron (target_class).
    This wrapper leaves the forward pass unchanged — Captum
    handles the indexing via the `target` argument.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # (B, 384)


# ── Normalisation helper ─────────────────────────────────────────
def _normalise(x):
    """Min-max normalise to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# ■■ SaliencyEngine ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class SaliencyEngine:
    """Unified attribution engine: AttnLRP → SmoothGrad fallback.

    Usage:
        from src.perception import DINOEncoder
        enc = DINOEncoder()
        engine = SaliencyEngine(enc.model)
        heatmap = engine.get_map(tensor_224, method="attnlrp")
    """

    def __init__(self, dino_model):
        """
        Args:
            dino_model: A DINOv2 backbone (e.g. from torch.hub).
                        Used for the SmoothGrad path.
        """
        # ── SmoothGrad path ──────────────────────────────────────
        wrapped = DINOWrapper(dino_model)
        self.smoothgrad = NoiseTunnel(Saliency(wrapped))

        # ── AttnLRP path ─────────────────────────────────────────
        self.attnlrp_model = None
        if HAS_LXT:
            try:
                self.attnlrp_model = dino_vits14_lrp(pretrained=True)
                self.attnlrp_model.eval()
                print("[SaliencyEngine] AttnLRP model loaded successfully")
            except Exception as e:
                print(f"[SaliencyEngine] AttnLRP load failed: {e} "
                      "— falling back to SmoothGrad")
        else:
            print("[SaliencyEngine] lxt not available — using SmoothGrad")

    # ── AttnLRP ──────────────────────────────────────────────────
    def attnlrp_map(self, input_tensor):
        """Generate AttnLRP relevance map.

        Args:
            input_tensor: (1, 3, 224, 224) — will be cloned internally

        Returns:
            numpy array (224, 224) in [0, 1]
        """
        if self.attnlrp_model is None:
            # Graceful fallback: silently use SmoothGrad
            return self.smoothgrad_map(input_tensor, target_class=0)

        # PITFALL: both .eval() AND .requires_grad_(True) required
        self.attnlrp_model.eval()
        x = input_tensor.clone().requires_grad_(True)

        out = self.attnlrp_model(x)
        target = out[0].sum()  # sum over CLS token dimensions
        target.backward()

        relevance = x.grad.abs().mean(dim=1).squeeze()  # (224, 224)
        return _normalise(relevance).detach().numpy()

    # ── SmoothGrad ───────────────────────────────────────────────
    def smoothgrad_map(self, input_tensor, target_class=0):
        """Generate SmoothGrad attribution map (Paper 5).

        Args:
            input_tensor: (1, 3, 224, 224)
            target_class: output neuron index to attribute (default 0)

        Returns:
            numpy array (224, 224) in [0, 1]
        """
        attrs = self.smoothgrad.attribute(
            input_tensor,
            nt_type="smoothgrad",
            nt_samples=50,      # spec: 50 noisy copies
            stdevs=0.15,        # spec: noise std dev
            target=target_class,
        )
        smap = attrs.abs().mean(dim=1).squeeze().detach().numpy()
        return _normalise(smap)

    # ── Unified entry point ──────────────────────────────────────
    def get_map(self, input_tensor, method="attnlrp"):
        """Get attribution map — used by dashboard and clever_hans.py.

        Args:
            input_tensor: (1, 3, 224, 224)
            method: "attnlrp" or "smoothgrad"

        Returns:
            numpy array (224, 224) in [0, 1]
        """
        if method == "attnlrp":
            return self.attnlrp_map(input_tensor)
        return self.smoothgrad_map(input_tensor)


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("saliency.py -- self-test")
    print("=" * 55)

    # Load DINOv2 backbone
    print("\n[1/3] Loading DINOv2 backbone...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino.eval()

    # Create engine
    print("\n[2/3] Creating SaliencyEngine...")
    engine = SaliencyEngine(dino)

    # Test input
    x = torch.randn(1, 3, 224, 224).requires_grad_(True)

    # Test SmoothGrad
    print("\n[3/3] Testing SmoothGrad (nt_samples=50)...")
    sg_map = engine.get_map(x, method="smoothgrad")
    assert sg_map.shape == (224, 224), f"SmoothGrad shape wrong: {sg_map.shape}"
    assert 0.0 <= sg_map.min() and sg_map.max() <= 1.0, "SmoothGrad values out of [0,1]"
    print(f"  [PASS] SmoothGrad -> {sg_map.shape}, "
          f"min={sg_map.min():.4f}, max={sg_map.max():.4f}")

    # Test AttnLRP (or fallback)
    print("\nTesting AttnLRP (or fallback)...")
    al_map = engine.get_map(x, method="attnlrp")
    assert al_map.shape == (224, 224), f"AttnLRP shape wrong: {al_map.shape}"
    assert 0.0 <= al_map.min() and al_map.max() <= 1.0, "AttnLRP values out of [0,1]"
    method_used = "AttnLRP" if engine.attnlrp_model is not None else "SmoothGrad (fallback)"
    print(f"  [PASS] {method_used} -> {al_map.shape}, "
          f"min={al_map.min():.4f}, max={al_map.max():.4f}")

    print("\n" + "=" * 55)
    print("All tests passed. SaliencyEngine is ready.")
    print(f"  AttnLRP available: {engine.attnlrp_model is not None}")
    print(f"  HAS_LXT flag:      {HAS_LXT}")
    print("=" * 55)
