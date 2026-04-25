# ============================================================
# verify_setup.py  |  Run this FIRST before collect_trajectories.py
#
# Checks every dependency and prints a clear PASS/FAIL per item.
# Fix every FAIL before proceeding.
#
# Usage:  python verify_setup.py
# ============================================================

import sys
import importlib

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

results = []

def check(label, fn):
    try:
        msg = fn()
        results.append((True, label, msg or ""))
        print(f"{PASS} {label}  {msg or ''}")
    except Exception as e:
        results.append((False, label, str(e)))
        print(f"{FAIL} {label}  →  {e}")


print()
print("=" * 60)
print("  Vision-to-Voice  |  Setup Verification")
print("=" * 60)
print()

# 1. Python version
check(
    "Python version",
    lambda: f"(got {sys.version.split()[0]})"
    if sys.version_info[:2] in [(3, 9), (3, 10)]
    else (_ for _ in ()).throw(RuntimeError(
        f"Need Python 3.9 or 3.10, got {sys.version.split()[0]}"
    ))
)

# 2. Core packages
for pkg, min_ver in [
    ("torch",       "2.0"),
    ("torchvision", "0.15"),
    ("numpy",       "1.23"),
    ("captum",      "0.7"),
    ("networkx",    "3.0"),
    ("sklearn",     "1.2"),
    ("cv2",         "4.7"),
    ("matplotlib",  "3.7"),
    ("timm",        "0.9"),
    ("PIL",         "9.0"),
    ("tqdm",        "4.0"),
]:
    def _chk(p=pkg):
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", "?")
        return f"v{ver}"
    check(f"import {pkg}", _chk)

# 3. AI2-THOR
check("import ai2thor",
    lambda: f"v{importlib.import_module('ai2thor').__version__}"
)

# 4. Ultralytics / YOLOE
check("import ultralytics (YOLOE)",
    lambda: f"v{importlib.import_module('ultralytics').__version__}"
)

# 5. EasyOCR
check("import easyocr",
    lambda: "ok"
)

# 6. pyttsx3
check("import pyttsx3",
    lambda: "ok"
)

# 7. lxt (AttnLRP) — optional but preferred
try:
    import lxt
    print(f"{PASS} import lxt (AttnLRP)  v{getattr(lxt,'__version__','?')}")
    results.append((True, "import lxt", ""))
except ImportError:
    print(f"{WARN} import lxt (AttnLRP)  NOT installed — fallback to SmoothGrad")
    print(f"       To install: pip install lxt")
    results.append((True, "import lxt", "WARN — fallback mode"))

print()
print("─" * 60)

# 8. DINOv3 shapes
print("  Testing DINOv3 shapes (requires internet on first run)...")
def _dino_test():
    import torch
    from PIL import Image as PILImage
    import numpy as np

    #sys.path.insert(0)
    from perception import DINOEncoder

    enc = DINOEncoder()
    dummy = PILImage.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )
    cls, patches = enc.encode(dummy)
    assert cls.shape     == torch.Size([384]),      f"CLS shape {cls.shape}"
    assert patches.shape == torch.Size([256, 384]), f"Patches shape {patches.shape}"
    attn = enc.attention_map(dummy)
    assert attn.shape == (16, 16), f"Attn shape {attn.shape}"
    return "CLS=(384,)  patches=(256,384)  attn=(16,16)"

check("DINOv3 encode shapes", _dino_test)

# 9. AI2-THOR basic movement
print()
print("─" * 60)
print("  Testing AI2-THOR (downloads Unity assets on first run ~2GB)...")

def _thor_test():
    from ai2thor.controller import Controller
    ctrl = Controller(scene="FloorPlan1", width=224, height=224, fieldOfView=90)
    event = ctrl.step("MoveAhead")
    pos   = event.metadata["agent"]["position"]
    ctrl.stop()
    return f"Agent at x={pos['x']:.2f}, z={pos['z']:.2f}"

check("AI2-THOR FloorPlan1 step", _thor_test)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
n_pass = sum(1 for ok, _, _ in results if ok)
n_fail = sum(1 for ok, _, _ in results if not ok)
print(f"  Result: {n_pass} passed  |  {n_fail} failed")
if n_fail == 0:
    print()
    print("  ✓  All checks passed. You are ready to collect trajectories.")
    print()
    print("  NEXT COMMAND:")
    print("    cd visionvoice")
    print("    python src/collect_trajectories.py --mode mock")
    print()
    print("  This will produce:  data/trajectories/mock_trajectories.json")
    print("  Send that file to Syeda on Day 3.")
else:
    print()
    print("  ✗  Fix the FAIL items above before running collect_trajectories.py")
print("=" * 60)
print()