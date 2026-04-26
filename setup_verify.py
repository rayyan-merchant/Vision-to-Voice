"""
setup_verify.py — Vision-to-Voice Track C
Environment verification script.

Run AFTER installing all dependencies to confirm everything is working.
Run this on ALL THREE LAPTOPS before Week 1 ends.

    python setup_verify.py
"""

import sys
import importlib

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")


def check_python():
    print(f"\n{BOLD}[1] Python Version{RESET}")
    v = sys.version_info
    if v.major == 3 and v.minor == 9:
        ok(f"Python {v.major}.{v.minor}.{v.micro} ✓")
    elif v.major == 3 and v.minor in (10, 11):
        warn(f"Python {v.major}.{v.minor}.{v.micro} — 3.9 recommended, but 3.10/3.11 may work")
    else:
        fail(f"Python {v.major}.{v.minor} — use Python 3.9 for best compatibility")


def check_import(name, package=None, min_ver=None):
    try:
        mod = importlib.import_module(package or name)
        ver = getattr(mod, "__version__", "unknown")
        if min_ver and ver != "unknown":
            from packaging.version import Version
            if Version(ver) < Version(min_ver):
                warn(f"{name} {ver} — recommended >= {min_ver}")
                return False
        ok(f"{name} {ver}")
        return True
    except ImportError:
        fail(f"{name} — NOT INSTALLED. Run: pip install {package or name}")
        return False


def check_numpy():
    print(f"\n{BOLD}[2] Core Libraries{RESET}")
    try:
        import numpy as np
        ver = np.__version__
        from packaging.version import Version
        if Version(ver) >= Version("2.0"):
            fail(f"numpy {ver} — captum requires numpy < 2.0. Run: pip install 'numpy>=1.23,<2.0'")
        else:
            ok(f"numpy {ver} (< 2.0 ✓)")
    except ImportError:
        fail("numpy — NOT INSTALLED")


def check_torch():
    try:
        import torch
        ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA not available — CPU only (fine for inference, slow for JEPA training)")
    except ImportError:
        fail("torch — NOT INSTALLED. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")


def check_track_c_libraries():
    print(f"\n{BOLD}[3] Track C Libraries (Riya){RESET}")
    results = {}
    results["captum"] = check_import("captum", min_ver="0.8.0")
    results["sklearn"] = check_import("scikit-learn", "sklearn")
    results["ultralytics"] = check_import("ultralytics", min_ver="8.3.0")
    results["easyocr"] = check_import("easyocr")
    results["pyttsx3"] = check_import("pyttsx3")
    results["cv2"] = check_import("opencv-python", "cv2")
    results["matplotlib"] = check_import("matplotlib")
    results["PIL"] = check_import("Pillow", "PIL")
    return results


def check_attnlrp():
    print(f"\n{BOLD}[4] AttnLRP (lxt — Optional but Important){RESET}")
    try:
        import lxt
        ok(f"lxt (AttnLRP) {getattr(lxt, '__version__', 'installed')}")
        try:
            from lxt.models.vit import dino_vits14_lrp
            ok("lxt.models.vit.dino_vits14_lrp importable ✓")
        except Exception as e:
            warn(f"dino_vits14_lrp import issue: {e}")
    except ImportError:
        warn("lxt not installed — AttnLRP unavailable. SmoothGrad will be used as fallback.")
        warn("To install: pip install lxt")
        warn("Or from GitHub: pip install git+https://github.com/rachtibat/LRP-eXplains-Transformers.git")


def check_yoloe():
    print(f"\n{BOLD}[5] YOLOE Verification{RESET}")
    try:
        from ultralytics import YOLOE
        import numpy as np

        print("  Attempting to load YOLOE small model ...")
        model = YOLOE("yoloe-11s-seg.pt")
        model.set_classes(["person", "door", "sign", "stairs", "water cooler"])

        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        ok(f"YOLOE inference OK. Classes: {list(model.names.values())[:5]}")
    except ImportError:
        fail("ultralytics not installed")
    except Exception as e:
        warn(f"YOLOE init issue: {e}")
        warn("If model download fails, check internet connection.")


def check_easyocr():
    print(f"\n{BOLD}[6] EasyOCR Verification{RESET}")
    try:
        import easyocr
        import numpy as np
        print("  Loading English reader (may download model on first run) ...")
        reader = easyocr.Reader(["en"], verbose=False)
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        _ = reader.readtext(dummy)
        ok("EasyOCR English reader OK")
    except ImportError:
        fail("easyocr not installed. Run: pip install easyocr")
    except Exception as e:
        warn(f"EasyOCR issue: {e}")


def check_pyttsx3():
    print(f"\n{BOLD}[7] TTS Verification{RESET}")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        ok(f"pyttsx3 OK. {len(voices)} system voice(s) found.")
        if voices:
            ok(f"Default voice: {voices[0].name}")
        # Don't actually speak in verification script
    except ImportError:
        fail("pyttsx3 not installed. Run: pip install pyttsx3")
    except Exception as e:
        warn(f"pyttsx3 issue: {e}")
        warn("On Linux: sudo apt-get install espeak")
        warn("On macOS: should work out of the box")
        warn("On Windows: should work out of the box")


def check_captum():
    print(f"\n{BOLD}[8] Captum SmoothGrad Verification{RESET}")
    try:
        import torch
        import torch.nn as nn
        from captum.attr import NoiseTunnel, Saliency

        class _Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 224 * 224, 1)
            def forward(self, x):
                return self.fc(x.flatten(1))

        model = _Tiny()
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        attrs = nt.attribute(x, nt_type="smoothgrad", nt_samples=5, stdevs=0.1, target=0)
        ok(f"SmoothGrad (Captum) OK. Attribution shape: {attrs.shape}")
    except ImportError:
        fail("captum not installed. Run: pip install captum==0.8.0")
    except Exception as e:
        warn(f"Captum test failed: {e}")


def check_track_c_modules():
    print(f"\n{BOLD}[9] Track C Module Imports{RESET}")
    src_path = "src"
    sys.path.insert(0, src_path)
    for module in ["detector", "saliency", "clever_hans", "narrator"]:
        try:
            importlib.import_module(module)
            ok(f"{module}.py importable")
        except ImportError as e:
            fail(f"{module}.py import failed: {e}")
        except Exception as e:
            warn(f"{module}.py import warning: {e}")


def print_summary():
    print(f"\n{'═' * 55}")
    print(f"{BOLD}SETUP SUMMARY{RESET}")
    print(f"{'═' * 55}")
    print("If all checks passed: you are ready for Week 1.")
    print()
    print("If lxt is missing:")
    print("  → AttnLRP unavailable. SmoothGrad will be the fallback.")
    print("  → Try again: pip install lxt")
    print()
    print("If pyttsx3 has issues (Linux):")
    print("  → sudo apt-get install espeak python3-espeak")
    print()
    print("If YOLOE model download fails:")
    print("  → Check internet. Model downloads on first use (~50MB).")
    print()
    print("Next step: run python tests/test_detector.py")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    print(f"\n{'═' * 55}")
    print(f"{BOLD}Vision-to-Voice — Track C Environment Verification{RESET}")
    print(f"Riya Bhart | FAST NUCES | v2.0")
    print(f"{'═' * 55}")

    check_python()
    check_numpy()
    check_torch()

    # Core imports
    print(f"\n{BOLD}[2b] Standard Libraries{RESET}")
    for lib in ["torchvision", "sklearn", "matplotlib", "cv2", "PIL", "tqdm", "timm"]:
        check_import(lib)

    check_track_c_libraries()
    check_attnlrp()
    check_yoloe()
    check_easyocr()
    check_pyttsx3()
    check_captum()
    check_track_c_modules()
    print_summary()
