# ============================================================
# detector.py  |  Track C  |  Owner: Riya Bhart
# Conditional YOLOE + EasyOCR object & text detection
#
# YOLOE is open-vocabulary — detects any object described in
# plain English text. No retraining needed.
# Only activates when JEPA surprise > threshold (conditional).
#
# Week 2: Build & test on 50+ static campus photos
# Week 3: Integrate with navigator.py surprise trigger
# ============================================================

import numpy as np
from PIL import Image

# ── YOLOE import (handle both class names) ────────────────────────
try:
    from ultralytics import YOLOE
    _YOLOE_CLS = YOLOE
except (ImportError, AttributeError):
    from ultralytics import YOLO
    _YOLOE_CLS = YOLO
    print("[detector] YOLOE alias not available — using YOLO class")

import easyocr


# ■■ ConditionalDetector ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class ConditionalDetector:
    """YOLOE + EasyOCR conditional detector for campus navigation.

    YOLOE detects objects described in plain text (open-vocabulary).
    EasyOCR reads signage, room numbers, and notices.
    Both only activate when JEPA surprise exceeds the threshold.

    Usage:
        det = ConditionalDetector()
        objects, ocr_text = det.run(pil_frame)
    """

    CAMPUS_CLASSES = [
        "person", "door", "sign", "notice board",
        "water cooler", "stairs", "wheelchair ramp",
        "fire extinguisher", "elevator", "pillar"
    ]

    def __init__(self, surprise_threshold=0.25):
        """
        Args:
            surprise_threshold: JEPA error above this triggers detection
        """
        self.threshold = surprise_threshold

        # ── YOLOE ────────────────────────────────────────────────
        print("[ConditionalDetector] Loading YOLOE model...")
        self.yoloe = _YOLOE_CLS("yoloe-11s-seg.pt")

        try:
            self.yoloe.set_classes(list(self.CAMPUS_CLASSES))
            print(f"[ConditionalDetector] Classes set: {self.yoloe.names}")
        except Exception as e:
            print(f"[ConditionalDetector] WARNING: set_classes() failed: {e}")
            print("  → This usually means mobileclip_blt.ts is corrupted.")
            print("  → Delete the .ts file and let ultralytics re-download it.")
            print("  → Detection will use default COCO classes as fallback.")

        # ── EasyOCR ──────────────────────────────────────────────
        print("[ConditionalDetector] Loading EasyOCR (English)...")
        self.ocr = easyocr.Reader(["en"])
        print("[ConditionalDetector] Ready.")

    # ── Primary detection method ─────────────────────────────────
    def run(self, pil_frame):
        """Run YOLOE + EasyOCR on a single frame.

        Args:
            pil_frame: PIL Image

        Returns:
            (objects_list, ocr_text_string)
            - objects_list: list of detected class names (conf > 0.5)
            - ocr_text_string: " | "-joined OCR texts (conf > 0.5, len > 2)
        """
        arr = np.array(pil_frame)

        # ── YOLOE detection ──────────────────────────────────────
        results = self.yoloe(arr, verbose=False)
        objects = []
        for box in results[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = self.yoloe.names[cls_id]
            if conf > 0.5:
                objects.append(cls_name)

        # ── EasyOCR ──────────────────────────────────────────────
        ocr_results = self.ocr.readtext(arr)
        texts = []
        for _, text, conf in ocr_results:
            # Filter: conf > 0.5, min 3 chars
            if conf <= 0.5 or len(text) <= 2:
                continue
            # Filter: ALLCAPS logos (≤4 chars, likely brand marks)
            if text.isupper() and len(text) <= 4:
                continue
            texts.append(text)

        ocr_text = " | ".join(texts) if texts else ""
        return objects, ocr_text

    # ── Dynamic class management ─────────────────────────────────
    def add_custom_class(self, new_class):
        """Add a new detectable class without retraining.

        Args:
            new_class: string — e.g. "bench", "trash can", "whiteboard"
        """
        if new_class not in self.CAMPUS_CLASSES:
            self.CAMPUS_CLASSES.append(new_class)
        try:
            self.yoloe.set_classes(list(self.CAMPUS_CLASSES))
            print(f"[ConditionalDetector] Added '{new_class}' — "
                  f"now {len(self.CAMPUS_CLASSES)} classes")
        except Exception as e:
            print(f"[ConditionalDetector] WARNING: set_classes() failed "
                  f"after adding '{new_class}': {e}")


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import glob

    print("=" * 60)
    print("detector.py — self-test on campus photos")
    print("=" * 60)

    # Create detector
    det = ConditionalDetector()

    # Find campus photos
    photo_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "campus_photos"
    )
    extensions = ("*.png", "*.jpg", "*.jpeg")
    photo_paths = []
    for ext in extensions:
        photo_paths.extend(glob.glob(os.path.join(photo_dir, ext)))
    photo_paths = sorted(photo_paths)

    print(f"\nFound {len(photo_paths)} campus photos in {photo_dir}")

    if not photo_paths:
        print("No photos found — creating dummy test image")
        dummy = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
        objects, ocr_text = det.run(dummy)
        print(f"  Objects: {objects}")
        print(f"  OCR:     '{ocr_text}'")
    else:
        # Run on all photos
        total_objects = 0
        total_texts = 0
        for i, path in enumerate(photo_paths):
            img = Image.open(path).convert("RGB")
            objects, ocr_text = det.run(img)
            total_objects += len(objects)
            total_texts += (1 if ocr_text else 0)

            basename = os.path.basename(path)
            obj_str = ", ".join(objects) if objects else "(none)"
            ocr_str = ocr_text if ocr_text else "(none)"
            print(f"  [{i+1:2d}/{len(photo_paths)}] {basename}")
            print(f"         Objects: {obj_str}")
            print(f"         OCR:     {ocr_str}")

        print(f"\n{'='*60}")
        print(f"Total detections:  {total_objects} objects across "
              f"{len(photo_paths)} images")
        print(f"Images with text:  {total_texts}/{len(photo_paths)}")

    # Test add_custom_class
    print(f"\nTesting add_custom_class('bench')...")
    det.add_custom_class("bench")
    assert "bench" in det.CAMPUS_CLASSES, "bench not added to class list"
    print(f"  [PASS] Classes now: {len(det.CAMPUS_CLASSES)}")

    print(f"\n{'='*60}")
    print("detector.py self-test complete.")
    print(f"  CAMPUS_CLASSES: {det.CAMPUS_CLASSES}")
    print(f"  OCR filtering:  conf > 0.5, len > 2, no ALLCAPS logos ≤ 4 chars")
    print("=" * 60)
