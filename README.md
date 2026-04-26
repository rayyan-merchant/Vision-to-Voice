# Vision-to-Voice — Track C (Riya Bhart)
**FAST NUCES | 6th Semester AI Project | v2.0**

> Interpretability & Detection — SmoothGrad · AttnLRP · YOLOE · EasyOCR · TTS

---

## What Riya Builds

Track C is the **interpretability, detection, and narration** layer of Vision-to-Voice. It has four modules:

| Module | Purpose | Paper |
|---|---|---|
| `saliency.py` | Generates attribution heatmaps showing WHY the agent made each decision | Paper 4 (AttnLRP) + Paper 5 (SmoothGrad) |
| `detector.py` | Detects objects and reads signs — only when JEPA surprise is high | Novel (YOLOE + EasyOCR) |
| `clever_hans.py` | Clusters 100+ attribution maps to find shortcut learning patterns | Paper 4 (Lapuschkin et al. 2019) |
| `narrator.py` | Speaks navigation events aloud to the visually impaired user | Novel (pyttsx3) |

### How They Fit in the Pipeline

```
DINOv3 frame (Rayyan) → JEPA surprise (Syeda)
                                │
                    surprise > 0.25?
                                │
                ┌───────────────┘
                │                    ┌─────────────────────────────┐
                ▼                    │         TRACK C (Riya)       │
          detector.py ──────────────►│  YOLOE detects objects       │
          (YOLOE + OCR)             │  EasyOCR reads sign text     │
                │                   │  → cognitive map gets label  │
                │                   │  → narrator speaks to user   │
                │                   └─────────────────────────────┘
                │
         saliency.py  ─── runs on EVERY frame ───►  Dashboard Screen 3
         (AttnLRP / SmoothGrad)
                │
         clever_hans.py ─── runs after 100+ maps ─►  Audit Report
         (KMeans clustering)
                │
          narrator.py ─── called by navigator.py ─►  User hears guidance
```

---

## Environment Setup

### Requirements
- Python **3.9** (strictly recommended)
- RAM: 8 GB minimum (16 GB preferred)
- GPU: not required (CPU inference works)
- Internet: required once for model downloads

### Step 1 — Create Virtual Environment
```bash
conda create -n visionvoice python=3.9 -y
conda activate visionvoice
```

### Step 2 — Install Dependencies
```bash
# Core torch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Numpy — captum requires < 2.0
pip install "numpy>=1.23,<2.0"

# Interpretability
pip install captum==0.8.0 scikit-learn

# Detection & OCR
pip install ultralytics>=8.3.0 easyocr

# AttnLRP (primary attribution method — Paper 4)
pip install lxt
# If lxt install fails:
# pip install git+https://github.com/rachtibat/LRP-eXplains-Transformers.git

# Visualization & utilities
pip install opencv-python matplotlib Pillow tqdm timm

# Text-to-speech
pip install pyttsx3

# Linux only (pyttsx3 dependency):
# sudo apt-get install espeak python3-espeak
```

### Step 3 — Install from requirements.txt
```bash
pip install -r requirements.txt
```

### Step 4 — Verify Everything
```bash
python setup_verify.py
```
All checks should pass (or warn for optional lxt).

---

## Project Structure
```
visionvoice/
├── src/
│   ├── detector.py          ← YOLOE + EasyOCR (this file)
│   ├── saliency.py          ← AttnLRP + SmoothGrad
│   ├── clever_hans.py       ← Attribution clustering audit
│   └── narrator.py          ← pyttsx3 TTS
├── data/
│   └── campus_photos/       ← Add real campus photos here
├── outputs/
│   ├── saliency_maps/       ← Saved .npy attribution maps + overlays
│   └── clusters/            ← Cluster PNGs + audit report
├── tests/
│   ├── test_detector.py
│   ├── test_saliency.py
│   ├── test_narrator.py
│   └── test_clever_hans.py
├── setup_verify.py          ← Run on all 3 laptops
└── requirements.txt
```

---

## Running Tests

```bash
cd visionvoice

# Test detector (YOLOE + EasyOCR)
python tests/test_detector.py

# Test saliency (AttnLRP + SmoothGrad)
python tests/test_saliency.py

# Test narrator (TTS)
python tests/test_narrator.py

# Test Clever Hans audit
python tests/test_clever_hans.py
```

---

## Running Individual Modules

### Detector — test on a campus photo
```bash
python src/detector.py data/campus_photos/hallway.jpg 0.5
```

### Saliency — generate a heatmap
```bash
python src/saliency.py data/campus_photos/hallway.jpg smoothgrad
python src/saliency.py data/campus_photos/hallway.jpg attnlrp   # if lxt installed
```

### Narrator — hear it speak
```bash
python src/narrator.py
```

### Clever Hans Audit — cluster collected maps
```bash
# After collecting 100+ maps from navigation runs:
python src/clever_hans.py \
    --maps_dir outputs/saliency_maps \
    --report_dir outputs/clusters \
    --k_min 3 --k_max 5
```

---

## Integration Notes for navigator.py (Rayyan + Syeda)

```python
from src.detector import ConditionalDetector
from src.saliency import SaliencyEngine, pil_to_tensor
from src.clever_hans import AttributionCollector
from src.narrator import make_narrator

# Init once at startup
detector   = ConditionalDetector(surprise_threshold=0.25)
saliency   = SaliencyEngine(dino_model=encoder.model)  # pass Rayyan's DINOv3
collector  = AttributionCollector(save_dir="outputs/saliency_maps")
narrator   = make_narrator(rate=160)

narrator.say("Vision to Voice system online.")

# Inside navigation loop:
for step in range(n_steps):
    # ... Rayyan's perception, Syeda's JEPA ...

    # Saliency — every frame
    tensor = pil_to_tensor(frame)
    heatmap = saliency.get_map(tensor, method="attnlrp")
    collector.record(frame, chosen_action, heatmap, step=step)

    # Conditional detection — only on surprise
    if detector.should_run(surprise_score):
        objects, ocr_text = detector.run(frame)
        narrator.say_detection(objects)
        if ocr_text:
            cog_map.tag_label(node_id, ocr_text)
            narrator.say_sign(ocr_text)

    # Regular narration
    if step % 5 == 0:
        narrator.say_navigation(action, len(cog_map.nodes))

collector.save_manifest()
narrator.shutdown()
```

---

## Paper Connections

| Component | Paper | Why |
|---|---|---|
| AttnLRP in `saliency.py` | **Paper 4** — Lapuschkin et al. 2019 + AttnLRP ICML 2024 | AttnLRP is the 2024 successor to SpRAy, co-authored by Lapuschkin. Faithfully propagates relevance through transformer attention layers. Standard LRP fails for ViTs. |
| SmoothGrad in `saliency.py` | **Paper 5** — Smilkov et al. 2017 | Direct implementation: 50 noisy copies, stdevs=0.15, averaged gradient attribution. |
| `clever_hans.py` clustering | **Paper 4** intent | Cluster attribution maps → find systematic shortcut patterns. A shortcut FINDING is a research result. |
| YOLOE + EasyOCR in `detector.py` | Novel addition | Open-vocabulary detection for accessibility use case. No custom training needed. |
| `narrator.py` | Novel addition | Closes the accessibility loop. Makes the system usable. |

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `lxt not found` | `pip install lxt` or `pip install git+https://github.com/rachtibat/LRP-eXplains-Transformers.git`. SmoothGrad works as fallback. |
| `YOLOE requires ultralytics >= 8.3.0` | `pip install --upgrade ultralytics` |
| `pyttsx3 no default output device` (Linux) | `sudo apt-get install espeak python3-espeak` |
| `numpy >= 2.0 conflict with captum` | `pip install "numpy>=1.23,<2.0"` |
| `YOLOE fires on every frame` | Increase `surprise_threshold` from 0.25 to 0.35-0.40 |
| `KMeans silhouette < 0.15` | Collect more navigation data (need 100+ diverse decisions) |
| `EasyOCR returning garbage` | Filter: `conf > 0.5` and `len(text) >= 3` (already implemented) |
