# Vision-to-Voice
### A Predictive & Socially Aware Navigation Assistant for Campus Accessibility

---


| | |
|---|---|
| [<img src="https://img.shields.io/badge/▶%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=googledrive" />](https://drive.google.com/file/d/1cW3rZMu2IV5sggFjb7kgtAt0j03ZKNJk/view?usp=sharing) | [<img src="https://img.shields.io/badge/📄%20Read%20Report-Google%20Drive-red?style=for-the-badge&logo=googledrive" />](https://drive.google.com/file/d/1X2Hk0NZnk3k4Ia_EqpMeWBymXmFf1l5q/view?usp=sharing) |


## What This Is

Vision-to-Voice is a memory-driven visual navigation assistant for visually impaired students on university campuses. It runs on a camera feed, builds a topological map of the environment from scratch, predicts what comes next before it is fully visible, detects unexpected changes, enforces socially appropriate navigation behaviour, and narrates everything to the user in plain language — without any pre-loaded map, without GPS, and without a human guide.

The system is demonstrated in the AI2-THOR simulation environment and implements five peer-reviewed research papers in a single integrated pipeline.

---

## The Problem

A visually impaired student enters a university building for the first time. There is no reliable indoor navigation system. Maps are static and go out of date. Corridors change. Rooms get reassigned. Signs are unreadable without sight. Existing systems require pre-built maps, have fixed object vocabularies, cannot read signage, and cannot explain their decisions.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  AI2-THOR Simulator (RGB frame 224×224 + pose)  │
└──────────────────┬──────────────────────────────┘
                   │
          ┌────────▼────────┐
          │  DINOv2 ViT-S   │  ← FROZEN, self-supervised
          │  (dinov2_vits14)│
          └──┬──────┬───────┘
             │      │
         CLS token  Patch tokens + Attention maps
             │      │
    ┌────────▼──┐  ┌▼──────────────────┐
    │  JEPA-lite│  │  Cognitive Map     │
    │  World    │  │  (NetworkX graph)  │
    │  Model    │  │  OCR landmark tags │
    └─────┬─────┘  └────────┬──────────┘
          │                 │
     Surprise score    Frontier nodes
          │                 │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ Hybrid Frontier  │
          │ Selection        │
          └────────┬────────┘
                   │
          ┌────────▼────────┐     High surprise?
          │  Scene Context  │────────────────────►  YOLOE + EasyOCR
          │  MLP (Paper 3)  │                        (conditional)
          └────────┬────────┘
                   │ Final action
                   ▼
              AI2-THOR
═══ INTERPRETABILITY ══════════════════════════════
  SmoothGrad (Captum)  →  live saliency heatmaps
  AttnLRP (lxt)        →  faithful ViT attribution
  KMeans clustering    →  Clever Hans audit report
  pyttsx3              →  voice narration
```

---

## Five Core Papers

| Paper | Contribution | Our Implementation |
|-------|-------------|-------------------|
| Mirowski et al. (2017) — *Learning to Navigate Without a Map* | Auxiliary predictive tasks force better spatial representations | JEPA-lite world model trained on 5000 trajectories |
| Gupta et al. (2017) — *Cognitive Mapping and Planning* | Differentiable mapper builds spatial belief map | Topological NetworkX graph with DINOv2 features per node |
| Epstein et al. (2019) — *Why Can't I Dance in the Mall?* | Action-place compatibility is LEARNED from visual features | Scene Context MLP trained on 267 labeled frames |
| Lapuschkin et al. (2019) — *Unmasking Clever Hans Predictors* | Cluster attribution maps to find systematic shortcut patterns | AttnLRP + KMeans clustering on 100+ navigation decisions |
| Smilkov et al. (2017) — *SmoothGrad* | Noise-averaged gradient saliency removes noise, reveals signal | Captum NoiseTunnel, live heatmap on Dashboard Screen 3 |

---

## Key Components

**DINOv2 ViT-S/14** — Frozen self-supervised backbone. Produces 384-dim CLS tokens for every frame. Ablation confirms it produces 2–3× more distinct cognitive map nodes than ResNet-18 (16–22 vs 3–7 over 200 steps).

**JEPA-lite World Model** — MLP trained to predict the next scene embedding given the current embedding and action. Held-out MSE: 0.1073 (target < 0.20). Surprise signal drives both conditional YOLOE detection and frontier selection.

**Topological Cognitive Map** — NetworkX graph where each node stores world position, DINOv2 features, JEPA surprise score, and OCR-derived landmark label. Builds in real time with no pre-loaded map.

**Scene Context MLP** — Three-class MLP (move_fast / stop_wait / navigate) trained on DINOv2 embeddings of labeled corridor screenshots. Accuracy: 89.89%. Reduces inappropriate navigation actions by 100% in ablation (1.95 → 0.00 per 10 steps).

**Conditional YOLOE** — Open-vocabulary object detector (Ultralytics YOLOE). Detects any object named in plain English — "wheelchair ramp", "notice board", "water cooler" — without retraining. Activates only when JEPA surprise exceeds a calibrated threshold (typically 10–30% of steps).

**EasyOCR** — Reads text from detected signs and door labels. OCR text becomes the semantic label of the nearest cognitive map node, converting an anonymous position graph into a named landmark graph.

**AttnLRP (lxt)** — ICML 2024 successor to SpRAy from the same research group, specifically designed for transformer attention layers. Generates faithful attribution maps from DINOv2 (a ViT). KMeans clustering on 100+ maps identifies legitimate vs shortcut decision patterns.

**SmoothGrad (Captum)** — Noise-averaged gradient saliency displayed live on Dashboard Screen 3.

**pyttsx3 Narrator** — Text-to-speech voice output narrating detected objects, OCR text, and navigation progress.

---

## Ablation Results

### Ablation 1 — JEPA-Biased vs Random Frontier Selection
*FloorPlan210 (corridor scene, 295 reachable positions, 200 steps)*

| Condition | Final Coverage |
|-----------|--------------|
| JEPA-biased | **21.0%** |
| Random | 9.5% |

JEPA outperformed random by 11.5 percentage points on the corridor scene. The surprise signal effectively identifies high-novelty frontier directions in branching environments.

### Ablation 2 — Scene Context Filter (Paper 3)
*FloorPlan210, 20 episodes × 10 steps*

| Condition | Inappropriate Actions / 10 Steps |
|-----------|----------------------------------|
| Filter ON | **0.00** |
| Filter OFF | 1.95 |

100% reduction. The MLP learned socially appropriate navigation constraints purely from DINOv2 visual features — no hand-coded rules.

### Ablation 3 — DINOv2 vs ResNet-18 Backbone
*FloorPlan1, 200 steps*

| Metric | DINOv2 | ResNet-18 |
|--------|--------|-----------|
| Map nodes produced | **16–22** | 3–7 |
| Avg JEPA surprise | **0.299** | 0.163 |
| Inter-frame cosine similarity | **0.701** | 0.838 |
| Encode time (ms) | 34.9 | **19.4** |

ResNet-18 is faster but its features don't distinguish nearby positions sufficiently, causing most map nodes to be deduplicated into the same location. DINOv2 is the required backbone.

---

## Quantitative Metrics

| Metric | Target | Result |
|--------|--------|--------|
| Scene MLP accuracy | > 75% | **89.89%** |
| F1 move_fast | > 0.70 | **0.898** |
| F1 stop_wait | > 0.70 | **0.980** |
| JEPA held-out MSE | < 0.20 | **0.1073** |
| Nodes mapped (200 steps) | > 50 | **82** |
| Edge:Node ratio | > 1.0 | **1.11** |
| YOLOE trigger rate | 10–30% | **20.5%** |



---


---

## Running the System

**Full navigation run (200 steps, FloorPlan210):**
```bash
PYTHONPATH=. python src/navigator.py
```

**Train JEPA world model:**
```bash
PYTHONPATH=. python src/predictor.py
```

**Train Scene Context MLP:**
```bash
PYTHONPATH=. python src/scene_classifier.py
```

**Run all ablations:**
```bash
# Ablation 1 — JEPA vs Random frontier
PYTHONPATH=. python src/ablation_frontier.py --scene FloorPlan210 --steps 200

# Ablation 2 — Paper 3 filter ON vs OFF
PYTHONPATH=. python src/ablation_paper3.py --scene FloorPlan210

# Ablation 3 — DINOv2 vs ResNet18
PYTHONPATH=. python src/ablation_compare.py --scene FloorPlan1 --steps 200
```

**Full metrics report:**
```bash
PYTHONPATH=. python src/metrics_summary.py
```

---




## 👥 Project Contributors  

<div align="center">  <a href="https://www.linkedin.com/in/rayyanmerchant2004/" target="_blank">    <img src="https://img.shields.io/badge/Rayyan%20Merchant-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Rayyan Merchant"/>  </a>  <a href="https://www.linkedin.com/in/syedarijaali" target="_blank">    <img src="https://img.shields.io/badge/Syeda%20Rija%20Ali-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Syeda Rija Ali"/>  </a>  <a href="https://www.linkedin.com/in/riya-bhart-339036287/" target="_blank">    <img src="https://img.shields.io/badge/Riya%20Bhart-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Riya Bhart"/>  </a></div>



---


---

*Vision-to-Voice — FAST NUCES AI Capstone 2026*
