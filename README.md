# Vision-to-Voice

AI-powered predictive navigation assistant for visually impaired students that enables map-free campus navigation using semantic scene understanding, cognitive mapping, world modeling, object detection, OCR-based landmark recognition, and real-time voice guidance.

## Overview

Vision-to-Voice is a smart accessibility system designed to help visually impaired students navigate university campuses independently. Unlike traditional navigation systems that require pre-built maps or GPS, our system builds its own understanding of the environment in real time using visual perception and predictive intelligence.

The system observes surroundings through a camera feed, understands spaces using DINOv3 semantic features, builds a topological cognitive map, predicts environmental changes using JEPA-lite world modeling, detects important objects using YOLOE, reads signs using EasyOCR, and narrates useful information through voice output.

It also includes explainability and trust verification using SmoothGrad and AttnLRP to ensure decisions are interpretable and free from shortcut learning.

## Core Features

* Map-free autonomous campus navigation
* DINOv3-based semantic scene understanding
* Topological cognitive mapping with landmark memory
* JEPA-lite predictive world model for surprise detection
* Conditional YOLOE open-vocabulary object detection
* EasyOCR sign and room number recognition
* Real-time text-to-speech voice narration
* SmoothGrad saliency visualization
* AttnLRP-based Clever Hans audit for trustworthy AI

## Tech Stack

* Python
* PyTorch
* DINOv3 (Distilled ViT Backbone)
* AI2-THOR
* YOLOE
* EasyOCR
* Captum
* LXT (AttnLRP)
* NetworkX
* OpenCV
* Matplotlib


## Current Status

Project currently in development. Initial setup, architecture planning, and module-wise implementation are in progress.

## Future Work

* Full real-time dashboard integration
* Gaussian Splatting campus simulation
* Advanced scene-context action filtering
* Extended real-world deployment testing
* Mobile and wearable device support

---
## 👥 Project Contributors  

<div align="center">  <a href="https://www.linkedin.com/in/rayyanmerchant2004/" target="_blank">    <img src="https://img.shields.io/badge/Rayyan%20Merchant-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Rayyan Merchant"/>  </a>  <a href="https://www.linkedin.com/in/rija-ali-731095296" target="_blank">    <img src="https://img.shields.io/badge/Syeda%20Rija%20Ali-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Syeda Rija Ali"/>  </a>  <a href="https://www.linkedin.com/in/riya-bhart-339036287/" target="_blank">    <img src="https://img.shields.io/badge/Riya%20Bhart-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Riya Bhart"/>  </a></div>

---

Building accessibility through trustworthy AI.
