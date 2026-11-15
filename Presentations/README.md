# 🦷 XRayBo Presentation & Defense Suite

Welcome to the **XRayBo SOTA Architecture Presentation Suite**. This directory contains all presentation artifacts, detailed markdown scripts, mathematical specifications, empirical benchmark matrices, and the generated PowerPoint presentation (`.pptx`) with native **Morph transitions**.

---

## 📁 Directory Structure & Deliverables

```
Presentations/
├── NOTEBOOKLM_MASTER_PRESENTATION_SOURCE.md # 🚀 MASTER SOURCE FILE FOR NOTEBOOKLM (Upload this!)
├── 01_SOTA_PRESENTATION_DECK_AND_SCRIPT.md  # Complete Slide-by-Slide Script, Bullet Points & Speech
├── 02_ENGINE_ARCHITECTURE_AND_IMAGE_PROCESSING_METHODS.md # Technical Manual on YOLO11 & CV Algorithms
├── 03_EMPIRICAL_BENCHMARK_AND_COMPARISON_MATRIX.md # Detailed Empirical Benchmarks (96% Specificity)
├── XRayBo_SOTA_AI_Dental_Diagnostics.pptx   # 16:9 Widescreen Presentation with Morph Transitions
├── images/                                  # 🖼️ All Visual Assets & Extracted Presentation Images
│   ├── cv_diagnostic_dashboard.png          # 6-Panel Multi-Spectral Clinical Dashboard
│   ├── annotated_quadrant_overlay.png       # Segmented Panoramic Quadrant Overlay
│   ├── app_preview.png                      # Clinical Web App UI
│   ├── robot_mascot_transparent.png         # Transparent 3D Bo Mascot
│   └── old_slides/                          # Extracted Slides from the 1st PDF (Slide 01-14)
│       ├── slide_01.png ... slide_14.png
├── generate_presentation.py                 # Standalone script to regenerate PPTX
└── robot_mascot_transparent.png             # Standalone 3D Mascot asset
```

---

## 🎯 What is New in This Presentation (vs. Old Baseline)?

1. **Ultralytics YOLO11 (`liodon_dental.onnx`)**:
   - Upgraded from legacy single-model detectors to **YOLO11 featuring `C3k2` bottleneck modules and `C2PSA` (Cross-Stage Partial with Self-Attention) heads**.
   - Forward pass in **$327\text{ ms}$ on CPU** ($2.7\times$ faster).

2. **The Dual-Brain Paradigm**:
   - Decoupled **Engine Reasoning** (inductive spatial box regression) from **Multimodal LLM Reasoning** (generative semantic verification).

3. **Panoramic Quadrant Decomposition**:
   - Solved the **$2.34\times$ horizontal aspect-ratio distortion** on $2:1$ panoramic X-rays by partitioning films into overlapping $Q_1\text{--}Q_4$ quadrants ($3\%\text{ horizontal} \times 4\%\text{ vertical}$ overlap) processed at native sub-pixel resolution.

4. **Second-Stage Anti-Overfitting Regularization**:
   - **Intersection over Minimum ($\text{IoM} \ge 0.35$)** suppression instead of standard IoU to prune redundant parent tooth boxes.
   - **Patch Z-Score Normalization** ($\Delta\mu > 0.08$) to mathematically reject non-radiolucent tissue.
   - **Spatial Visual Prompting**: $25\%$ contextual tooth crop stamped with a $2\text{px}$ red target box to eliminate maxillary sinus and bone marrow false alarms.

5. **Deterministic Image Processing Methods**:
   - **CLAHE** ($8 \times 8$ grid, clip limit $3.0$) for local contrast balancing without blowing out metallic restorations.
   - **Morphological Black-Hat Transform** ($(I \bullet B) - I$) with a $15 \times 15$ elliptical structuring element to isolate dark cavitations and periapical bone resorption halos.
   - **Morphological Top-Hat Transform** ($I - (I \circ B)$) to isolate hyper-dense enamel and metallic amalgams.
   - **Sobel Differential Gradient** ($G = \sqrt{G_x^2 + G_y^2}$) to quantify enamel boundary breakdown and differentiate true caries from cervical burnout.

6. **Definitive Empirical Benchmark Table**:
   - Highlights the jump from **$0.0\%$ periapical lesion recall** (legacy baseline failure) to **$50.0\%$ periapical recall** and **$100.0\%$ caries sensitivity**.
   - Clean film specificity soaring to **$96.0\%$** with an **$86.7\%$ reduction in false alarms** ($0.67/\text{film}$ vs $5.00/\text{film}$).

7. **Native Morph Transitions**:
   - Every slide includes OpenXML `<p14:morph option="byObject"/>` transition instructions for fluid, modern slide morphing in PowerPoint / Office 365.

---

## 🚀 How to Run and Present

### 1. Open the Presentation
Double-click:
`Presentations/XRayBo_SOTA_AI_Dental_Diagnostics.pptx`
- Open in **Microsoft PowerPoint** or **Office 365** to experience the native **Morph transitions**.
- Press **F5** (or click Slide Show) to begin presenting.

### 2. Regenerate the Presentation Anytime
If you ever want to re-run the generator:
```bash
uv run --with python-pptx python Presentations/generate_presentation.py
```

---
*Created for Karrar Al-Mayaly & Mohammed Majed • XRayBo Clinical AI Engine*
