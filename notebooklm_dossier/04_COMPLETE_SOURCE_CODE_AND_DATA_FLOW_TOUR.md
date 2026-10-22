# 💻 Document 04: Complete Source Code & Data Flow Tour

This document provides a line-by-line, module-by-module architectural walkthrough of every key file in the **XRayBo** repository.

---

## 1. High-Level Architectural Tree

```
XRayBo/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint & middleware
│   │   ├── api/
│   │   │   └── routes.py            # REST endpoints (/detect, /generate-report, /tts)
│   │   ├── core/
│   │   │   └── config.py            # Pydantic BaseSettings & environment variables
│   │   ├── models/
│   │   │   └── detection.py         # Pydantic schemas (Detection, DiagnosticReport)
│   │   └── services/
│   │       ├── local_yolo_service.py        # OpenCV DNN C++ ONNX multi-model runtime
│   │       ├── multi_scanner_service.py     # 4-stage multi-scanning orchestrator
│   │       ├── verification_service.py      # Gemini Vision multimodal crop verification
│   │       ├── image_processing_service.py  # CLAHE, Black-Hat, Top-Hat & Sobel filters
│   │       ├── diagnostic_service.py        # 500+ word structured clinical reports
│   │       ├── preprocessing_service.py     # Letterboxing & DICOM pixel array unpacking
│   │       └── tts_service.py               # Fish Audio voice generation & MD5 caching
│   └── models/weights/                      # ONNX neural network weights (.onnx)
├── frontend/
│   ├── src/
│   │   ├── app/page.tsx             # Next.js 15 root page
│   │   ├── components/
│   │   │   ├── dental-dashboard.tsx # Main clinical dashboard UI & state machine
│   │   │   ├── detection/
│   │   │   │   ├── detection-overlay.tsx       # Canvas 2D bounding-box renderer
│   │   │   │   └── detection-image-preview.tsx # Radiograph viewer with zoom/pan
│   │   │   └── dicom/               # DICOM metadata inspector & tag viewer
│   │   └── hooks/
│   │       └── use-dicom-detection.ts # React Query / Fetch hook for /detect API
├── scripts/
│   ├── start-backend.js             # Node.js runner activating backend/.venv
│   └── start-frontend.js            # Node.js runner launching Next.js Turbopack
├── package.json                     # Root orchestrator: `npm run dev`
└── README.md                        # Master GitHub documentation with Mermaid diagrams
```

---

## 2. Granular Module Walkthrough

### 2.1 `backend/app/services/multi_scanner_service.py`
The master coordinator for the 4-stage pipeline.

```python
class MultiScanningService:
    def __init__(self):
        self.preprocessor = get_preprocessing_service()
        self.verifier = get_verification_service()
        self.ip_service = get_image_processing_service()
        self.local_yolo = get_local_yolo_service()
```

#### Key Methods:
- `scan_dental_image(image_path, model_id)`:
  1. Reads image via `cv2.imread(image_path)`.
  2. Calculates aspect ratio: $\text{aspect\_ratio} = \frac{W}{H}$.
  3. Executes **Stage 0** Pre-flight: `audit_result = await self.verifier.audit_radiograph(image_path)`.
  4. Executes **Stage 1**: `stage1_candidates = await self._run_stage1_diolo(img, img_w, img_h, is_panoramic)`.
  5. Evaluates consensus: If both Stage 0 finds no pathology and Stage 1 finds zero candidates, the image is confirmed healthy immediately.
  6. Executes **Stage 2**: `verified_candidates = await self._run_stage2_recheck(img, stage1_candidates)`.
  7. Executes **Stage 3**: `ip_analysis = self.ip_service.analyze_segments(img, is_panoramic)`.
  8. Packages and returns the comprehensive diagnostic telemetry payload.

---

### 2.2 `backend/app/services/local_yolo_service.py`
High-throughput, on-device neural network execution using OpenCV DNN.

#### Key Implementation Details:
- **Model Initialization (`_load_models`)**:
  ```python
  self.net_caries = cv2.dnn.readNetFromONNX(self.caries_path)
  self.net_caries.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  self.net_caries.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  ```
  Initializes the C++ graph on CPU without importing PyTorch or allocating CUDA contexts.
- **Letterbox Transform (`detect`)**:
  Resizes the input radiograph to $640 \times 640$ while maintaining aspect ratio, padding the remaining boundary with gray (`[114, 114, 114]`).
- **Dynamic Relative Filtering**:
  ```python
  # Discards anatomical false alarms that span unrealistic proportions of the film
  if (bw / img_w > 0.35) or (bh / img_h > 0.40):
      continue
  ```
  *(Crucial fix: Replaced hardcoded pixel size limits that previously discarded adult teeth on high-resolution panoramic films).*

---

### 2.3 `backend/app/services/verification_service.py`
Multimodal Vision-Language Model interface for Stage 0 Audit and Stage 2 Crop Verification.

#### Key Implementation Details:
- **Global Radiologist Audit (`audit_radiograph`)**:
  Downscales panoramic images to a max dimension of $1024\text{ px}$ for fast network transfer ($<1\text{ second}$) and issues the `AUDIT_PROMPT`.
- **Targeted Crop Verification (`verify_detections`)**:
  For each Stage 1 candidate bounding box:
  ```python
  # 1. Expand region by 25% for anatomical context
  margin_x = int(box_w * 0.25)
  margin_y = int(box_h * 0.25)
  crop = img[y1 - margin_y : y2 + margin_y, x1 - margin_x : x2 + margin_x]

  # 2. Draw a vivid RED bounding box around the candidate lesion
  cv2.rectangle(crop_with_box, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

  # 3. Encode to JPEG base64 and send to Gemini Vision with VERIFICATION_PROMPT
  ```
- **Strict Clinical Schema Enforcement**:
  Enforces JSON output containing:
  ```json
  {
    "verified": true,
    "corrected_class": "Cavity",
    "actual_condition": "Demineralization of occlusal enamel",
    "confidence_adjustment": 0.95,
    "reasoning": "Lesion is strictly on the tooth crown; not bone marrow."
  }
  ```

---

### 2.4 `backend/app/services/image_processing_service.py`
Classical signal processing and mathematical morphology.

```python
class ImageProcessingService:
    def __init__(self):
        self.clahe_clip = 3.0
        self.clahe_grid = (8, 8)
        self.morph_kernel_size = (15, 15)
```

#### Algorithms Implemented:
1. `cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))`: Equalizes contrast adaptively across $64$ local sub-tiles.
2. `cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)`: Isolates localized radiolucent cavities using an elliptical structuring element.
3. `cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)`: Isolates radiopaque restorations and dense enamel cusps.
4. `cv2.magnitude(sobelx, sobely)`: Calculates the 2D gradient magnitude map to evaluate enamel edge smoothness.
5. Base64 Heatmap Generation: Color-maps the morphology matrix using `cv2.COLORMAP_JET` and returns a data URL for real-time frontend canvas rendering.

---

### 2.5 `backend/app/services/diagnostic_service.py`
Clinical diagnostic reporting conforming to board-certified radiologist standards.

- Model: `gemini-3.5-flash-lite` (or configured Gemini Vision model).
- Enforces an absolute **$500+$ word non-negotiable report requirement** broken down into:
  1. **Radiographic Analysis**: Specific tooth coordinates, depth into dentin/enamel, radiolucency shape.
  2. **Clinical Significance**: Bacterial etiology (*S. mutans*), risk of pulp necrosis, periapical spread timeline ($3\text{--}6\text{ months}$).
  3. **Recommended Treatment Protocol**: Anesthesia (2% lidocaine 1:100,000 epi), etching with 37% phosphoric acid, universal bonding agent, incremental composite resin placement.
  4. **Prognosis & Follow-Up**: Expected restoration longevity ($7\text{--}15\text{ years}$), recall schedule.
  5. **Home Care Instructions**: Fluoride concentration ($1450\text{ ppm}$), modified Bass brushing technique, interdental flossing.

---

### 2.6 `backend/app/services/tts_service.py`
Text-to-Speech audio synthesis for patient and doctor communication.

- API Provider: Fish Audio API (`s2.1-pro-free` model) with reference voice ID `0b74ead073f2474a904f69033535b98e` (calm, authoritative clinical tone).
- Caching Strategy:
  - In-memory dictionary: `_audio_cache[cleaned_text] -> bytes`.
  - Disk persistence: Computes `hashlib.md5(text).hexdigest()` and writes to `backend/audio_cache/<hash>.mp3`. Subsequent requests for identical diagnoses return instantaneously with zero API cost.

---

### 2.7 `frontend/src/components/dental-dashboard.tsx`
The primary clinical user interface built on Next.js 15 and React 19.

#### Key Features:
- **Multi-Format Ingestion**: Supports `.dcm` (DICOM), `.png`, `.jpg`, and `.jpeg`.
- **Dual-Layer Canvas Renderer**:
  - Base layer: High-resolution radiographic film with pan/zoom transformation matrix.
  - Overlay layer: Vectorized bounding boxes color-coded by pathology:
    - 🔴 **Red**: Active Dental Caries / Cavity
    - 🟣 **Purple**: Periapical Lesion / Apical Periodontitis
    - 🟡 **Yellow**: Impacted / Unerupted Tooth
    - 🔵 **Blue**: Dental Restoration / Filling
- **Heatmap Toggle**: Instantly superimposes the Stage 3 morphological Black-Hat diagnostic heatmap over the film with adjustable opacity.
- **Audio Diagnostic Player**: Integrated HTML5 Audio player streaming MP3 audio from the backend TTS cache.
- **Client-Side PDF Generation**: Uses `jspdf` and `jspdf-autotable` to generate printable, high-resolution diagnostic reports complete with hospital headers, tooth diagrams, and clinical recommendations.
