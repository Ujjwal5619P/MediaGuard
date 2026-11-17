# MediaGuard — Verify Before You Trust

MediaGuard is an **AI-assisted multimodal image forensics platform** pairing a multi-page React + Vite + TypeScript + Tailwind CSS forensic studio with an independent Python FastAPI + PyTorch + SQLite backend.

## Multi-Page Forensic Studio (Frontend)

1. **Dashboard** — Real SQLite metrics (`Total Images Analyzed`, `AI-Generated Flagged` via Smaug Primary, `Face Manipulation Flagged`, `Mean Inference Latency`), daily activity breakdown chart, 4-model readiness summary, and recent investigations table.
2. **New Investigation** — Central high-resolution forensic image canvas with drag-and-drop upload, zoom (`100%`–`200%`), drag-to-pan, Face Grad-CAM & DINOv2 patch-attention split/opacity overlay comparison, EXIF/SHA-256 inspector, task-separated indicator cards clearly distinguishing **Smaug-AI-Image-Detector (Primary)** from **UnivFD (Secondary Comparison)** and **Face Manipulation (EfficientNet-B0 FF++)**, JSON report export, and print summary.
3. **Investigation History** — Paginated SQLite ledger displaying primary Smaug scores, secondary UnivFD comparison scores, model & device telemetry, inline Investigation Details modal, single-click workspace loading, attribution artifact purge, record deletion, and CSV/JSON export.
4. **AI Models** — Live checkpoint manifest displaying roles (`Primary AI-Generated`, `Secondary Comparison`, `Face Manipulation`, `Feature Extractor`), architectures, training datasets, licenses, provenance URLs, SHA-256 digests, active devices, VRAM/latency metrics, and checkpoint setup trigger.
5. **Analytics & Dataset Evaluation** — Segmented view combining live SQLite Investigation Telemetry (Smaug probability histogram, 14-day volume chart, device/format breakdowns) with an offline **Dataset Evaluation** benchmark panel supporting local `dataset/ai_generated/` and `dataset/real/` folders, Accuracy/Precision/Recall/F1/FPR/FNR, 2×2 Confusion Matrix, and CSV/JSON export without automatic training or threshold tuning.
6. **Methodology** — Transparent documentation of preprocessing (`SmaugImageDetect` 3×528×528 multi-crop, `bf16`/`fp32`), UnivFD comparison reference, EfficientNet-B0 Grad-CAM, DINOv2 patch attention, dataset evaluation split governance, and failure modes.
7. **System Settings** — API endpoint configuration, `/api/v1/health` probe, CUDA telemetry, storage retention TTLs (`30d` records, `7d` attribution artifacts, originals deleted immediately), manual expired-record purge, API v1 contract table, and local launch commands.

## Implemented Detection Pipelines

1. **Pipeline A (Primary) — AI-Generated Image Detection (`abacusai/Smaug-AI-Image-Detector`)** — **Apache-2.0 License** (~909.8 MB / 867.6 MiB)
   - Official loader & inference: `SmaugImageDetect.from_pretrained(...)` using three `528×528` views (`views(img, crops=3)`, `BUDGET=140` vision tokens) on an `EmbeddingGemma-2` vision transformer (439M params).
   - Runs in `bfloat16` on supported CUDA GPUs or `float32` on CPU (never `fp16` per official model constraints).
2. **Pipeline A (Secondary Comparison) — UniversalFakeDetect / UnivFD (`Arjunkalliyadath/ai-image-detector`)** — **MIT License**
   - Retained explicitly as an independent comparison reference (`openai/clip-vit-large-patch14` + `fc_weights.pth`). Never fused or blended with Smaug.
3. **Pipeline B — Face Manipulation Detection (`Xicor9/efficientnet-b0-ffpp-c23`)** — **MIT / FaceForensics++ Research-Only**
   - OpenCV Haar cascade face detection (+25% margin) -> `torchvision.models.efficientnet_b0` trained on FaceForensics++ C23 (`DeepFake`, `FaceSwap`, `Face2Face`, `NeuralTextures`) + genuine CNN **Grad-CAM** overlays.
   - Returns `no_face_detected` explicitly when no face is present.
4. **Feature Extraction & Saliency (`facebook/dinov2-small`)** — **Apache-2.0**
   - Extracts real 384-d `[CLS]` embeddings (`POST /api/v1/features/dinov2` & `/api/v1/analyze/image`) and experimental `[CLS]`-to-patch attention overlays. Strictly non-diagnostic.
5. **Video & Audio Modalities**: Queued for post-v1 milestones (`501 Not Implemented`).

## Local Dataset Evaluation

Organize strictly local test/validation images as:

```text
dataset/
  ai_generated/   # Ground truth: AI_GENERATED (.jpg, .jpeg, .png, .webp)
  real/           # Ground truth: REAL (.jpg, .jpeg, .png, .webp)
```

Evaluate via the **Analytics → Dataset Evaluation** UI tab or offline CLI (never trains models or alters thresholds from evaluation images; tune thresholds only on `validation` splits and keep `final_test` untouched):

```powershell
cd backend
conda activate mediaguard
python evaluate_dataset.py --dataset-dir ..\dataset --split-role evaluation --threshold 0.50 --export-csv eval_report.csv --export-json eval_report.json
```

## Windows PowerShell Quick Start

### 1. Python FastAPI ML Backend (Existing `mediaguard` Conda Env · RTX 2050 4 GB VRAM)

> **Python 3.10 Compatibility Note**: `abacusai/Smaug-AI-Image-Detector` requires `transformers >= 5.20.0`, while `transformers >= 5.21.0` requires Python `>= 3.11`. `backend/requirements.txt` pins `transformers>=5.20.0,<5.21.0` and `sentence-transformers>=6.1.0,<7.0.0` so existing Python 3.10 Conda environments upgrade cleanly.

```powershell
cd backend
conda activate mediaguard

# Upgrade dependencies cleanly (includes transformers>=5.20.0,<5.21.0 & sentence-transformers>=6.1.0)
pip install -r requirements.txt

# Download & SHA-256 verify all 4 models into backend/models/ (resumable, migrates legacy UnivFD cache automatically)
python setup_models.py

# Run backend tests (unit + security + SQLite retention + dataset evaluation + real Smaug/UnivFD/EffNet/DINOv2 inference)
pytest -v

# Start single-worker FastAPI server strictly bound to 127.0.0.1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

If creating a fresh Conda environment from scratch:

```powershell
cd backend
conda create -n mediaguard python=3.10 -y
conda activate mediaguard
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python setup_models.py
pytest -v
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

### 2. React + Vite Frontend

```powershell
# In project root
npm install
npm run build
npm run dev
```

Open `http://localhost:3000` in your browser and verify `/api/v1/health` in **System Settings**.
