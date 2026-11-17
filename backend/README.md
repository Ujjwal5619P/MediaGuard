# MediaGuard — Verify Before You Trust (FastAPI Backend)

MediaGuard is an **AI-assisted image forensics** backend built with FastAPI, PyTorch, Transformers, Sentence-Transformers, OpenCV, and SQLite. It executes **`abacusai/Smaug-AI-Image-Detector`** as the primary AI-generated image detector alongside **UnivFD** (secondary comparison), **EfficientNet-B0 FF++ C23** (face manipulation + Grad-CAM), and **DINOv2 ViT-S/14** (feature extraction + patch attention), optimized sequentially for an **NVIDIA RTX 2050 (4 GB VRAM)** with active GPU-to-CPU offloading and automatic **CPU fallback**.

## Verified Pretrained Models

1. **Pipeline A (Primary): `abacusai/Smaug-AI-Image-Detector`** — **Apache-2.0 License** (~909.8 MB / 867.6 MiB)
   - Official `SmaugImageDetect.from_pretrained(...)` loader (`EmbeddingGemma-2` vision encoder, 439M params).
   - Preprocessing & Inference: Shortest side 576px (upscaling sub-528px inputs), three `528×528` crops (`views(img, crops=3)`), `BUDGET=140` tokens -> `torch.sigmoid(logits.mean(1))`. Uses `bfloat16` on supported CUDA GPUs or `float32` on CPU (never `fp16`).
2. **Pipeline A (Secondary Comparison): UniversalFakeDetect / UnivFD (`Arjunkalliyadath/ai-image-detector`)** — **MIT License**
   - Pinned `openai/clip-vit-large-patch14` (`224×224` Bicubic CLIP normalization) -> L2-normalized 768-d embedding -> linear head `siddharthksah/deepsafe-weights` (`universalfakedetect/fc_weights.pth`). Retained strictly as a secondary comparison indicator (`univfd_indicator`), never blended with Smaug.
3. **Pipeline B: Face Manipulation Detector (`Xicor9/efficientnet-b0-ffpp-c23`)** — **MIT / FaceForensics++ Research-Only**
   - `torchvision.models.efficientnet_b0` trained on FaceForensics++ C23 (`DeepFake`, `FaceSwap`, `Face2Face`, `NeuralTextures`; `0=Real, 1=Fake`), executed on OpenCV Haar-detected face crops (+25% margin) with genuine CNN **Grad-CAM** overlays on `features[-1]`.
4. **DINOv2 ViT-S/14 Feature Extractor (`facebook/dinov2-small`)** — **Apache-2.0**
   - Extracts real 384-dimensional `[CLS]` embeddings and optional `[CLS]`-to-patch attention overlays (`POST /api/v1/features/dinov2` and `POST /api/v1/analyze/image`). Strictly non-diagnostic.

## Windows PowerShell + Conda Setup

> **Security Notice**: Run this service strictly bound to `127.0.0.1` during local development. Original uploaded images are deleted immediately after inference.
> **Python 3.10 Compatibility**: `backend/requirements.txt` pins `transformers>=5.20.0,<5.21.0` and `sentence-transformers>=6.1.0,<7.0.0` so Python 3.10 Conda environments satisfy Smaug without requiring Python 3.11+.

### Existing `mediaguard` Environment (Recommended)

```powershell
cd backend
conda activate mediaguard
pip install -r requirements.txt
python setup_models.py
pytest -v
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

### Fresh Environment

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

## Offline Local Dataset Evaluation CLI

Evaluates local images stored in `<dataset_dir>/ai_generated/` (`AI_GENERATED`) and `<dataset_dir>/real/` (`REAL`) without any training, fine-tuning, or threshold auto-tuning:

```powershell
python evaluate_dataset.py --dataset-dir ..\dataset --split-role evaluation --threshold 0.50 --export-csv eval.csv --export-json eval.json
```

## Normalized API v1 Endpoints

- `GET /api/v1/health` — Liveness, CUDA verification, measured VRAM, version (`1.2.0`), and per-model readiness.
- `GET /api/v1/models/status` — Persistent checkpoint manifest (4 models), SHA-256 verification, active device, and last latency/VRAM telemetry.
- `POST /api/v1/models/setup` — Resumable download & SHA-256 verification of any missing checkpoints.
- `POST /api/v1/features/dinov2` — Real 384-d DINOv2 ViT-S/14 feature extraction.
- `POST /api/v1/analyze/image` — Full image analysis pipeline (magic-byte/dimension/pixel-bomb validation, immediate temp-file deletion, sequential Smaug primary + UnivFD comparison + EfficientNet-B0 + DINOv2 overlays, SQLite persistence).
- `GET /api/v1/analytics` — Real aggregate SQLite detection & runtime statistics (Smaug score histogram, UnivFD comparison count, daily activity, device/MIME breakdown).
- `GET /api/v1/evaluation/inspect-folder`, `POST /api/v1/evaluation/run`, `GET /api/v1/evaluation/runs`, `GET /api/v1/evaluation/runs/{evaluation_id}`, `GET /api/v1/evaluation/runs/{evaluation_id}/export` — Local dataset evaluation, confusion matrix metrics, and CSV/JSON export.
- `GET /api/v1/artifacts/{filename}` — Serves generated Grad-CAM / DINOv2 patch attention PNGs.
- `GET /api/v1/scans`, `GET /api/v1/scans/{scan_id}`, `DELETE /api/v1/scans/{scan_id}`, `DELETE /api/v1/scans/{scan_id}/artifact`, `POST /api/v1/scans/purge-expired`, `GET /api/v1/scans/export` — SQLite scan history & split retention controls.
- `POST /api/v1/analyze/video` & `POST /api/v1/analyze/audio` — Queued modalities (`501 Not Implemented`).
