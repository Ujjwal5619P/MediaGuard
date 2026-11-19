# MediaGuard — Verify Before You Trust

MediaGuard is an image-forensics demo with a React + Vite frontend and a FastAPI backend. It records investigations in SQLite and reports probabilistic AI-image and face-manipulation indicators. It is an investigative aid, not proof of authenticity.

## What it includes

- Dashboard, image investigation, metadata and SHA-256 inspection, history, analytics, exports, model status, and methodology pages.
- Upload validation, EXIF orientation handling, bounded file size and dimensions, immediate deletion of the uploaded original, and SQLite retention.
- Primary AI-generated-image detector: **Imager-1-Tiny**, a ResNet-50 classifier, always run in FP32 on CPU.
- Optional local pipelines: UnivFD comparison, EfficientNet-B0 face manipulation analysis with Grad-CAM, and DINOv2 features/patch attention. These stay enabled in normal local mode and can be disabled together for a small CPU deployment.

The existing database tables and API v1 response structure remain in place. Historical `smaug_*` database and evaluation field names are retained for compatibility; the primary model recorded in new scans is Imager-1-Tiny.

## Detector and preprocessing

The primary checkpoint is downloaded from [Imagers/Imager-1-tiny](https://huggingface.co/Imagers/Imager-1-tiny) at revision `3aeaac5` and checked against SHA-256 `80e01b0a5c180ed2391c5763accbc2b9b4821c5f12299f95b08fcbe09442bd8d`. Its 94.4 MB `.pth` checkpoint carries metadata specifying ResNet-50, 224px input, class mapping `{real: 0, ai: 1}`, and threshold `0.385`. MediaGuard verifies this metadata before loading the weights.

The released checkpoint has a two-logit classifier head, although the model card's sample loader describes a one-logit head. MediaGuard follows the actual checkpoint shape, uses softmax class 1 as the AI score, and applies the checkpoint's 0.385 threshold. Loading is strict; unexpected weights or metadata cause a clear detector error instead of a prediction.

Preprocessing is EXIF-transposed RGB, resize the shorter edge to 256px, center-crop to 224×224, tensor conversion, and ImageNet normalization (mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`). The primary detector does not call CUDA, even if a GPU is installed.

The model card reports 97.03% validation accuracy, 0.9737 macro F1, 0.9608 AI-class F1, and 0.982 ROC-AUC. These are author-reported results, not independent testing or a promise of performance on unseen images. Its training data lists Hemg's research-intended dataset and Defactify; review those dataset terms separately from the Apache-2.0 model-weight license. Scores are uncalibrated signals and may change with generators, compression, editing, cropping, or domain.

## Local setup — existing Windows Conda environment

Use the existing `mediaguard` environment. No environment recreation or package upgrade is needed when its current dependencies are present.

### 1. Prepare the primary checkpoint

```powershell
cd "C:\Users\ujjwa\Downloads\mediaguard (3)\backend"
conda activate mediaguard
python setup_models.py
```

This downloads/verifies the primary detector only. To also download optional local models if they are missing, run `python setup_models.py --include-optional`. That includes the much larger UnivFD/CLIP checkpoint. Existing model files are not removed by setup.

### 2. Start FastAPI

Open a PowerShell window:

```powershell
cd "C:\Users\ujjwa\Downloads\mediaguard (3)\backend"
conda activate mediaguard
$env:MEDIAGUARD_AUTO_DOWNLOAD_MODELS = "false"
$env:MEDIAGUARD_LIGHTWEIGHT_MODE = "false"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Normal local mode keeps the existing optional UnivFD, face, and DINOv2 pipelines available. The primary Imager-1-Tiny detector still runs on CPU. For a CPU-only primary-detector demonstration, set `MEDIAGUARD_LIGHTWEIGHT_MODE="true"`; UnivFD, face analysis, and DINOv2 then report unavailable explicitly.

### 3. Start the React frontend

In another PowerShell window:

```powershell
cd "C:\Users\ujjwa\Downloads\mediaguard (3)"
npm run dev -- --host 127.0.0.1
```

Open [http://localhost:3000](http://localhost:3000). During local Vite development, the API defaults to `http://127.0.0.1:8000`. Check **System Settings → API Health** if the backend does not connect.

### Existing environment versions checked

- `mediaguard`: Python 3.10.22, PyTorch 2.5.1+cu121, Transformers 4.51.3, torchvision 0.20.1+cu121.
- `mediaguard-smaug`: Python 3.10.22, PyTorch 2.6.0+cu124, Transformers 5.20.2, torchvision 0.21.0+cu124.

The project requirements now keep Transformers below 5 and Hugging Face Hub below 1 to remain compatible with the original `mediaguard` environment. Do not run a dependency upgrade just to start the demo. If a package is missing, inspect its installed version first.

## Environment variables

Backend settings use the `MEDIAGUARD_` prefix. See [backend/.env.example](backend/.env.example). The useful deployment/local settings are:

| Variable | Purpose |
| --- | --- |
| `MEDIAGUARD_HOST` / `MEDIAGUARD_PORT` | Backend bind values; the local command above binds to loopback. |
| `MEDIAGUARD_CORS_ORIGINS` | JSON array of allowed frontend origins. Add the exact Vercel origin if hosted. |
| `MEDIAGUARD_MAX_UPLOAD_BYTES` | Upload cap; default 15 MiB. |
| `MEDIAGUARD_MAX_IMAGE_PIXELS` / `MEDIAGUARD_MAX_DIMENSION` | Image safety limits. |
| `MEDIAGUARD_DECISION_THRESHOLD` | Primary threshold; default 0.385. |
| `MEDIAGUARD_LIGHTWEIGHT_MODE` | When true, loads only Imager-1-Tiny and reports optional detectors unavailable. |
| `MEDIAGUARD_AUTO_DOWNLOAD_MODELS` | Download the primary checkpoint at startup when it is absent. |

Frontend configuration uses `VITE_API_BASE_URL`; see [.env.example](.env.example). Leave it unset for local Vite development. For a deployed frontend, set it to the HTTPS backend URL in the frontend host's build environment, then rebuild/redeploy. No secret belongs in a `VITE_*` variable.

## Tests and evaluation

Run the focused primary checkpoint test:

```powershell
cd backend
conda activate mediaguard
python -m pytest -q tests/test_lightweight_cpu.py
```

To measure quality on labeled samples, place images into `dataset/ai_generated/` and `dataset/real/`, with sources and generators documented. Then run:

```powershell
cd backend
python evaluate_dataset.py --dataset-dir ..\dataset --split-role evaluation --threshold 0.385 --no-univfd --export-csv eval.csv --export-json eval.json
```

The evaluator reports accuracy, precision, recall, F1, false-positive/false-negative rates, and ROC-AUC when scores permit. Keep a final test split untouched until threshold selection is complete. No labeled sample dataset is included in this repository, so this project run does not claim independent accuracy, precision, recall, F1, or ROC-AUC.

## CPU benchmark from this workspace

The primary model loaded successfully in `mediaguard` and inferred on CPU while CUDA was installed. A 256×256 local smoke image took 294 ms in a direct engine run; an HTTP upload took 151 ms in one warm server request. These are single-image observations on this laptop, not a general benchmark or dataset evaluation. Peak process RAM was not measured. The checkpoint file occupies 94.4 MB on disk.

ONNX Runtime was not installed in either existing environment. The tested PyTorch CPU path was already below 0.3 seconds for the smoke image, so no ONNX export or quantization was added without a validated need. No CUDA call is made by the primary detector. Real-vs-AI labeled image examples were not present locally; the API smoke test only verified a valid image request, not a correct ground-truth classification.

## Cloud deployment assessment

Vercel Hobby can serve the React frontend and its Python runtime can run FastAPI functions. However, its function request/response body limit is 4.5 MB, below MediaGuard's configured 15 MiB upload limit. Serverless local SQLite is also not durable or shared across instances, so scan history and analytics would not be reliable without adding a persistent database and changing the upload path/limit. Vercel Hobby functions currently have 2 GB memory and 1 vCPU, with included monthly usage caps; this does not solve those data-contract issues. See [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python), [Vercel function limits](https://vercel.com/docs/functions/limitations), and [Vercel Hobby plan](https://vercel.com/docs/plans/hobby).

Render's free web service has 512 MB RAM and 0.1 CPU, sleeps after 15 minutes without traffic, loses local filesystem changes on restart/spin-down, and takes about a minute to wake. That is not a reliable host for this PyTorch service plus its model and SQLite history. See [Render free services](https://render.com/docs/free) and [Render compute plans](https://render.com/docs/compute-plans).

The Render container preparation and measured memory blocker are documented in [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md). No Render Blueprint is included while the measured runtime exceeds the Free memory limit.

Conclusion: the Vercel frontend can be deployed on its free Hobby plan, but no currently verified free backend configuration meets the existing upload size, model runtime, and durable SQLite history requirements. No backend was deployed, and no billing was enabled. To use Vercel Functions, the application would need a durable external database and a changed upload design/limit; that is a separate architecture change.

## 5-minute demonstration guide

1. Start the backend and frontend using the commands above.
2. Open the dashboard and point out service health, model readiness, and recent investigation history.
3. Open **New Investigation**, choose a valid JPEG, PNG, or WebP, and run analysis. Point out the primary score, 0.385 threshold, CPU device, SHA-256, and extracted metadata.
4. Explain that the score is probabilistic and that lightweight mode disables optional pipelines rather than fabricating results.
5. Open **History** and select the new record; then show **Analytics** and the detector/methodology pages.
6. If a face image is available and normal local mode is on, demonstrate the separate face-manipulation indicator and Grad-CAM. If not, explain its scope and show its readiness status.

## Troubleshooting

- **Primary model unavailable:** Run `python setup_models.py` in `backend`. If offline, connect once so Hugging Face can supply the checkpoint.
- **Checkpoint load error:** The loader checks the architecture, class mapping, input size, threshold, and SHA-256. Do not bypass a mismatch; rerun setup and inspect `backend/models/manifest.json`.
- **Backend connection or wake delay:** Confirm Uvicorn is running on port 8000. A hosted sleeping service may need a minute to wake; retry the health check. The UI reports backend-unavailable errors.
- **CORS error after frontend hosting:** Add the exact HTTPS frontend origin to `MEDIAGUARD_CORS_ORIGINS` and restart the backend.
- **Optional models unavailable:** Normal local mode requires their cached weights. `python setup_models.py --include-optional` downloads them. Lightweight mode deliberately disables those pipelines.
- **Invalid upload:** Only JPEG, PNG, and WebP signatures are accepted. Default limits are 15 MiB, 36 MP, and 8192 pixels per dimension.
- **Existing Conda environment:** Do not recreate it. Check `python --version`, `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`, and `python -m pip show fastapi torchvision safetensors` before considering any package installation.
