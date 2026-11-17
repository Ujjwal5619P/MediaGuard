# MediaGuard — Multimodal AI-Assisted Media Analysis & Forensics System

MediaGuard is an **AI-assisted media analysis** platform and research workspace designed to help everyday users, journalists, and researchers inspect images across two distinct forensic tasks—**AI-generated image detection** and **face manipulation detection**—using verifiable deep learning pipelines, transparent model readiness states, and zero fabricated predictions.

## User Review & Critical Decisions

> [!IMPORTANT]
> **Task Separation & Local Hardware Boundary**: MediaGuard explicitly separates **AI-generated image detection** (fully or substantially synthetic images) from **face manipulation detection** (face swaps and facial alterations in otherwise real images). It will **never** collapse both tasks into a single combined probability unless a dedicated, validated multi-task model supports it, nor assume a deepfake-trained XceptionNet checkpoint detects fully AI-generated images. Initial development runs strictly locally on **Windows using Conda and an NVIDIA RTX 2050 (4 GB VRAM)** with the React frontend and Python FastAPI backend kept completely separate, without exposing `localhost` or the development laptop to the public internet. **Milestone 1 completes and passes verification with zero pretrained model checkpoints required**, reporting all model states explicitly as `unavailable`.

- **Visual Direction (Confirmed)**: Dark forensic studio aesthetic using deep obsidian surfaces (`#07090E`), slate structural panels (`#111827`), hairline 1px borders, and calibrated signal cyan/emerald/amber/crimson accents.
- **v1 Detection Modality (Confirmed)**: Image analysis is the **only active detection modality** in v1; Video and Audio pipelines remain cleanly queued in the interface and backend router.
- **SaaS & Persistence Scope (Confirmed)**: Open public scanner UI backed by a local SQLite scan history ledger, searchable archive, and JSON/CSV report export (no user auth wall in v1).
- **Codebase Deliverables (Confirmed)**: Separate React + TypeScript + Tailwind CSS frontend and standalone Python FastAPI + PyTorch + SQLite backend codebase with Conda environment configuration.

---

## 1. Overview & Core Concept

- **What It Does**: Accepts uploaded images, performs strict magic-byte signature and decompression-bomb validation alongside deterministic metadata inspection, dispatches validated inputs to a decoupled Python FastAPI service running task-specific PyTorch pipelines, and returns separate **experimental forensic indicators** (`ai_generated_indicator` and `face_manipulation_indicator`), optional experimental attribution visualizations, and SQLite audit records.
- **Two Independent Image Analysis Tasks**:
  1. **AI-Generated Image Detection**: Identifies whether an entire image (or substantial scene content) was synthetically generated (e.g., diffusion or GAN generators) using representations from `DINOv2 ViT-S/14` paired with a task-specific trained classification head.
  2. **Face Manipulation Detection**: Identifies face swaps, expression reenactment, or localized facial alterations in otherwise real photographs using a verified deepfake-trained `XceptionNet` checkpoint (audited for its exact training dataset, intended task, architecture, preprocessing, and license) and/or task-specific head.
- **Key Value**: Enforces strict scientific honesty—no untrained-head predictions, no cross-task conflation, no unverified probability claims, automatic deletion of original uploaded media, and sequential GPU execution tuned for a 4 GB VRAM RTX 2050.

---

## 2. User Experience & Visual Design

### Frontend Page Structure & Navigation

The interface uses a single-row 3-zone top bar (`MediaGuard` wordmark — 4 concise views: `Analyze`, `History`, `Methodology`, `System` — primary `New Scan` action) paired with an asymmetric forensic workspace below.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MediaGuard         Analyze   History   Methodology   System   [New Scan]│
├─────────────────────────────────────────────────────────────────────────┤
│  Modality Bar:  (● Image Analysis)    (Video · Queued)   (Audio · Queued)│
├──────────────────────┬──────────────────────────────────────────────────┤
│ LEFT INSPECTOR       │ MAIN INSPECTION STAGE                            │
│ (340px Fixed)        │ (Flex-1 Viewport)                                │
│                      │                                                  │
│ • Media Dropzone     │  ┌────────────────────────────────────────────┐  │
│ • Privacy Notice     │  │  Synchronized Original / Attribution Canvas│  │
│   (Originals deleted │  │  • Split-curtain slider & opacity blend    │  │
│    after analysis)   │  │  • Crosshair coordinate probe (x, y)       │  │
│ • Deterministic Facts│  │  • Experimental Attribution Layer Selector:│  │
│   (SHA-256, MIME,    │  │    Original / Xception Grad-CAM /          │  │
│    Dimensions, EXIF) │  │    DINOv2 Patch Attention (Non-Diagnostic) │  │
│ • Analysis Controls  │  └────────────────────────────────────────────┘  │
│ • Backend Connection ├──────────────────────────────────────────────────┤
│   (Live              │ TASK-SEPARATED EXPERIMENTAL INDICATORS TRAY      │
│   /api/v1/health)    │ • Track A: ai_generated_indicator                │
│                      │   (Ready / Inconclusive / Unavailable state)     │
│                      │ • Track B: face_manipulation_indicator           │
│                      │   (Ready / No Face / Inconclusive / Unavailable) │
│                      │ • Disclaimer: Experimental Indicators ≠ Proof    │
│                      │ • Export Analysis Report (JSON / Print Summary)  │
└──────────────────────┴──────────────────────────────────────────────────┘
```

#### Core Views

1. **Image Analyzer (`Analyze`)**:
   - **Zero-State**: Upload dropzone enforcing PNG/JPEG/WebP, upload size limits, and privacy notice explaining that original uploaded images are deleted immediately after analysis and never used for training without explicit consent.
   - **Active Inspection State**: Left panel shows deterministic file inspection (SHA-256 hash, verified magic-byte MIME, dimensions, EXIF summary). Center stage provides a zoomable image viewer with optional **Experimental Attribution** overlays (XceptionNet Grad-CAM or DINOv2 patch attention visualization) explicitly labeled as non-diagnostic saliency aids rather than confirmed manipulation proof.
   - **Task-Separated Indicator Cards & Honest States**: Displays two clearly separated indicator cards—**AI-Generated Image Indicator** (`ai_generated_indicator`) and **Face Manipulation Indicator** (`face_manipulation_indicator`). Each card independently reports `unavailable`, `inconclusive`, or an experimental score once its respective task-specific model/checkpoint is verified and evaluated. In Milestone 1 (before checkpoints exist), both tracks cleanly display `unavailable` with zero fabricated scores or fake heatmaps.

2. **Scan History & Archive (`History`)**:
   - High-density tabular ledger (`36px–44px` rows) backed by SQLite, displaying scan timestamps, SHA-256 prefix, filename, model readiness snapshot, separate `ai_generated_indicator` and `face_manipulation_indicator` columns, and independent retention status for scan records vs. experimental attribution artifacts.
   - Supports search, indicator filtering, reopening archived reports, deleting individual records or purging artifacts separately, and CSV/JSON export.

3. **Methodology & Limitations (`Methodology`)**:
   - Documents the strict separation between AI-generated image detection and face manipulation detection, explains why `DINOv2 ViT-S/14` is a self-supervised feature extractor rather than a classifier, lists verified checkpoint provenance/license/preprocessing details (once audited in Milestone 3), details task-specific datasets and evaluation protocols (Milestones 4–5), calibration status, and known failure modes (heavy recompression, resizing, out-of-distribution generators).

4. **System & Hardware Telemetry (`System`)**:
   - Configures the local FastAPI endpoint URL (default `http://127.0.0.1:8000` for local Windows Conda development) and polls `/api/v1/health`.
   - Displays **live measured** PyTorch CUDA telemetry (`device_name`, `allocated_vram_mb`, `peak_vram_mb`, `free_vram_mb`), per-task model/checkpoint readiness states (`dinov2_extractor`, `ai_generated_head`, `xception_face_detector`, `calibration_profile`), retention policy settings, and normalized REST API documentation.

### Visual Identity & Theme

- **Aesthetic Direction**: Dark Forensic Studio — high-contrast obsidian surfaces, crisp 1px borders, single-elevation flat panels, zero static pill badges, and zero decorative pseudo-technical clutter.
- **Color Palette (60-30-10 Budget)**:
  - *60% Primary Canvas*: Deep Space Obsidian (`#07090E`) and Void Charcoal (`#0B0E17`).
  - *30% Structural Surfaces*: Slate Panels (`#111827`), Hairline Borders (`#1E293B`), Muted Secondary Text (`#94A3B8`).
  - *10% Signal Accents*: Laser Cyan (`#06B6D4`) for active interactive controls and coordinate crosshairs; Semantic status colors always paired with explicit text labels and icons: Emerald (`#10B981` — Low Signal / Nominal), Telemetry Amber (`#F59E0B` — Inconclusive / Unavailable), Crimson (`#EF4444` — High Synthetic or Manipulation Signal / Error).
- **Typography**: `Plus Jakarta Sans` (display & navigation), `DM Sans` (body prose), and `JetBrains Mono` (`tabular-nums`) for hashes, VRAM measurements, latencies, and numeric indicator outputs.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Strict Separation of AI-Generated vs. Face Manipulation Tasks**
  - *Chosen Approach*: Keep separate datasets, evaluation protocols, classification heads, and API indicator fields (`ai_generated_indicator` and `face_manipulation_indicator`). Never combine them into a single probability unless a dedicated, properly trained and validated joint model supports it.
  - *Why*: A deepfake XceptionNet checkpoint trained on facial forgery benchmarks (e.g., FaceForensics++) cannot be assumed to generalize to full-scene diffusion/GAN synthesis, and vice versa.

- **Decision 2: Checkpoint-Free Milestone 1 & Explicit `unavailable` States**
  - *Chosen Approach*: Milestone 1 implements and verifies the React UI, FastAPI security/validation pipeline, SQLite persistence, and normalized `/api/v1/health` contract without requiring any pretrained weights; all unverified or untrained ML stages return explicit `unavailable` states.
  - *Why*: Guarantees deterministic, testable engineering foundations before introducing GPU/ML dependencies and prevents untrained heads from emitting random scores.

- **Decision 3: Experimental Attribution Framing**
  - *Chosen Approach*: Grad-CAM (XceptionNet) and patch attention maps (DINOv2) are introduced in Milestone 5, strictly labeled as **Experimental Attribution (Not Proof of Manipulation)**, and never simulated if the underlying model is inactive.
  - *Why*: DINOv2 attention maps highlight semantic salience rather than confirmed forgery boundaries; users must never confuse saliency visualizations with proof.

- **Decision 4: Measured VRAM Governance & Sequential Execution on RTX 2050 (4 GB)**
  - *Chosen Approach*: Single GPU inference worker protected by an `asyncio.Semaphore(1)`, `torch.inference_mode()`, mixed precision (`torch.amp.autocast`) where supported, explicit CUDA OOM exception handling (`torch.cuda.empty_cache()` + safe CPU fallback), and runtime VRAM profiling rather than assumed memory numbers.
  - *Why*: Windows WDDM shares the 4 GB RTX 2050 VRAM with the desktop compositor; concurrent model loading or multi-worker inference risks hard CUDA OOM crashes.

- **Decision 5: Privacy-First Ephemeral Media & Split Retention**
  - *Chosen Approach*: Original uploaded files are validated via magic bytes in isolated temp storage and deleted immediately after analysis by default. SQLite scan records and optional attribution overlay artifacts follow separate retention/deletion policies and TTL controls, and uploaded media is never used for training without explicit opt-in consent.

---

## 4. Technical Architecture & Data Strategy

### Complete System Architecture

```
┌──────────────────────────────────┐       HTTP / Multipart / JSON        ┌──────────────────────────────────────────┐
│   FRONTEND (React + Vite + TS)   │ ◄──────────────────────────────────► │   BACKEND (Python FastAPI · Windows/Conda)│
│   (Local Dev / Static Client)    │                                      │   (Bound to localhost during dev)        │
│                                  │   GET  /api/v1/health                │                                          │
│ • Image Analyzer Console         │   POST /api/v1/features/dinov2       │ • FastAPI Router, Rate Limit & Safe Errors│
│ • Task-Separated Indicator Cards │   POST /api/v1/analyze/image         │ • Magic-Byte MIME & Bomb Validation      │
│ • Attribution Overlay Canvas     │   GET  /api/v1/scans                 │ • Safe Temp File + Immediate Auto-Delete │
│ • Deterministic EXIF/Hash Panel  │   GET/DEL /api/v1/scans/{id}         │ • Single-Worker GPU Semaphore + OOM Guard│
│ • SQLite Scan History Ledger     │   DEL  /api/v1/scans/{id}/artifact   │                                          │
│ • Methodology & Provenance View  │   GET  /api/v1/scans/export          │ ┌──────────────────────────────────────┐ │
│ • System & Measured VRAM Console │                                      │ │ Task-Separated Sequential ML Engine  │ │
└──────────────────────────────────┘                                      │ │ • torch.inference_mode() + AMP/CPU   │ │
                                                                          │ │ 1. DINOv2 ViT-S/14 Feature Extractor │ │
                                                                          │ │    └─► AI-Generated Head (M4+)       │ │
                                                                          │ │ 2. Verified XceptionNet Checkpoint   │ │
                                                                          │ │    └─► Face Manipulation Track (M3+) │ │
                                                                          │ └──────────────────┬───────────────────┘ │
                                                                          │                    ▼                     │
                                                                          │ ┌──────────────────────────────────────┐ │
                                                                          │ │ Persistence & Retention (SQLite WAL) │ │
                                                                          │ │ • scans table (independent TTL)      │ │
                                                                          │ │ • artifacts store (independent TTL)  │ │
                                                                          │ │ • Original upload: deleted on finish │ │
                                                                          │ └──────────────────────────────────────┘ │
└──────────────────────────────────┘                                      └──────────────────────────────────────────┘
```

### Backend Module Structure

1. **API & Security Layer**:
   - FastAPI lifecycle management, CORS whitelist, configurable rate limiter middleware (required before any future public deployment), and sanitized exception handlers stripping internal stack traces and local paths.
   - Normalized API routes across frontend, backend, and README using `/api/v1/health` and `/api/v1/*` prefixes consistently.
2. **Ingestion, Validation & Privacy Guard**:
   - Inspects leading magic bytes to verify actual file signatures (PNG/JPEG/WebP) against declared MIME types.
   - Enforces maximum byte size (e.g., 15–20 MB), maximum pixel dimensions, and `PIL.Image.MAX_IMAGE_PIXELS` decompression-bomb limits before full decoding; rejects malformed/truncated streams safely.
   - Writes uploads only to isolated randomized temporary files inside a `try/finally` block that deletes the original uploaded image immediately after analysis completes.
3. **ML Engine & Hardware Telemetry Layer (Task-Separated)**:
   - **Device & Memory Manager**: Probes CUDA availability, records live `memory_allocated` / `max_memory_allocated` on the RTX 2050, enforces `asyncio.Semaphore(1)` sequential execution, and catches `torch.cuda.OutOfMemoryError` to clear cache and fall back to CPU or return HTTP `503`.
   - **DINOv2 Feature Extractor Module**: Loads `dinov2_vits14` solely as a frozen feature extractor (`384-d` `[CLS]` + spatial patch tokens). Does not emit classification verdicts.
   - **AI-Generated Image Head Module**: Task-specific lightweight classifier trained on labeled real vs. AI-generated image datasets (Milestone 4). Remains `unavailable` until trained and evaluated.
   - **Face Manipulation Detector Module (XceptionNet)**: Integrates an audited deepfake-trained XceptionNet checkpoint only after verifying its training dataset, intended facial-manipulation task, architecture, preprocessing (aligned face cropping), and license (Milestone 3). Remains `unavailable` if unverified or missing.
   - **Calibration & Experimental Explainability Module**: Applies task-specific probability calibration only after empirical evaluation (Milestone 5) and generates non-diagnostic XceptionNet Grad-CAM / DINOv2 patch attention visualizations.
4. **Persistence & Retention Layer**:
   - SQLite (`WAL` mode) repository storing scan metadata (`scan_id`, `sha256`, `ai_generated_indicator`, `face_manipulation_indicator`, `model_versions`, `record_expires_at`, `artifact_expires_at`) with separate retention purge routines for scan records vs. attribution PNG artifacts.

### API Design (v1 Contract)

All frontend clients, backend routers, and README documentation standardize on the `/api/v1/*` namespace (specifically `/api/v1/health` for liveness and readiness checks).

| Method & Endpoint | Purpose | Key Behavior & Constraints |
| :--- | :--- | :--- |
| `GET /api/v1/health` | Liveness, measured VRAM & per-model readiness | Reports CUDA device, measured allocated/peak VRAM in MB, and explicit readiness per component (`dinov2_extractor`, `ai_generated_head`, `xception_face_detector`, `calibration_profile` as `ready` or `unavailable`). |
| `POST /api/v1/features/dinov2` | Milestone 2 feature extraction verification | Validates image, runs DINOv2 ViT-S/14 in `inference_mode()`, returns embedding dimension, L2 norm, latency, and measured VRAM delta (no classification score). |
| `POST /api/v1/analyze/image` | Task-separated image analysis pipeline | Validates signature/dimensions, deletes original upload in `finally`, and returns explicit `ai_generated_indicator` and `face_manipulation_indicator` objects (each independently reporting `status`: `ready`, `inconclusive`, `no_face_detected`, or `unavailable`; Milestone 1 returns `unavailable` for both). |
| `GET /api/v1/scans` | Paginated SQLite scan history | Filter by indicator status, search by filename or SHA-256 prefix; surfaces independent record vs. artifact retention timestamps. |
| `GET /api/v1/scans/{scan_id}` | Retrieve archived scan record | Returns stored metadata, task-separated indicator payloads, model provenance snapshot, and artifact availability. |
| `DELETE /api/v1/scans/{scan_id}` | Delete scan record (with optional artifact purge) | Supports separate deletion of experimental attribution artifacts (`DELETE /api/v1/scans/{scan_id}/artifact`) vs. full SQLite scan records. |
| `GET /api/v1/scans/export` | Export filtered history | Streams sanitized CSV or JSON report preserving separate indicator columns. |
| `POST /api/v1/analyze/video` & `/audio` | Queued future modalities | Returns HTTP `501 Not Implemented` with milestone status. |

#### Task-Separated Response Shape (`POST /api/v1/analyze/image`)

```
{
  "scan_id": "...",
  "sha256": "...",
  "original_deleted": true,
  "file_metadata": { "mime": "image/jpeg", "width": 1024, "height": 1024, "exif_summary": {...} },
  "ai_generated_indicator": {
    "task": "ai_generated_image_detection",
    "status": "unavailable" | "inconclusive" | "evaluated",
    "score": null | float,
    "calibrated": false,
    "model_id": "dinov2_vits14 + ai_gen_head_v1",
    "reason": "Model checkpoint not loaded (Milestone 1)"
  },
  "face_manipulation_indicator": {
    "task": "face_manipulation_detection",
    "status": "unavailable" | "no_face_detected" | "inconclusive" | "evaluated",
    "score": null | float,
    "calibrated": false,
    "model_id": "xception_face_ffpp_v1",
    "reason": "Verified deepfake checkpoint not loaded (Milestone 1)"
  },
  "experimental_attribution": {
    "available": false,
    "disclaimer": "Experimental model saliency aid only; not proof of manipulation."
  }
}
```

### ML Integration Strategy (Task-Separated, Measured & Verified)

1. **DINOv2 ViT-S/14 Feature Extraction (Milestone 2)**: Strictly extracts `384-d` `[CLS]` and spatial patch tokens in `torch.inference_mode()`. No probabilities are emitted from DINOv2 alone.
2. **Verified Deepfake XceptionNet Integration (Milestone 3)**: Before writing model weights into the pipeline, audit and document the checkpoint's training dataset (e.g., FaceForensics++), intended task (face manipulation only—NOT general AI-generated image detection), architecture match, face-cropping/normalization preprocessing, and license. Missing or unverified checkpoints keep `face_manipulation_indicator.status = "unavailable"`.
3. **Task-Specific Datasets & Separate Classification Heads (Milestone 4)**: Curate separate labeled datasets and evaluation splits for (a) AI-generated image detection and (b) face manipulation detection (zero user-upload data without explicit consent). Train separate lightweight heads rather than merging both tasks into one combined score unless a dedicated, validated multi-task model is explicitly trained and benchmarked.
4. **Evaluation, Calibration & Experimental Explainability (Milestone 5)**: Evaluate AUROC, F1, false-positive rates across JPEG/resize perturbations, and Expected Calibration Error (ECE) independently per task before enabling `calibrated: true`. Implement XceptionNet Grad-CAM and DINOv2 patch attention strictly as experimental attribution tools.
5. **RTX 2050 4 GB Runtime Discipline**: Measure actual baseline, peak, and post-inference VRAM on the RTX 2050 during Milestone 2 and Milestone 3 tests. Enforce sequential single-worker execution (`asyncio.Semaphore(1)`), `torch.inference_mode()`, supported `autocast` mixed precision, OOM recovery, and CPU fallback.

---

## 5. Development & Verification Milestones

No milestone is marked complete until its dedicated verification tests pass.

- **Milestone 1 — React Frontend, FastAPI Foundation, API Contracts & SQLite (Zero Checkpoints Required)**
  - *Scope*: Build the complete dark forensic React + Vite + TypeScript + Tailwind CSS UI (`Analyze`, `History`, `Methodology`, `System`), normalized `/api/v1/health` and `/api/v1/*` FastAPI contracts, magic-byte/decompression-bomb upload validation, immediate original-file deletion, and SQLite WAL persistence with split retention fields. All ML model states explicitly report `unavailable` with zero pretrained checkpoints required.
  - *Verification Gate*: Frontend TypeScript/Vite build checks + `pytest` suite verifying `/api/v1/health`, `/api/v1/analyze/image` `unavailable` task responses, magic-byte validation, bomb rejection, temp-file cleanup, and SQLite CRUD/retention—passing completely without any model checkpoints present.

- **Milestone 2 — DINOv2 Installation & Actual Feature Extraction**
  - *Scope*: Configure local Windows Conda + PyTorch CUDA environment, load `dinov2_vits14` in `torch.inference_mode()` behind the sequential GPU semaphore, expose `/api/v1/features/dinov2`, and **measure actual RTX 2050 VRAM usage** (FP32 vs. AMP FP16 vs. CPU fallback).
  - *Verification Gate*: `pytest` verifying real `384-d` feature extraction shapes/norms, determinism, OOM/CPU fallback behavior, and recorded RTX 2050 VRAM benchmark logs.

- **Milestone 3 — Verified Deepfake-Trained XceptionNet Integration**
  - *Scope*: Identify, audit, and document a real deepfake-trained XceptionNet checkpoint (verifying training dataset, face-manipulation task scope, architecture, face-crop preprocessing, and license), integrate sequential inference with graceful `unavailable` fallback when absent, and measure combined sequential RTX 2050 VRAM usage.
  - *Verification Gate*: `pytest` confirming checkpoint integrity/metadata checks, face-crop preprocessing behavior, missing-checkpoint `unavailable` fallback, and measured RTX 2050 VRAM bounds.

- **Milestone 4 — Dataset Preparation & Training of Task-Specific Classification / Fusion Heads**
  - *Scope*: Prepare separate, documented, licensed datasets for AI-generated image detection vs. face manipulation detection (with strict train/val/test splits and zero user-upload contamination), train task-specific lightweight classification heads (`ai_generated_head` and, if needed, a face-manipulation/fusion head without conflating tasks).
  - *Verification Gate*: Reproducible training tests confirming split isolation, loss convergence, saved head checkpoint provenance, and zero cross-task leakage.

- **Milestone 5 — Evaluation, Calibration & Experimental Explainability**
  - *Scope*: Evaluate held-out AUROC, F1, compression robustness, and Expected Calibration Error (ECE) separately per task; apply probability calibration only after empirical verification; implement XceptionNet Grad-CAM and DINOv2 patch attention visualizations as experimental attribution tools.
  - *Verification Gate*: Automated per-task evaluation reports verifying calibration metrics before setting `calibrated: true`, plus attribution output shape/bounds tests.

- **Milestone 6 — End-to-End Image Analysis Integration**
  - *Scope*: Wire verified Milestone 2–5 task pipelines into `POST /api/v1/analyze/image` and connect the React `Analyze` workspace to live `ai_generated_indicator`, `face_manipulation_indicator`, experimental attribution overlays, and `/api/v1/health` telemetry.
  - *Verification Gate*: End-to-end integration tests from React upload → FastAPI validation → sequential RTX 2050 inference → immediate original file deletion → SQLite persistence → UI rendering.

- **Milestone 7 — History, Exports, Security Hardening & Final Testing (Pre-Public Gate)**
  - *Scope*: Complete History search/filtering, CSV/JSON exports, independent record vs. attribution artifact TTL purge jobs, rate limiting, strict header/CORS/error sanitization, and full regression testing prior to any future separate public hosting milestone (keeping localhost/development laptop private).
  - *Verification Gate*: Security & rate-limit `pytest` suite, retention purge verification, OOM/load stress tests on RTX 2050, and full frontend/backend verification.
