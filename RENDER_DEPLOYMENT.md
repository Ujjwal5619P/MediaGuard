# Render backend preparation

## Readiness decision

**Do not deploy this build to Render Free.** Render's current [Free web service plan](https://render.com/docs/compute-plans) provides 512 MB RAM and 0.1 CPU. A live local lightweight backend running in the existing `mediaguard` Conda environment measured:

- Working set after loading Imager-1-Tiny: **590.6 MiB**.
- Maximum working set sampled every 20 ms during a real API inference: **645.3 MiB** (18 samples).
- Process peak working set high-water mark: **646.5 MiB**.
- The inference request itself took **156 ms** and the measured detector pipeline took **158 ms** on this computer.
- Startup to primary-model readiness took **7.96 seconds** with the checkpoint already cached locally; this does not measure an uncached model download.

The current process already exceeds the 512 MiB limit before allowing any safety margin. These are measurements from the existing Windows/Conda runtime, not a Docker or Render measurement; they cannot establish a safe Linux container footprint. A container measurement must show a peak below the service limit with headroom before Free can be reconsidered. Upload handling also buffers up to 15 MiB and permits images up to 36 megapixels, which can add substantial decoded-image memory.

## Prepared runtime files

- `Dockerfile` uses Python 3.10, CPU-only PyTorch/torchvision, lightweight mode, runtime model download, `/tmp` cache/data paths, and the injected `PORT` (local default 8000).
- `backend/requirements-render.txt` omits test tooling, Transformers, OpenCV, and safetensors for the lightweight image. OpenCV and NumPy imports used by optional full-mode code are lazy; lightweight mode does not call that code. The general local `backend/requirements.txt` remains unchanged so existing environments and full mode retain their dependencies.
- `.dockerignore` excludes local environment files, data, model weights, and build caches. Weights are not copied into the image.
- `backend/run_server.py` binds to `0.0.0.0` in the container and uses Render's `PORT`, falling back to configured local port 8000.
- No `render.yaml` is provided yet: a Free plan declaration would make an unverified, already over-limit configuration look ready to deploy. Once the container is proven within a safe memory envelope, add a Blueprint with the exact Vercel origin in `MEDIAGUARD_CORS_ORIGINS`.

The primary checkpoint is 94,371,929 bytes. At runtime, `huggingface_hub` downloads it into `MEDIAGUARD_MODELS_DIR`; `HF_HOME` is set to `/tmp/huggingface`. `MEDIAGUARD_AUTO_DOWNLOAD_MODELS=true` begins download/load at startup. Render Free's filesystem is ephemeral, so expect the checkpoint download again after a restart or idle spin-down.

## Runtime checks and known behavior

- `/api/v1/health` reports model readiness and CPU mode. It currently returns HTTP 200 even when health is `degraded`, so it is not a strict model-ready deployment gate. Startup model download/load runs in a background thread.
- Upload limits are 15 MiB, 36 million pixels, and 8192 pixels per dimension. Large decoded inputs can exceed memory available on a 512 MiB service.
- Default CORS allows only local Vite origins. Set `MEDIAGUARD_CORS_ORIGINS` to a JSON array containing the exact deployed Vercel origin, for example `["https://your-app.vercel.app"]`; do not use a wildcard for a credential-free but user-data API.
- Warm local inference measured 156 ms for the detector. There is no application-level inference timeout; execution is dispatched to a worker thread and the request remains open until analysis finishes. Render documents support for long-running web responses, but client/browser timeouts and cold-start delays still apply.
- Investigation history is SQLite at `MEDIAGUARD_DB_PATH`. Render Free has no persistent disk: scans and evaluation history in SQLite are lost on restart, redeploy, or idle spin-down. The model cache is also lost. Do not present this service as durable history storage. A database migration is required for durable history; Render's free Postgres database expires after 30 days.
- Render Free [spins down after 15 minutes without traffic and takes about a minute to wake](https://render.com/docs/free). Cold wake includes model download and loading, so first-request availability and latency are not reliable.

## Local validation steps before any deployment

1. Start Docker Desktop and confirm `docker info` works.
2. From the repository root, build the prepared image:

   ```powershell
   docker build -f Dockerfile -t mediaguard-api:render-check .
   ```

3. Run it with a temporary writable volume for local smoke testing, `MEDIAGUARD_CORS_ORIGINS` set to the frontend origin, and a published local port.
4. Measure container memory after startup, checkpoint load, an inference, and the largest accepted upload. Include all container processes and preserve a meaningful safety margin below 512 MiB. If it exceeds that budget, use a larger plan or reduce the model/runtime footprint and repeat the measurements.
5. Test `/api/v1/health`, image analysis, invalid uploads, CORS preflight, and restart behavior. Confirm the expected loss of local SQLite data on a clean ephemeral run.
6. Only after memory and history requirements are resolved, create/sync a Render Web Service Blueprint. Set `VITE_API_BASE_URL` in Vercel to the Render HTTPS origin and set the backend CORS allowlist to the exact Vercel origin.

The Docker CLI is installed locally, but the Docker Desktop Linux engine was not running during preparation (`docker version` could not connect to its named pipe), so image build status is **not run**.
