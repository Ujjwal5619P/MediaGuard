# MediaGuard FastAPI backend

See the root [README](../README.md) for the setup guide, detector description, exact Conda commands, test/evaluation instructions, cloud hosting constraints, and demonstration steps.

## Runtime

The primary detector is Imager-1-Tiny, downloaded into `backend/models/imager_1_tiny/` and verified by SHA-256 and checkpoint metadata. It uses torchvision ResNet-50, class-1 softmax, ImageNet normalization, threshold 0.385, and FP32 CPU inference.

Normal local mode also loads the cached UnivFD comparison, EfficientNet-B0 face manipulation, and DINOv2 feature models. Set `MEDIAGUARD_LIGHTWEIGHT_MODE=true` to load only the primary detector; disabled pipelines report unavailable and do not claim a result.

The setup command downloads the primary model only:

```powershell
conda activate mediaguard
python setup_models.py
```

Use `python setup_models.py --include-optional` to download the larger comparison/face/feature checkpoints when needed. Setup never removes files outside the selected checkpoint set and does not rewrite existing `.env` files.

## API v1

- `GET /api/v1/health` — readiness, CPU/CUDA mode, and retention policy.
- `GET /api/v1/models/status` and `POST /api/v1/models/setup` — checkpoint verification and primary setup.
- `POST /api/v1/analyze/image` — validated image analysis and SQLite persistence using the existing response contract.
- `GET /api/v1/scans`, `/api/v1/analytics`, `/api/v1/evaluation/*` — history, analytics, and dataset evaluation.
- Existing feature, artifact, deletion, purge, and export routes remain available.

Some database and evaluation field identifiers retain historical `smaug_*` naming for API compatibility. New scan model metadata identifies Imager-1-Tiny.
