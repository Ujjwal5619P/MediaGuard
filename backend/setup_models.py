"""
MediaGuard Model Setup & Verification CLI

Downloads and SHA-256 verifies all required pretrained model checkpoints into
`backend/models/` without duplicating already-verified downloads.

Usage:
    python setup_models.py
    python setup_models.py --verify-only
"""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.ml.registry import ModelRegistry
from app.ml.specs import MODEL_CATALOG


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify MediaGuard pretrained vision models."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing local weights without downloading missing files.",
    )
    parser.add_argument(
        "--no-load-test",
        action="store_true",
        help="Skip test-loading models into PyTorch after SHA-256 verification.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download the larger UnivFD, face detector, and DINOv2 models.",
    )
    args = parser.parse_args()

    settings = get_settings()
    registry = ModelRegistry(
        models_dir=settings.models_dir,
        artifacts_dir=settings.artifacts_dir,
        low_vram_offload=settings.low_vram_offload,
        decision_threshold=settings.decision_threshold,
        lightweight_mode=settings.lightweight_mode,
    )

    print(f"MediaGuard Model Storage: {settings.models_dir}")
    if args.verify_only:
        registry.manager.verify_existing_models()
    else:
        print("Downloading and verifying checkpoints (resumable, skips already verified files)...")
        keys = list(MODEL_CATALOG) if args.include_optional else ["ai_generated_detector"]
        registry.manager.download_all_models(keys=keys)

    if not args.no_load_test:
        print("Testing PyTorch model loading...")
        registry.load_cached_models()

    status = registry.get_status() if hasattr(registry, "get_status") else registry.manager.get_status()
    cuda_ok, device_name, _, vram = registry.probe_hardware()
    print(f"\nCompute Device: {device_name} (CUDA={cuda_ok})")
    if vram:
        print(
            f"VRAM: total={vram.total_mb} MB, free={vram.free_mb} MB, allocated={vram.allocated_mb} MB"
        )

    print("\nModel Manifest Summary:")
    print("-" * 88)
    for entry in status.models:
        print(
            f"- {entry.key} ({entry.repo_id})\n"
            f"  Status: {entry.status} | Verified: {entry.verified} | Loaded: {entry.loaded_in_memory} | Size: {entry.size_mb} MB\n"
            f"  License: {entry.license}\n"
            f"  SHA-256: {entry.actual_sha256 or 'not verified'}"
        )
        if entry.error_message:
            print(f"  Error: {entry.error_message}")
    print("-" * 88)

    primary = next((m for m in status.models if m.key == "ai_generated_detector"), None)
    if not primary or not primary.verified:
        print("ERROR: Primary Imager-1-Tiny checkpoint failed verification.", file=sys.stderr)
        return 1

    if not args.no_load_test and not primary.loaded_in_memory:
        print("ERROR: Imager-1-Tiny did not load in PyTorch.", file=sys.stderr)
        return 1

    print("SUCCESS: Imager-1-Tiny is verified and ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
