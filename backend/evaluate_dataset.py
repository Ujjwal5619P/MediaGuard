"""
MediaGuard Offline Local Dataset Evaluation CLI

Evaluates local images organized in:
  <dataset_dir>/ai_generated/   (ground truth: AI_GENERATED)
  <dataset_dir>/real/           (ground truth: REAL)

Runs official abacusai/Smaug-AI-Image-Detector (Primary) and optionally
UnivFD CLIP ViT-L/14 (Comparison) without any model training, fine-tuning,
or automatic threshold tuning.

Usage:
    python evaluate_dataset.py
    python evaluate_dataset.py --dataset-dir ../dataset --split-role final_test --threshold 0.50
    python evaluate_dataset.py --dataset-dir ./my_val_split --split-role validation --export-csv eval.csv --export-json eval.json
"""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.db.database import ScanRepository
from app.ml.evaluator import (
    evaluate_local_dataset,
    inspect_dataset_folder,
    resolve_dataset_dir,
)
from app.ml.registry import ModelRegistry


def _fmt_pct(val: float | None) -> str:
    return f"{val * 100:.2f}%" if val is not None else "N/A"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate MediaGuard AI-image detectors on a local dataset folder."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path containing `ai_generated/` and `real/` folders (default: ../dataset).",
    )
    parser.add_argument(
        "--split-role",
        choices=["validation", "evaluation", "final_test"],
        default="evaluation",
        help="Split designation for audit clarity. Thresholds must never be tuned on final_test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Fixed decision threshold in (0, 1) (default: 0.50). Never auto-tuned.",
    )
    parser.add_argument(
        "--no-univfd",
        action="store_true",
        help="Evaluate only primary Smaug-AI-Image-Detector without UnivFD comparison.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of images evaluated.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional file path to write per-image CSV results.",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional file path to write full JSON evaluation report.",
    )
    parser.add_argument(
        "--no-save-db",
        action="store_true",
        help="Do not persist the evaluation run to SQLite.",
    )
    args = parser.parse_args()

    if not (0.0 < args.threshold < 1.0):
        print("ERROR: --threshold must be strictly between 0.0 and 1.0.", file=sys.stderr)
        return 2

    settings = get_settings()
    resolved_dir = resolve_dataset_dir(settings.dataset_dir, args.dataset_dir)
    if not resolved_dir.is_dir():
        print(f"ERROR: Dataset directory does not exist: {resolved_dir}", file=sys.stderr)
        return 1

    inspection = inspect_dataset_folder(resolved_dir)
    print(f"Dataset Directory : {inspection.dataset_dir}")
    print(
        f"Supported Images  : {inspection.total_supported_count} "
        f"(AI_GENERATED={inspection.ai_generated_count}, REAL={inspection.real_count}, "
        f"Skipped={len(inspection.unsupported_or_skipped)})"
    )
    if inspection.total_supported_count == 0:
        print(
            "ERROR: Place images inside `ai_generated/` and `real/` folders before running.",
            file=sys.stderr,
        )
        return 1

    registry = ModelRegistry(
        models_dir=settings.models_dir,
        artifacts_dir=settings.artifacts_dir,
        low_vram_offload=settings.low_vram_offload,
        decision_threshold=args.threshold,
    )
    registry.load_cached_models()

    if registry.get_readiness_map().ai_generated_detector.status != "ready":
        print(
            "ERROR: Smaug-AI-Image-Detector is not verified/loaded. Run `python setup_models.py` first.",
            file=sys.stderr,
        )
        return 1

    detail = evaluate_local_dataset(
        engine=registry.engine,
        dataset_dir=resolved_dir,
        split_role=args.split_role,
        decision_threshold=args.threshold,
        include_univfd=not args.no_univfd,
        max_images=args.max_images,
    )

    repo = ScanRepository(db_path=settings.db_path, artifacts_dir=settings.artifacts_dir)
    repo.initialize()
    if not args.no_save_db:
        repo.insert_evaluation(detail)

    sm = detail.smaug_metrics
    cm = sm.confusion_matrix
    print("\n=== Smaug AI Image Detector (Primary · abacusai/Smaug-AI-Image-Detector) ===")
    print(f"Split Role : {detail.split_role} | Fixed Threshold: {detail.decision_threshold:.2f} (Auto-tuned: {detail.threshold_auto_tuned})")
    print(f"Evaluated  : {sm.evaluated_count} | Errors: {sm.error_count} | Device: {detail.execution_device}")
    print(f"Accuracy   : {_fmt_pct(sm.accuracy)} | Precision: {_fmt_pct(sm.precision)} | Recall: {_fmt_pct(sm.recall)} | F1: {_fmt_pct(sm.f1_score)}")
    print(f"FPR / FNR  : {_fmt_pct(sm.false_positive_rate)} / {_fmt_pct(sm.false_negative_rate)}")
    print(f"Confusion  : TP={cm.tp}  FP={cm.fp}  TN={cm.tn}  FN={cm.fn}")
    print(f"Latency    : mean={sm.mean_inference_ms} ms | total={sm.total_duration_ms if hasattr(sm, 'total_duration_ms') else detail.total_duration_ms} ms | Peak VRAM={sm.peak_vram_mb} MB")

    if detail.univfd_metrics:
        um = detail.univfd_metrics
        ucm = um.confusion_matrix
        print("\n=== UnivFD CLIP ViT-L/14 (Secondary Comparison) ===")
        print(f"Evaluated  : {um.evaluated_count} | Errors: {um.error_count}")
        print(f"Accuracy   : {_fmt_pct(um.accuracy)} | Precision: {_fmt_pct(um.precision)} | Recall: {_fmt_pct(um.recall)} | F1: {_fmt_pct(um.f1_score)}")
        print(f"FPR / FNR  : {_fmt_pct(um.false_positive_rate)} / {_fmt_pct(um.false_negative_rate)}")
        print(f"Confusion  : TP={ucm.tp}  FP={ucm.fp}  TN={ucm.tn}  FN={ucm.fn}")
        print(f"Latency    : mean={um.mean_inference_ms} ms | Peak VRAM={um.peak_vram_mb} MB")

    # Save exports directly from detail (or repo export helper)
    if args.export_json:
        Path(args.export_json).write_text(detail.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nExported JSON report -> {args.export_json}")
    if args.export_csv:
        if args.no_save_db:
            # Save in-memory via temporary SQLite or direct helper
            import csv
            import io

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "evaluation_id",
                    "split_role",
                    "decision_threshold",
                    "filename",
                    "relative_path",
                    "ground_truth",
                    "smaug_status",
                    "smaug_score",
                    "smaug_prediction",
                    "smaug_correct",
                    "smaug_inference_ms",
                    "univfd_status",
                    "univfd_score",
                    "univfd_prediction",
                    "univfd_correct",
                    "univfd_inference_ms",
                    "execution_device",
                    "error_message",
                ]
            )
            for item in detail.items:
                writer.writerow(
                    [
                        detail.evaluation_id,
                        detail.split_role,
                        detail.decision_threshold,
                        item.filename,
                        item.relative_path,
                        item.ground_truth,
                        item.smaug_status,
                        item.smaug_score if item.smaug_score is not None else "",
                        item.smaug_prediction or "",
                        item.smaug_correct if item.smaug_correct is not None else "",
                        item.smaug_inference_ms if item.smaug_inference_ms is not None else "",
                        item.univfd_status or "",
                        item.univfd_score if item.univfd_score is not None else "",
                        item.univfd_prediction or "",
                        item.univfd_correct if item.univfd_correct is not None else "",
                        item.univfd_inference_ms if item.univfd_inference_ms is not None else "",
                        item.execution_device,
                        item.error_message or "",
                    ]
                )
            csv_text = buf.getvalue()
        else:
            csv_text = repo.export_evaluation(detail.evaluation_id, fmt="csv") or ""
        Path(args.export_csv).write_text(csv_text, encoding="utf-8")
        print(f"Exported CSV report  -> {args.export_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
