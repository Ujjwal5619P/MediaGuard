import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageOps

from app.ml.inference import InferenceEngine
from app.schemas import (
    ConfusionMatrixCounts,
    DatasetEvaluationDetail,
    DatasetInspectResponse,
    DatasetSplitRole,
    DetectorEvaluationMetrics,
    EvaluationImageResult,
    GroundTruthLabel,
    TaskStatus,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_dataset_dir(default_dir: Path, requested_dir: Optional[str] = None) -> Path:
    if requested_dir and requested_dir.strip():
        candidate = Path(requested_dir.strip()).expanduser()
        if not candidate.is_absolute():
            # Allow relative to project root or backend dir cleanly
            project_root = default_dir.parent
            candidate = (project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    if default_dir.exists():
        return default_dir.resolve()
    backend_fallback = Path(__file__).resolve().parent.parent.parent / "dataset"
    if backend_fallback.exists():
        return backend_fallback.resolve()
    return default_dir.resolve()


def _collect_folder_images(
    folder: Path, ground_truth: GroundTruthLabel
) -> Tuple[List[Tuple[Path, GroundTruthLabel]], List[str]]:
    supported: List[Tuple[Path, GroundTruthLabel]] = []
    skipped: List[str] = []
    if not folder.exists() or not folder.is_dir():
        return supported, skipped

    for item in sorted(folder.iterdir()):
        if not item.is_file():
            continue
        if item.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append((item, ground_truth))
        else:
            skipped.append(f"{ground_truth.lower()}/{item.name}")
    return supported, skipped


def inspect_dataset_folder(dataset_dir: Path) -> DatasetInspectResponse:
    ai_dir = dataset_dir / "ai_generated"
    real_dir = dataset_dir / "real"
    ai_items, ai_skipped = _collect_folder_images(ai_dir, "AI_GENERATED")
    real_items, real_skipped = _collect_folder_images(real_dir, "REAL")

    return DatasetInspectResponse(
        dataset_dir=str(dataset_dir),
        ai_generated_dir=str(ai_dir),
        real_dir=str(real_dir),
        ai_generated_exists=ai_dir.is_dir(),
        real_exists=real_dir.is_dir(),
        ai_generated_count=len(ai_items),
        real_count=len(real_items),
        total_supported_count=len(ai_items) + len(real_items),
        unsupported_or_skipped=ai_skipped + real_skipped,
    )


def _compute_metrics(
    model_id: str,
    model_name: str,
    items: List[EvaluationImageResult],
    use_univfd: bool = False,
    vram_peaks: Optional[List[float]] = None,
) -> DetectorEvaluationMetrics:
    tp = fp = tn = fn = 0
    evaluated = 0
    errors = 0
    latencies: List[int] = []

    for row in items:
        status: Optional[TaskStatus] = row.univfd_status if use_univfd else row.smaug_status
        pred: Optional[GroundTruthLabel] = (
            row.univfd_prediction if use_univfd else row.smaug_prediction
        )
        ms = row.univfd_inference_ms if use_univfd else row.smaug_inference_ms

        if status != "evaluated" or pred is None:
            errors += 1
            continue

        evaluated += 1
        if ms is not None:
            latencies.append(ms)

        if row.ground_truth == "AI_GENERATED" and pred == "AI_GENERATED":
            tp += 1
        elif row.ground_truth == "REAL" and pred == "AI_GENERATED":
            fp += 1
        elif row.ground_truth == "REAL" and pred == "REAL":
            tn += 1
        elif row.ground_truth == "AI_GENERATED" and pred == "REAL":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = round((tp + tn) / total, 4) if total > 0 else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    f1 = (
        round((2 * precision * recall) / (precision + recall), 4)
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else None
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else None

    total_ms = sum(latencies)
    mean_ms = round(total_ms / len(latencies), 1) if latencies else None
    peak_vram = round(max(vram_peaks), 2) if vram_peaks else None

    return DetectorEvaluationMetrics(
        model_id=model_id,
        model_name=model_name,
        evaluated_count=evaluated,
        error_count=errors,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        confusion_matrix=ConfusionMatrixCounts(tp=tp, fp=fp, tn=tn, fn=fn),
        mean_inference_ms=mean_ms,
        total_inference_ms=total_ms,
        peak_vram_mb=peak_vram,
    )


def evaluate_local_dataset(
    engine: InferenceEngine,
    dataset_dir: Path,
    split_role: DatasetSplitRole = "evaluation",
    decision_threshold: float = 0.50,
    include_univfd: bool = True,
    max_images: Optional[int] = None,
) -> DatasetEvaluationDetail:
    """
    Evaluates local images in `<dataset_dir>/ai_generated/` (AI_GENERATED) and
    `<dataset_dir>/real/` (REAL) without modifying, training, fine-tuning, or
    auto-tuning model weights or decision thresholds.
    """
    started = time.perf_counter()
    ai_dir = dataset_dir / "ai_generated"
    real_dir = dataset_dir / "real"

    ai_items, ai_skipped = _collect_folder_images(ai_dir, "AI_GENERATED")
    real_items, real_skipped = _collect_folder_images(real_dir, "REAL")
    combined = ai_items + real_items
    if max_images is not None:
        combined = combined[:max_images]

    results: List[EvaluationImageResult] = []
    smaug_vrams: List[float] = []
    univfd_vrams: List[float] = []
    devices_seen: List[str] = []
    skipped_count = len(ai_skipped) + len(real_skipped)

    # Temporarily evaluate using explicit caller threshold without mutating engine permanently
    for img_path, gt in combined:
        rel_path = f"{'ai_generated' if gt == 'AI_GENERATED' else 'real'}/{img_path.name}"
        try:
            with Image.open(img_path) as raw_img:
                pil_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")
        except Exception as exc:
            skipped_count += 1
            results.append(
                EvaluationImageResult(
                    filename=img_path.name,
                    relative_path=rel_path,
                    ground_truth=gt,
                    smaug_status="error",
                    smaug_score=None,
                    smaug_prediction=None,
                    smaug_correct=None,
                    smaug_inference_ms=None,
                    execution_device="cpu",
                    error_message=f"Corrupt or unreadable image: {str(exc)[:120]}",
                )
            )
            continue

        smaug_ind, smaug_vram, smaug_dev = engine.run_ai_generated_detection(pil_rgb)
        if smaug_vram is not None:
            smaug_vrams.append(smaug_vram)
        if smaug_dev != "unavailable":
            devices_seen.append(smaug_dev)

        smaug_pred: Optional[GroundTruthLabel] = None
        smaug_correct: Optional[bool] = None
        if smaug_ind.status == "evaluated" and smaug_ind.score is not None:
            smaug_pred = (
                "AI_GENERATED" if smaug_ind.score >= decision_threshold else "REAL"
            )
            smaug_correct = smaug_pred == gt

        univfd_status: Optional[TaskStatus] = None
        univfd_score: Optional[float] = None
        univfd_pred: Optional[GroundTruthLabel] = None
        univfd_correct: Optional[bool] = None
        univfd_ms: Optional[int] = None

        if include_univfd:
            u_ind, u_vram, u_dev = engine.run_univfd_detection(pil_rgb)
            if u_vram is not None:
                univfd_vrams.append(u_vram)
            if u_dev != "unavailable":
                devices_seen.append(u_dev)
            univfd_status = u_ind.status
            univfd_score = u_ind.score
            univfd_ms = u_ind.processing_ms
            if u_ind.status == "evaluated" and u_ind.score is not None:
                univfd_pred = (
                    "AI_GENERATED" if u_ind.score >= decision_threshold else "REAL"
                )
                univfd_correct = univfd_pred == gt

        results.append(
            EvaluationImageResult(
                filename=img_path.name,
                relative_path=rel_path,
                ground_truth=gt,
                smaug_status=smaug_ind.status,
                smaug_score=smaug_ind.score,
                smaug_prediction=smaug_pred,
                smaug_correct=smaug_correct,
                smaug_inference_ms=smaug_ind.processing_ms,
                univfd_status=univfd_status,
                univfd_score=univfd_score,
                univfd_prediction=univfd_pred,
                univfd_correct=univfd_correct,
                univfd_inference_ms=univfd_ms,
                execution_device=smaug_dev,
                error_message=None
                if smaug_ind.status == "evaluated"
                else smaug_ind.reason,
            )
        )

    smaug_metrics = _compute_metrics(
        model_id="abacusai/Smaug-AI-Image-Detector",
        model_name="Smaug AI Image Detector (Primary)",
        items=results,
        use_univfd=False,
        vram_peaks=smaug_vrams,
    )
    univfd_metrics = (
        _compute_metrics(
            model_id="univfd-clip-vit-l14",
            model_name="UniversalFakeDetect (UnivFD · Comparison)",
            items=results,
            use_univfd=True,
            vram_peaks=univfd_vrams,
        )
        if include_univfd
        else None
    )

    total_duration_ms = int((time.perf_counter() - started) * 1000)
    exec_device = devices_seen[0] if devices_seen else "cpu"

    if split_role == "validation":
        policy_note = (
            f"Validation split evaluated at fixed threshold {decision_threshold:.2f}. "
            "Thresholds may be selected strictly on validation data; never tune on final_test."
        )
    elif split_role == "final_test":
        policy_note = (
            f"Untouched final test split evaluated at pre-selected threshold {decision_threshold:.2f}. "
            "Zero fine-tuning and zero threshold modification performed."
        )
    else:
        policy_note = (
            f"Local evaluation dataset scored at fixed threshold {decision_threshold:.2f} "
            "(no automatic training or threshold tuning applied)."
        )

    return DatasetEvaluationDetail(
        evaluation_id=f"eval_{uuid.uuid4().hex[:14]}",
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_dir=str(dataset_dir),
        split_role=split_role,
        decision_threshold=decision_threshold,
        threshold_auto_tuned=False,
        threshold_policy_note=policy_note,
        models_trained_or_finetuned=False,
        include_univfd=include_univfd,
        total_images=len(results),
        ai_generated_count=sum(1 for r in results if r.ground_truth == "AI_GENERATED"),
        real_count=sum(1 for r in results if r.ground_truth == "REAL"),
        skipped_count=skipped_count,
        execution_device=exec_device,
        total_duration_ms=total_duration_ms,
        smaug_metrics=smaug_metrics,
        univfd_metrics=univfd_metrics,
        items=results,
    )
