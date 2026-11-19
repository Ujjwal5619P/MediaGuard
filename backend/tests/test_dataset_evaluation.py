from pathlib import Path
from typing import Optional, Tuple

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.ml.evaluator import evaluate_local_dataset, inspect_dataset_folder
from app.ml.model_manager import ModelManager
from app.schemas import TaskIndicator


class _StubEngine:
    """Deterministic unit-test engine stub for confusion-matrix & export verification."""

    def unload_models(self) -> None:
        return None

    def run_ai_generated_detection(
        self, pil_rgb: Image.Image
    ) -> Tuple[TaskIndicator, Optional[float], str]:
        # Bright average pixel -> high AI score (0.85), dark average -> low AI score (0.15)
        pixels = pil_rgb.get_flattened_data() if hasattr(pil_rgb, "get_flattened_data") else pil_rgb.getdata()
        mean_r = sum(p[0] for p in pixels) / max(1, pil_rgb.width * pil_rgb.height)
        score = 0.85 if mean_r >= 128 else 0.15
        return (
            TaskIndicator(
                task="ai_generated_image_detection",
                status="evaluated",
                score=score,
                decision_threshold=0.50,
                prediction_label="AI-Generated Signal" if score >= 0.50 else "Human / Natural Signal",
                calibrated=False,
                model_id="abacusai/Smaug-AI-Image-Detector",
                model_name="Smaug AI Image Detector (Primary)",
                architecture="EmbeddingGemma-2 + 3x528 Crops",
                model_source="https://huggingface.co/abacusai/Smaug-AI-Image-Detector",
                model_version="c7c95951",
                execution_device="cpu",
                processing_ms=12,
                reason="Unit test evaluation signal",
            ),
            None,
            "cpu",
        )

    def run_univfd_detection(
        self, pil_rgb: Image.Image
    ) -> Tuple[TaskIndicator, Optional[float], str]:
        pixels = pil_rgb.get_flattened_data() if hasattr(pil_rgb, "get_flattened_data") else pil_rgb.getdata()
        mean_r = sum(p[0] for p in pixels) / max(1, pil_rgb.width * pil_rgb.height)
        score = 0.72 if mean_r >= 200 else 0.22
        return (
            TaskIndicator(
                task="ai_generated_image_detection",
                status="evaluated",
                score=score,
                decision_threshold=0.50,
                prediction_label="AI-Generated Signal" if score >= 0.50 else "Human / Natural Signal",
                calibrated=False,
                model_id="univfd-clip-vit-l14",
                model_name="UniversalFakeDetect (UnivFD · Comparison)",
                architecture="CLIP ViT-L/14",
                model_source="https://huggingface.co/Arjunkalliyadath/ai-image-detector",
                model_version="main",
                execution_device="cpu",
                processing_ms=9,
                reason="Unit test comparison signal",
            ),
            None,
            "cpu",
        )


def _create_sample_dataset(root: Path) -> Path:
    ai_dir = root / "ai_generated"
    real_dir = root / "real"
    ai_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # 2 bright AI images (TP=2), 1 dark AI image (FN=1)
    Image.new("RGB", (64, 64), (240, 200, 200)).save(ai_dir / "ai_1.png")
    Image.new("RGB", (64, 64), (220, 180, 180)).save(ai_dir / "ai_2.jpg")
    Image.new("RGB", (64, 64), (30, 40, 50)).save(ai_dir / "ai_hard.png")

    # 2 dark REAL images (TN=2), 1 bright REAL image (FP=1)
    Image.new("RGB", (64, 64), (20, 30, 40)).save(real_dir / "real_1.png")
    Image.new("RGB", (64, 64), (40, 50, 60)).save(real_dir / "real_2.webp")
    Image.new("RGB", (64, 64), (235, 220, 210)).save(real_dir / "real_fp.jpg")

    # Unsupported file should be skipped cleanly
    (real_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    return root


def test_inspect_dataset_folder_and_metrics_math(tmp_path: Path) -> None:
    ds = _create_sample_dataset(tmp_path / "dataset")
    inspected = inspect_dataset_folder(ds)
    assert inspected.ai_generated_exists is True
    assert inspected.real_exists is True
    assert inspected.ai_generated_count == 3
    assert inspected.real_count == 3
    assert inspected.total_supported_count == 6
    assert inspected.unsupported_or_skipped == ["real/notes.txt"]

    detail = evaluate_local_dataset(
        engine=_StubEngine(),  # type: ignore[arg-type]
        dataset_dir=ds,
        split_role="final_test",
        decision_threshold=0.50,
        include_univfd=True,
    )

    assert detail.split_role == "final_test"
    assert detail.threshold_auto_tuned is False
    assert detail.models_trained_or_finetuned is False
    assert "Never tune" in detail.threshold_policy_note or "Zero fine-tuning" in detail.threshold_policy_note
    assert detail.total_images == 6
    assert detail.skipped_count == 1

    sm = detail.smaug_metrics
    assert sm.confusion_matrix.tp == 2
    assert sm.confusion_matrix.fn == 1
    assert sm.confusion_matrix.tn == 2
    assert sm.confusion_matrix.fp == 1
    assert sm.accuracy == pytest.approx(4 / 6, abs=1e-4)
    assert sm.precision == pytest.approx(2 / 3, abs=1e-4)
    assert sm.recall == pytest.approx(2 / 3, abs=1e-4)
    assert sm.f1_score == pytest.approx(2 / 3, abs=1e-4)
    assert sm.false_positive_rate == pytest.approx(1 / 3, abs=1e-4)
    assert sm.false_negative_rate == pytest.approx(1 / 3, abs=1e-4)

    assert detail.univfd_metrics is not None
    assert detail.univfd_metrics.evaluated_count == 6


def test_evaluation_endpoints_sqlite_persistence_and_exports(
    tmp_path: Path, test_settings: Settings
) -> None:
    ds = _create_sample_dataset(tmp_path / "eval_ds")
    app = create_app(test_settings, load_models_on_startup=False)
    app.state.registry.engine = _StubEngine()  # type: ignore[assignment]

    with TestClient(app) as client:
        insp = client.get(
            "/api/v1/evaluation/inspect-folder",
            params={"dataset_dir": str(ds)},
        )
        assert insp.status_code == 200
        assert insp.json()["total_supported_count"] == 6

        run_resp = client.post(
            "/api/v1/evaluation/run",
            json={
                "dataset_dir": str(ds),
                "split_role": "validation",
                "decision_threshold": 0.50,
                "include_univfd": True,
            },
        )
        assert run_resp.status_code == 200
        body = run_resp.json()
        eval_id = body["evaluation_id"]
        assert body["split_role"] == "validation"
        assert body["threshold_auto_tuned"] is False
        assert body["smaug_metrics"]["confusion_matrix"] == {
            "tp": 2,
            "fp": 1,
            "tn": 2,
            "fn": 1,
        }

        runs_resp = client.get("/api/v1/evaluation/runs")
        assert runs_resp.status_code == 200
        assert any(r["evaluation_id"] == eval_id for r in runs_resp.json())

        detail_resp = client.get(f"/api/v1/evaluation/runs/{eval_id}")
        assert detail_resp.status_code == 200
        assert len(detail_resp.json()["items"]) == 6

        csv_resp = client.get(
            f"/api/v1/evaluation/runs/{eval_id}/export", params={"format": "csv"}
        )
        assert csv_resp.status_code == 200
        assert "smaug_score" in csv_resp.text
        assert "univfd_score" in csv_resp.text

        json_resp = client.get(
            f"/api/v1/evaluation/runs/{eval_id}/export", params={"format": "json"}
        )
        assert json_resp.status_code == 200
        assert eval_id in json_resp.text

        # Non-existent dataset path returns clear 400 error
        missing_resp = client.post(
            "/api/v1/evaluation/run",
            json={"dataset_dir": str(tmp_path / "does_not_exist")},
        )
        assert missing_resp.status_code == 400
        assert missing_resp.json()["detail"]["code"] == "ERR_DATASET_DIR_NOT_FOUND"


def _real_models_ready() -> bool:
    settings = Settings()
    mgr = ModelManager(settings.models_dir)
    mgr.verify_existing_models()
    return mgr.get_status().all_verified


@pytest.mark.skipif(
    not _real_models_ready(),
    reason="Real checkpoints not downloaded yet; run `python setup_models.py` first.",
)
def test_real_smaug_and_univfd_dataset_evaluation_pipeline(tmp_path: Path) -> None:
    ds = tmp_path / "real_eval_ds"
    (ds / "ai_generated").mkdir(parents=True)
    (ds / "real").mkdir(parents=True)

    # Create two valid local test images cleanly
    Image.new("RGB", (512, 512), (240, 120, 40)).save(ds / "ai_generated" / "sample_ai.png")
    Image.new("RGB", (512, 512), (40, 90, 180)).save(ds / "real" / "sample_real.jpg")

    base = Settings()
    settings = Settings(
        models_dir=base.models_dir,
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "eval_real.db",
        temp_dir=tmp_path / "data" / "tmp",
        artifacts_dir=tmp_path / "data" / "artifacts",
        auto_download_models=False,
        low_vram_offload=True,
    )
    app = create_app(settings, load_models_on_startup=True)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/evaluation/run",
            json={
                "dataset_dir": str(ds),
                "split_role": "evaluation",
                "decision_threshold": 0.50,
                "include_univfd": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 2
        assert data["smaug_metrics"]["evaluated_count"] == 2
        assert data["univfd_metrics"]["evaluated_count"] == 2
        assert data["threshold_auto_tuned"] is False
        assert data["models_trained_or_finetuned"] is False
