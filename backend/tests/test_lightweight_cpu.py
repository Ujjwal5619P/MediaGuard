from pathlib import Path
import subprocess
import sys

import pytest
from unittest.mock import patch
from PIL import Image
from unittest.mock import patch

from app.config import Settings
from app.ml.inference import InferenceEngine
from app.ml.model_manager import ModelManager


MODEL_FILE = Settings().models_dir / "imager_1_tiny" / "model_imager-1-tiny.pth"


def test_api_import_does_not_eagerly_import_optional_model_stacks() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.main; print(','.join(name for name in ('torch', 'cv2', 'transformers') if name in sys.modules))",
        ],
        cwd=backend_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_server_launcher_honors_render_port_and_host(monkeypatch) -> None:
    import run_server

    captured = {}

    def fake_run(app, **kwargs):
        captured.update(app=app, **kwargs)

    monkeypatch.setenv("PORT", "9137")
    monkeypatch.setenv("MEDIAGUARD_HOST", "0.0.0.0")
    monkeypatch.setattr(run_server.uvicorn, "run", fake_run)
    run_server.main()

    assert captured["port"] == 9137
    assert captured["host"] == "0.0.0.0"


@pytest.mark.skipif(not MODEL_FILE.is_file(), reason="Imager-1-Tiny checkpoint is not installed")
def test_imager_tiny_loads_and_infers_on_cpu() -> None:
    settings = Settings()
    manager = ModelManager(settings.models_dir)
    manager.verify_existing_models(keys=["ai_generated_detector"])
    engine = InferenceEngine(manager, settings.artifacts_dir)

    assert engine._load_primary_detector_if_verified() is True
    result, _, device = engine.run_ai_generated_detection(
        Image.new("RGB", (256, 256), color=(70, 110, 150))
    )

    assert result.status == "evaluated"
    assert result.execution_device == "cpu"
    assert device == "cpu"
    assert result.decision_threshold == 0.385
    assert result.score is not None and 0.0 <= result.score <= 1.0
    assert result.processing_ms is not None and result.processing_ms > 0


def test_lightweight_pipeline_only_runs_primary_and_reports_only_its_telemetry() -> None:
    settings = Settings(lightweight_mode=True)
    manager = ModelManager(settings.models_dir)
    manager.verify_existing_models(keys=["ai_generated_detector"])
    engine = InferenceEngine(manager, settings.artifacts_dir)
    engine.lightweight_mode = True

    with (
        patch.object(engine, "run_ai_generated_detection", return_value=(object(), 7.0, "cpu")) as primary,
        patch.object(engine, "run_dinov2", side_effect=AssertionError("DINOv2 executed")),
        patch.object(engine, "run_univfd_detection", side_effect=AssertionError("UnivFD executed")),
        patch.object(engine, "run_face_manipulation_detection", side_effect=AssertionError("EfficientNet executed")),
    ):
        image = Image.new("RGB", (32, 32))
        from io import BytesIO
        payload = BytesIO()
        image.save(payload, format="PNG")
        response = engine.analyze_image_pipeline(payload.getvalue(), "scan_test", False)

    primary.assert_called_once()
    assert response[0].status == "unavailable"
    assert response[2].status == "unavailable"
    assert response[3].status == "unavailable"
    assert response[4].available is False
    assert response[7] == 7.0
    assert response[8] == "cpu"
