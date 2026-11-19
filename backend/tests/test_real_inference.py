"""
Real Checkpoint Inference Integration Tests

These tests run against the actual persistent `backend/models/` directory and
downloaded weights (not mocks). If weights have not been downloaded yet via
`python setup_models.py`, the tests are skipped cleanly with an explicit reason
so CI/unit tests remain honest.
"""

import io
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.ml.model_manager import ModelManager


def _real_models_available() -> bool:
    settings = Settings()
    mgr = ModelManager(settings.models_dir)
    mgr.verify_existing_models()
    return mgr.get_status().all_verified


@pytest.fixture(scope="module")
def real_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp_path = tmp_path_factory.mktemp("real_inference")
    base = Settings()
    settings = Settings(
        models_dir=base.models_dir,
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "real_test.db",
        temp_dir=tmp_path / "data" / "tmp",
        artifacts_dir=tmp_path / "data" / "artifacts",
        auto_download_models=False,
        low_vram_offload=True,
    )
    app = create_app(settings, load_models_on_startup=True)
    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(
    not _real_models_available(),
    reason="Real model checkpoints not downloaded yet; run `python setup_models.py` first.",
)
def test_real_checkpoints_load_and_infer_non_face_png_and_jpeg(
    real_client: TestClient,
) -> None:
    health = real_client.get("/api/v1/health").json()
    assert health["status"] == "ok"
    assert health["models"]["dinov2_extractor"]["status"] == "ready"
    assert health["models"]["ai_generated_detector"]["status"] == "ready"
    assert (
        health["models"]["ai_generated_detector"]["model_id"]
        == "Imagers/Imager-1-tiny"
    )
    assert health["models"]["univfd_detector"]["status"] == "ready"
    assert health["models"]["face_manipulation_detector"]["status"] == "ready"

    # 1. Test gradient PNG without a face -> DINOv2 + Imager primary + UnivFD evaluated, face = no_face_detected
    img = Image.new("RGB", (256, 256))
    for y in range(256):
        for x in range(256):
            img.putpixel((x, y), (x, y, (x + y) // 2))
    png_buf = io.BytesIO()
    img.save(png_buf, format="PNG")
    jpg_buf = io.BytesIO()
    img.save(jpg_buf, format="JPEG", quality=92)

    for fname, payload, mime in [
        ("gradient_no_face.png", png_buf.getvalue(), "image/png"),
        ("gradient_no_face.jpg", jpg_buf.getvalue(), "image/jpeg"),
    ]:
        resp = real_client.post(
            "/api/v1/analyze/image",
            files={"file": (fname, payload, mime)},
            data={"include_attribution": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["original_deleted"] is True

        assert body["dinov2_features"]["status"] == "evaluated"
        assert body["dinov2_features"]["embedding_dim"] == 384
        assert body["dinov2_features"]["l2_norm"] > 0

        ai = body["ai_generated_indicator"]
        assert ai["status"] == "evaluated"
        assert ai["model_id"] == "Imagers/Imager-1-tiny"
        assert "Imager-1-Tiny" in ai["model_name"]
        assert "Imagers/Imager-1-tiny" in ai["model_source"]
        assert 0.0 <= ai["score"] <= 1.0
        assert ai["calibrated"] is False

        univfd = body["univfd_indicator"]
        assert univfd is not None
        assert univfd["status"] == "evaluated"
        assert univfd["model_id"] == "univfd-clip-vit-l14"
        assert "CLIP" in univfd["architecture"]
        assert 0.0 <= univfd["score"] <= 1.0
        assert univfd["calibrated"] is False

        face = body["face_manipulation_indicator"]
        assert face["status"] == "no_face_detected"
        assert face["score"] is None
        assert face["faces_detected"] == 0

        attr = body["experimental_attribution"]
        assert attr["patch_attention_available"] is True
        assert attr["gradcam_available"] is False
        art_resp = real_client.get(attr["patch_attention_url"])
        assert art_resp.status_code == 200
        assert art_resp.headers["content-type"] == "image/png"

    feat_resp = real_client.post(
        "/api/v1/features/dinov2",
        files={"file": ("gradient_features.jpg", jpg_buf.getvalue(), "image/jpeg")},
    )
    assert feat_resp.status_code == 200
    feat_body = feat_resp.json()
    assert feat_body["status"] == "evaluated"
    assert feat_body["embedding_dim"] == 384
    assert len(feat_body["embedding_preview"]) == 8


@pytest.mark.skipif(
    not _real_models_available(),
    reason="Real model checkpoints not downloaded yet; run `python setup_models.py` first.",
)
def test_real_checkpoints_face_detection_and_gradcam_on_portrait(
    real_client: TestClient,
) -> None:
    req = urllib.request.Request(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        headers={"User-Agent": "MediaGuardTest/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            jpeg_bytes = r.read()
    except Exception:
        pytest.skip("Unable to fetch sample portrait image for face test.")

    resp = real_client.post(
        "/api/v1/analyze/image",
        files={"file": ("lena_portrait.jpg", jpeg_bytes, "image/jpeg")},
        data={"include_attribution": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["ai_generated_indicator"]["status"] == "evaluated"
    assert body["ai_generated_indicator"]["model_id"] == "Imagers/Imager-1-tiny"
    assert 0.0 <= body["ai_generated_indicator"]["score"] <= 1.0

    assert body["univfd_indicator"] is not None
    assert body["univfd_indicator"]["status"] == "evaluated"
    assert body["univfd_indicator"]["model_id"] == "univfd-clip-vit-l14"
    assert 0.0 <= body["univfd_indicator"]["score"] <= 1.0

    face = body["face_manipulation_indicator"]
    assert face["status"] == "evaluated"
    assert face["faces_detected"] >= 1
    assert 0.0 <= face["score"] <= 1.0

    attr = body["experimental_attribution"]
    assert attr["gradcam_available"] is True
    assert attr["patch_attention_available"] is True
    assert real_client.get(attr["gradcam_url"]).status_code == 200
    assert real_client.get(attr["patch_attention_url"]).status_code == 200

    scans_resp = real_client.get("/api/v1/scans")
    assert scans_resp.status_code == 200
    assert scans_resp.json()["total"] >= 3
    assert len(scans_resp.json()["items"]) >= 3

    analytics_resp = real_client.get("/api/v1/analytics")
    assert analytics_resp.status_code == 200
    assert analytics_resp.json()["total_scans"] >= 3
    assert analytics_resp.json()["ai_evaluated_count"] >= 3
    assert analytics_resp.json()["univfd_evaluated_count"] >= 3
