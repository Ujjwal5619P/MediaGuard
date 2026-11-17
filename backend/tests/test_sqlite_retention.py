from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.schemas import (
    AnalyzeImageResponse,
    Dinov2FeatureSummary,
    ExifSummary,
    ExperimentalAttribution,
    FileMetadata,
    TaskIndicator,
)
from tests.conftest import make_png_bytes


def test_sqlite_history_crud_and_exports(client: TestClient) -> None:
    png_bytes = make_png_bytes(48, 48)
    create_resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("history_item.png", png_bytes, "image/png")},
    )
    assert create_resp.status_code == 200
    scan_id = create_resp.json()["scan_id"]

    list_resp = client.get("/api/v1/scans", params={"search": "history_item"})
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 1
    assert list_resp.json()["items"][0]["scan_id"] == scan_id

    detail_resp = client.get(f"/api/v1/scans/{scan_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["scan_id"] == scan_id

    analytics_resp = client.get("/api/v1/analytics")
    assert analytics_resp.status_code == 200
    assert analytics_resp.json()["total_scans"] == 1
    assert analytics_resp.json()["originals_deleted_count"] == 1

    csv_resp = client.get("/api/v1/scans/export", params={"format": "csv"})
    assert csv_resp.status_code == 200
    assert "ai_generated_status,ai_generated_score" in csv_resp.text
    assert "history_item.png" in csv_resp.text

    json_resp = client.get("/api/v1/scans/export", params={"format": "json"})
    assert json_resp.status_code == 200
    exported_rows = json_resp.json()
    assert len(exported_rows) == 1
    assert exported_rows[0]["scan_id"] == scan_id

    del_resp = client.delete(f"/api/v1/scans/{scan_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    missing_resp = client.get(f"/api/v1/scans/{scan_id}")
    assert missing_resp.status_code == 404


def test_independent_artifact_deletion_and_ttl_purge(client: TestClient) -> None:
    repo = client.app.state.repo
    artifacts_dir = client.app.state.settings.artifacts_dir
    artifact_file = artifacts_dir / "scan_0123456789abcdef_gradcam.png"
    artifact_file.write_bytes(b"\x89PNG\r\n\x1a\nfake-artifact")

    now = datetime.now(timezone.utc)
    record = AnalyzeImageResponse(
        scan_id="scan_0123456789abcdef",
        created_at=now.isoformat(),
        sha256="a" * 64,
        original_deleted=True,
        inference_ms=10,
        execution_device="cpu",
        vram_peak_mb=None,
        file_metadata=FileMetadata(
            filename="ttl_test.png",
            detected_mime="image/png",
            declared_mime="image/png",
            byte_size=123,
            width=64,
            height=64,
            exif_summary=ExifSummary(has_exif=False),
        ),
        dinov2_features=Dinov2FeatureSummary(
            status="unavailable",
            model_id="facebook/dinov2-small",
            reason="Test",
        ),
        ai_generated_indicator=TaskIndicator(
            task="ai_generated_image_detection",
            status="unavailable",
            score=None,
            decision_threshold=0.5,
            calibrated=False,
            model_id="Ateeqq/ai-vs-human-image-detector",
            model_version="test",
            limitations="Test",
            reason="Test",
        ),
        face_manipulation_indicator=TaskIndicator(
            task="face_manipulation_detection",
            status="unavailable",
            score=None,
            decision_threshold=0.5,
            calibrated=False,
            model_id="Xicor9/efficientnet-b0-ffpp-c23",
            model_version="test",
            limitations="Test",
            reason="Test",
        ),
        experimental_attribution=ExperimentalAttribution(
            available=True,
            gradcam_available=True,
            gradcam_url="/api/v1/artifacts/scan_0123456789abcdef_gradcam.png",
            method="gradcam",
            artifact_url="/api/v1/artifacts/scan_0123456789abcdef_gradcam.png",
            reason="Test artifact",
        ),
        record_expires_at=(now + timedelta(days=30)).isoformat(),
        artifact_expires_at=(now - timedelta(hours=1)).isoformat(),
    )
    repo.insert_scan(record, gradcam_path=str(artifact_file))

    # Artifact TTL expired, record TTL still valid
    purge_resp = client.post("/api/v1/scans/purge-expired")
    assert purge_resp.status_code == 200
    assert purge_resp.json()["expired_artifacts_purged"] == 1
    assert purge_resp.json()["expired_records_deleted"] == 0
    assert not artifact_file.exists()

    # Record still exists
    preserved = client.get("/api/v1/scans/scan_0123456789abcdef")
    assert preserved.status_code == 200
    assert preserved.json()["experimental_attribution"]["available"] is False

    # Test manual artifact delete endpoint preserves record
    artifact_file.write_bytes(b"\x89PNG\r\n\x1a\nmanual-purge")
    repo.delete_scan("scan_0123456789abcdef")
    repo.insert_scan(record, gradcam_path=str(artifact_file))
    del_art_resp = client.delete("/api/v1/scans/scan_0123456789abcdef/artifact")
    assert del_art_resp.status_code == 200
    assert del_art_resp.json()["artifact_deleted"] is True
    assert del_art_resp.json()["record_preserved"] is True
    assert not artifact_file.exists()
    assert client.get("/api/v1/scans/scan_0123456789abcdef").status_code == 200
