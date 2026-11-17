from fastapi.testclient import TestClient

from app.config import Settings
from tests.conftest import make_jpeg_bytes, make_png_bytes


def test_analyze_valid_png_and_jpeg_clean_temp_files(
    client: TestClient, test_settings: Settings
) -> None:
    for fname, payload, mime in [
        ("sample.png", make_png_bytes(96, 64), "image/png"),
        ("sample.jpg", make_jpeg_bytes(80, 80), "image/jpeg"),
    ]:
        resp = client.post(
            "/api/v1/analyze/image",
            files={"file": (fname, payload, mime)},
            data={"include_attribution": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["original_deleted"] is True
        assert list(test_settings.temp_dir.glob("*")) == []
        assert body["file_metadata"]["detected_mime"] == mime
        assert body["ai_generated_indicator"]["status"] == "unavailable"
        assert body["ai_generated_indicator"]["score"] is None
        assert body["face_manipulation_indicator"]["status"] == "unavailable"
        assert body["face_manipulation_indicator"]["score"] is None
        assert body["experimental_attribution"]["available"] is False


def test_rejects_magic_byte_spoofing(client: TestClient) -> None:
    fake_payload = b"MZ\x90\x00This is not actually a PNG file"
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("evil.png", fake_payload, "image/png")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "ERR_UNSUPPORTED_SIGNATURE"


def test_rejects_mime_signature_mismatch(client: TestClient) -> None:
    png_bytes = make_png_bytes()
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("mislabeled.jpg", png_bytes, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "ERR_MIME_SIGNATURE_MISMATCH"


def test_rejects_malformed_image_stream(client: TestClient) -> None:
    corrupt_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("corrupt.jpg", corrupt_jpeg, "image/jpeg")},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "ERR_MALFORMED_IMAGE"


def test_rejects_oversized_upload(client: TestClient) -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + (b"\x00" * (1024 * 1024 + 100))
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("huge.png", oversized, "image/png")},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "ERR_FILE_TOO_LARGE"


def test_rejects_dimension_limit_exceeded(client: TestClient) -> None:
    too_wide = make_png_bytes(width=1200, height=32)
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("wide.png", too_wide, "image/png")},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "ERR_DIMENSION_LIMIT_EXCEEDED"


def test_dinov2_feature_endpoint_reports_unavailable_without_weights(
    client: TestClient,
) -> None:
    jpg_bytes = make_jpeg_bytes()
    resp = client.post(
        "/api/v1/features/dinov2",
        files={"file": ("photo.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "ERR_DINOV2_UNAVAILABLE"


def test_rate_limiter_triggers_429(client: TestClient) -> None:
    jpg_bytes = make_jpeg_bytes(32, 32)
    statuses = []
    for _ in range(35):
        r = client.post(
            "/api/v1/analyze/image",
            files={"file": ("burst.jpg", jpg_bytes, "image/jpeg")},
        )
        statuses.append(r.status_code)
    assert 429 in statuses
