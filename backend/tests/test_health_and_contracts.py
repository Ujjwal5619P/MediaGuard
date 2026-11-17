from fastapi.testclient import TestClient


def test_health_and_model_status_without_weights(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "degraded"
    assert data["sequential_gpu_lock"] is True
    assert data["max_concurrent_gpu_workers"] == 1

    models = data["models"]
    assert models["dinov2_extractor"]["status"] == "unavailable"
    assert models["dinov2_extractor"]["checkpoint_verified"] is False
    assert models["ai_generated_detector"]["status"] == "unavailable"
    assert models["ai_generated_detector"]["model_id"] == "abacusai/Smaug-AI-Image-Detector"
    assert models["univfd_detector"]["status"] == "unavailable"
    assert "openai/clip-vit-large-patch14" in models["univfd_detector"]["model_id"]
    assert models["face_manipulation_detector"]["status"] == "unavailable"
    assert models["calibration_profile"]["status"] == "unavailable"

    retention = data["retention_policy"]
    assert retention["delete_original_after_analysis"] is True
    assert retention["never_train_without_consent"] is True

    mstatus = client.get("/api/v1/models/status")
    assert mstatus.status_code == 200
    mdata = mstatus.json()
    assert mdata["all_verified"] is False
    assert len(mdata["models"]) == 4
    assert all(m["status"] == "not_downloaded" for m in mdata["models"])


def test_queued_video_and_audio_return_501(client: TestClient) -> None:
    video_resp = client.post("/api/v1/analyze/video")
    assert video_resp.status_code == 501
    assert video_resp.json()["detail"]["modality"] == "video"
    assert video_resp.json()["detail"]["status"] == "queued"

    audio_resp = client.post("/api/v1/analyze/audio")
    assert audio_resp.status_code == 501
    assert audio_resp.json()["detail"]["modality"] == "audio"
    assert audio_resp.json()["detail"]["status"] == "queued"
