import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test_mediaguard.db",
        temp_dir=tmp_path / "data" / "tmp",
        artifacts_dir=tmp_path / "data" / "artifacts",
        max_upload_bytes=1024 * 1024,  # 1 MB for fast tests
        max_image_pixels=1_000_000,
        max_dimension=1024,
        rate_limit_per_min=30,
        auto_download_models=False,
        lightweight_mode=True,
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    app = create_app(test_settings, load_models_on_startup=False)
    with TestClient(app) as test_client:
        yield test_client


def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(20, 90, 140))
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(140, 60, 25))
    img.save(buf, format="JPEG")
    return buf.getvalue()
