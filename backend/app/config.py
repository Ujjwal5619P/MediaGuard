from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDIAGUARD_",
        env_file=".env",
        extra="ignore",
    )

    service_name: str = "mediaguard-api"
    milestone: str = "imager_tiny_cpu_primary_v1_3"
    host: str = "127.0.0.1"
    # Render injects PORT; keep local runs on 8000 by default.
    port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Upload security bounds
    max_upload_bytes: int = 15 * 1024 * 1024  # 15 MB
    max_image_pixels: int = 36_000_000  # 36 MP decompression bomb guard
    max_dimension: int = 8192

    # Privacy & split retention
    delete_original_after_analysis: bool = True
    never_train_without_consent: bool = True
    record_retention_days: int = 30
    artifact_retention_days: int = 7

    # Rate limiting
    rate_limit_per_min: int = 60

    # ML Model Management & RTX 2050 4 GB VRAM optimization
    auto_download_models: bool = True
    low_vram_offload: bool = True
    decision_threshold: float = 0.385

    # When enabled, use the lightweight primary and leave optional forensic
    # models unloaded on resource-constrained hosts.
    lightweight_mode: bool = False

    # Local storage directories (relative to backend root by default)
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    db_path: Path = Path(__file__).resolve().parent.parent / "data" / "mediaguard.db"
    temp_dir: Path = Path(__file__).resolve().parent.parent / "data" / "tmp"
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "data" / "artifacts"
    dataset_dir: Path = Path(__file__).resolve().parent.parent.parent / "dataset"


def get_settings() -> Settings:
    return Settings()
