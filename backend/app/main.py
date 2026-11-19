import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.db.database import ScanRepository
from app.ml.registry import ModelRegistry
from app.routers import analyze, evaluation, health, models, scans
from app.security import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)


def _download_primary_model(registry: ModelRegistry) -> None:
    """Run startup download off-thread and expose unexpected failures in status."""
    try:
        registry.setup_and_load_all()
        primary = registry.manager.get_entry("ai_generated_detector")
        if primary.verified:
            logger.info("Startup Imager-1-Tiny download completed and checkpoint verified")
        else:
            logger.error(
                "Startup Imager-1-Tiny download finished without a verified checkpoint: %s",
                primary.error_message or primary.status,
            )
    except Exception as exc:
        logger.exception("Startup Imager-1-Tiny download worker failed")
        registry.manager.mark_error("ai_generated_detector", f"Startup download failed: {exc}")


def create_app(
    custom_settings: Optional[Settings] = None,
    load_models_on_startup: bool = True,
) -> FastAPI:
    settings = custom_settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        app.state.repo.initialize()

        if load_models_on_startup:
            logger.info(
                "Model startup configuration: lightweight_mode=%s auto_download_models=%s",
                settings.lightweight_mode,
                settings.auto_download_models,
            )
            app.state.registry.load_cached_models()
            primary = app.state.registry.manager.get_entry("ai_generated_detector")
            logger.info(
                "Imager-1-Tiny startup verification: verified=%s status=%s",
                primary.verified,
                primary.status,
            )
            if settings.auto_download_models and not primary.verified:
                logger.info("Starting background download for Imager-1-Tiny")
                worker = threading.Thread(
                    target=_download_primary_model,
                    args=(app.state.registry,),
                    name="mediaguard-imager-download",
                    daemon=True,
                )
                try:
                    worker.start()
                    logger.info("Background Imager-1-Tiny download thread started")
                except Exception as exc:
                    logger.exception("Could not start Imager-1-Tiny download thread")
                    app.state.registry.manager.mark_error(
                        "ai_generated_detector", f"Could not start startup download: {exc}"
                    )
            elif not settings.auto_download_models:
                logger.info("Automatic model download is disabled by configuration")
            else:
                logger.info("Imager-1-Tiny checkpoint is already verified; download skipped")
        yield
        app.state.registry.engine.unload_models()

    app = FastAPI(
        title="MediaGuard API",
        description="MediaGuard — Verify Before You Trust. AI-assisted image forensics backend.",
        version="1.3.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.repo = ScanRepository(
        db_path=settings.db_path, artifacts_dir=settings.artifacts_dir
    )
    app.state.registry = ModelRegistry(
        models_dir=settings.models_dir,
        artifacts_dir=settings.artifacts_dir,
        low_vram_offload=settings.low_vram_offload,
        decision_threshold=settings.decision_threshold,
        lightweight_mode=settings.lightweight_mode,
    )
    app.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_per_min,
        window_seconds=60,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def sanitized_unhandled_exception(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "ERR_INTERNAL_SERVER",
                    "message": "An unexpected internal error occurred while processing the request.",
                }
            },
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(analyze.router)
    app.include_router(scans.router)
    app.include_router(evaluation.router)

    return app


app = create_app()
