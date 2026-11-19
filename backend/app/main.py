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
            app.state.registry.load_cached_models()
            status = app.state.registry.manager.get_status()
            primary = app.state.registry.manager.get_entry("ai_generated_detector")
            if settings.auto_download_models and not primary.verified:
                threading.Thread(
                    target=app.state.registry.setup_and_load_all,
                    daemon=True,
                ).start()
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
