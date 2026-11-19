from fastapi import APIRouter, Request

from app.schemas import HealthResponse, RetentionPolicySummary

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    registry = request.app.state.registry
    cuda_available, device_name, execution_mode, vram_measured = registry.probe_hardware()
    readiness = registry.get_readiness_map()

    all_ready = readiness.ai_generated_detector.status == "ready" and (
        settings.lightweight_mode
        or (
            readiness.dinov2_extractor.status == "ready"
            and readiness.face_manipulation_detector.status == "ready"
        )
    )

    return HealthResponse(
        status="ok" if all_ready else "degraded",
        service=settings.service_name,
        milestone=settings.milestone,
        version="1.3.0",
        timestamp_utc=registry.utc_now_iso(),
        cuda_available=cuda_available,
        device_name=device_name,
        execution_mode=execution_mode,
        vram_measured=vram_measured,
        sequential_gpu_lock=True,
        max_concurrent_gpu_workers=1,
        low_vram_offload=settings.low_vram_offload,
        models=readiness,
        retention_policy=RetentionPolicySummary(
            delete_original_after_analysis=settings.delete_original_after_analysis,
            never_train_without_consent=settings.never_train_without_consent,
            record_retention_days=settings.record_retention_days,
            artifact_retention_days=settings.artifact_retention_days,
            max_upload_bytes=settings.max_upload_bytes,
            max_image_pixels=settings.max_image_pixels,
            max_dimension=settings.max_dimension,
        ),
    )
