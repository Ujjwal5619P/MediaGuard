import asyncio
import io
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, ImageOps

from app.db.database import compute_expiry_iso
from app.schemas import (
    AnalyzeImageResponse,
    Dinov2FeatureSummary,
    FileMetadata,
    QueuedModalityResponse,
)
from app.security import (
    get_client_key,
    inspect_image_safely,
    read_validated_upload,
)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/features/dinov2", response_model=Dinov2FeatureSummary)
async def extract_dinov2_features(
    request: Request,
    file: UploadFile = File(...),
) -> Dinov2FeatureSummary:
    settings = request.app.state.settings
    rate_limiter = request.app.state.rate_limiter
    registry = request.app.state.registry
    rate_limiter.check(get_client_key(request))

    raw_bytes, detected_mime, _ = await read_validated_upload(file, settings)
    inspect_image_safely(raw_bytes, detected_mime, settings)

    with Image.open(io.BytesIO(raw_bytes)) as img:
        pil_rgb = ImageOps.exif_transpose(img).convert("RGB")

    async with registry.gpu_semaphore:
        summary, _, _, _ = await asyncio.to_thread(
            registry.engine.run_dinov2,
            pil_rgb,
            f"feat_{uuid.uuid4().hex[:10]}",
            False,
        )

    if summary.status == "unavailable":
        raise HTTPException(
            status_code=503,
            detail={
                "code": "ERR_DINOV2_UNAVAILABLE",
                "message": summary.reason,
            },
        )
    return summary


@router.post("/analyze/image", response_model=AnalyzeImageResponse)
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    include_attribution: bool = Form(default=True),
) -> AnalyzeImageResponse:
    settings = request.app.state.settings
    rate_limiter = request.app.state.rate_limiter
    registry = request.app.state.registry
    repo = request.app.state.repo

    rate_limiter.check(get_client_key(request))

    raw_bytes, detected_mime, safe_filename = await read_validated_upload(file, settings)
    width, height, sha256_hex, exif_summary, original_deleted = inspect_image_safely(
        raw_bytes, detected_mime, settings
    )

    scan_id = f"scan_{uuid.uuid4().hex[:16]}"

    async with registry.gpu_semaphore:
        (
            dinov2_summary,
            ai_indicator,
            univfd_indicator,
            face_indicator,
            attribution,
            gradcam_path,
            patch_path,
            peak_vram,
            execution_device,
        ) = await asyncio.to_thread(
            registry.engine.analyze_image_pipeline,
            raw_bytes,
            scan_id,
            include_attribution,
        )

    # inference_ms represents detector pipeline execution, not upload validation
    # or database persistence latency.
    elapsed_ms = registry.engine.last_pipeline_ms or 0
    created_at = registry.utc_now_iso()
    record_expires_at = compute_expiry_iso(settings.record_retention_days)
    artifact_expires_at = (
        compute_expiry_iso(settings.artifact_retention_days)
        if attribution.available
        else None
    )

    response = AnalyzeImageResponse(
        scan_id=scan_id,
        created_at=created_at,
        sha256=sha256_hex,
        original_deleted=original_deleted,
        inference_ms=elapsed_ms,
        execution_device=execution_device,
        vram_peak_mb=peak_vram,
        file_metadata=FileMetadata(
            filename=safe_filename,
            detected_mime=detected_mime,
            declared_mime=file.content_type,
            byte_size=len(raw_bytes),
            width=width,
            height=height,
            exif_summary=exif_summary,
        ),
        dinov2_features=dinov2_summary,
        ai_generated_indicator=ai_indicator,
        univfd_indicator=univfd_indicator,
        face_manipulation_indicator=face_indicator,
        experimental_attribution=attribution,
        record_expires_at=record_expires_at,
        artifact_expires_at=artifact_expires_at,
    )

    repo.insert_scan(
        response,
        gradcam_path=str(gradcam_path) if gradcam_path else None,
        patch_attention_path=str(patch_path) if patch_path else None,
    )
    return response


@router.post("/analyze/video")
async def analyze_video_queued() -> QueuedModalityResponse:
    raise HTTPException(
        status_code=501,
        detail=QueuedModalityResponse(
            modality="video",
            status="queued",
            milestone="Post-v1 Video Milestone",
            message="Video deepfake & temporal consistency analysis is queued. Image analysis is the sole active modality in v1.",
        ).model_dump(),
    )


@router.post("/analyze/audio")
async def analyze_audio_queued() -> QueuedModalityResponse:
    raise HTTPException(
        status_code=501,
        detail=QueuedModalityResponse(
            modality="audio",
            status="queued",
            milestone="Post-v1 Audio Milestone",
            message="Audio voice-cloning & synthetic speech analysis is queued. Image analysis is the sole active modality in v1.",
        ).model_dump(),
    )
