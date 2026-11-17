import re
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from app.schemas import (
    AnalyticsResponse,
    AnalyzeImageResponse,
    DeleteArtifactResponse,
    DeleteScanResponse,
    PurgeExpiredResponse,
    ScanListResponse,
    TaskStatus,
)

router = APIRouter(prefix="/api/v1", tags=["scans"])

SAFE_ARTIFACT_NAME = re.compile(r"^scan_[a-f0-9]{8,32}_(gradcam|dinov2_patch)\.png$")


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_scan_analytics(request: Request) -> AnalyticsResponse:
    repo = request.app.state.repo
    return repo.get_analytics()


@router.get("/artifacts/{filename}")
async def get_attribution_artifact(request: Request, filename: str):
    if not SAFE_ARTIFACT_NAME.match(filename):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ERR_INVALID_ARTIFACT_NAME",
                "message": "Invalid artifact filename.",
            },
        )
    artifacts_dir = request.app.state.settings.artifacts_dir.resolve()
    target = (artifacts_dir / filename).resolve()
    if target.parent != artifacts_dir or not target.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_ARTIFACT_NOT_FOUND",
                "message": "Attribution artifact not found or expired.",
            },
        )
    return FileResponse(target, media_type="image/png")


@router.get("/scans", response_model=ScanListResponse)
async def list_scans(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None, max_length=120),
    status: Optional[TaskStatus] = Query(default=None),
) -> ScanListResponse:
    repo = request.app.state.repo
    return repo.list_scans(limit=limit, offset=offset, search=search, status=status)


@router.get("/scans/export")
async def export_scans(
    request: Request,
    format: Literal["csv", "json"] = Query(default="csv"),
    status: Optional[TaskStatus] = Query(default=None),
) -> Response:
    repo = request.app.state.repo
    payload = repo.export_scans(fmt=format, status=status)
    media_type = "text/csv" if format == "csv" else "application/json"
    filename = f"mediaguard-scans.{format}"
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/scans/purge-expired", response_model=PurgeExpiredResponse)
async def purge_expired_records(request: Request) -> PurgeExpiredResponse:
    repo = request.app.state.repo
    records_deleted, artifacts_purged = repo.purge_expired()
    return PurgeExpiredResponse(
        expired_records_deleted=records_deleted,
        expired_artifacts_purged=artifacts_purged,
    )


@router.get("/scans/{scan_id}", response_model=AnalyzeImageResponse)
async def get_scan(request: Request, scan_id: str) -> AnalyzeImageResponse:
    repo = request.app.state.repo
    record = repo.get_scan(scan_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_SCAN_NOT_FOUND",
                "message": "Scan record not found.",
            },
        )
    return record


@router.delete("/scans/{scan_id}/artifact", response_model=DeleteArtifactResponse)
async def delete_scan_artifact(
    request: Request, scan_id: str
) -> DeleteArtifactResponse:
    repo = request.app.state.repo
    found, artifact_deleted = repo.delete_artifact(scan_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_SCAN_NOT_FOUND",
                "message": "Scan record not found.",
            },
        )
    return DeleteArtifactResponse(
        artifact_deleted=artifact_deleted,
        scan_id=scan_id,
        record_preserved=True,
    )


@router.delete("/scans/{scan_id}", response_model=DeleteScanResponse)
async def delete_scan(request: Request, scan_id: str) -> DeleteScanResponse:
    repo = request.app.state.repo
    found, artifact_deleted = repo.delete_scan(scan_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_SCAN_NOT_FOUND",
                "message": "Scan record not found.",
            },
        )
    return DeleteScanResponse(
        deleted=True,
        scan_id=scan_id,
        artifact_deleted=artifact_deleted,
    )
