import asyncio
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from app.ml.evaluator import (
    evaluate_local_dataset,
    inspect_dataset_folder,
    resolve_dataset_dir,
)
from app.schemas import (
    DatasetEvaluationDetail,
    DatasetEvaluationRequest,
    DatasetEvaluationSummary,
    DatasetInspectResponse,
)

router = APIRouter(prefix="/api/v1/evaluation", tags=["dataset-evaluation"])


@router.get("/inspect-folder", response_model=DatasetInspectResponse)
async def inspect_evaluation_folder(
    request: Request,
    dataset_dir: Optional[str] = Query(default=None),
) -> DatasetInspectResponse:
    settings = request.app.state.settings
    resolved = resolve_dataset_dir(settings.dataset_dir, dataset_dir)
    return inspect_dataset_folder(resolved)


@router.post("/run", response_model=DatasetEvaluationDetail)
async def run_dataset_evaluation(
    request: Request,
    body: DatasetEvaluationRequest,
) -> DatasetEvaluationDetail:
    settings = request.app.state.settings
    registry = request.app.state.registry
    repo = request.app.state.repo

    resolved = resolve_dataset_dir(settings.dataset_dir, body.dataset_dir)
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ERR_DATASET_DIR_NOT_FOUND",
                "message": (
                    f"Dataset directory does not exist: {resolved}. "
                    "Create `dataset/ai_generated/` and `dataset/real/` locally with supported images."
                ),
            },
        )

    inspection = inspect_dataset_folder(resolved)
    if inspection.total_supported_count == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ERR_EMPTY_DATASET",
                "message": (
                    f"No supported images (.jpg, .jpeg, .png, .webp) found inside "
                    f"`{resolved}/ai_generated/` or `{resolved}/real/`."
                ),
            },
        )

    async with registry.gpu_semaphore:
        detail = await asyncio.to_thread(
            evaluate_local_dataset,
            registry.engine,
            resolved,
            body.split_role,
            body.decision_threshold,
            body.include_univfd,
            body.max_images,
        )

    repo.insert_evaluation(detail)
    return detail


@router.get("/runs", response_model=List[DatasetEvaluationSummary])
async def list_evaluation_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> List[DatasetEvaluationSummary]:
    repo = request.app.state.repo
    return repo.list_evaluations(limit=limit)


@router.get("/runs/{evaluation_id}", response_model=DatasetEvaluationDetail)
async def get_evaluation_run(
    request: Request,
    evaluation_id: str,
) -> DatasetEvaluationDetail:
    repo = request.app.state.repo
    detail = repo.get_evaluation(evaluation_id)
    if not detail:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_EVALUATION_NOT_FOUND",
                "message": f"Evaluation run '{evaluation_id}' was not found.",
            },
        )
    return detail


@router.get("/runs/{evaluation_id}/export")
async def export_evaluation_run(
    request: Request,
    evaluation_id: str,
    format: Literal["csv", "json"] = Query(default="csv"),
) -> PlainTextResponse:
    repo = request.app.state.repo
    payload = repo.export_evaluation(evaluation_id, fmt=format)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ERR_EVALUATION_NOT_FOUND",
                "message": f"Evaluation run '{evaluation_id}' was not found.",
            },
        )
    media_type = "application/json" if format == "json" else "text/csv"
    return PlainTextResponse(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=mediaguard-eval-{evaluation_id}.{format}"
        },
    )
