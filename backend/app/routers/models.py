import asyncio

from fastapi import APIRouter, Request

from app.schemas import ModelsStatusResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/status", response_model=ModelsStatusResponse)
async def get_models_status(request: Request) -> ModelsStatusResponse:
    registry = request.app.state.registry
    return registry.manager.get_status()


@router.post("/setup", response_model=ModelsStatusResponse)
async def setup_and_load_models(request: Request) -> ModelsStatusResponse:
    registry = request.app.state.registry
    async with registry.gpu_semaphore:
        await asyncio.to_thread(registry.setup_and_load_all)
    return registry.manager.get_status()
