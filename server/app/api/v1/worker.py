import logging

from app.core.state import ml_state
from app.core.worker_errors import WorkerError
from app.core.worker_types import StopReason
from fastapi import APIRouter
from fastapi import HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/worker/status")
async def worker_status():
    if ml_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    try:
        status = await ml_state.worker_manager.get_worker_status()
        return status.model_dump(mode="json")
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/worker/warmup")
async def worker_warmup():
    if ml_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    try:
        await ml_state.worker_manager.warmup()
        status = await ml_state.worker_manager.get_worker_status()
        return {"ok": True, "status": status.model_dump(mode="json")}
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/worker/stop")
async def worker_stop():
    if ml_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    await ml_state.worker_manager.stop(StopReason.MANUAL)
    status = await ml_state.worker_manager.get_worker_status()
    return {"ok": True, "status": status.model_dump(mode="json")}
