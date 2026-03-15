import logging

from app.core.runtime.state import app_state
from app.core.runtime.worker_client.errors import WorkerError
from app.core.runtime.worker_client.types import StopReason
from fastapi import APIRouter
from fastapi import HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/worker/status")
async def worker_status():
    """
    Return current worker process status snapshot.

    Uses the runtime worker manager to fetch local/remote worker state and returns
    a JSON-serializable status model.

    Returns:
      dict: Worker status payload including state, pid, inflight count,
      config version, restart counters, and related telemetry fields.

    Raises:
      HTTPException:
          - 503 if worker manager is not initialized.
          - 503 if worker runtime reports an operational error.
    """

    if app_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    try:
        status = await app_state.worker_manager.get_worker_status()
        return status.model_dump(mode="json")
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/worker/warmup")
async def worker_warmup():
    """
    Start or wake the worker and perform warmup initialization.

    Triggers worker startup (if needed), executes warmup routine, and returns
    the resulting worker status.

    Returns:
      dict:
          {
            "ok": True,
            "status": <worker status payload>
          }

    Raises:
      HTTPException:
          - 503 if worker manager is not initialized.
          - 503 if warmup/startup fails with a worker runtime error.
    """

    if app_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    try:
        await app_state.worker_manager.warmup()
        status = await app_state.worker_manager.get_worker_status()
        return {"ok": True, "status": status.model_dump(mode="json")}
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/worker/stop")
async def worker_stop():
    """
    Manually stop the worker process and return post-stop status.

    Sends a manual stop reason to worker manager, then retrieves and returns
    the latest status snapshot.

    Returns:
    dict:
    {
    "ok": True,
    "status": <worker status payload>
    }

    Raises:
    HTTPException:
    - 503 if worker manager is not initialized.
    """

    if app_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    await app_state.worker_manager.stop(StopReason.MANUAL)
    status = await app_state.worker_manager.get_worker_status()
    return {"ok": True, "status": status.model_dump(mode="json")}
