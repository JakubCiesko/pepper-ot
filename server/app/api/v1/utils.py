import base64
import io
import json
import logging
from pathlib import Path
import time
from typing import Any

from app.core.state import MLState
from app.core.state import ml_state
from app.core.storage import save_last_image_async
from app.core.storage import save_last_state_async
from app.core.worker_errors import WorkerError
from app.core.ws_manager import ws_manager
from app.inference.types import PipelineResult
from app.schemas.robot import PersonMetadata
from app.schemas.robot import RobotMetadata
from app.schemas.scene import SceneState
from fastapi import HTTPException
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# dependencies
def get_pipeline():
    """[DEPRECATED]: Safely injects the visual pipeline into endpoints."""
    if ml_state.pipeline is None:
        raise HTTPException(
            status_code=503, detail="AI Pipeline is currently warming up. Please wait."
        )
    return ml_state.pipeline


def get_chat_service():
    """Safely injects the chat/RAG service into endpoints."""
    if ml_state.chat_service is None:
        raise HTTPException(status_code=503, detail="Chat Service is not initialized.")
    return ml_state.chat_service


def get_memory():
    if ml_state.pipeline is None or ml_state.pipeline.memory is None:
        logger.warning("Memory requested but not initialized")
        raise HTTPException(status_code=503, detail="Memory not initialized")
    return ml_state.pipeline.memory


# detect utils


def parse_robot_metadata(
    head_yaw: float,
    head_pitch: float,
    body_yaw: float | None,
    camera_hfov: float | None,
    camera_vfov: float | None,
    image_width: int | None,
    image_height: int | None,
    timestamp: float | None,
    frame_id: str | None,
    scan_id: str | None,
    people: str | None,
) -> RobotMetadata:
    """
    Parses form fields and constructs a RobotMetadata object.

    Args:
        head_yaw: Head yaw angle.
        head_pitch: Head pitch angle.
        body_yaw: Body yaw angle.
        camera_hfov: Camera horizontal field of view.
        camera_vfov: Camera vertical field of view.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        timestamp: Epoch timestamp.
        frame_id: Unique frame identifier.
        scan_id: Scan identifier.
        people: JSON string of people metadata.

    Returns:
        RobotMetadata object with parsed people list.
    """
    people_list = []
    if people:
        try:
            parsed = json.loads(people)
            people_list.extend([PersonMetadata(**item) for item in parsed])
        except Exception as e:
            logger.warning(f"Failed to parse people metadata: {e}")
            people_list = []

    return RobotMetadata(
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        body_yaw=body_yaw,
        camera_hfov=camera_hfov,
        camera_vfov=camera_vfov,
        image_width=image_width,
        image_height=image_height,
        timestamp=timestamp,
        frame_id=frame_id,
        scan_id=scan_id,
        people=people_list,
    )


# TODO: think about scene graph, right now in inference.types only; should be in schemas? that is output only
async def upload_data_to_dashboard(
    result: PipelineResult, current_scene_state: SceneState, objects: list[dict]
) -> dict[str, Any]:
    """
    Converts SoM image to base64 and broadcasts the detection payload to the dashboard.

    Args:
        result: PipelineResult containing all outputs from processing.
        current_scene_state: Current scene state saved.
        objects: List of serialized detection objects.

    Returns:
        Payload dict sent to WebSocket dashboard.
    """
    som_image = result.som_image
    metrics = result.metrics
    scene_graph = result.scene_graph
    executed_stages = result.executed_stages
    scene_graph_dict = scene_graph.as_dict() if scene_graph else []
    logger.debug(f"Current Scene State : {current_scene_state}")
    logger.debug(f"Current scene graph: {scene_graph_dict}")
    logger.info("Uploading data to dashboard...")
    display_image = som_image
    if display_image is None:
        display_image = np.array(result.raw_image)
    som_pil = Image.fromarray(display_image.astype("uint8"))
    buf = io.BytesIO()
    som_pil.save(buf, format="JPEG")
    som_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    payload = {
        "type": "detection",
        "image": som_b64,
        "objects": objects,
        "scene_graph": scene_graph_dict,
        "memory": (
            current_scene_state.model_dump()
            if current_scene_state
            else {"objects": [], "relationships": [], "timestamp": time.time()}
        ),
        "metrics": metrics,
        "executed_stages": executed_stages,
    }
    await ws_manager.broadcast(payload)
    return payload


# TODO: MAYBE DO NOT PASS ML_STATE?
async def update_and_persist_appstate(ml_state: MLState, payload: dict[str, Any]):
    """
    Updates global ML state and persists to disk if configured.

    Args:
        ml_state: The global ML application state.
        payload: Detection payload to save.
    """
    ml_state.last_state = payload
    if ml_state.config and ml_state.config.storage.persist_last_state:
        base_dir = (
            ml_state.config._config_path.parent
            if ml_state.config._config_path is not None
            else Path.cwd()
        )
        state_path = base_dir / ml_state.config.storage.last_state_path
        persist_payload = dict(payload)
        if ml_state.config.storage.store_image:
            image_path = state_path.with_suffix(".jpg")
            persist_payload["image"] = None
            persist_payload["image_path"] = str(image_path.relative_to(base_dir))
            await save_last_image_async(image_path, payload["image"])
        else:
            persist_payload["image"] = None
            persist_payload.pop("image_path", None)
        await save_last_state_async(state_path, persist_payload)


async def upload_worker_payload(payload: dict[str, Any]):
    dashboard_payload = {
        "type": "detection",
        "image": payload.get("image"),
        "objects": payload.get("objects", []),
        "scene_graph": payload.get("scene_graph", []),
        "memory": payload.get(
            "memory", {"objects": [], "relationships": [], "timestamp": time.time()}
        ),
        "metrics": payload.get("metrics", {}),
        "executed_stages": payload.get("executed_stages", []),
    }
    await ws_manager.broadcast(dashboard_payload)


# worker utils
def worker_enabled() -> bool:
    return bool(
        ml_state.config
        and ml_state.config.worker.enabled
        and ml_state.worker_manager is not None
    )


async def worker_request(
    method: str,
    path: str,
    *,
    json_payload: dict | None = None,
    params: dict | None = None,
) -> dict:
    if ml_state.worker_manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not initialized")
    try:
        clean_params = (
            {k: v for k, v in (params or {}).items() if v is not None}
            if params is not None
            else None
        )
        return await ml_state.worker_manager.request(
            method, path, json=json_payload, params=clean_params
        )
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Worker request failed: {exc}"
        ) from exc
