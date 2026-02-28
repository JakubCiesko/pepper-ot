import base64
import io
import json
import logging
from pathlib import Path
import time
from typing import Any

from app.api.v1.dependencies import get_pipeline
from app.core.state import MLState
from app.core.state import ml_state
from app.core.storage import save_last_image_async
from app.core.storage import save_last_state_async
from app.core.ws_manager import ws_manager
from app.inference.types import PipelineResult
from app.schemas.robot import PersonMetadata
from app.schemas.robot import RobotMetadata
from app.schemas.scene import DetectionResponse
from app.schemas.scene import SceneState
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()


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
    scene_graph_dict = scene_graph.as_dict() if scene_graph else {}
    logger.debug(f"Current Scene State : {current_scene_state}")
    logger.debug(f"Current scene graph: {scene_graph_dict}")
    logger.info("Uploading data to dashboard...")
    som_pil = Image.fromarray(som_image.astype("uint8"))
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
    }
    await ws_manager.broadcast(payload)
    return payload


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


@router.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    head_yaw: float = Form(0.0),
    head_pitch: float = Form(0.0),
    body_yaw: float | None = Form(None),
    camera_hfov: float | None = Form(None),
    camera_vfov: float | None = Form(None),
    image_width: int | None = Form(None),
    image_height: int | None = Form(None),
    timestamp: float | None = Form(None),
    frame_id: str | None = Form(None),
    scan_id: str | None = Form(None),
    people: str | None = Form(None),
    pipeline=Depends(get_pipeline),
    publish: bool = True,
):
    """
    Run the See-Track-Understand detection loop.

    Steps:
    1. Read uploaded image and parse metadata.
    2. Run detection pipeline.
    3. Serialize detections and optionally publish to dashboard.
    4. Update global ML state and persist if configured.

    Args:
        file: Uploaded image file.
        head_yaw: Head yaw angle.
        head_pitch: Head pitch angle.
        body_yaw: Body yaw angle.
        camera_hfov: Camera horizontal field of view.
        camera_vfov: Camera vertical field of view.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        timestamp: Optional timestamp.
        frame_id: Frame identifier.
        scan_id: Scan identifier.
        people: JSON string with people metadata.
        pipeline: Detection pipeline dependency.
        publish: Whether to send results to dashboard.

    Returns:
        DetectionResponse containing detected objects and image info.
    """

    logger.info(f"Running detection endpoint on file: {file.filename}")

    # Prepare inputs: image + robot metadata
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file") from e

    robot_metadata = parse_robot_metadata(
        head_yaw,
        head_pitch,
        body_yaw,
        camera_hfov,
        camera_vfov,
        image_width,
        image_height,
        timestamp,
        frame_id,
        scan_id,
        people,
    )

    # Inference
    # TODO: will have to play with scheme and type detectionobject, redundant...
    try:
        result = await pipeline.process(image, robot_metadata)
        objects = [
            {
                "label": det.label,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "object_id": det.object_id,
            }
            for det in result.detections
        ]
        logger.info(
            f"Pipeline processing finished. Detected {len(objects)} objs. Detection metrics: {result.metrics}"
        )
        logger.debug(f"Detected objects: {objects}")
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise HTTPException(status_code=500, detail="Detection pipeline failed") from e
    # If publishing results to dashboard, prepare SoM Image for display

    if publish:
        current_scene_state = (
            pipeline.memory.scene_state() if hasattr(pipeline, "memory") else None
        )
        payload = await upload_data_to_dashboard(result, current_scene_state, objects)
        await update_and_persist_appstate(ml_state, payload)

    return DetectionResponse(
        objects=objects,
        timestamp=time.time(),
        image_width=image.width,
        image_height=image.height,
    )
