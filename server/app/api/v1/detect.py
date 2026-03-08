import io
import logging
import time

from app.api.v1.utils import parse_robot_metadata
from app.api.v1.utils import update_and_persist_appstate
from app.api.v1.utils import upload_data_to_dashboard
from app.api.v1.utils import upload_worker_payload
from app.core.state import ml_state
from app.core.worker_errors import WorkerError
from app.schemas.robot import RobotMetadata
from app.schemas.scene import DetectionResponse
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()


async def detect_in_worker(
    img_bytes: bytes, robot_metadata: RobotMetadata, publish: bool
) -> tuple[list[dict], dict]:
    try:
        worker_result = await ml_state.worker_manager.detect(img_bytes, robot_metadata)
        objects = worker_result.get("objects", [])
        payload = {
            "type": "detection",
            "image": worker_result.get("image_b64"),
            "objects": objects,
            "scene_graph": worker_result.get("scene_graph", []),
            "memory": worker_result.get("memory", {}),
            "metrics": worker_result.get("metrics", {}),
            "executed_stages": worker_result.get("executed_stages", []),
        }
        if publish:
            await upload_worker_payload(payload)
            await update_and_persist_appstate(ml_state, payload)
        return objects, payload["metrics"]
    except WorkerError as exc:
        logger.error(f"Worker inference failed: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Worker inference failed unexpectedly: {exc}")
        raise HTTPException(
            status_code=500, detail="Detection pipeline failed"
        ) from exc


async def detect(
    image: Image.Image, robot_metadata: RobotMetadata, publish: bool
) -> tuple[list[dict], dict]:
    pipeline = ml_state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="AI Pipeline is currently warming up. Please wait."
        )
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
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise HTTPException(status_code=500, detail="Detection pipeline failed") from e
    if publish:
        current_scene_state = (
            pipeline.memory.scene_state() if hasattr(pipeline, "memory") else None
        )
        payload = await upload_data_to_dashboard(result, current_scene_state, objects)
        await update_and_persist_appstate(ml_state, payload)
    return objects, result.metrics


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

    use_worker = bool(
        ml_state.config
        and ml_state.config.worker.enabled
        and ml_state.worker_manager is not None
    )
    objects: list[dict] = []
    if use_worker:
        objects, metrics = await detect_in_worker(img_bytes, robot_metadata, publish)
    else:
        objects, metrics = await detect(image, robot_metadata, publish)
    logger.info(
        f"Pipeline processing finished. Detected {len(objects)} objs. Detection metrics: {metrics}"
    )
    logger.debug(f"Detected objects: {objects}")
    response = DetectionResponse(
        objects=objects,
        timestamp=time.time(),
        image_width=image.width,
        image_height=image.height,
    )
    logger.info(f"Detection endpoint finished for detection: {response.id}")
    return response
