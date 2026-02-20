import base64
import io
import logging
import time

from app.api.v1.dependencies import get_pipeline
from app.core.ws_manager import ws_manager
from app.schemas.robot import RobotMetadata
from app.schemas.scene import DetectionObject
from app.schemas.scene import DetectionResponse
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()


async def upload_data_to_dashboard(som_image, scene_state, objects, scene_graph):
    logger.info("Uploading data to dashboard...")
    som_pil = Image.fromarray(som_image.astype("uint8"))
    buf = io.BytesIO()
    som_pil.save(buf, format="JPEG")
    som_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    await ws_manager.broadcast(
        {
            "type": "detection",
            "image": som_b64,
            "objects": objects,
            "scene_graph": scene_graph.as_dict(),
            "memory": (
                scene_state.model_dump()
                if scene_state
                else {"objects": [], "relationships": [], "timestamp": time.time()}
            ),
        }
    )


@router.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    head_yaw: float = Form(0.0),
    head_pitch: float = Form(0.0),
    pipeline=Depends(get_pipeline),
    publish: bool = True,
):
    """API endpoint which runs the See-Track-Understand loop."""
    logger.info("Running detection endpoint...")

    # 1. Prepare Inputs
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    robot_metadata = RobotMetadata(
        head_yaw=head_yaw,
        head_pitch=head_pitch,
    )

    # 2. Run Engine (No globals!)
    # TODO: will have to play with scheme and type detectionobject, redundant...
    result = await pipeline.process(image, robot_metadata)

    objects = [
        DetectionObject(
            label=det.label,
            confidence=det.confidence,
            bbox=det.bbox,
            object_id=det.object_id,
        )
        for det in result.detections
    ]
    logger.info(f"Detected ({len(objects)}) objects : {objects}")
    # 3. Prepare SoM image for dashboard if publishable
    if publish:
        scene_state = (
            pipeline.memory.scene_state() if hasattr(pipeline, "memory") else None
        )
        scene_graph = result.scene_graph
        logger.info(f"Current Scene State : {scene_state}")
        logger.info(f"Current scene graph: {scene_state.model_dump()}")
        await upload_data_to_dashboard(
            result.som_image, scene_state, objects, scene_graph
        )

    return DetectionResponse(
        objects=objects,
        timestamp=time.time(),
        image_width=image.width,
        image_height=image.height,
    )
