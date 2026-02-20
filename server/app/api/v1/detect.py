import base64
import io
import logging
import time

from app.api.v1.dependencies import get_pipeline
from app.core.ws_manager import ws_manager
from app.schemas.scene import DetectionResponse
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from PIL import Image

# We assume you moved annotate_image to a visualization utility file
# from app.engine.visualization import annotate_image, get_color_encoding

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    head_yaw: float = Form(0.0),
    head_pitch: float = Form(0.0),
    pipeline=Depends(get_pipeline),
):
    """API endpoint which runs the See-Track-Understand loop."""
    logger.info("Running detection endpoint...")

    # 1. Prepare Inputs
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    _ = (head_yaw, head_pitch)

    # 2. Run Engine (No globals!)
    result = await pipeline.process(image)

    # 3. Prepare SoM image for dashboard
    som_pil = Image.fromarray(result.som_image.astype("uint8"))
    buf = io.BytesIO()
    som_pil.save(buf, format="JPEG")
    som_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    objects = [
        {
            "label": det.label,
            "confidence": det.confidence,
            "bbox": det.bbox,
            "object_id": det.object_id,
        }
        for det in result.detections
    ]

    await ws_manager.broadcast(
        {
            "type": "detection",
            "image": som_b64,
            "objects": objects,
            "scene_graph": result.scene_graph.as_dict(),
            "memory": pipeline.memory.snapshot() if hasattr(pipeline, "memory") else [],
        }
    )

    return DetectionResponse(
        objects=objects,
        timestamp=time.time(),
        image_width=image.width,
        image_height=image.height,
    )
