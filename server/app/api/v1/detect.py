import io
import logging
import time

from app.api.v1.dependencies import get_pipeline
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

    # 3. Handle WebSockets (Background Task)
    # colors = get_color_encoding(result.detections)
    # annotated_image_b64 = annotate_image(img_bytes, result.detections, colors)
    #
    # asyncio.create_task(ws_manager.broadcast({
    #     "type": "detection",
    #     "objects": [d.model_dump() for d in result.detections],
    #     "image": annotated_image_b64,
    #     "colors": colors
    # }))

    objects = [
        {
            "label": det.label,
            "confidence": det.confidence,
            "bbox": det.bbox,
            "object_id": det.object_id,
        }
        for det in result.detections
    ]

    return DetectionResponse(
        objects=objects,
        timestamp=time.time(),
        image_width=image.width,
        image_height=image.height,
    )
