import logging

from app.core.runtime.state import ml_state
from app.orchestration.detect_service import DetectService
from app.schemas.scene import DetectionResponse
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    publish: bool = Form(True),
):
    logger.info("Detection endpoint called")
    image_bytes = await file.read()
    service = DetectService(ml_state)
    robot_metadata = service.parse_metadata(metadata)
    response = await service.process(image_bytes, robot_metadata, publish)
    logger.info("Detection endpoint completed: %s", response.id)
    return response
