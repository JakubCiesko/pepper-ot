import logging
from typing import Annotated

from app.core.runtime.state import app_state
from app.orchestration.detect_service import DetectService
from app.schemas.detect import DetectFormRequest
from app.schemas.detect import DetectionResponse
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()

default_form = DetectFormRequest()


@router.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    form: Annotated[DetectFormRequest, Form()] = default_form,
):
    """
    Run the perception pipeline on an uploaded image and return detected objects.

    Behavior:
    - Reads the uploaded image bytes.
    - Parses optional robot metadata from JSON form field.
    - Executes detection via DetectService (local pipeline or worker-backed runtime).
    - Optionally publishes resulting state/events to websocket subscribers.

    Args:
        file: Uploaded image file (multipart/form-data).
        metadata: Optional JSON string with robot metadata (pose, camera info, frame IDs).
        publish: If True, allows service-layer broadcasting/persistence side effects.

    Returns:
        DetectionResponse including:
            - request/response id
            - detected objects
            - timestamp
            - image dimensions

    Raises:
        HTTPException: Propagated from DetectService for invalid image/metadata
        or runtime processing errors.
    """

    logger.info(
        "Detection endpoint called with file=%s, metadata=%s, publish=%s",
        file.filename,
        form.metadata,
        form.publish,
    )
    image_bytes = await file.read()
    service = DetectService(app_state)
    robot_metadata = service.parse_metadata(form.metadata)
    logger.info("Running detection with received robot metadata: %s", robot_metadata)
    response = await service.process(image_bytes, robot_metadata, form.publish)
    logger.info("Detection endpoint completed: %s", response.id)
    return response
