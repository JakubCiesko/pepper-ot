import logging

from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import app_state
from app.orchestration.detect_service import DetectService
from app.schemas.caption import CaptionResponse
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/caption", response_model=CaptionResponse)
async def caption_endpoint(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    prompt: str | None = Form(None),
    run_detect: bool = Form(True),
    publish: bool = Form(True),
):
    """
    Generate a caption for an uploaded image and optionally trigger background detection.

    This endpoint is optimized for fast "what do you see?" responses while preserving
    grounding quality through optional asynchronous detect/memory updates.

    Args:
        file: Uploaded image file (multipart/form-data).
        metadata: Optional JSON string with robot metadata (head pose, frame info, etc.).
            Parsed using DetectService metadata parser for consistency with detect routes.
        prompt: Optional per-request caption prompt override.
        run_detect: If True, starts full detect pipeline in background so memory/chat
            context can be refreshed for follow-up turns.
        publish: If True, broadcasts caption event to dashboard websocket clients.

    Returns:
        CaptionResponse: Caption text plus provider/model metadata and detect trigger flags.

    Raises:
        HTTPException: 503 if caption service is unavailable in runtime state.
    """
    if app_state.caption_service is None:
        raise HTTPException(
            status_code=503, detail="Caption service is not initialized."
        )

    image_bytes = await file.read()
    detect_service = DetectService(app_state)
    robot_metadata = detect_service.parse_metadata(metadata)

    result = await app_state.caption_service.caption_with_optional_detect(
        image_bytes,
        metadata=robot_metadata,
        run_detect=run_detect,
        publish=publish,
        prompt_override=prompt,
    )
    logger.info(f"Returning caption: {result['caption']}")
    if publish:
        await ws_manager.broadcast({"type": "caption", "text": result["caption"]})
    return CaptionResponse(**result)
