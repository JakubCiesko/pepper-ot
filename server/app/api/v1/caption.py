import logging

import app.api.v1.image_utils as img_utils
from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import app_state
from app.orchestration.services.detection import DetectService
from app.schemas.caption import CaptionFormRequest
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
    language: str | None = Form(None),
    resize_image: bool = Form(True),
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
    form = CaptionFormRequest(
        metadata=metadata,
        prompt=prompt,
        run_detect=run_detect,
        publish=publish,
        language=language,
        resize_image=resize_image,
    )
    logger.info(
        "Caption endpoint triggered with args filename=%s, "
        "metadata=%s, prompt=%s, run_detect=%s, publish=%s",
        file.filename,
        form.metadata,
        form.prompt,
        form.run_detect,
        form.publish,
    )
    image_bytes = await file.read()
    image_bytes, (w, h) = (
        (
            img_utils.resize_image_bytes(image_bytes, debug_show=True)
            if form.resize_image
            else image_bytes
        ),
        (None, None),
    )
    robot_metadata = DetectService.parse_metadata(form.metadata)

    if w is not None and h is not None:
        robot_metadata.image_width, robot_metadata.image_height = w, h

    result = await app_state.caption_service.caption_with_optional_detect(
        image_bytes,
        metadata=robot_metadata,
        run_detect=form.run_detect,
        publish=form.publish,
        prompt_override=form.prompt,
        language=form.language,
    )
    logger.info("Returning caption: %s", result)
    if form.publish:
        await ws_manager.broadcast({"type": "caption", "text": result["caption"]})
    return CaptionResponse(**result)
