import logging

from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import ml_state
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
    if ml_state.caption_service is None:
        raise HTTPException(
            status_code=503, detail="Caption service is not initialized."
        )

    image_bytes = await file.read()
    detect_service = DetectService(ml_state)
    robot_metadata = detect_service.parse_metadata(metadata)

    result = await ml_state.caption_service.caption_with_optional_detect(
        image_bytes,
        metadata=robot_metadata,
        run_detect=run_detect,
        publish=publish,
        prompt_override=prompt,
    )
    if publish:
        await ws_manager.broadcast({"type": "caption", "text": result["caption"]})
    return CaptionResponse(**result)
