import logging

import app.api.v1.image_utils as img_utils
from app.core.runtime.state import app_state
from app.orchestration.runtime_adapter import resolve_runtime_adapter
from app.providers.translation import enforce_output_language
from app.schemas.vision_chat import VisionChatFormRequest
from app.schemas.vision_chat import VisionChatResponse
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/vision_chat", response_model=VisionChatResponse)
async def vision_chat_endpoint(
    file: UploadFile = File(...),
    form: VisionChatFormRequest = Depends(VisionChatFormRequest.as_form),
):
    if form.query is None:
        raise HTTPException(status_code=400, detail="query is required")

    query = form.query
    system_prompt = (
        form.system_prompt.strip()
        if form.system_prompt
        else app_state.chat_service.system_prompt
    )
    logger.info(
        "Vision Chat Endpoint Active with query=%s, system_prompt=%s",
        query,
        system_prompt,
    )
    image_bytes = await file.read()

    if form.resize_image:
        image_bytes = img_utils.resize_image_bytes(image_bytes)

    adapter = resolve_runtime_adapter(app_state)
    payload = await adapter.vision_chat(
        image_bytes,
        user_prompt=query,
        system_prompt=system_prompt.strip() if system_prompt else None,
    )

    output_language = (
        form.language
        if form.language is not None
        else (
            app_state.config.system.get("output_language")
            if app_state.config is not None
            and isinstance(app_state.config.system, dict)
            else None
        )
    )
    response_text = await enforce_output_language(
        payload.get("answer", ""), output_language
    )

    logger.info(
        "Vision chat completed answer=%s provider=%s model_id=%s language=%s",
        response_text,
        payload.get("provider"),
        payload.get("model_id"),
        output_language,
    )
    return VisionChatResponse(
        answer=response_text,
        provider=str(payload.get("provider", "")),
        model_id=str(payload.get("model_id", "")),
    )
