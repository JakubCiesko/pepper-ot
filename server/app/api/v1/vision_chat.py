import logging

import app.api.v1.image_utils as img_utils
from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import app_state
from app.orchestration.adapters.runtime import resolve_runtime_adapter
from app.orchestration.services.conversation import ConversationService
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


def _format_history(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    lines = []
    for role, text in history:
        role_name = "User" if role == "user" else "Assistant"
        lines.append(f"{role_name}: {text}")
    return "\n".join(lines)


@router.post("/vision_chat", response_model=VisionChatResponse)
async def vision_chat_endpoint(
    file: UploadFile = File(...),
    form: VisionChatFormRequest = Depends(VisionChatFormRequest.as_form),
):
    if form.query is None:
        raise HTTPException(status_code=400, detail="query is required")
    if app_state.conversation_service is None or app_state.chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation or Chat Service is not initialized.",
        )
    conversations: ConversationService = app_state.conversation_service
    conversation = await conversations.ensure_conversation(
        form.chat_id or form.conversation_id
    )
    chat_id = conversation.chat_id
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

    query_original = form.query
    query_translated, query_languages = await enforce_output_language(
        text=query_original,
        output_language=output_language,
        return_languages=True,
    )
    query_language = query_languages[0]
    user_message = await conversations.add_message(
        chat_id=chat_id,
        role="user",
        text_original=query_original,
        text_model=query_translated,
        language_original=query_language,
        language_model=output_language,
        translation_applied=query_language != output_language,
    )
    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": conversations.serialize_message(user_message),
        }
    )
    history = await conversations.prompt_history_model(chat_id, include_last_user=False)
    history_text = _format_history(history)
    user_prompt = (
        "Conversation so far:\n"
        f"{history_text}\n\n"
        "Current user message:\n"
        f"{query_translated}"
        if history_text
        else query_translated
    )
    system_prompt = (
        form.system_prompt.strip()
        if form.system_prompt
        else app_state.chat_service.system_prompt
    )
    logger.info(
        "Vision Chat Endpoint Active with chat_id=%s original query=%s translated query=%s system_prompt=%s",
        chat_id,
        query_original,
        query_translated,
        system_prompt,
    )

    image_bytes = await file.read()

    image_bytes, (_, _) = (
        (
            img_utils.resize_image_bytes(image_bytes, debug_show=True)
            if form.resize_image
            else image_bytes
        ),
        (None, None),
    )

    adapter = resolve_runtime_adapter(app_state)
    payload = await adapter.vision_chat(
        image_bytes,
        user_prompt=user_prompt,
        system_prompt=system_prompt.strip() if system_prompt else None,
    )

    model_response = payload.get("answer", "")
    response_text, response_languages = await enforce_output_language(
        model_response,
        output_language,
        return_languages=True,
    )
    response_language = response_languages[0]
    assistant_message = await conversations.add_message(
        chat_id=chat_id,
        role="assistant",
        text_original=response_text,
        text_model=model_response,
        language_original=response_language,
        language_model=output_language,
        translation_applied=response_language != output_language,
    )
    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": conversations.serialize_message(assistant_message),
        }
    )

    logger.info(
        "Vision chat completed answer=%s provider=%s model_id=%s language=%s query_language=%s response_language=%s",
        response_text,
        payload.get("provider"),
        payload.get("model_id"),
        output_language,
        query_language,
        response_language,
    )
    return VisionChatResponse(
        chat_id=chat_id,
        answer=response_text,
        provider=str(payload.get("provider", "")),
        model_id=str(payload.get("model_id", "")),
    )
