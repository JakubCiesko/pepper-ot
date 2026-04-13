import logging

from app.api.v1.memory_route_utils import run_memory_action
from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import app_state
from app.orchestration.adapters.runtime import resolve_runtime_adapter
from app.orchestration.services.conversation import ConversationService
from app.orchestration.services.memory import MemoryService
from app.providers.translation import enforce_output_language
from app.schemas.chat import ChatMode
from app.schemas.chat import ChatRequest
from app.schemas.chat import ChatResponse
from app.schemas.chat import PregeneratedQAPair
from app.schemas.chat import PregeneratedQAPairs
from app.schemas.chat import PregeneratedQARequest
from app.schemas.chat import PregeneratedQAResponse
from fastapi import APIRouter
from fastapi import Body
from fastapi import HTTPException
import asyncio 


logger = logging.getLogger(__name__)
router = APIRouter()


# used for long conversations for the robot when no chat id given
DEFAULT_CONVERSATION_ID = "-1"


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a user chat turn, maintain conversation state, and return Pepper's reply.

    Behavior:
    - Ensures a conversation exists (creates one when no chat_id is provided).
    - Stores the incoming user message in conversation memory.
    - Broadcasts user message to websocket subscribers.
    - Builds conversational history and calls the chat service for response generation.
    - Stores and broadcasts assistant response.
    - Returns the assistant reply with the active chat_id.

    Args:
        request: ChatRequest containing user query and optional chat_id.

    Returns:
        ChatResponse with:
            - chat_id: active conversation identifier
            - sentence: assistant response text
            - source_object_ids: currently empty placeholder list
            - confidence: currently fixed response confidence value

    Raises:
        HTTPException: 503 when chat or conversation services are not initialized.
    """
    if app_state.chat_service is None or app_state.conversation_service is None:
        raise HTTPException(status_code=503, detail="Chat Service is not initialized.")
    logger.info("Chat endpoint triggered with request: %s", request)
    if request.chat_id is None:
        logger.info(
            "No chat id provided in request: %s. Defaulting chat id to default %s",
            request,
            DEFAULT_CONVERSATION_ID,
        )
        request.chat_id = DEFAULT_CONVERSATION_ID
        request.conversation_id = DEFAULT_CONVERSATION_ID
    conversations: ConversationService = app_state.conversation_service
    conversation = await conversations.ensure_conversation(request.chat_id)
    chat_id = conversation.chat_id

    output_language = (
        request.language
        if request.language is not None
        else (
            app_state.config.system.get("output_language")
            if app_state.config is not None
            and isinstance(app_state.config.system, dict)
            else None
        )
    )

    query_original = request.query
    query_translated, query_languages = await enforce_output_language(
        text=query_original,
        output_language=output_language,
        return_languages=True,
    )
    query_language = query_languages[0]
    user_message_dict = {
        "chat_id": chat_id,
        "role": "user",
        "text_original": query_original,
        "text_model": query_translated,
        "language_original": query_language,
        "language_model": output_language,
        "translation_applied": query_language != output_language,
    }
    logger.info(
        "Adding user message to conversation %s, message: %s",
        chat_id,
        user_message_dict,
    )
    user_message = await conversations.add_message(**user_message_dict)
    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": conversations.serialize_message(user_message),
        }
    )

    history = await conversations.prompt_history_model(chat_id, include_last_user=False)

    mode = request.mode or ChatMode.GENERAL
    source_object_ids: list[int] = []
    crop_fallback_used_ids: list[int] = []
    resolved_object_label: str | None = None

    async def _caption_crop(crop_bytes: bytes) -> str:
        if app_state.caption_service is None:
            return ""
        return await app_state.caption_service.caption(
            crop_bytes,
            prompt_override="Describe only this object in one short sentence.",
            language=output_language,
        )

    model_response = ""
    match mode:
        case ChatMode.OBJECT:
            (
                model_response,
                source_object_ids,
                crop_fallback_used_ids,
                resolved_object_label,
            ) = await app_state.chat_service.object_chat(
                query_translated,
                object_label=request.object_label or "",
                conversation_history=history,
                max_instances=request.max_instances,
                max_crop_fallbacks=request.max_crop_fallbacks,
                caption_crop_callback=(
                    _caption_crop if app_state.caption_service is not None else None
                ),
            )
        case ChatMode.GENERAL:
            model_response = await app_state.chat_service.chat(
                query_translated,
                conversation_history=history,
            )
        case _:
            mode = ChatMode.GENERAL
            model_response = await app_state.chat_service.chat(
                query_translated,
                conversation_history=history,
            )

    # this preferably does no translation and model answers in the output language...
    model_response_translated, model_response_languages = await enforce_output_language(
        text=model_response,
        output_language=output_language,
        return_languages=True,
    )
    model_response_language = model_response_languages[0]
    model_message_dict = {
        "chat_id": chat_id,
        "role": "assistant",
        "text_original": model_response_translated,
        "text_model": model_response,
        "language_original": model_response_language,
        "language_model": output_language,
        "translation_applied": model_response_language != output_language,
    }
    logger.info("Received LLM response: %s", model_response)
    logger.info(
        "Adding assistant message to conversation %s, message: %s",
        chat_id,
        model_message_dict,
    )
    assistant_message = await conversations.add_message(**model_message_dict)
    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": conversations.serialize_message(assistant_message),
        }
    )
    metadata: dict[str, object] = {
        "model_id": app_state.config.chat.model_id,
        "provider": app_state.config.chat.provider,
        "mode": str(mode),
        "input_language": query_language or "",
        "output_language": output_language or "",
        "conversation_in_language": output_language,
        "input_translation_applied": str(query_language != output_language),
        "output_translation_applied": str(model_response_language != output_language),
        "object_label_requested": request.object_label,
        "object_label_resolved": resolved_object_label,
        "matched_object_ids": source_object_ids,
        "matched_object_count": len(source_object_ids),
        "crop_fallback_used_ids": crop_fallback_used_ids,
    }
    logger.info(
        "CHAT: %s RESPONSE: %s METADATA: %s",
        chat_id,
        model_response_translated,
        metadata,
    )
    return ChatResponse(
        chat_id=chat_id,
        sentence=model_response_translated,
        source_object_ids=source_object_ids,
        confidence=1.0,
        metadata=metadata,
    )


@router.get("/chat/conversations")
async def list_conversations(limit: int = 20):
    """
    List recent conversation sessions with lightweight metadata.

    Args:
        limit: Maximum number of conversation summaries to return.

    Returns:
        dict: {"items": [...]} where each item contains conversation metadata
        such as chat_id, timestamps, message count, and last message summary.

    Raises:
        HTTPException: 503 when conversation service is not initialized.
    """

    if app_state.conversation_service is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation service is not initialized.",
        )
    conversations: ConversationService = app_state.conversation_service
    return {"items": await conversations.list_conversations(limit=limit)}


@router.get("/chat/conversations/{chat_id}")
async def get_conversation(chat_id: str):
    """
    Retrieve a full serialized conversation by chat_id.

    Args:
        chat_id: Conversation identifier.

    Returns:
        dict: Serialized conversation payload including all stored messages.

    Raises:
        HTTPException:
            - 503 when conversation service is not initialized.
            - 404 when the conversation does not exist.
    """

    if app_state.conversation_service is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation service is not initialized.",
        )
    conversations: ConversationService = app_state.conversation_service
    conversation = await conversations.get_conversation(chat_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversations.serialize_conversation(conversation)


@router.post("/chat/conversations/{chat_id}/reset")
async def reset_conversation(chat_id: str):
    """
    Clear all messages in an existing conversation while preserving its chat_id.

    Args:
        chat_id: Conversation identifier to reset.

    Returns:
        dict: {"ok": True} on successful reset.

    Raises:
        HTTPException:
            - 503 when conversation service is not initialized.
            - 404 when the conversation does not exist.
    """

    if app_state.conversation_service is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation service is not initialized.",
        )
    conversations: ConversationService = app_state.conversation_service
    ok = await conversations.reset_conversation(chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@router.delete("/chat/conversations/{chat_id}")
async def delete_conversation(chat_id: str):
    """
    Delete an existing conversation session permanently.

    Args:
        chat_id: Conversation identifier to delete.

    Returns:
        dict: {"ok": True} when deletion succeeds.

    Raises:
        HTTPException:
            - 503 when conversation service is not initialized.
            - 404 when the conversation does not exist.
    """

    if app_state.conversation_service is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation service is not initialized.",
        )
    conversations: ConversationService = app_state.conversation_service
    ok = await conversations.delete_conversation(chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@router.post("/chat/pregenerate_qa", response_model=PregeneratedQAResponse)
async def get_memory_pregenerated_qa(
    request: PregeneratedQARequest | None = Body(default=None),
):
    request = request or PregeneratedQARequest()
    logger.info(
        "PregenerateQA for Current Memory state requested with: %s",
        request,
    )

    if app_state.chat_service is None:
        raise HTTPException(status_code=503, detail="Chat Service is not initialized.")

    def _service() -> MemoryService:
        return MemoryService(resolve_runtime_adapter(app_state))

    memory_service = _service()
    total_memory_description = await run_memory_action(
        lambda: memory_service.build_text_description()
    )
    logger.info(
        "PregenerateQA: Total Memory Description requested: %s",
        total_memory_description,
    )

    output_language = (
        request.output_language
        if request.output_language is not None
        else (
            request.language
            if request.language is not None
            else (
                app_state.config.system.get("output_language")
                if app_state.config is not None
                and isinstance(app_state.config.system, dict)
                else None
            )
        )
    )

    number_of_pairs = request.requested_number_of_pairs or 3
    user_prompt = (
        f"Generate exactly {number_of_pairs} concise question-answer pairs about the current scene.\n"
        "Return only structured data matching the schema.\n"
        f"Write every question and every answer in {output_language}.\n"
        "Use only facts supported by the provided scene description.\n"
        "Keep each answer short and concrete.\n\n"
        "Scene description:\n"
        f"{total_memory_description}"
    )
    structured = await app_state.chat_service.chat_structured(
        user_prompt,
        output_schema=PregeneratedQAPairs,
        conversation_history=None,
        user_prompt_override=user_prompt,
    )
    async def translate_pair(pair: PregeneratedQAPair) -> PregeneratedQAPair:
        question, answer = pair.question, pair.answer
        task1 = enforce_output_language(question, output_language or "english")
        task2 = enforce_output_language(answer, output_language or "english")
        translations = await asyncio.gather(task1, task2)
        question = translations[0] if isinstance(translations[0], str) else translations[0][0]
        answer = translations[1] if isinstance(translations[1], str) else translations[1][0]
        return PregeneratedQAPair(question=question, answer=answer)

    pregenerated_pairs = [
        PregeneratedQAPair(
            question=item.question.strip(),
            answer=item.answer.strip(),
        )
        for item in structured.items
        if item.question.strip() and item.answer.strip()
    ]
    logger.info("PregeneratedQA Pre-Translation: %s", pregenerated_pairs)
    if output_language is not None and output_language != "default":
        pregenerated_pairs = await asyncio.gather(*[
            translate_pair(pair) for pair in pregenerated_pairs
        ])
        logger.info("PregeneratedQA Final Output After Translation %s", pregenerated_pairs)

    metadata: dict[str, object] = {
        "model_id": app_state.config.chat.model_id,
        "provider": app_state.config.chat.provider,
        "output_language": output_language or "",
        "requested_pair_count": number_of_pairs,
        "generated_pair_count": len(pregenerated_pairs),
    }
    logger.info(
        "PregenerateQA: GENERATED_PAIRS=%s METADATA=%s",
        pregenerated_pairs,
        metadata,
    )
    return PregeneratedQAResponse(
        pregenerated_qa=pregenerated_pairs,
        metadata=metadata,
    )
