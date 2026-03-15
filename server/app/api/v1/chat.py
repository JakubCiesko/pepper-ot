import logging

from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import app_state
from app.orchestration.conversation_service import ConversationService
from app.schemas.chat import ChatRequest
from app.schemas.chat import ChatResponse
from fastapi import APIRouter
from fastapi import HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


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

    conversations: ConversationService = app_state.conversation_service
    conversation = await conversations.ensure_conversation(request.chat_id)
    chat_id = conversation.chat_id

    user_message = await conversations.add_message(chat_id, "user", request.query)
    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": conversations.serialize_message(user_message),
        }
    )

    history = await conversations.prompt_history(chat_id, include_last_user=False)
    response_text = await app_state.chat_service.chat(
        request.query,
        conversation_history=history,
    )
    assistant_message = await conversations.add_message(
        chat_id, "assistant", response_text
    )

    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": conversations.serialize_message(assistant_message),
        }
    )

    return ChatResponse(
        chat_id=chat_id,
        sentence=response_text,
        source_object_ids=[],
        confidence=1.0,
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
