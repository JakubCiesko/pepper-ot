import logging

from app.api.v1.utils import get_chat_service
from app.core.ws_manager import ws_manager
from app.schemas.chat import ChatRequest
from app.schemas.chat import ChatResponse
from fastapi import APIRouter
from fastapi import Depends

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_service=Depends(get_chat_service)):
    """
    Handle a user chat request and return Pepper's generated sentence.

    This endpoint:
    1. Accepts a `ChatRequest` containing the user's query text.
    2. Uses the injected chat service to generate a response sentence.
    3. Broadcasts the generated sentence to connected dashboard clients via WebSocket.
    4. Returns a structured `ChatResponse` payload for API consumers.

    Args:
      request: Incoming chat payload with the user query.
      chat_service: Chat backend dependency resolved by `get_chat_service`.

    Returns:
      ChatResponse: Generated sentence plus metadata fields (`source_object_ids`,
      `confidence`).

    Raises:
      HTTPException: If chat service dependency is unavailable (raised by dependency
    layer) or if downstream generation fails.
    """

    logger.info(f"Received chat query: {request.query}")
    # Call the RAG engine
    response_text = await chat_service.chat(request.query)
    logger.info(f"Received chat response: {response_text}")

    await ws_manager.broadcast({"type": "sentence", "text": response_text})

    return ChatResponse(
        sentence=response_text,
        source_object_ids=[],  # Can be filled by the chat service later
        confidence=1.0,
    )
