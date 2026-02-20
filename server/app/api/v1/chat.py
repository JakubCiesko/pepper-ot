import logging

from app.api.v1.dependencies import get_chat_service
from app.schemas.chat import ChatRequest
from app.schemas.chat import ChatResponse
from fastapi import APIRouter
from fastapi import Depends

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_service=Depends(get_chat_service)):
    """API endpoint for Pepper's dialogue."""
    logger.info(f"Received chat query: {request.query}")
    # Call the RAG engine
    response_text = await chat_service.chat(request.query)
    logger.info(f"Received chat response: {response_text}")

    return ChatResponse(
        sentence=response_text,
        source_object_ids=[],  # Can be filled by the chat service later
        confidence=1.0,
    )
