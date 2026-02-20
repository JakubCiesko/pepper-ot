from app.core.state import ml_state
from fastapi import HTTPException


def get_pipeline():
    """Safely injects the visual pipeline into endpoints."""
    if ml_state.pipeline is None:
        raise HTTPException(
            status_code=503, detail="AI Pipeline is currently warming up. Please wait."
        )
    return ml_state.pipeline


def get_chat_service():
    """Safely injects the chat/RAG service into endpoints."""
    if ml_state.chat_service is None:
        raise HTTPException(status_code=503, detail="Chat Service is not initialized.")
    return ml_state.chat_service
