from pydantic import BaseModel
from pydantic import Field


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    use_rag: bool = True


class ChatResponse(BaseModel):
    sentence: str
    source_object_ids: list[int] = Field(default_factory=list)
    confidence: float
