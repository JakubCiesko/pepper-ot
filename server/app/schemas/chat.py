from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class ChatRequest(BaseModel):
    query: str
    chat_id: str | None = None
    conversation_id: str | None = None
    language: str | None = None

    @model_validator(mode="after")
    def normalize_chat_id(self):
        if self.chat_id is None and self.conversation_id is not None:
            self.chat_id = self.conversation_id
        return self


class ChatResponse(BaseModel):
    chat_id: str
    sentence: str
    source_object_ids: list[int] = Field(default_factory=list)
    confidence: float
    metadata: dict[str, str]
