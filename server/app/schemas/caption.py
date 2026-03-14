from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


class CaptionResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    caption: str
    provider: str
    model_id: str
    detect_started: bool = False
    detect_request_id: str | None = None
    timestamp: float
