from typing import Literal
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


class CaptionFormRequest(BaseModel):
    metadata: str | None = Field(
        default=None,
        description="JSON string with RobotMetadata payload",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional prompt override for caption generation",
    )
    run_detect: bool = Field(
        default=True,
        description="Run full detect pipeline in background",
    )
    publish: bool = Field(
        default=True,
        description="Broadcast caption event to dashboard/ws",
    )
    language: Literal["default", "english", "czech"] | None = Field(
        default=None,
        description="Optional per-request output language override",
    )
    resize_image: bool = Field(
        default=True,
        description="Resize image before running detection to mitigate GPU load",
    )
