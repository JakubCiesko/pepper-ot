from typing import Literal
from uuid import UUID
from uuid import uuid4

from fastapi import Form
from pydantic import BaseModel
from pydantic import Field


class VisionChatResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    answer: str
    provider: str
    model_id: str


class VisionChatFormRequest(BaseModel):
    query: str = Field(
        description="prompt for caption generation",
    )
    system_prompt: str | None = None

    language: Literal["default", "english", "czech"] | None = Field(
        default=None,
        description="Optional per-request output language override",
    )

    resize_image: bool = Field(
        default=True,
        description="Resize image before running language model to mitigate GPU load",
    )

    @classmethod
    def as_form(
        cls,
        query: str = Form(...),
        system_prompt: str | None = Form(None),
        language: Literal["default", "english", "czech"] | None = Form(None),
        resize_image: bool = Form(True),
    ) -> "VisionChatFormRequest":
        return cls(
            query=query,
            system_prompt=system_prompt,
            language=language,
            resize_image=resize_image,
        )
