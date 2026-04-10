from typing import Literal
from uuid import UUID
from uuid import uuid4

from fastapi import Form
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class VisionChatResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: str
    answer: str
    provider: str
    model_id: str


class VisionChatFormRequest(BaseModel):
    query: str = Field(
        description="prompt for caption generation",
    )
    chat_id: str | None = None
    conversation_id: str | None = None
    system_prompt: str | None = None

    language: Literal["default", "english", "czech"] | None = Field(
        default=None,
        description="Optional per-request output language override",
    )

    resize_image: bool = Field(
        default=True,
        description="Resize image before running language model to mitigate GPU load",
    )

    @model_validator(mode="after")
    def normalize_chat_id(self):
        if self.chat_id is None and self.conversation_id is not None:
            self.chat_id = self.conversation_id
        return self

    @classmethod
    def as_form(
        cls,
        query: str = Form(...),
        chat_id: str | None = Form(None),
        conversation_id: str | None = Form(None),
        system_prompt: str | None = Form(None),
        language: Literal["default", "english", "czech"] | None = Form(None),
        resize_image: bool = Form(True),
    ) -> "VisionChatFormRequest":
        return cls(
            query=query,
            chat_id=chat_id,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            language=language,
            resize_image=resize_image,
        )
