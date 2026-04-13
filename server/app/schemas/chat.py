from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class ChatMode(StrEnum):
    GENERAL = "general"
    OBJECT = "object"


class ChatRequest(BaseModel):
    query: str
    chat_id: str | None | int = None
    conversation_id: str | None | int = None
    language: str | None = None
    input_language: str | None = None
    output_language: str | None = None
    mode: ChatMode | None = None
    object_label: str | None = None
    max_instances: int | None = Field(default=None, ge=1)
    max_crop_fallbacks: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def normalize_chat_id(self):
        if self.chat_id is None and self.conversation_id is not None:
            self.chat_id = self.conversation_id
        if self.chat_id is not None:
            self.chat_id = str(self.chat_id)
        if self.conversation_id is not None:
            self.conversation_id = str(self.conversation_id)
        return self

    @model_validator(mode="after")
    def normalize_languages(self):
        if self.language is not None:
            if self.input_language is None:
                self.input_language = self.language
            if self.output_language is None:
                self.output_language = self.language
        return self

    @model_validator(mode="after")
    def normalize_mode(self):
        if self.object_label is not None:
            object_label = self.object_label.strip()
            self.object_label = object_label or None
        if self.mode is None and self.object_label is not None:
            self.mode = ChatMode.OBJECT
        if self.mode == ChatMode.OBJECT and not self.object_label:
            raise ValueError("object_label is required when mode=object")
        return self


class ChatResponse(BaseModel):
    chat_id: str
    sentence: str
    source_object_ids: list[int] = Field(default_factory=list)
    confidence: float
    metadata: dict[str, Any]


class PregeneratedQARequest(BaseModel):
    language: str | None = None
    input_language: str | None = None
    output_language: str | None = None
    requested_number_of_pairs: int | None = None


class PregeneratedQAPair(BaseModel):
    question: str
    answer: str


class PregeneratedQAPairs(BaseModel):
    items: list[PregeneratedQAPair] = Field(default_factory=list)


class PregeneratedQAResponse(BaseModel):
    pregenerated_qa: list[PregeneratedQAPair] = Field(default_factory=list)
    metadata: dict[str, Any]
