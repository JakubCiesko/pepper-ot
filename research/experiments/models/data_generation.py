from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from .common import DataPathConfig
from .common import OntologyConfig


class TeacherModelConfig(BaseModel):
    ontology_config: OntologyConfig
    box_threshold: float = Field(
        0.35, ge=0.0, le=1.0, description="Confidence to accept a box"
    )
    # text_threshold: float = Field(0.25, ge=0.0, le=1.0)
    device: str = "cuda"


class LLMLabelerConfig(BaseModel):
    """Configuration for the VLM Scene Graph ground truth Generator (ex. GPT-4o)."""

    # TODO: Right now, vendor-locked for OpenAI, might change for future
    model_id: str = "gpt-4o"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(512, gt=0)

    system_prompt: str = (
        "You are a robotic scene graph generator. "
        "Analyze the provided Set-of-Mark (SoM) image where objects are marked with numerical IDs. "
        "Output a JSON list of spatial or other relationships "
        # "using ONLY the allowed predicates. " # Provide predicates into prompt if ClosedVocab, else dont mention
        "Unary predicates are represented by the same subject and object "
        "Format: [{'sub': 'ID', 'rel': 'PREDICATE', 'obj': 'ID'}]. "
        "Example: [{'sub': '1', 'rel': 'holding', 'obj': '2'}, {'sub': '1', 'rel': 'red', 'obj': '1'}]."
    )


class DataGenConfig(BaseModel):
    """Settings for the Synthetic Data Generation."""

    teacher: TeacherModelConfig | None = None
    llm_labeler: LLMLabelerConfig | None = None
    paths: DataPathConfig
    target_resolution: tuple[int, int] | None = (
        640,
        480,
    )  # if None, no resizing happening
    generate_masks: bool = False

    @model_validator(mode="after")
    def check_at_least_one_source_config(self):
        if self.teacher or self.llm_labeler:
            # ! can be both
            return self
        raise ValueError(
            "Specify at least one data source: Teacher model or LLM Labeler"
        )
