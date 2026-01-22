from pydantic import BaseModel
from pydantic import Field

from .common import DataPathConfig
from .common import OntologyConfig


class TeacherModelConfig(BaseModel):
    ontology_config: OntologyConfig
    box_threshold: float = Field(
        0.35, ge=0.0, le=1.0, description="Confidence to accept a box"
    )
    # text_threshold: float = Field(0.25, ge=0.0, le=1.0)
    device: str = "cuda"


class DataGenConfig(BaseModel):
    """Settings for the Synthetic Data Generation."""

    teacher: TeacherModelConfig
    paths: DataPathConfig
    target_resolution: tuple[int, int] | None = (
        640,
        480,
    )  # if None, no resizing happening
    generate_masks: bool = False


# TODO: rename this file from distillation to something more like data.py and add also the GPT ground truth for Scene graph generation. Basically DataPathConfig and thats all, really, so maybe no config needed.
