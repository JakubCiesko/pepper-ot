from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
import yaml


class DetectionConfig(BaseModel):
    backend: Literal["yolo", "rt_detr", "rf_detr"]
    weights_path: str | None = None
    confidence_threshold: float = 0.5


class AssociationConfig(BaseModel):
    visual_weight: float = 0.8
    geometry_weight: float = 0.2
    match_threshold: float = 0.4


class TrackingConfig(BaseModel):
    reid_model: str
    max_dormant_frames: int = 30
    association: AssociationConfig


class OntologyConfig(BaseModel):
    predicates: list[str]


class UnderstandingConfig(BaseModel):
    backend: Literal["openai", "local", "local_4bit"]
    model_id: str
    inference: dict = Field(default_factory=dict)
    ontology: OntologyConfig


class VisConfig(BaseModel):
    show_bbox: bool = True
    show_mask: bool = False
    show_labels: bool = True
    line_thickness: int = 2


class AppConfig(BaseModel):
    system: dict
    detection: DetectionConfig
    tracking: TrackingConfig
    understanding: UnderstandingConfig
    visualization: VisConfig

    @classmethod
    def load(cls, path: str = "config.yaml") -> "AppConfig":
        if not Path(path).exists():
            # Fallback to looking one level up if running from inside app/
            path = "../config.yaml"

        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
