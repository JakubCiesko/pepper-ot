from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
import torch
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


DEFAULT_MODEL_NAME = "rtdetr-x.pt"
DEFAULT_LANGUAGE = "en"
DETECTION_CONFIG = {
    "confidence_threshold": None,
    "model": DEFAULT_MODEL_NAME,
    "language": DEFAULT_LANGUAGE,
}


class YOLOSettings(BaseSettings):
    """Settable by enviroment variables PEPPER_{var}"""

    model_config = SettingsConfigDict(env_prefix="PEPPER_")
    model_name: str = Field(DEFAULT_MODEL_NAME, description="Model name")
    model_url: str = Field(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
        description="Download URL",
    )
    fuse_model: bool = Field(
        True, description="boolean flag whether to fuse model or not"
    )
    device: str | None = Field(
        None,
        description="device for loading and using model, default to cuda if not set and available",
    )
    # will not let user set this, as this should be set by the one who sets up the server
    imgsz: int = Field(640, description="image size")
    language: str = Field(DEFAULT_LANGUAGE, description="language of labels")

    @property
    def device_actual(self):
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_path(self) -> Path:
        # server_code_path / detection_models / model_name
        server_code_path = Path(__file__).parent.parent.parent
        return server_code_path / "detection_models" / self.model_name
