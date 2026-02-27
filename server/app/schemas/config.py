from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import model_validator
import yaml


class DetectionConfig(BaseModel):
    backend: Literal["yolo", "rt_detr", "rf_detr", "owl_v2"]
    weights_path: str | None = None
    confidence_threshold: float = 0.5


class LLMConfig(BaseModel):
    backend: Literal["openai", "local", "local_4bit"]
    model_id: str
    inference: dict = Field(default_factory=dict)


class AssociationConfig(BaseModel):
    visual_weight: float = 0.8
    geometry_weight: float = 0.2
    match_threshold: float = 0.4


class TrackingConfig(BaseModel):
    reid_model: str
    max_dormant_frames: int = 30
    association: AssociationConfig
    memory_max_age_seconds: int = 60
    memory_max_objects: int = 200
    memory_max_relations: int = 500


class PromptSource(BaseModel):
    text: str | None = None
    path: Path | None = None

    @model_validator(mode="after")
    def ensure_one_of(self):
        if (self.text is None and self.path is None) or (
            self.text is not None and self.path is not None
        ):
            raise ValueError("PromptSource requires exactly one of 'text' or 'path'.")
        return self

    def resolve(self, base_dir: Path) -> str:
        if self.path is not None:
            data = (base_dir / self.path).read_text(encoding="utf-8")
            return data.strip()
        return (self.text or "").strip()


class OntologySource(BaseModel):
    predicates: list[str] | None = None
    objects: dict[str, str] | None = None
    path: Path | None = None

    def resolve(self, base_dir: Path) -> tuple[list[str] | None, dict[str, str] | None]:
        predicates = self.predicates
        objects = self.objects

        if self.path is not None:
            raw = (
                yaml.safe_load((base_dir / self.path).read_text(encoding="utf-8")) or {}
            )
            file_predicates = raw.get("predicates")
            file_objects = raw.get("objects")

            if predicates is None:
                predicates = file_predicates
            if objects is None:
                objects = file_objects

        return predicates, objects


class UnderstandingConfig(LLMConfig):
    system_prompt: PromptSource
    user_prompt: PromptSource | None = None
    ontology: OntologySource


class ChatConfig(LLMConfig):
    system_prompt: PromptSource
    context_template: PromptSource | None = None


class VisConfig(BaseModel):
    show_bbox: bool = True
    show_mask: bool = False
    show_labels: bool = True
    show_polygon: bool = False
    line_thickness: int = 2
    mask_opacity: float = 0.5
    color_lookup: Literal["index", "class", "track"] = "index"


class StorageConfig(BaseModel):
    persist_last_state: bool = False
    last_state_path: Path = Path("state/last_state.json")
    store_image: bool = True


class SGGRuleConstraints(BaseModel):
    subject_labels: list[str] | None = None
    object_labels: list[str] | None = None
    labels_any: list[str] | None = None


class SGGRule(BaseModel):
    predicate: str
    type: str
    thresholds: dict = Field(default_factory=dict)
    constraints: SGGRuleConstraints | None = None


class SGGRulesConfig(BaseModel):
    enabled: bool = True
    rule_list: list[SGGRule] = Field(default_factory=list)


class SGGConfig(BaseModel):
    mode: Literal["vlm", "rules", "hybrid"] = "hybrid"
    rules: SGGRulesConfig = Field(default_factory=SGGRulesConfig)


class FusionConfig(BaseModel):
    person_bbox_match_threshold_px: float = 10.0
    estimated_person_bbox_base_px: float = 80.0
    estimated_person_bbox_min_px: float = 40.0
    estimated_person_bbox_max_px: float = 200.0


class AppConfig(BaseModel):
    system: dict
    detection: DetectionConfig
    tracking: TrackingConfig
    understanding: UnderstandingConfig
    chat: ChatConfig
    visualization: VisConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sgg: SGGConfig = Field(default_factory=SGGConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    _config_path: Path | None = PrivateAttr(None)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "AppConfig":
        if not Path(path).exists():
            # Fallback to looking one level up if running from inside app/
            path = "../config.yaml"

        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls(**raw)
        cfg._config_path = Path(path)
        return cfg
