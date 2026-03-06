from pathlib import Path
from typing import Any
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
    device: None | str = None
    ontology: list[str] | None = None
    ontology_path: Path | None = None

    def resolve_ontology(self, base_dir: Path) -> list[str] | None:
        if self.ontology is not None:
            return self.ontology
        if self.ontology_path is None:
            return None
        raw = yaml.safe_load(
            (base_dir / self.ontology_path).read_text(encoding="utf-8")
        )
        if raw is None:
            return None
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, dict):
            values = raw.get("objects")
            if isinstance(values, list):
                return [str(item).strip() for item in values if str(item).strip()]
        return None


class StructuredOutputConfig(BaseModel):
    mode: Literal["provider_native", "parse_output"] = "parse_output"
    strict: bool = True


class LLMConfig(BaseModel):
    provider: Literal[
        "openai",
        "gemini",
        "openai_compatible",
        "local_hf",
        "local_4bit",
    ] = "openai"
    model_id: str
    device: str | None = None

    # Connection/credentials
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float | None = None

    # Provider-specific passthrough knobs
    client_init_kwargs: dict[str, Any] = Field(default_factory=dict)
    call_kwargs: dict[str, Any] = Field(default_factory=dict)
    structured_output: StructuredOutputConfig = Field(
        default_factory=StructuredOutputConfig
    )

    # Legacy fields kept for migration compatibility.
    backend: str | None = Field(default=None, exclude=True)
    inference: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # TODO: this will be removed
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)

        backend = data.get("backend")
        if "provider" not in data and backend:
            backend_map = {
                "openai": "openai",
                "gemini": "gemini",
                "local": "local_hf",
                "local_hf": "local_hf",
                "local_4bit": "local_4bit",
            }
            data["provider"] = backend_map.get(backend, "openai")

        inference = data.get("inference")
        if isinstance(inference, dict):
            call_kwargs = dict(data.get("call_kwargs") or {})
            client_init_kwargs = dict(data.get("client_init_kwargs") or {})

            backend_kwargs = inference.get("backend_kwargs")
            if isinstance(backend_kwargs, dict):
                if isinstance(backend_kwargs.get("client_init_kwargs"), dict):
                    client_init_kwargs.update(backend_kwargs["client_init_kwargs"])
                if isinstance(backend_kwargs.get("call_kwargs"), dict):
                    call_kwargs.update(backend_kwargs["call_kwargs"])
                else:
                    passthrough = {
                        k: v
                        for k, v in backend_kwargs.items()
                        if k not in {"client_init_kwargs", "call_kwargs"}
                    }
                    call_kwargs.update(passthrough)

            call_kwargs.update(
                {k: v for k, v in inference.items() if k != "backend_kwargs"}
            )
            data["client_init_kwargs"] = client_init_kwargs
            data["call_kwargs"] = call_kwargs

        structured = data.get("structured_output")
        if isinstance(structured, dict):
            mode = structured.get("mode")
            if mode is None:
                strategy = structured.get("strategy")
                if strategy in {"auto", "native"}:
                    structured["mode"] = "provider_native"
                elif strategy in {"json_mode", "prompt_only"}:
                    structured["mode"] = "parse_output"
            if isinstance(structured.get("enabled"), bool) and not structured.get(
                "enabled"
            ):
                structured["mode"] = "parse_output"
            structured.pop("strategy", None)
            structured.pop("enabled", None)
            data["structured_output"] = structured

        return data


class AssociationConfig(BaseModel):
    visual_weight: float = 0.8
    geometry_weight: float = 0.2
    match_threshold: float = 0.4


class FeatureExtractionConfig(BaseModel):
    reid_model: str | None = None
    device: str | None = None
    target_size: tuple[int, int] | None = None
    resampling_method: str | None = None


class TrackingConfig(BaseModel):
    feature_extraction: FeatureExtractionConfig = FeatureExtractionConfig()
    max_dormant_frames: int = 30
    association: AssociationConfig = AssociationConfig()
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


# TODO: use this in object detection
class OntologySource(BaseModel):
    predicates: list[str] | None = None
    objects: list[str] | None = None
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


class SceneGraphVLMConfig(LLMConfig):
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


class SceneGraphConfig(BaseModel):
    mode: Literal["vlm", "rules", "hybrid"] = "hybrid"
    vlm: SceneGraphVLMConfig
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
    scene_graph: SceneGraphConfig
    chat: ChatConfig
    visualization: VisConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
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
