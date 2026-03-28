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
    mode: Literal["provider_native", "parse_output", "instructor"] = "parse_output"
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
    class LocalVLMHints(BaseModel):
        prompt_template_style: Literal["auto", "chatml", "plain"] = "auto"
        image_token_strategy: Literal["auto", "single", "multi"] = "auto"

    system_prompt: PromptSource
    user_prompt: PromptSource | None = None
    ontology: OntologySource
    structured_schema: Literal["scene_graph", "relationship_list"] = "scene_graph"
    local_vlm_hints: LocalVLMHints = Field(default_factory=LocalVLMHints)


class ChatConfig(LLMConfig):
    system_prompt: PromptSource
    context_template: PromptSource | None = None


class CaptionConfig(LLMConfig):
    mode: Literal["unconditional", "prompted"] = "prompted"
    max_words: int | None = None
    system_prompt: PromptSource = Field(
        default_factory=lambda: PromptSource(
            text=(
                "You are Pepper robot vision caption module. "
                "Describe the visible scene simply and briefly."
            )
        )
    )
    user_prompt: PromptSource | None = Field(
        default_factory=lambda: PromptSource(text="What do you see?")
    )
    provider: Literal[
        "openai",
        "gemini",
        "openai_compatible",
        "local_hf",
        "local_4bit",
    ] = "local_hf"
    model_id: str = "Salesforce/blip-image-captioning-large"


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


class WorkerRuntimeConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    idle_timeout_seconds: int = Field(default=600, gt=0)
    idle_check_interval_seconds: float = Field(default=2.0, gt=0)
    startup_timeout_seconds: float = Field(default=120.0, gt=0)
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    shutdown_grace_seconds: float = Field(default=15.0, gt=0)
    max_startup_queue: int = Field(default=32, ge=1)
    healthcheck_interval_seconds: float = Field(default=2.0, gt=0)
    restart_max_attempts: int = Field(default=3, ge=1)
    restart_window_seconds: int = Field(default=60, gt=0)
    restart_backoff_seconds: list[float] = Field(
        default_factory=lambda: [1.0, 3.0, 10.0], min_length=1
    )
    circuit_breaker_cooldown_seconds: int = Field(default=30, gt=0)
    auto_warmup_on_startup: bool = False

    @model_validator(mode="after")
    def validate_restart_backoff_seconds(self):
        if any(delay <= 0 for delay in self.restart_backoff_seconds):
            raise ValueError("restart_backoff_seconds values must be > 0")
        return self


class PipelineControls(BaseModel):
    preset: Literal[
        "full",
        "detect_only",
        "vlm_only",
        "rules_only",
        "minimal",
        "custom",
    ] = "full"
    detect: bool = True
    track_memory: bool = True
    paint_som: bool = True
    scene_graph: bool = True
    update_scene_memory: bool = True

    @staticmethod
    def preset_map() -> dict[str, dict[str, bool]]:
        return {
            "full": {
                "detect": True,
                "track_memory": True,
                "paint_som": True,
                "scene_graph": True,
                "update_scene_memory": True,
            },
            "detect_only": {
                "detect": True,
                "track_memory": False,
                "paint_som": False,
                "scene_graph": False,
                "update_scene_memory": False,
            },
            "vlm_only": {
                "detect": False,
                "track_memory": False,
                "paint_som": False,
                "scene_graph": True,
                "update_scene_memory": False,
            },
            "rules_only": {
                "detect": True,
                "track_memory": True,
                "paint_som": False,
                "scene_graph": True,
                "update_scene_memory": True,
            },
            "minimal": {
                "detect": True,
                "track_memory": False,
                "paint_som": False,
                "scene_graph": False,
                "update_scene_memory": False,
            },
        }

    @model_validator(mode="after")
    def apply_preset(self):
        if self.preset != "custom":
            mapped = self.preset_map().get(self.preset)
            if mapped is not None:
                self.detect = mapped["detect"]
                self.track_memory = mapped["track_memory"]
                self.paint_som = mapped["paint_som"]
                self.scene_graph = mapped["scene_graph"]
                self.update_scene_memory = mapped["update_scene_memory"]
                return self

        # Promote to a named preset when toggles match exactly, otherwise keep custom.
        for name, flags in self.preset_map().items():
            if (
                self.detect == flags["detect"]
                and self.track_memory == flags["track_memory"]
                and self.paint_som == flags["paint_som"]
                and self.scene_graph == flags["scene_graph"]
                and self.update_scene_memory == flags["update_scene_memory"]
            ):
                self.preset = name
                return self
        self.preset = "custom"
        return self


class AppConfig(BaseModel):
    system: dict
    detection: DetectionConfig
    tracking: TrackingConfig
    scene_graph: SceneGraphConfig
    chat: ChatConfig
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    visualization: VisConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    worker: WorkerRuntimeConfig = Field(default_factory=WorkerRuntimeConfig)
    pipeline_controls: PipelineControls = Field(default_factory=PipelineControls)
    _config_path: Path | None = PrivateAttr(None)

    @model_validator(mode="after")
    def validate_pipeline_controls(self):
        controls = self.pipeline_controls
        if controls.track_memory and not controls.detect:
            raise ValueError("pipeline_controls.track_memory requires detect=true")
        if controls.paint_som and not controls.detect:
            raise ValueError("pipeline_controls.paint_som requires detect=true")
        if controls.update_scene_memory and not controls.scene_graph:
            raise ValueError(
                "pipeline_controls.update_scene_memory requires scene_graph=true"
            )
        if controls.update_scene_memory and not controls.track_memory:
            raise ValueError(
                "pipeline_controls.update_scene_memory requires track_memory=true"
            )
        if (
            self.scene_graph.mode == "rules"
            and controls.scene_graph
            and not controls.detect
        ):
            raise ValueError(
                "scene_graph.mode=rules requires detect=true when scene_graph stage is enabled"
            )
        return self

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
