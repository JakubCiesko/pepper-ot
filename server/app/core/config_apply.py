from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter

from app.core.state import MLState
from app.schemas.config import AppConfig


@dataclass
class ConfigDiff:
    hot: list[str]
    hard: list[str]


def _changed(old, new) -> bool:
    return old != new


_RULES: list[tuple[str, Callable, str]] = [
    # Detection
    ("detection.backend", attrgetter("detection.backend"), "hard"),
    ("detection.weights_path", attrgetter("detection.weights_path"), "hard"),
    (
        "detection.confidence_threshold",
        attrgetter("detection.confidence_threshold"),
        "hot",
    ),
    ("detection.device", attrgetter("detection.device"), "hot"),
    # objects
    ("detection.ontology", attrgetter("detection.ontology"), "hot"),
    # Scene Graph (VLM)
    ("scene_graph.vlm.backend", attrgetter("scene_graph.vlm.backend"), "hard"),
    ("scene_graph.vlm.model_id", attrgetter("scene_graph.vlm.model_id"), "hard"),
    ("scene_graph.vlm.device", attrgetter("scene_graph.vlm.device"), "hard"),
    (
        "scene_graph.vlm.system_prompt",
        attrgetter("scene_graph.vlm.system_prompt"),
        "hot",
    ),
    ("scene_graph.vlm.user_prompt", attrgetter("scene_graph.vlm.user_prompt"), "hot"),
    ("scene_graph.vlm.ontology", attrgetter("scene_graph.vlm.ontology"), "hot"),
    # Visualization -- not only this, it affects sgg!
    ("visualization", attrgetter("visualization"), "hot"),
    # SGG
    ("scene_graph.mode", attrgetter("scene_graph.mode"), "hot"),
    ("scene_graph.rules", attrgetter("scene_graph.rules"), "hot"),
    # Chat
    ("chat.backend", attrgetter("chat.backend"), "hard"),
    ("chat.model_id", attrgetter("chat.model_id"), "hard"),
    ("chat.system_prompt", attrgetter("chat.system_prompt"), "hot"),
    ("chat.context_template", attrgetter("chat.context_template"), "hot"),
    # Tracking
    ("tracking.reid_model", attrgetter("tracking.reid_model"), "hard"),
    (
        "tracking.memory_max_age_seconds",
        attrgetter("tracking.memory_max_age_seconds"),
        "hot",
    ),
    ("tracking.memory_max_objects", attrgetter("tracking.memory_max_objects"), "hot"),
    (
        "tracking.memory_max_relations",
        attrgetter("tracking.memory_max_relations"),
        "hot",
    ),
    # Storage
    ("storage.persist_last_state", attrgetter("storage.persist_last_state"), "hot"),
    ("storage.last_state_path", attrgetter("storage.last_state_path"), "hot"),
    ("storage.store_image", attrgetter("storage.store_image"), "hot"),
]


def diff_config(old, new) -> ConfigDiff:
    hot, hard = [], []

    for path, getter, category in _RULES:
        if getter(old) != getter(new):
            (hot if category == "hot" else hard).append(path)

    return ConfigDiff(hot=hot, hard=hard)


def _update_pipeline(ml_state: MLState, new: AppConfig):
    ml_state.pipeline.detector.threshold = new.detection.confidence_threshold
    ml_state.pipeline.detector.device = new.detection.device
    ml_state.pipeline.detector.ontology = new.detection.ontology
    ml_state.pipeline.vis_config = new.visualization
    ml_state.pipeline.fusion_config = new.fusion

    # Update memory pruning settings
    if hasattr(ml_state.pipeline, "memory") and ml_state.pipeline.memory:
        ml_state.pipeline.memory.set_limits(
            new.tracking.memory_max_age_seconds,
            new.tracking.memory_max_objects,
            new.tracking.memory_max_relations,
        )

    # Update scene graph mode and runtime VLM/rules settings
    ml_state.pipeline.scene_graph_service.mode = new.scene_graph.mode

    base_dir = new._config_path.parent if new._config_path is not None else None
    if ml_state.pipeline.scene_graph_service:
        sg_service = ml_state.pipeline.scene_graph_service
        vlm_backend = sg_service.vlm_backend
        system_prompt = (
            new.scene_graph.vlm.system_prompt.resolve(base_dir)
            if base_dir is not None
            else vlm_backend.system_prompt
        )
        user_prompt = (
            new.scene_graph.vlm.user_prompt.resolve(base_dir)
            if base_dir is not None and new.scene_graph.vlm.user_prompt is not None
            else None
        )
        predicates, objects = (
            new.scene_graph.vlm.ontology.resolve(base_dir)
            if base_dir is not None
            else (vlm_backend.predicates, vlm_backend.objects)
        )
        vlm_backend.update_runtime(
            config=new.scene_graph.vlm,
            predicates=predicates,
            objects=objects,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            rebuild_client=False,
        )
        sg_service.rule_backend.rules_config = new.scene_graph.rules


def _update_chat(ml_state: MLState, new: AppConfig):
    base_dir = new._config_path.parent if new._config_path is not None else None
    if base_dir is not None:
        ml_state.chat_service.system_prompt = new.chat.system_prompt.resolve(base_dir)
        ml_state.chat_service.context_template = (
            new.chat.context_template.resolve(base_dir)
            if new.chat.context_template is not None
            else None
        )
    ml_state.chat_service.llm.config.inference = new.chat.inference
    ml_state.chat_service.llm.config.device = new.chat.device


async def apply_hot_config(ml_state: MLState, new: AppConfig) -> None:
    ml_state.config = new

    if ml_state.pipeline:
        _update_pipeline(ml_state, new)

    if ml_state.chat_service:
        _update_chat(ml_state, new)
