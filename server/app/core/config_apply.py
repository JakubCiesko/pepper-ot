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
    # Understanding
    ("understanding.backend", attrgetter("understanding.backend"), "hard"),
    ("understanding.model_id", attrgetter("understanding.model_id"), "hard"),
    ("understanding.system_prompt", attrgetter("understanding.system_prompt"), "hot"),
    ("understanding.user_prompt", attrgetter("understanding.user_prompt"), "hot"),
    # predicates
    ("understanding.ontology", attrgetter("understanding.ontology"), "hot"),
    # Visualization -- not only this, it affects sgg!
    ("visualization", attrgetter("visualization"), "hot"),
    # SGG
    ("sgg.mode", attrgetter("sgg.mode"), "hot"),
    ("sgg.rules", attrgetter("sgg.rules"), "hot"),
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
    ml_state.pipeline.vis_config = new.visualization
    ml_state.pipeline.sgg_mode = new.sgg.mode
    ml_state.pipeline.fusion_config = new.fusion

    # Update memory pruning settings
    if hasattr(ml_state.pipeline, "memory") and ml_state.pipeline.memory:
        ml_state.pipeline.memory.set_limits(
            new.tracking.memory_max_age_seconds,
            new.tracking.memory_max_objects,
            new.tracking.memory_max_relations,
        )

    # Update VLM prompts/ontology + inference params
    base_dir = new._config_path.parent if new._config_path is not None else None
    if ml_state.pipeline.sgg:
        sgg = ml_state.pipeline.sgg
        if base_dir is not None:
            sgg.system_prompt = new.understanding.system_prompt.resolve(base_dir)
            sgg.user_prompt = (
                new.understanding.user_prompt.resolve(base_dir)
                if new.understanding.user_prompt is not None
                else None
            )
            predicates, objects = new.understanding.ontology.resolve(base_dir)
            sgg.predicates = predicates
            sgg.objects = objects

        sgg.config.temperature = new.understanding.inference.get(
            "temperature", sgg.config.temperature
        )
        sgg.config.max_tokens = new.understanding.inference.get(
            "max_tokens", sgg.config.max_tokens
        )
    if ml_state.pipeline.rules_sgg:
        ml_state.pipeline.rules_sgg.rules_config = new.sgg.rules


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


async def apply_hot_config(ml_state: MLState, new: AppConfig) -> None:
    ml_state.config = new

    if ml_state.pipeline:
        _update_pipeline(ml_state, new)

    if ml_state.chat_service:
        _update_chat(ml_state, new)
