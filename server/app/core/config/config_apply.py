from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter

from app.core.runtime.state import MLState
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
    ("detection.ontology_path", attrgetter("detection.ontology_path"), "hot"),
    # Scene Graph (VLM)
    ("scene_graph.vlm.provider", attrgetter("scene_graph.vlm.provider"), "hard"),
    ("scene_graph.vlm.model_id", attrgetter("scene_graph.vlm.model_id"), "hard"),
    ("scene_graph.vlm.base_url", attrgetter("scene_graph.vlm.base_url"), "hard"),
    (
        "scene_graph.vlm.timeout_seconds",
        attrgetter("scene_graph.vlm.timeout_seconds"),
        "hard",
    ),
    (
        "scene_graph.vlm.api_key_env",
        attrgetter("scene_graph.vlm.api_key_env"),
        "hard",
    ),
    (
        "scene_graph.vlm.client_init_kwargs",
        attrgetter("scene_graph.vlm.client_init_kwargs"),
        "hard",
    ),
    ("scene_graph.vlm.device", attrgetter("scene_graph.vlm.device"), "hard"),
    ("scene_graph.vlm.call_kwargs", attrgetter("scene_graph.vlm.call_kwargs"), "hot"),
    (
        "scene_graph.vlm.structured_output",
        attrgetter("scene_graph.vlm.structured_output"),
        "hot",
    ),
    (
        "scene_graph.vlm.structured_schema",
        attrgetter("scene_graph.vlm.structured_schema"),
        "hot",
    ),
    (
        "scene_graph.vlm.local_vlm_hints",
        attrgetter("scene_graph.vlm.local_vlm_hints"),
        "hot",
    ),
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
    ("chat.provider", attrgetter("chat.provider"), "hard"),
    ("chat.model_id", attrgetter("chat.model_id"), "hard"),
    ("chat.base_url", attrgetter("chat.base_url"), "hard"),
    ("chat.timeout_seconds", attrgetter("chat.timeout_seconds"), "hard"),
    ("chat.api_key_env", attrgetter("chat.api_key_env"), "hard"),
    ("chat.client_init_kwargs", attrgetter("chat.client_init_kwargs"), "hard"),
    ("chat.device", attrgetter("chat.device"), "hot"),
    ("chat.call_kwargs", attrgetter("chat.call_kwargs"), "hot"),
    ("chat.structured_output", attrgetter("chat.structured_output"), "hot"),
    ("chat.system_prompt", attrgetter("chat.system_prompt"), "hot"),
    ("chat.context_template", attrgetter("chat.context_template"), "hot"),
    # Caption
    ("caption.provider", attrgetter("caption.provider"), "hard"),
    ("caption.model_id", attrgetter("caption.model_id"), "hard"),
    ("caption.base_url", attrgetter("caption.base_url"), "hard"),
    ("caption.timeout_seconds", attrgetter("caption.timeout_seconds"), "hard"),
    ("caption.api_key_env", attrgetter("caption.api_key_env"), "hard"),
    ("caption.client_init_kwargs", attrgetter("caption.client_init_kwargs"), "hard"),
    ("caption.device", attrgetter("caption.device"), "hard"),
    ("caption.call_kwargs", attrgetter("caption.call_kwargs"), "hot"),
    ("caption.structured_output", attrgetter("caption.structured_output"), "hot"),
    ("caption.mode", attrgetter("caption.mode"), "hot"),
    ("caption.max_words", attrgetter("caption.max_words"), "hot"),
    ("caption.system_prompt", attrgetter("caption.system_prompt"), "hot"),
    ("caption.user_prompt", attrgetter("caption.user_prompt"), "hot"),
    # Tracking
    (
        "tracking.feature_extraction.reid_model",
        attrgetter("tracking.feature_extraction.reid_model"),
        "hard",
    ),
    (
        "tracking.feature_extraction.device",
        attrgetter("tracking.feature_extraction.device"),
        "hot",
    ),
    (
        "tracking.feature_extraction.target_size",
        attrgetter("tracking.feature_extraction.target_size"),
        "hot",
    ),
    (
        "tracking.feature_extraction.resampling_method",
        attrgetter("tracking.feature_extraction.resampling_method"),
        "hot",
    ),
    ("tracking.max_dormant_frames", attrgetter("tracking.max_dormant_frames"), "hot"),
    ("tracking.association", attrgetter("tracking.association"), "hot"),
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
    # Worker process runtime
    ("worker.enabled", attrgetter("worker.enabled"), "hard"),
    ("worker.host", attrgetter("worker.host"), "hard"),
    ("worker.port", attrgetter("worker.port"), "hard"),
    (
        "worker.startup_timeout_seconds",
        attrgetter("worker.startup_timeout_seconds"),
        "hard",
    ),
    (
        "worker.shutdown_grace_seconds",
        attrgetter("worker.shutdown_grace_seconds"),
        "hard",
    ),
    ("worker.max_startup_queue", attrgetter("worker.max_startup_queue"), "hard"),
    (
        "worker.healthcheck_interval_seconds",
        attrgetter("worker.healthcheck_interval_seconds"),
        "hard",
    ),
    ("worker.restart_max_attempts", attrgetter("worker.restart_max_attempts"), "hard"),
    (
        "worker.restart_window_seconds",
        attrgetter("worker.restart_window_seconds"),
        "hard",
    ),
    (
        "worker.restart_backoff_seconds",
        attrgetter("worker.restart_backoff_seconds"),
        "hard",
    ),
    (
        "worker.circuit_breaker_cooldown_seconds",
        attrgetter("worker.circuit_breaker_cooldown_seconds"),
        "hard",
    ),
    ("worker.idle_timeout_seconds", attrgetter("worker.idle_timeout_seconds"), "hot"),
    (
        "worker.idle_check_interval_seconds",
        attrgetter("worker.idle_check_interval_seconds"),
        "hot",
    ),
    (
        "worker.request_timeout_seconds",
        attrgetter("worker.request_timeout_seconds"),
        "hot",
    ),
    ("pipeline_controls", attrgetter("pipeline_controls"), "hot"),
]


def diff_config(old, new) -> ConfigDiff:
    hot, hard = [], []

    for path, getter, category in _RULES:
        if getter(old) != getter(new):
            (hot if category == "hot" else hard).append(path)

    return ConfigDiff(hot=hot, hard=hard)


def _update_pipeline(ml_state: MLState, new: AppConfig):
    base_dir = new._config_path.parent if new._config_path is not None else None
    ml_state.pipeline.detector.threshold = new.detection.confidence_threshold
    ml_state.pipeline.detector.device = new.detection.device
    ml_state.pipeline.detector.ontology = (
        new.detection.resolve_ontology(base_dir)
        if base_dir is not None
        else new.detection.ontology
    )
    ml_state.pipeline.vis_config = new.visualization
    ml_state.pipeline.fusion_config = new.fusion
    ml_state.pipeline.pipeline_controls = new.pipeline_controls

    # Update memory pruning settings
    if hasattr(ml_state.pipeline, "memory") and ml_state.pipeline.memory:
        ml_state.pipeline.memory.set_limits(
            new.tracking.memory_max_age_seconds,
            new.tracking.memory_max_objects,
            new.tracking.memory_max_relations,
        )
        ml_state.pipeline.memory.set_max_dormant_frames(new.tracking.max_dormant_frames)
        ml_state.pipeline.memory.set_association_config(new.tracking.association)
        ml_state.pipeline.memory.set_feature_extraction_config(
            new.tracking.feature_extraction
        )

    # Update scene graph mode and runtime VLM/rules settings
    ml_state.pipeline.scene_graph_service.mode = new.scene_graph.mode

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
    ml_state.chat_service.llm.update_runtime(new.chat)


def _update_caption(ml_state: MLState, new: AppConfig):
    if ml_state.caption_service is None:
        return
    base_dir = new._config_path.parent if new._config_path is not None else None
    system_prompt = (
        new.caption.system_prompt.resolve(base_dir)
        if base_dir is not None
        else ml_state.caption_service.system_prompt
    )
    user_prompt = (
        new.caption.user_prompt.resolve(base_dir)
        if base_dir is not None and new.caption.user_prompt is not None
        else (
            ml_state.caption_service.user_prompt
            if new.caption.user_prompt is None
            else None
        )
    )
    ml_state.caption_service.update_runtime(
        new.caption,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        rebuild_client=False,
    )


async def apply_hot_config(ml_state: MLState, new: AppConfig) -> None:
    ml_state.config = new
    ml_state.config_version += 1

    if ml_state.pipeline:
        _update_pipeline(ml_state, new)

    if ml_state.chat_service:
        _update_chat(ml_state, new)
    if ml_state.caption_service:
        _update_caption(ml_state, new)

    if ml_state.worker_manager:
        await ml_state.worker_manager.apply_hot_config(new, ml_state.config_version)
