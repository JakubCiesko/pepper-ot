from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from app.core.config.runtime_mutations import apply_pipeline_runtime_updates
from app.core.config.runtime_mutations import apply_scene_graph_runtime_updates
from app.core.config.runtime_mutations import resolve_base_dir
from app.core.config.runtime_mutations import resolve_prompt_text
from app.schemas.config import AppConfig

ReloadMode = Literal["hot", "hard"]


if TYPE_CHECKING:
    from app.core.runtime.state import AppState


@dataclass(frozen=True)
class ReloadRule:
    path: str
    getter: Callable[[Any], Any]
    mode: ReloadMode
    apply_hot: Callable[["AppState", AppConfig], None] | None = None


@dataclass(frozen=True)
class ConfigDiff:
    hot: list[str]
    hard: list[str]


def _resolve_base_dir(cfg: AppConfig) -> Path | None:
    return resolve_base_dir(cfg)


def _apply_pipeline_group(ml_state: "AppState", new: AppConfig):
    if ml_state.pipeline is None:
        return
    base_dir = _resolve_base_dir(new)
    apply_pipeline_runtime_updates(ml_state.pipeline, new, base_dir)
    if ml_state.pipeline.scene_graph_service:
        apply_scene_graph_runtime_updates(
            ml_state.pipeline.scene_graph_service,
            new,
            base_dir,
        )


def _apply_chat_group(ml_state: "AppState", new: AppConfig):
    if ml_state.chat_service is None:
        return
    base_dir = _resolve_base_dir(new)
    ml_state.chat_service.system_prompt = resolve_prompt_text(
        new.chat.system_prompt,
        base_dir,
        fallback=ml_state.chat_service.system_prompt,
    )
    ml_state.chat_service.context_template = resolve_prompt_text(
        new.chat.context_template,
        base_dir,
        fallback=None,
    )
    ml_state.chat_service.llm.update_runtime(new.chat)


def _apply_caption_group(ml_state: "AppState", new: AppConfig):
    if ml_state.caption_service is None:
        return
    base_dir = _resolve_base_dir(new)
    system_prompt = resolve_prompt_text(
        new.caption.system_prompt,
        base_dir,
        fallback=ml_state.caption_service.system_prompt,
    )
    user_prompt = resolve_prompt_text(
        new.caption.user_prompt,
        base_dir,
        fallback=ml_state.caption_service.user_prompt,
    )
    ml_state.caption_service.update_runtime(
        new.caption,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        rebuild_client=False,
    )


RELOAD_RULES: list[ReloadRule] = [
    # Detection
    ReloadRule("detection.backend", attrgetter("detection.backend"), "hard"),
    ReloadRule("detection.weights_path", attrgetter("detection.weights_path"), "hard"),
    ReloadRule(
        "detection.confidence_threshold",
        attrgetter("detection.confidence_threshold"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "detection.device",
        attrgetter("detection.device"),
        "hot",
        _apply_pipeline_group,
    ),
    # objects
    ReloadRule(
        "detection.ontology",
        attrgetter("detection.ontology"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "detection.ontology_path",
        attrgetter("detection.ontology_path"),
        "hot",
        _apply_pipeline_group,
    ),
    # Scene Graph (VLM)
    ReloadRule(
        "scene_graph.vlm.provider", attrgetter("scene_graph.vlm.provider"), "hard"
    ),
    ReloadRule(
        "scene_graph.vlm.model_id", attrgetter("scene_graph.vlm.model_id"), "hard"
    ),
    ReloadRule(
        "scene_graph.vlm.base_url", attrgetter("scene_graph.vlm.base_url"), "hard"
    ),
    ReloadRule(
        "scene_graph.vlm.timeout_seconds",
        attrgetter("scene_graph.vlm.timeout_seconds"),
        "hard",
    ),
    ReloadRule(
        "scene_graph.vlm.api_key_env",
        attrgetter("scene_graph.vlm.api_key_env"),
        "hard",
    ),
    ReloadRule(
        "scene_graph.vlm.client_init_kwargs",
        attrgetter("scene_graph.vlm.client_init_kwargs"),
        "hard",
    ),
    ReloadRule("scene_graph.vlm.device", attrgetter("scene_graph.vlm.device"), "hard"),
    ReloadRule(
        "scene_graph.vlm.call_kwargs",
        attrgetter("scene_graph.vlm.call_kwargs"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.structured_output",
        attrgetter("scene_graph.vlm.structured_output"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.structured_schema",
        attrgetter("scene_graph.vlm.structured_schema"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.local_vlm_hints",
        attrgetter("scene_graph.vlm.local_vlm_hints"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.system_prompt",
        attrgetter("scene_graph.vlm.system_prompt"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.user_prompt",
        attrgetter("scene_graph.vlm.user_prompt"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.ontology",
        attrgetter("scene_graph.vlm.ontology"),
        "hot",
        _apply_pipeline_group,
    ),
    # Visualization -- not only this, it affects sgg!
    ReloadRule(
        "visualization", attrgetter("visualization"), "hot", _apply_pipeline_group
    ),
    # SGG
    ReloadRule(
        "scene_graph.mode", attrgetter("scene_graph.mode"), "hot", _apply_pipeline_group
    ),
    ReloadRule(
        "scene_graph.rules",
        attrgetter("scene_graph.rules"),
        "hot",
        _apply_pipeline_group,
    ),
    # Chat
    ReloadRule("chat.provider", attrgetter("chat.provider"), "hard"),
    ReloadRule("chat.model_id", attrgetter("chat.model_id"), "hard"),
    ReloadRule("chat.base_url", attrgetter("chat.base_url"), "hard"),
    ReloadRule("chat.timeout_seconds", attrgetter("chat.timeout_seconds"), "hard"),
    ReloadRule("chat.api_key_env", attrgetter("chat.api_key_env"), "hard"),
    ReloadRule(
        "chat.client_init_kwargs", attrgetter("chat.client_init_kwargs"), "hard"
    ),
    ReloadRule("chat.device", attrgetter("chat.device"), "hot", _apply_chat_group),
    ReloadRule(
        "chat.call_kwargs", attrgetter("chat.call_kwargs"), "hot", _apply_chat_group
    ),
    ReloadRule(
        "chat.structured_output",
        attrgetter("chat.structured_output"),
        "hot",
        _apply_chat_group,
    ),
    ReloadRule(
        "chat.system_prompt", attrgetter("chat.system_prompt"), "hot", _apply_chat_group
    ),
    ReloadRule(
        "chat.context_template",
        attrgetter("chat.context_template"),
        "hot",
        _apply_chat_group,
    ),
    # Caption
    ReloadRule("caption.provider", attrgetter("caption.provider"), "hard"),
    ReloadRule("caption.model_id", attrgetter("caption.model_id"), "hard"),
    ReloadRule("caption.base_url", attrgetter("caption.base_url"), "hard"),
    ReloadRule(
        "caption.timeout_seconds", attrgetter("caption.timeout_seconds"), "hard"
    ),
    ReloadRule("caption.api_key_env", attrgetter("caption.api_key_env"), "hard"),
    ReloadRule(
        "caption.client_init_kwargs", attrgetter("caption.client_init_kwargs"), "hard"
    ),
    ReloadRule("caption.device", attrgetter("caption.device"), "hard"),
    ReloadRule(
        "caption.call_kwargs",
        attrgetter("caption.call_kwargs"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule(
        "caption.structured_output",
        attrgetter("caption.structured_output"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule("caption.mode", attrgetter("caption.mode"), "hot", _apply_caption_group),
    ReloadRule(
        "caption.max_words",
        attrgetter("caption.max_words"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule(
        "caption.system_prompt",
        attrgetter("caption.system_prompt"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule(
        "caption.user_prompt",
        attrgetter("caption.user_prompt"),
        "hot",
        _apply_caption_group,
    ),
    # Tracking
    ReloadRule(
        "tracking.feature_extraction.reid_model",
        attrgetter("tracking.feature_extraction.reid_model"),
        "hard",
    ),
    ReloadRule(
        "tracking.feature_extraction.device",
        attrgetter("tracking.feature_extraction.device"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.feature_extraction.target_size",
        attrgetter("tracking.feature_extraction.target_size"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.feature_extraction.resampling_method",
        attrgetter("tracking.feature_extraction.resampling_method"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.max_dormant_frames",
        attrgetter("tracking.max_dormant_frames"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.association",
        attrgetter("tracking.association"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.memory_max_age_seconds",
        attrgetter("tracking.memory_max_age_seconds"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.memory_max_objects",
        attrgetter("tracking.memory_max_objects"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "tracking.memory_max_relations",
        attrgetter("tracking.memory_max_relations"),
        "hot",
        _apply_pipeline_group,
    ),
    # Storage
    ReloadRule(
        "storage.persist_last_state", attrgetter("storage.persist_last_state"), "hot"
    ),
    ReloadRule("storage.last_state_path", attrgetter("storage.last_state_path"), "hot"),
    ReloadRule("storage.store_image", attrgetter("storage.store_image"), "hot"),
    # Worker process runtime
    ReloadRule("worker.enabled", attrgetter("worker.enabled"), "hard"),
    ReloadRule("worker.host", attrgetter("worker.host"), "hard"),
    ReloadRule("worker.port", attrgetter("worker.port"), "hard"),
    ReloadRule(
        "worker.startup_timeout_seconds",
        attrgetter("worker.startup_timeout_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.shutdown_grace_seconds",
        attrgetter("worker.shutdown_grace_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.max_startup_queue", attrgetter("worker.max_startup_queue"), "hard"
    ),
    ReloadRule(
        "worker.healthcheck_interval_seconds",
        attrgetter("worker.healthcheck_interval_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.restart_max_attempts", attrgetter("worker.restart_max_attempts"), "hard"
    ),
    ReloadRule(
        "worker.restart_window_seconds",
        attrgetter("worker.restart_window_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.restart_backoff_seconds",
        attrgetter("worker.restart_backoff_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.circuit_breaker_cooldown_seconds",
        attrgetter("worker.circuit_breaker_cooldown_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.idle_timeout_seconds", attrgetter("worker.idle_timeout_seconds"), "hot"
    ),
    ReloadRule(
        "worker.idle_check_interval_seconds",
        attrgetter("worker.idle_check_interval_seconds"),
        "hot",
    ),
    ReloadRule(
        "worker.request_timeout_seconds",
        attrgetter("worker.request_timeout_seconds"),
        "hot",
    ),
    ReloadRule(
        "pipeline_controls",
        attrgetter("pipeline_controls"),
        "hot",
        _apply_pipeline_group,
    ),
]


def _assert_unique_paths():
    seen: set[str] = set()
    dupes: set[str] = set()
    for rule in RELOAD_RULES:
        if rule.path in seen:
            dupes.add(rule.path)
        seen.add(rule.path)
    if dupes:
        raise ValueError(f"Duplicate reload rule paths: {sorted(dupes)}")


_assert_unique_paths()


def rule_index() -> dict[str, ReloadRule]:
    return {rule.path: rule for rule in RELOAD_RULES}


def diff_config(old: AppConfig, new: AppConfig) -> ConfigDiff:
    hot: list[str] = []
    hard: list[str] = []
    for rule in RELOAD_RULES:
        if rule.getter(old) != rule.getter(new):
            (hot if rule.mode == "hot" else hard).append(rule.path)
    return ConfigDiff(hot=hot, hard=hard)


def hard_reload_fields() -> list[str]:
    return [rule.path for rule in RELOAD_RULES if rule.mode == "hard"]


async def apply_hot_changes(
    ml_state: "AppState", old: AppConfig, new: AppConfig
) -> ConfigDiff:
    diff = diff_config(old, new)

    ml_state.config = new
    ml_state.config_version += 1

    idx = rule_index()
    applied_handlers: set[int] = set()
    for path in diff.hot:
        rule = idx.get(path)
        if rule is None or rule.apply_hot is None:
            continue
        marker = id(rule.apply_hot)
        if marker in applied_handlers:
            continue
        rule.apply_hot(ml_state, new)
        applied_handlers.add(marker)

    if ml_state.worker_manager:
        await ml_state.worker_manager.apply_hot_config(new, ml_state.config_version)
    return diff
