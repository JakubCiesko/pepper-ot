from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from app.core.config.mutations.runtime import apply_pipeline_runtime_updates
from app.core.config.mutations.runtime import apply_scene_graph_runtime_updates
from app.core.config.mutations.runtime import resolve_base_dir
from app.core.config.mutations.runtime import resolve_prompt_text
from app.core.runtime.state import AppState
from app.schemas.config import AppConfig

ReloadMode = Literal["hot", "hard"]


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


def _apply_qa_group(ml_state: "AppState", new: AppConfig):
    if ml_state.qa_pool_service is not None:
        ml_state.qa_pool_service.set_max_entries(new.qa_generation.pool_max_entries)
    _apply_pipeline_group(ml_state, new)


def _apply_chat_group(ml_state: "AppState", new: AppConfig):
    if ml_state.chat_service is None:
        return
    base_dir = _resolve_base_dir(new)
    ml_state.chat_service.system_prompt = resolve_prompt_text(
        new.chat.system_prompt,
        base_dir,
        fallback=ml_state.chat_service.system_prompt,
    )
    ml_state.chat_service.user_prompt = resolve_prompt_text(
        new.chat.user_prompt,
        base_dir,
        fallback=ml_state.chat_service.user_prompt,
    )
    ml_state.chat_service.object_system_prompt = resolve_prompt_text(
        new.chat.object_system_prompt,
        base_dir,
        fallback=ml_state.chat_service.object_system_prompt,
    )
    ml_state.chat_service.object_user_prompt = resolve_prompt_text(
        new.chat.object_user_prompt,
        base_dir,
        fallback=ml_state.chat_service.object_user_prompt,
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
