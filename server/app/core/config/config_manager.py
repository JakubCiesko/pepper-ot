import io
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from app.core.config.llm_contracts import provider_capability_matrix
from app.schemas.config import AppConfig
from app.schemas.config import PipelineControls


def config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config.yaml"


def load_config(path: Path | None = None) -> AppConfig:
    path = path or config_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig(**raw)
    cfg._config_path = path
    return cfg


def dump_config(cfg: AppConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def dump_config_yaml(cfg: AppConfig) -> str:
    data = dump_config(cfg)
    return yaml.safe_dump(data, sort_keys=False)


def resolve_config(cfg: AppConfig) -> dict[str, Any]:
    base_dir = cfg._config_path.parent if cfg._config_path else Path.cwd()

    resolved = dump_config(cfg)
    resolved_detection = resolved.get("detection", {})
    resolved_detection["resolved_ontology"] = cfg.detection.resolve_ontology(base_dir)
    resolved["detection"] = resolved_detection

    resolved_scene_graph = resolved.get("scene_graph", {})
    resolved_vlm = resolved_scene_graph.get("vlm", {})
    resolved_vlm["resolved_system_prompt"] = cfg.scene_graph.vlm.system_prompt.resolve(
        base_dir
    )
    resolved_vlm["resolved_user_prompt"] = (
        cfg.scene_graph.vlm.user_prompt.resolve(base_dir)
        if cfg.scene_graph.vlm.user_prompt is not None
        else None
    )
    predicates, objects = cfg.scene_graph.vlm.ontology.resolve(base_dir)
    resolved_vlm["resolved_ontology"] = {
        "predicates": predicates,
        "objects": objects,
    }
    resolved_scene_graph["vlm"] = resolved_vlm
    resolved["scene_graph"] = resolved_scene_graph

    resolved_chat = resolved.get("chat", {})
    resolved_chat["resolved_system_prompt"] = cfg.chat.system_prompt.resolve(base_dir)
    resolved["chat"] = resolved_chat

    resolved_caption = resolved.get("caption", {})
    resolved_caption["resolved_system_prompt"] = cfg.caption.system_prompt.resolve(
        base_dir
    )
    resolved_caption["resolved_user_prompt"] = (
        cfg.caption.user_prompt.resolve(base_dir)
        if cfg.caption.user_prompt is not None
        else None
    )
    resolved["caption"] = resolved_caption

    return resolved


def behavior_contracts() -> dict[str, Any]:
    return {
        **provider_capability_matrix(),
        "pipeline_presets": PipelineControls.preset_map(),
        "worker_state_values": [
            "STOPPED",
            "STARTING",
            "READY",
            "BUSY",
            "DRAINING",
            "STOPPING",
            "FAILED",
        ],
        "worker_defaults": {
            "enabled": False,
            "idle_timeout_seconds": 600,
            "startup_timeout_seconds": 120.0,
            "shutdown_grace_seconds": 15.0,
        },
    }


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    replace_on_patch_keys = {"call_kwargs", "client_init_kwargs"}
    for key, value in patch.items():
        if key in replace_on_patch_keys:
            base[key] = value
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            if "text" in value or "path" in value:
                base[key] = value
            else:
                base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_patch(cfg: AppConfig, patch: dict[str, Any]) -> AppConfig:
    merged = deep_merge(dump_config(cfg), patch)
    try:
        new_cfg = AppConfig(**merged)
    except ValidationError as exc:
        pieces: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "invalid value")
            pieces.append(f"{loc}: {msg}")
        if not pieces:
            raise ValueError(str(exc)) from exc
        raise ValueError("; ".join(pieces)) from exc
    new_cfg._config_path = cfg._config_path
    return new_cfg


def parse_uploaded_yaml(content: bytes) -> AppConfig:
    raw = yaml.safe_load(io.BytesIO(content)) or {}
    try:
        cfg = AppConfig(**raw)
    except ValidationError as exc:
        pieces: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "invalid value")
            pieces.append(f"{loc}: {msg}")
        if not pieces:
            raise ValueError(str(exc)) from exc
        raise ValueError("; ".join(pieces)) from exc
    cfg._config_path = config_path()
    _validate_paths(cfg)
    return cfg


def _is_safe_rel_path(path: Path, allowed_roots: set[str]) -> bool:
    if path.is_absolute():
        return False
    if ".." in path.parts:
        return False
    return path.parts and path.parts[0] in allowed_roots


def _validate_paths(cfg: AppConfig):
    prompt_roots = {"prompts"}
    ontology_roots = {"ontology"}

    def check(path: Path | None, roots: set[str], label: str):
        if path is None:
            return
        if not _is_safe_rel_path(path, roots):
            raise ValueError(f"Unsafe path in {label}: {path}")

    check(
        cfg.scene_graph.vlm.system_prompt.path,
        prompt_roots,
        "scene_graph.vlm.system_prompt",
    )
    if cfg.scene_graph.vlm.user_prompt is not None:
        check(
            cfg.scene_graph.vlm.user_prompt.path,
            prompt_roots,
            "scene_graph.vlm.user_prompt",
        )
    check(
        cfg.scene_graph.vlm.ontology.path,
        ontology_roots,
        "scene_graph.vlm.ontology",
    )
    check(cfg.chat.system_prompt.path, prompt_roots, "chat.system_prompt")
    check(cfg.caption.system_prompt.path, prompt_roots, "caption.system_prompt")
    if cfg.caption.user_prompt is not None:
        check(cfg.caption.user_prompt.path, prompt_roots, "caption.user_prompt")
    check(cfg.detection.ontology_path, ontology_roots, "detection.ontology_path")
