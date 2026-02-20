import io
from pathlib import Path
from typing import Any

import yaml

from app.schemas.config import AppConfig


def config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.yaml"


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

    resolved_understanding = resolved.get("understanding", {})
    resolved_understanding["resolved_system_prompt"] = (
        cfg.understanding.system_prompt.resolve(base_dir)
    )
    resolved_understanding["resolved_user_prompt"] = (
        cfg.understanding.user_prompt.resolve(base_dir)
        if cfg.understanding.user_prompt is not None
        else None
    )
    predicates, objects = cfg.understanding.ontology.resolve(base_dir)
    resolved_understanding["resolved_ontology"] = {
        "predicates": predicates,
        "objects": objects,
    }
    resolved["understanding"] = resolved_understanding

    resolved_chat = resolved.get("chat", {})
    resolved_chat["resolved_system_prompt"] = cfg.chat.system_prompt.resolve(base_dir)
    resolved_chat["resolved_context_template"] = (
        cfg.chat.context_template.resolve(base_dir)
        if cfg.chat.context_template is not None
        else None
    )
    resolved["chat"] = resolved_chat

    return resolved


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
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
    new_cfg = AppConfig(**merged)
    new_cfg._config_path = cfg._config_path
    return new_cfg


def parse_uploaded_yaml(content: bytes) -> AppConfig:
    raw = yaml.safe_load(io.BytesIO(content)) or {}
    cfg = AppConfig(**raw)
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
        cfg.understanding.system_prompt.path,
        prompt_roots,
        "understanding.system_prompt",
    )
    if cfg.understanding.user_prompt is not None:
        check(
            cfg.understanding.user_prompt.path,
            prompt_roots,
            "understanding.user_prompt",
        )
    check(cfg.understanding.ontology.path, ontology_roots, "understanding.ontology")
    check(cfg.chat.system_prompt.path, prompt_roots, "chat.system_prompt")
    if cfg.chat.context_template is not None:
        check(cfg.chat.context_template.path, prompt_roots, "chat.context_template")
