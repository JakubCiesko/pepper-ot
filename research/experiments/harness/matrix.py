from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any

import yaml

from research.experiments.config.models import ExperimentConfig
from research.experiments.io import save_json
from research.experiments.io.metadata import start_run
from research.experiments.workflows.experiments import run_all_phases


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_variant_config(run_dir: Path, raw_config: dict[str, Any]) -> None:
    with (run_dir / "variant_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw_config, f, sort_keys=False, allow_unicode=True)


def _copy_reusable_artifacts(
    run_dir: Path, reuse_config: dict[str, Any] | None
) -> tuple[str | None, list[str]]:
    if not reuse_config:
        return None, []

    source_raw = reuse_config.get("from_run")
    if not source_raw:
        raise RuntimeError("reuse_artifacts.from_run is required")
    source_dir = Path(str(source_raw))
    if not source_dir.is_absolute():
        source_dir = (Path.cwd() / source_dir).resolve()
    if not source_dir.exists():
        raise RuntimeError(f"Reusable artifact source does not exist: {source_dir}")

    files = reuse_config.get("files") or []
    if not isinstance(files, list) or not files:
        raise RuntimeError("reuse_artifacts.files must be a non-empty list")

    copied: list[str] = []
    for file_name in files:
        relative = Path(str(file_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Reusable artifact path must be relative within the run: {file_name}"
            )
        source_path = source_dir / relative
        if not source_path.exists():
            raise RuntimeError(f"Missing reusable artifact: {source_path}")
        target_path = run_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied.append(str(relative))
    return str(source_dir), copied


async def run_matrix(matrix_path: Path) -> list[dict[str, Any]]:
    matrix = _load_yaml(matrix_path)
    base_config_path = Path(matrix["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = (matrix_path.parent / base_config_path).resolve()
    base_raw = _load_yaml(base_config_path)

    common_overrides = matrix.get("common_overrides", {})
    reuse_config = matrix.get("reuse_artifacts")
    variants = matrix.get("variants") or []
    if not variants:
        raise RuntimeError(f"No variants defined in {matrix_path}")

    results: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        name = str(variant.get("name") or f"variant_{idx:02d}")
        overrides = variant.get("overrides") or {}
        raw_config = _deep_merge(base_raw, common_overrides)
        raw_config = _deep_merge(raw_config, overrides)
        raw_config["experiment_id"] = raw_config.get("experiment_id") or name
        raw_config["name"] = raw_config.get("name") or matrix.get("name", "matrix")

        config = ExperimentConfig(**raw_config)
        run = start_run(
            config.paths.output_root,
            config.name,
            config.experiment_id,
            raw_config,
            command=f"run-matrix:{name}",
        )
        _write_variant_config(run.run_dir, raw_config)
        if config.paths.manifest_file:
            manifest_path = Path(config.paths.manifest_file)
            if manifest_path.exists():
                (run.run_dir / "manifest.jsonl").write_text(
                    manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
                )

        reused_from = None
        reused_artifacts: list[str] = []
        try:
            reused_from, reused_artifacts = _copy_reusable_artifacts(
                run.run_dir, reuse_config
            )
            outputs = await run_all_phases(config, run)
            result = {
                "variant": name,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "ok": True,
                "outputs": list(outputs.keys()),
                "reused_artifacts_from": reused_from,
                "reused_artifacts": reused_artifacts,
            }
        except Exception as exc:
            run.logger.exception("Variant failed: %s", name)
            result = {
                "variant": name,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "ok": False,
                "error": str(exc),
                "reused_artifacts_from": reused_from,
                "reused_artifacts": reused_artifacts,
            }
        save_json(run.run_dir / "matrix_result.json", result)
        results.append(result)
    return results


def run_matrix_sync(matrix_path: Path) -> list[dict[str, Any]]:
    return asyncio.run(run_matrix(matrix_path))
