from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
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


async def run_matrix(matrix_path: Path) -> list[dict[str, Any]]:
    matrix = _load_yaml(matrix_path)
    base_config_path = Path(matrix["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = (matrix_path.parent / base_config_path).resolve()
    base_raw = _load_yaml(base_config_path)

    common_overrides = matrix.get("common_overrides", {})
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

        try:
            outputs = await run_all_phases(config, run)
            result = {
                "variant": name,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "ok": True,
                "outputs": list(outputs.keys()),
            }
        except Exception as exc:
            run.logger.exception("Variant failed: %s", name)
            result = {
                "variant": name,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "ok": False,
                "error": str(exc),
            }
        save_json(run.run_dir / "matrix_result.json", result)
        results.append(result)
    return results


def run_matrix_sync(matrix_path: Path) -> list[dict[str, Any]]:
    return asyncio.run(run_matrix(matrix_path))
