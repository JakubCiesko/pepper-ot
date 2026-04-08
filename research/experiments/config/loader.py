from pathlib import Path
from typing import Any

import yaml

from .models import ExperimentConfig


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = ExperimentConfig(**raw)
    return config, raw
