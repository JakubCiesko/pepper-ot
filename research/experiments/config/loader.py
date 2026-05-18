from pathlib import Path
from typing import Any

import yaml

from .models import ExperimentConfig


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load and validate an experiment YAML config.

    Args:
        path: YAML config path.

    Returns:
        Tuple of validated ExperimentConfig and the raw parsed dictionary saved
        later into run metadata for reproducibility.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = ExperimentConfig(**raw)
    return config, raw
