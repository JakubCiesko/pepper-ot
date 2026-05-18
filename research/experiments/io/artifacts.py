import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: Any) -> None:
    """Write an experiment artifact as pretty UTF-8 JSON.

    Args:
        path: Destination artifact path.
        payload: JSON-serializable payload.

    Side Effects:
        Creates the parent directory and writes the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Any = None) -> Any:
    """Load an experiment JSON artifact with a default for missing files.

    Args:
        path: Artifact path to read.
        default: Value returned when the file is missing. If None, an empty
            dictionary is returned for backwards compatibility.

    Returns:
        Parsed JSON payload or the selected default value.
    """
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
