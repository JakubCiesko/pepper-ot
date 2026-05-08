from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.experiments.io import load_json
from research.experiments.io import save_json


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class JsonCache:
    def __init__(self, root: Path):
        self.root = root

    def key_path(self, stage: str, key: str) -> Path:
        return self.root / stage / key[:2] / f"{key}.json"

    def load(self, stage: str, key: str) -> Any | None:
        path = self.key_path(stage, key)
        if not path.exists():
            return None
        return load_json(path, default=None)

    def save(self, stage: str, key: str, payload: Any) -> Path:
        path = self.key_path(stage, key)
        save_json(path, payload)
        return path
