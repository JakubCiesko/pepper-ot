import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_paths(images_dir: Path):
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def iter_manifest_image_paths(manifest_file: Path):
    with manifest_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = row.get("image_path")
            if image_path:
                yield Path(image_path)


def iter_config_image_paths(images_dir: Path, manifest_file: Path | None = None):
    if manifest_file is not None:
        yield from iter_manifest_image_paths(manifest_file)
        return
    yield from iter_image_paths(images_dir)
