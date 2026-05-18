import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_paths(images_dir: Path):
    """Yield supported image files from a directory tree.

    Args:
        images_dir: Root directory scanned recursively.

    Yields:
        Image paths with supported extensions in stable sorted order.
    """
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def iter_manifest_image_paths(manifest_file: Path):
    """Yield image paths referenced by a manifest JSONL file.

    Args:
        manifest_file: JSONL file with image_path fields.

    Yields:
        Path objects for non-empty image_path values.
    """
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
    """Yield image paths from either an explicit manifest or an image directory.

    Args:
        images_dir: Directory fallback when no manifest is provided.
        manifest_file: Optional manifest JSONL path.

    Yields:
        Paths from manifest_file when present, otherwise discovered image paths.
    """
    if manifest_file is not None:
        yield from iter_manifest_image_paths(manifest_file)
        return
    yield from iter_image_paths(images_dir)
