from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil

from research.experiments.io.dataset import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class ManifestRow:
    """One image row in an experiment manifest JSONL file.

    Attributes:
        image_id: Stable ID used to align artifacts and metrics.
        image_path: Absolute or relative path to the image file.
        dataset: Dataset name used for provenance and reporting.
        split: Dataset split label, usually eval.
        tags: Optional labels for filtering or provenance.
        scene_graph_source: Optional upstream source for ground-truth graphs.
    """

    image_id: str
    image_path: str
    dataset: str
    split: str = "eval"
    tags: list[str] | None = None
    scene_graph_source: str | None = None

    def to_json(self) -> dict:
        """Serialize the manifest row without None-valued fields."""
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def _iter_images(images_dir: Path) -> Iterable[Path]:
    """Yield supported image files from a directory tree in stable order.

    Args:
        images_dir: Root directory to scan recursively.

    Returns:
        Resolved image paths with extensions supported by the experiment IO
        layer.
    """
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path.resolve()


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load manifest rows from JSONL.

    Args:
        path: Manifest file containing one JSON object per non-empty line.

    Returns:
        List of ManifestRow instances in file order.
    """
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(ManifestRow(**payload))
    return rows


def save_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    """Write manifest rows to JSONL.

    Args:
        path: Destination JSONL path.
        rows: Manifest rows to serialize.

    Side Effects:
        Creates the parent directory and writes one sorted-key JSON object per
        row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_local_manifest(
    *,
    images_dir: Path,
    out: Path,
    dataset: str = "local",
    split: str = "eval",
    max_samples: int | None = None,
    seed: int = 42,
    tags: list[str] | None = None,
) -> list[ManifestRow]:
    """Create a manifest from local image files.

    Args:
        images_dir: Directory scanned recursively for supported image files.
        out: Manifest JSONL path to write.
        dataset: Dataset label used in image IDs and row metadata.
        split: Split label written to each row.
        max_samples: Optional maximum number of images to keep.
        seed: Sampling seed used when max_samples truncates the image set.
        tags: Optional tags assigned to every row.

    Returns:
        Manifest rows written to out.

    Side Effects:
        Writes the manifest JSONL file.
    """
    image_paths = list(_iter_images(images_dir))
    if max_samples is not None and len(image_paths) > max_samples:
        rng = random.Random(seed)
        image_paths = sorted(rng.sample(image_paths, max_samples))
    rows = [
        ManifestRow(
            image_id=f"{dataset}_{idx:05d}",
            image_path=str(path),
            dataset=dataset,
            split=split,
            tags=tags or [dataset],
        )
        for idx, path in enumerate(image_paths, start=1)
    ]
    save_manifest(out, rows)
    return rows


def write_gqa_manifest(
    *,
    out: Path,
    image_root: Path,
    max_samples: int = 10,
    seed: int = 42,
    split: str = "eval",
) -> list[ManifestRow]:
    """Create a manifest by streaming samples from the GQA scene graph dataset.

    Args:
        out: Manifest JSONL path to write.
        image_root: Directory where sampled images are materialized.
        max_samples: Maximum number of streamed samples to inspect.
        seed: Reserved for API symmetry with local manifest creation.
        split: Split label written to manifest rows.

    Returns:
        Manifest rows for samples whose images were available and written.

    Raises:
        RuntimeError: If the optional datasets package is not installed.

    Side Effects:
        Creates image_root, saves or copies sampled JPEG images, and writes the
        manifest JSONL file.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional Hugging Face datasets package to create a GQA "
            "manifest: pip install datasets"
        ) from exc

    dataset = load_dataset("Voxel51/GQA-Scene-Graph", split="train", streaming=True)
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = []
    for idx, item in enumerate(dataset):
        if idx >= max_samples:
            break
        image = item.get("image")
        image_id = str(item.get("id") or item.get("image_id") or f"gqa_{idx:05d}")
        image_path = image_root / f"{image_id}.jpg"
        if image is None:
            continue
        if hasattr(image, "save"):
            image.convert("RGB").save(image_path, format="JPEG", quality=95)
        else:
            source = Path(str(image))
            if source.exists():
                shutil.copyfile(source, image_path)
            else:
                continue
        rows.append(
            ManifestRow(
                image_id=image_id,
                image_path=str(image_path.resolve()),
                dataset="gqa",
                split=split,
                tags=["gqa", "scene_graph"],
                scene_graph_source="Voxel51/GQA-Scene-Graph",
            )
        )
    save_manifest(out, rows)
    return rows
