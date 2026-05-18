from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.adapters.utils import resize_pil
from research.experiments.config.models import ExperimentConfig


def render_template(template: str, values: dict[str, Any]) -> str:
    """Replace simple `{name}` placeholders in a prompt template.

    Args:
        template: Prompt template text. Empty or None-like values render as an
            empty string.
        values: Placeholder names and replacement values.

    Returns:
        Rendered prompt text. Placeholders without a matching value are left
        unchanged.
    """
    rendered = template or ""
    for key, value in values.items():
        placeholder = "{" + key + "}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
    return rendered


def to_detection_objects(raw_rows: list[dict]):
    """Convert serialized detection rows into server inference objects.

    Args:
        raw_rows: Detection dictionaries containing bbox, label, confidence,
            optional class_id, and optional object_id.

    Returns:
        List of InferenceDetectionObject instances suitable for the server SoM
        painter. Rows without a four-value bbox are skipped.

    Side Effects:
        Ensures the server application package is importable before importing
        the inference type.
    """
    ensure_server_app_importable()
    from app.inference.types import InferenceDetectionObject

    objects: list[InferenceDetectionObject] = []
    for idx, row in enumerate(raw_rows, start=1):
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        object_id = row.get("object_id")
        if object_id is None:
            object_id = idx
        objects.append(
            InferenceDetectionObject(
                class_id=int(row.get("class_id", idx)),
                label=str(row.get("label", "object")),
                confidence=float(row.get("confidence", 0.0)),
                bbox=[float(v) for v in bbox],
                object_id=object_id,
            )
        )
    return objects


def pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image as high-quality RGB JPEG bytes.

    Args:
        image: PIL image to send to a VLM adapter.

    Returns:
        JPEG-encoded bytes after converting the image to RGB.
    """
    with BytesIO() as buf:
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def objects_for_prompt(detected_rows: list[dict]) -> list[dict]:
    """Build the compact object list inserted into SGG prompts.

    Args:
        detected_rows: Detection rows loaded from the run detection artifact.

    Returns:
        List of dictionaries with id, label, and bbox fields. The function keeps
        the detector identifiers as-is so prompts match downstream evaluation
        object IDs.
    """
    return [
        {
            "id": det.get("object_id"),
            "label": det.get("label"),
            "bbox": det.get("bbox"),
        }
        for det in detected_rows
    ]


def vocabulary_for_prompt(vocabulary: dict, vocab_mode: str) -> dict | str | list[str]:
    """Format vocabulary according to the configured prompt mode.

    Args:
        vocabulary: Final or sliced vocabulary with predicates and attributes.
        vocab_mode: One of open, soft, list, or the default structured mode.

    Returns:
        Empty string for open mode, a soft-guidance wrapper for soft mode, a
        shuffled flat term list for list mode, or the original vocabulary
        dictionary for structured modes.
    """
    import random

    if vocab_mode == "open":
        return ""
    if vocab_mode == "soft":
        return {"suggested": vocabulary, "mode": "soft_guidance"}
    if vocab_mode == "list":
        vocab = []
        predicates = vocabulary.get("predicates")
        attributes = vocabulary.get("attributes")
        if predicates:
            vocab.extend(predicates)
        if attributes:
            vocab.extend(attributes)
        random.shuffle(vocab)
        return vocab
    return vocabulary


def build_prompt_image(
    *,
    image_path: Path,
    detected_rows: list[dict],
    config: ExperimentConfig,
    painter,
) -> tuple[Image.Image, Image.Image | None]:
    """Load and optionally paint the image used for draft SGG prompting.

    Args:
        image_path: Source image path from the run artifacts.
        detected_rows: Detection rows used to draw SoM marks.
        config: Experiment configuration controlling visual mode, SoM overlays,
            and maximum image size.
        painter: Server SoMPainter instance used when visual_mode is som.

    Returns:
        A tuple of the prompt image sent to the model and the full SoM image.
        The SoM image is None when raw visual mode is used.

    Side Effects:
        Reads the image from disk and may invoke the SoM painter with masks,
        boxes, polygons, and labels according to config.
    """
    with Image.open(image_path) as img:
        pil_image = img.convert("RGB")

    som_image: Image.Image | None = None
    visual_mode = config.draft_scene_graph.visual_mode
    if not config.draft_scene_graph.use_som_image:
        visual_mode = "raw"

    if visual_mode == "som":
        som_image_np = painter.paint(
            np.asarray(pil_image),
            to_detection_objects(detected_rows),
            bbox=config.draft_scene_graph.som_show_bbox,
            mask=config.draft_scene_graph.som_show_mask,
            polygon=config.draft_scene_graph.som_show_polygon,
            class_names=config.draft_scene_graph.som_show_labels,
            grab_cut_scale=config.draft_scene_graph.som_grab_cut_scale,
            grab_cut_iter_count=config.draft_scene_graph.som_grab_cut_iter_count,
            use_roi_grab_cut=config.draft_scene_graph.som_use_roi_grab_cut,
            max_mask_workers=config.draft_scene_graph.som_max_mask_workers,
        )
        som_image = Image.fromarray(som_image_np.astype(np.uint8))
        prompt_image = som_image
    else:
        prompt_image = pil_image

    if config.draft_scene_graph.max_image_size:
        prompt_image = resize_pil(prompt_image, config.draft_scene_graph.max_image_size)
    return prompt_image, som_image
