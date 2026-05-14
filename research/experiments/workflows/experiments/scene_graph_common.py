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
    rendered = template or ""
    for key, value in values.items():
        placeholder = "{" + key + "}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
    return rendered


def to_detection_objects(raw_rows: list[dict]):
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
    with BytesIO() as buf:
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def objects_for_prompt(detected_rows: list[dict]) -> list[dict]:
    return [
        {
            "id": det.get("object_id"),
            "label": det.get("label"),
            "bbox": det.get("bbox"),
        }
        for det in detected_rows
    ]


def vocabulary_for_prompt(vocabulary: dict, vocab_mode: str) -> dict | str | list[str]:
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
