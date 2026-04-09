import asyncio
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from research.experiments.adapters import ServerVLMAdapter
from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.adapters.utils import resize_pil
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft


def _render_template(template: str, values: dict[str, Any]) -> str:
    rendered = template or ""
    for key, value in values.items():
        placeholder = "{" + key + "}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
    return rendered


def _to_detection_objects(raw_rows: list[dict]):
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


def _pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    with BytesIO() as buf:
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()


async def run_draft_scene_graph(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting draft scene graph phase")
    stage_metrics = StageMetrics(stage="draft_scene_graph")
    descriptions = load_json(run.run_dir / config.paths.descriptions_file, default={})
    run.logger.info(
        "Loaded descriptions for %d images from file %s",
        len(descriptions),
        run.run_dir / config.paths.descriptions_file,
    )
    detections = load_json(run.run_dir / config.paths.detections_file, default={})
    run.logger.info(
        "Loaded detections for %d images from file %s",
        len(detections),
        run.run_dir / config.paths.detections_file,
    )
    vocabulary = load_json(run.run_dir / config.paths.vocabulary_final_file, default={})
    run.logger.info(
        "Loaded vocabulary with %d predicates, %d attributes from file %s",
        len(vocabulary.get("predicates", [])),
        len(vocabulary.get("attributes", [])),
        run.run_dir / config.paths.vocabulary_final_file,
    )

    if not descriptions or not vocabulary:
        raise RuntimeError(
            "Descriptions or vocabulary missing. Run previous phases first."
        )

    ensure_server_app_importable()
    from app.inference.scene_graph.som import SoMPainter

    vlm = ServerVLMAdapter(
        provider=config.draft_sgg_model.provider,
        model_id=config.draft_sgg_model.model_id,
        structured_mode=config.draft_sgg_model.structured_mode,
        device=config.draft_scene_graph.som_device,
    )

    painter = SoMPainter(
        line_thickness=config.draft_scene_graph.som_line_thickness,
        color_lookup=config.draft_scene_graph.som_color_lookup,
        mask_opacity=config.draft_scene_graph.som_mask_opacity,
        mask_backend=config.draft_scene_graph.som_mask_backend,
        device=config.draft_scene_graph.som_device,
    )

    som_dir = run.run_dir / config.draft_scene_graph.som_output_dir
    if config.draft_scene_graph.save_som_images:
        som_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.draft_scene_graph.max_concurrent_batches)
    drafts: dict[str, dict] = {}
    progress = tqdm(total=len(descriptions), desc="draft_sgg", unit="img")

    async def process_one(image_path: str, payload: dict):
        t0 = perf_counter()
        async with semaphore:
            try:
                path = Path(image_path)
                if not path.exists():
                    stage_metrics.record_skipped("missing_image_path")
                    drafts[image_path] = {"relationships": []}
                    return

                caption = str(payload.get("text", "")).strip()
                detected_rows = detections.get(image_path, [])
                objects = [
                    {
                        "id": det.get("object_id"),
                        "label": det.get("label"),
                        "bbox": det.get("bbox"),
                    }
                    for det in detected_rows
                ]

                render_values = {
                    "caption": (
                        caption
                        if config.prompting.include_caption_in_sgg_prompt
                        else ""
                    ),
                    "vocabulary": vocabulary,
                    "objects": objects,
                }
                system_prompt = _render_template(
                    config.draft_scene_graph.system_prompt, render_values
                )
                user_prompt = _render_template(
                    config.draft_scene_graph.user_prompt_template, render_values
                )

                with Image.open(path) as img:
                    pil_image = img.convert("RGB")

                image_np = np.asarray(pil_image)
                det_objects = _to_detection_objects(detected_rows)
                som_image_np = painter.paint(
                    image_np,
                    det_objects,
                    bbox=config.draft_scene_graph.som_show_bbox,
                    mask=config.draft_scene_graph.som_show_mask,
                    polygon=config.draft_scene_graph.som_show_polygon,
                    class_names=config.draft_scene_graph.som_show_labels,
                )
                som_image = Image.fromarray(som_image_np.astype(np.uint8))

                som_path: str | None = None
                if config.draft_scene_graph.save_som_images:
                    som_file = som_dir / f"som_{path.name}"
                    som_image.save(som_file)
                    som_path = str(som_file)
                if config.draft_scene_graph.max_image_size:
                    som_image = resize_pil(
                        som_image, config.draft_scene_graph.max_image_size
                    )
                raw_text, parsed = await vlm.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_bytes=_pil_to_jpeg_bytes(som_image),
                    output_schema=SceneGraphDraft,
                )
                parsed_payload = (
                    parsed.model_dump() if parsed else {"relationships": []}
                )

                drafts[image_path] = {
                    "image_path": image_path,
                    "som_image_path": som_path,
                    "caption": caption,
                    "objects": objects,
                    "vocabulary": vocabulary,
                    **parsed_payload,
                }
                if config.draft_scene_graph.include_raw_response:
                    drafts[image_path]["raw_response"] = raw_text
                if parsed is None:
                    stage_metrics.record_failed(
                        "structured_parse_missing", perf_counter() - t0
                    )
                else:
                    stage_metrics.record_ok(perf_counter() - t0)
            except Exception:
                drafts[image_path] = {"relationships": []}
                stage_metrics.record_failed(
                    "draft_generation_error", perf_counter() - t0
                )
            finally:
                progress.update(1)

    await asyncio.gather(
        *(process_one(path, payload) for path, payload in descriptions.items())
    )
    progress.close()
    stage_metrics.finish()
    save_json(run.run_dir / config.paths.draft_scene_graph_file, drafts)
    save_json(run.run_dir / "metrics_draft_scene_graph.json", stage_metrics.to_dict())
    run.logger.info("Saved draft scene graphs for %d images", len(drafts))
    run.logger.info(
        "Draft SGG summary ok=%d failed=%d skipped=%d duration=%.3fs",
        stage_metrics.ok,
        stage_metrics.failed,
        stage_metrics.skipped,
        stage_metrics.duration_s,
    )
    return drafts
