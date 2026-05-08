import asyncio
import random
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from research.experiments.adapters import ServerVLMAdapter
from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.adapters.utils import resize_pil
from research.experiments.config.models import ExperimentConfig
from research.experiments.eval import evaluate_graph_pair
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft


def _build_vocab_slices(vocab: dict, min_size: int, step: int, strategy: str, seed: int) -> list[dict]:
    predicates = list(vocab.get("predicates", []))
    attributes = list(vocab.get("attributes", []))
    if strategy in {"random", "random_drop"}:
        rng = random.Random(seed)
        rng.shuffle(predicates)
        rng.shuffle(attributes)
    max_len = len(predicates) + len(attributes)
    sizes = list(range(min_size, max_len + 1, step))
    if not sizes or sizes[-1] != max_len:
        sizes.append(max_len)
    out: list[dict] = []
    for size in sizes:
        # TODO: Why this split? 
        keep_pred = min(len(predicates), max(1, size // 2))
        keep_attr = min(len(attributes), size - keep_pred)
        out.append(
            {
                "predicates": predicates[:keep_pred],
                "attributes": attributes[:keep_attr],
            }
        )
    return out


def _render_template(template: str, values: dict) -> str:
    rendered = template or ""
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _to_detection_objects(raw_rows: list[dict]):
    from app.inference.types import InferenceDetectionObject

    objects = []
    for idx, row in enumerate(raw_rows, start=1):
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        objects.append(
            InferenceDetectionObject(
                class_id=int(row.get("class_id", idx)),
                label=str(row.get("label", "object")),
                confidence=float(row.get("confidence", 0.0)),
                bbox=[float(v) for v in bbox],
                object_id=row.get("object_id", idx),
            )
        )
    return objects


def _pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    with BytesIO() as buf:
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()


async def run_context_rot(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting context-rot phase")
    stage_metrics = StageMetrics(stage="context_rot")
    descriptions = load_json(run.run_dir / config.paths.descriptions_file, default={})
    detections = load_json(run.run_dir / config.paths.detections_file, default={})
    vocabulary = load_json(run.run_dir / config.paths.vocabulary_final_file, default={})
    ground_truth = load_json(
        run.run_dir / config.paths.ground_truth_scene_graph_file, default={}
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

    random.seed(config.seed)
    vocab_slices = _build_vocab_slices(
        vocabulary,
        min_size=config.context_rot.min_vocab_size,
        step=config.context_rot.step,
        strategy=config.context_rot.strategy,
        seed=config.seed,
    )

    results: dict[str, dict] = {}
    sample_items = list(descriptions.items())
    has_ground_truth = (
        config.context_rot.evaluate_against_ground_truth
        and isinstance(ground_truth, dict)
        and bool(ground_truth)
    )

    async def evaluate_one(image_path: str, payload: dict, sliced_vocab: dict):
        t0 = perf_counter()
        try:
            path = Path(image_path)
            if not path.exists():
                stage_metrics.record_skipped("missing_image_path")
                return {"relationship_count": 0, "parse_failed": 1}
            caption = str(payload.get("text", "")).strip()
            detected_rows = detections.get(image_path, [])
            objects = [
                {"id": det.get("object_id"), "label": det.get("label"), "bbox": det.get("bbox")}
                for det in detected_rows
            ]
            # TODO: clearly split predicates and attributes.
            render_values = {
                "objects": objects,
                "vocabulary": sliced_vocab,
                "caption": caption if config.prompting.include_caption_in_sgg_prompt else "",
            }
            system_prompt = _render_template(
                config.draft_scene_graph.system_prompt, render_values
            )
            user_prompt = _render_template(
                config.draft_scene_graph.user_prompt_template, render_values
            )

            with Image.open(path) as img:
                pil_image = img.convert("RGB")
            det_objects = _to_detection_objects(detected_rows)
            som_image_np = painter.paint(
                np.asarray(pil_image),
                det_objects,
                bbox=config.draft_scene_graph.som_show_bbox,
                mask=config.draft_scene_graph.som_show_mask,
                polygon=config.draft_scene_graph.som_show_polygon,
                class_names=config.draft_scene_graph.som_show_labels,
            )
            prompt_image = (
                Image.fromarray(som_image_np.astype(np.uint8))
                if config.draft_scene_graph.use_som_image
                else pil_image
            )
            if config.draft_scene_graph.max_image_size:
                prompt_image = resize_pil(
                    prompt_image, config.draft_scene_graph.max_image_size
                )
            _, parsed = await vlm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=_pil_to_jpeg_bytes(prompt_image),
                output_schema=SceneGraphDraft,
            )
            rel_count = len(parsed.relationships) if parsed else 0
            out = {"relationship_count": rel_count}
            if has_ground_truth:
                gt_payload = ground_truth.get(image_path)
                if gt_payload is not None and parsed is not None:
                    pair_metrics = evaluate_graph_pair(
                        gt_payload=gt_payload,
                        pred_payload=parsed.model_dump(),
                        normalize_ids=config.evaluation.normalize_ids,
                        normalize_relations=config.evaluation.normalize_relations,
                        compute_ged=False,
                    )
                    out["triplet_f1"] = pair_metrics["strict_triplet"]["f1"]
                    out["attribute_f1"] = pair_metrics["attribute"]["f1"]
                    out["pair_f1"] = pair_metrics["pair"]["f1"]
                else:
                    out["triplet_f1"] = 0.0
                    out["attribute_f1"] = 0.0
                    out["pair_f1"] = 0.0
            if parsed is None:
                stage_metrics.record_failed(
                    "structured_parse_missing", perf_counter() - t0
                )
            else:
                stage_metrics.record_ok(perf_counter() - t0)
            return out
        except Exception:
            stage_metrics.record_failed(
                "context_rot_request_error", perf_counter() - t0
            )
            return {"relationship_count": 0}

    slices_progress = tqdm(vocab_slices, desc="context_rot_slices", unit="slice")
    for sliced in slices_progress:
        key = f"vocab_{len(sliced.get('predicates', [])) + len(sliced.get('attributes', []))}"
        stats: dict[str, float | int] = {"images": 0, "relationship_count_sum": 0}
        if has_ground_truth:
            stats["triplet_f1_sum"] = 0.0
            stats["attribute_f1_sum"] = 0.0
            stats["pair_f1_sum"] = 0.0
        for _ in range(config.context_rot.rounds_per_size):
            tasks = [
                evaluate_one(path, payload, sliced) for path, payload in sample_items
            ]
            round_progress = tqdm(
                total=len(tasks),
                desc=f"{key}_round",
                unit="img",
                leave=False,
            )
            for future in asyncio.as_completed(tasks):
                item = await future
                stats["images"] += 1
                stats["relationship_count_sum"] += item["relationship_count"]
                if has_ground_truth:
                    stats["triplet_f1_sum"] += float(item.get("triplet_f1", 0.0))
                    stats["attribute_f1_sum"] += float(item.get("attribute_f1", 0.0))
                    stats["pair_f1_sum"] += float(item.get("pair_f1", 0.0))
                round_progress.update(1)
            round_progress.close()
        stats["relationship_count_avg"] = (
            stats["relationship_count_sum"] / stats["images"]
            if stats["images"]
            else 0.0
        )
        if has_ground_truth:
            stats["triplet_f1_avg"] = (
                stats["triplet_f1_sum"] / stats["images"] if stats["images"] else 0.0
            )
            stats["attribute_f1_avg"] = (
                stats["attribute_f1_sum"] / stats["images"] if stats["images"] else 0.0
            )
            stats["pair_f1_avg"] = (
                stats["pair_f1_sum"] / stats["images"] if stats["images"] else 0.0
            )
        results[key] = stats
        slices_progress.set_postfix(
            avg_rel=round(stats["relationship_count_avg"], 2),
            images=stats["images"],
        )
    slices_progress.close()

    save_json(run.run_dir / config.paths.context_rot_file, results)
    stage_metrics.finish()
    save_json(run.run_dir / "metrics_context_rot.json", stage_metrics.to_dict())
    run.logger.info("Saved context-rot results")
    run.logger.info(
        "Context-rot summary ok=%d failed=%d skipped=%d duration=%.3fs",
        stage_metrics.ok,
        stage_metrics.failed,
        stage_metrics.skipped,
        stage_metrics.duration_s,
    )
    return results
