import asyncio
from pathlib import Path
import random
from time import perf_counter

from tqdm.auto import tqdm

from research.experiments.adapters import ServerVLMAdapter
from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.config.models import ExperimentConfig
from research.experiments.eval import evaluate_graph_pair
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft

from .scene_graph_common import build_prompt_image
from .scene_graph_common import objects_for_prompt
from .scene_graph_common import pil_to_jpeg_bytes
from .scene_graph_common import render_template


def _frequency_order(items: list[str], counts: dict[str, int]) -> list[str]:
    return sorted(items, key=lambda item: (-int(counts.get(item, 0)), item))


def _semantic_order(items: list[str]) -> list[str]:
    groups: dict[str, list[str]] = {}
    for item in items:
        key = item.split("_", 1)[0] if "_" in item else item[:4]
        groups.setdefault(key, []).append(item)
    for values in groups.values():
        values.sort()
    ordered: list[str] = []
    while any(groups.values()):
        ordered.extend([groups[key].pop(0) for key in sorted(groups) if groups[key]])
    return ordered


def _slice_terms(
    predicates: list[str], attributes: list[str], size: int
) -> tuple[list[str], list[str]]:
    total = len(predicates) + len(attributes)
    if total <= size:
        return predicates, attributes
    pred_target = round(size * (len(predicates) / total)) if total else 0
    pred_target = min(len(predicates), max(0, pred_target))
    attr_target = min(len(attributes), size - pred_target)
    while pred_target + attr_target < size and pred_target < len(predicates):
        pred_target += 1
    while pred_target + attr_target < size and attr_target < len(attributes):
        attr_target += 1
    return predicates[:pred_target], attributes[:attr_target]


def _build_vocab_slices(
    vocab: dict, min_size: int, step: int, strategy: str, seed: int, rounds: int
) -> list[dict]:
    predicates_base = list(vocab.get("predicates", []))
    attributes_base = list(vocab.get("attributes", []))
    provenance = (
        vocab.get("provenance", {}) if isinstance(vocab.get("provenance"), dict) else {}
    )
    predicate_counts = provenance.get("predicate_counts", {})
    attribute_counts = provenance.get("attribute_counts", {})
    max_len = len(predicates_base) + len(attributes_base)
    sizes = list(range(min_size, max_len + 1, step))
    if not sizes or sizes[-1] != max_len:
        sizes.append(max_len)

    out: list[dict] = []
    for size in sizes:
        for round_idx in range(max(1, rounds)):
            predicates = predicates_base[:]
            attributes = attributes_base[:]
            if strategy in {"random", "random_drop"}:
                rng = random.Random(seed + round_idx)
                rng.shuffle(predicates)
                rng.shuffle(attributes)
            elif strategy == "frequency":
                predicates = _frequency_order(predicates, predicate_counts)
                attributes = _frequency_order(attributes, attribute_counts)
            elif strategy in {"semantic", "llm_drop"}:
                predicates = _semantic_order(predicates)
                attributes = _semantic_order(attributes)
            keep_predicates, keep_attributes = _slice_terms(
                predicates, attributes, size
            )
            out.append(
                {
                    "size": len(keep_predicates) + len(keep_attributes),
                    "requested_size": size,
                    "round": round_idx + 1,
                    "strategy": strategy,
                    "vocabulary": {
                        "predicates": keep_predicates,
                        "attributes": keep_attributes,
                    },
                }
            )
    return out


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
        base_url=config.draft_sgg_model.base_url,
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
        rounds=config.context_rot.rounds_per_size,
    )
    save_json(run.run_dir / "context_rot_vocab_slices.json", vocab_slices)

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
            objects = objects_for_prompt(detected_rows)
            render_values = {
                "objects": objects,
                "vocabulary": sliced_vocab,
                "caption": (
                    caption if config.prompting.include_caption_in_sgg_prompt else ""
                ),
            }
            system_prompt = render_template(
                config.draft_scene_graph.system_prompt, render_values
            )
            user_prompt = render_template(
                config.draft_scene_graph.user_prompt_template, render_values
            )
            if config.draft_scene_graph.vocab_mode == "open":
                system_prompt = system_prompt.replace(
                    "Use ONLY the values in the provided PREDICATES and ATTRIBUTES lists. Do not invent new ones.",
                    "Use concise relation and attribute names grounded in the image. Do not invent objects.",
                )
                user_prompt = user_prompt.replace(
                    "Allowed predicates and attributes: \n", ""
                )

            prompt_image, _ = build_prompt_image(
                image_path=path,
                detected_rows=detected_rows,
                config=config,
                painter=painter,
            )
            _, parsed = await vlm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=pil_to_jpeg_bytes(prompt_image),
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
    for sliced_record in slices_progress:
        sliced = sliced_record["vocabulary"]
        key = (
            f"{sliced_record['strategy']}_size_{sliced_record['size']}"
            f"_round_{sliced_record['round']}"
        )
        stats: dict[str, float | int] = {"images": 0, "relationship_count_sum": 0}
        if has_ground_truth:
            stats["triplet_f1_sum"] = 0.0
            stats["attribute_f1_sum"] = 0.0
            stats["pair_f1_sum"] = 0.0
        tasks = [evaluate_one(path, payload, sliced) for path, payload in sample_items]
        round_progress = tqdm(
            total=len(tasks),
            desc=key,
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
        stats["slice"] = sliced_record
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
