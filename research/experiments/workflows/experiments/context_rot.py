import asyncio
from collections import Counter
from pathlib import Path
import random
from time import perf_counter

from tqdm.auto import tqdm
import yaml

from research.experiments.adapters import ServerVLMAdapter
from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.config.models import ExperimentConfig
from research.experiments.eval import evaluate_graph_pair
from research.experiments.eval.metrics_scene_graph import summarize_per_image
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft

from .scene_graph_common import build_prompt_image
from .scene_graph_common import objects_for_prompt
from .scene_graph_common import pil_to_jpeg_bytes
from .scene_graph_common import render_template
from .scene_graph_common import vocabulary_for_prompt


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


def _as_terms(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _load_manual_vocab_levels(levels_file: Path, full_vocab: dict) -> list[dict]:
    with levels_file.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    raw_levels = payload.get("levels", [])
    if not isinstance(raw_levels, list) or not raw_levels:
        raise RuntimeError(f"No vocabulary levels defined in {levels_file}")

    full_predicates = set(_as_terms(full_vocab.get("predicates")))
    full_attributes = set(_as_terms(full_vocab.get("attributes")))
    out: list[dict] = []
    for idx, raw in enumerate(raw_levels, start=1):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Vocabulary level {idx} is not an object")
        name = str(raw.get("name") or f"level_{idx:02d}").strip()
        predicates = _as_terms(raw.get("predicates"))
        attributes = _as_terms(raw.get("attributes"))
        if not name or not predicates and not attributes:
            raise RuntimeError(f"Vocabulary level {idx} is missing name or terms")

        predicate_map = {
            str(k).strip(): str(v).strip()
            for k, v in dict(raw.get("predicate_map") or {}).items()
            if str(k).strip() and str(v).strip()
        }
        attribute_map = {
            str(k).strip(): str(v).strip()
            for k, v in dict(raw.get("attribute_map") or {}).items()
            if str(k).strip() and str(v).strip()
        }

        bad_predicate_sources = sorted(set(predicate_map) - full_predicates)
        bad_attribute_sources = sorted(set(attribute_map) - full_attributes)
        if bad_predicate_sources:
            raise RuntimeError(
                f"Level {name} predicate_map sources are not in full predicates: "
                f"{bad_predicate_sources}"
            )
        if bad_attribute_sources:
            raise RuntimeError(
                f"Level {name} attribute_map sources are not in full attributes: "
                f"{bad_attribute_sources}"
            )
        bad_predicate_targets = sorted(set(predicate_map.values()) - set(predicates))
        bad_attribute_targets = sorted(set(attribute_map.values()) - set(attributes))
        if bad_predicate_targets:
            raise RuntimeError(
                f"Level {name} predicate_map targets are not in reduced predicates: "
                f"{bad_predicate_targets}"
            )
        if bad_attribute_targets:
            raise RuntimeError(
                f"Level {name} attribute_map targets are not in reduced attributes: "
                f"{bad_attribute_targets}"
            )

        out.append(
            {
                "name": name,
                "size": len(predicates) + len(attributes),
                "requested_size": raw.get(
                    "requested_size", len(predicates) + len(attributes)
                ),
                "round": int(raw.get("round", 1)),
                "strategy": str(raw.get("strategy", "manual")),
                "drop_unmapped": bool(raw.get("drop_unmapped", True)),
                "predicate_map": predicate_map,
                "attribute_map": attribute_map,
                "vocabulary": {
                    "predicates": predicates,
                    "attributes": attributes,
                },
            }
        )
    return out


def _relationship_rows(payload: object) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("relationships", "edges", "no_label_edges"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    if all(key in payload for key in ("sub", "rel", "obj")):
        return [payload]
    return []


def _remap_graph_payload(
    payload: object,
    *,
    vocabulary: dict,
    predicate_map: dict[str, str],
    attribute_map: dict[str, str],
    drop_unmapped: bool,
) -> dict:
    allowed_predicates = set(_as_terms(vocabulary.get("predicates")))
    allowed_attributes = set(_as_terms(vocabulary.get("attributes")))
    rows = []
    dropped = Counter()
    for row in _relationship_rows(payload):
        if not all(key in row for key in ("sub", "rel", "obj")):
            dropped["malformed"] += 1
            continue
        sub = str(row.get("sub")).strip()
        obj = str(row.get("obj")).strip()
        rel = str(row.get("rel")).strip().lower().replace(" ", "_")
        if not sub or not obj or not rel:
            dropped["empty"] += 1
            continue
        if sub == obj:
            mapped_rel = rel if rel in allowed_attributes else attribute_map.get(rel)
            if not mapped_rel:
                if drop_unmapped:
                    dropped["unmapped_attribute"] += 1
                    continue
                mapped_rel = rel
            if drop_unmapped and mapped_rel not in allowed_attributes:
                dropped["oov_attribute"] += 1
                continue
        else:
            mapped_rel = rel if rel in allowed_predicates else predicate_map.get(rel)
            if not mapped_rel:
                if drop_unmapped:
                    dropped["unmapped_predicate"] += 1
                    continue
                mapped_rel = rel
            if drop_unmapped and mapped_rel not in allowed_predicates:
                dropped["oov_predicate"] += 1
                continue
        rows.append({"sub": sub, "rel": mapped_rel, "obj": obj})
    unique_rows = []
    seen = set()
    for row in rows:
        key = (row["sub"], row["rel"], row["obj"])
        if key in seen:
            dropped["duplicate_after_remap"] += 1
            continue
        seen.add(key)
        unique_rows.append(row)
    return {"relationships": unique_rows, "dropped": dict(dropped)}


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
    if config.context_rot.levels_file is not None:
        vocab_slices = _load_manual_vocab_levels(
            config.context_rot.levels_file, vocabulary
        )
    else:
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

    async def evaluate_one(image_path: str, payload: dict, sliced_record: dict):
        t0 = perf_counter()
        try:
            sliced_vocab = sliced_record["vocabulary"]
            path = Path(image_path)
            if not path.exists():
                stage_metrics.record_skipped("missing_image_path")
                return {"relationship_count": 0, "parse_failed": 1}
            caption = str(payload.get("text", "")).strip()
            detected_rows = detections.get(image_path, [])
            objects = objects_for_prompt(detected_rows)
            prompt_vocabulary = vocabulary_for_prompt(
                sliced_vocab, config.draft_scene_graph.vocab_mode
            )
            render_values = {
                "objects": objects,
                "vocabulary": prompt_vocabulary,
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
            pred_payload = parsed.model_dump() if parsed else {"relationships": []}
            remapped_pred = _remap_graph_payload(
                pred_payload,
                vocabulary=sliced_vocab,
                predicate_map=sliced_record.get("predicate_map", {}),
                attribute_map=sliced_record.get("attribute_map", {}),
                drop_unmapped=bool(sliced_record.get("drop_unmapped", True)),
            )
            rel_count = len(remapped_pred["relationships"])
            out = {
                "image_path": image_path,
                "relationship_count": rel_count,
                "prompt_vocabulary": prompt_vocabulary,
                "raw_prediction": pred_payload,
                "remapped_prediction": {
                    "relationships": remapped_pred["relationships"],
                },
                "prediction_remap_dropped": remapped_pred["dropped"],
            }
            if has_ground_truth:
                gt_payload = ground_truth.get(image_path)
                if gt_payload is not None:
                    remapped_gt = _remap_graph_payload(
                        gt_payload,
                        vocabulary=sliced_vocab,
                        predicate_map=sliced_record.get("predicate_map", {}),
                        attribute_map=sliced_record.get("attribute_map", {}),
                        drop_unmapped=bool(sliced_record.get("drop_unmapped", True)),
                    )
                    pair_metrics = evaluate_graph_pair(
                        gt_payload={"relationships": remapped_gt["relationships"]},
                        pred_payload={"relationships": remapped_pred["relationships"]},
                        normalize_ids=config.evaluation.normalize_ids,
                        normalize_relations=config.evaluation.normalize_relations,
                        compute_ged=False,
                    )
                    out["remapped_ground_truth"] = {
                        "relationships": remapped_gt["relationships"],
                    }
                    out["ground_truth_remap_dropped"] = remapped_gt["dropped"]
                    out["pair_metrics"] = pair_metrics
                    out["triplet_f1"] = pair_metrics["strict_triplet"]["f1"]
                    out["attribute_f1"] = pair_metrics["attribute"]["f1"]
                    out["pair_ordered_f1"] = pair_metrics["pair_ordered"]["f1"]
                    out["pair_unordered_f1"] = pair_metrics["pair_unordered"]["f1"]
                else:
                    out["remapped_ground_truth"] = {"relationships": []}
                    out["ground_truth_remap_dropped"] = {}
                    out["pair_metrics"] = evaluate_graph_pair(
                        gt_payload={"relationships": []},
                        pred_payload={"relationships": remapped_pred["relationships"]},
                        normalize_ids=config.evaluation.normalize_ids,
                        normalize_relations=config.evaluation.normalize_relations,
                        compute_ged=False,
                    )
                    out["triplet_f1"] = 0.0
                    out["attribute_f1"] = 0.0
                    out["pair_ordered_f1"] = 0.0
                    out["pair_unordered_f1"] = 0.0
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
        key = str(
            sliced_record.get(
                "name",
                (f"vocab_{sliced_record['size']}" f"_round_{sliced_record['round']}"),
            )
        )
        stats: dict[str, float | int] = {"images": 0, "relationship_count_sum": 0}
        per_image: dict[str, dict] = {}
        raw_predictions: dict[str, dict] = {}
        remapped_predictions: dict[str, dict] = {}
        remapped_ground_truth: dict[str, dict] = {}
        prompt_vocabularies: dict[str, object] = {}
        tasks = [
            evaluate_one(path, payload, sliced_record) for path, payload in sample_items
        ]
        round_progress = tqdm(
            total=len(tasks),
            desc=key,
            unit="img",
            leave=False,
        )
        for future in asyncio.as_completed(tasks):
            item = await future
            image_path = str(item.get("image_path", ""))
            stats["images"] += 1
            stats["relationship_count_sum"] += item["relationship_count"]
            if image_path:
                prompt_vocabularies[image_path] = item.get("prompt_vocabulary")
                raw_predictions[image_path] = item.get(
                    "raw_prediction", {"relationships": []}
                )
                remapped_predictions[image_path] = item.get(
                    "remapped_prediction", {"relationships": []}
                )
                if "remapped_ground_truth" in item:
                    remapped_ground_truth[image_path] = item.get(
                        "remapped_ground_truth", {"relationships": []}
                    )
                if "pair_metrics" in item:
                    per_image[image_path] = item["pair_metrics"]
            round_progress.update(1)
        round_progress.close()
        stats["slice"] = sliced_record
        stats["relationship_count_avg"] = (
            stats["relationship_count_sum"] / stats["images"]
            if stats["images"]
            else 0.0
        )
        if has_ground_truth:
            summary = summarize_per_image(
                per_image,
                include_per_predicate=config.evaluation.compute_per_predicate,
            )
            stats["metrics_summary"] = summary
            stats["triplet_f1_avg"] = summary["strict_triplet_micro"]["f1"]
            stats["attribute_f1_avg"] = summary["attribute_micro"]["f1"]
            stats["pair_ordered_f1_avg"] = summary["pair_ordered_micro"]["f1"]
            stats["pair_unordered_f1_avg"] = summary["pair_unordered_micro"]["f1"]
        level_dir = run.run_dir / "context_rot_levels" / key
        level_dir.mkdir(parents=True, exist_ok=True)
        save_json(level_dir / "vocabulary.json", sliced)
        save_json(level_dir / "prompt_vocabularies.json", prompt_vocabularies)
        save_json(level_dir / "raw_predictions.json", raw_predictions)
        save_json(level_dir / "remapped_predictions.json", remapped_predictions)
        if has_ground_truth:
            save_json(level_dir / "remapped_ground_truth.json", remapped_ground_truth)
            save_json(level_dir / "metrics_scene_graph_per_image.json", per_image)
            save_json(
                level_dir / "metrics_scene_graph_summary.json",
                stats["metrics_summary"],
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
