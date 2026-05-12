import json
from pathlib import Path
from time import perf_counter

from research.experiments.config.models import ExperimentConfig
from research.experiments.eval import bootstrap_metric_ci
from research.experiments.eval import build_prompt_sensitivity_table
from research.experiments.eval import build_vocab_sensitivity_curve
from research.experiments.eval import compute_image_potency
from research.experiments.eval import evaluate_graph_pair
from research.experiments.eval import graph_diagnostics
from research.experiments.eval import per_predicate_counts
from research.experiments.eval import summarize_per_image
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json


def _manifest_aliases(config: ExperimentConfig, run: RunContext) -> dict[str, str]:
    manifest_path = run.run_dir / "manifest.jsonl"
    if not manifest_path.exists() and config.paths.manifest_file:
        manifest_path = Path(config.paths.manifest_file)
    if not manifest_path.exists():
        return {}
    aliases: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        image_id = str(row.get("image_id"))
        image_path = Path(str(row.get("image_path")))
        aliases[image_id] = image_id
        aliases[str(image_path)] = image_id
        aliases[str(image_path.resolve())] = image_id
    return aliases


def _canonicalize_payload_keys(
    items: dict[str, object], aliases: dict[str, str]
) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in items.items():
        canonical = aliases.get(str(key)) or aliases.get(str(Path(key))) or str(key)
        out[canonical] = value
    return out


def _valid_ids(detections: object, *, normalize_ids: bool) -> set[str]:
    if not isinstance(detections, list):
        return set()
    ids: set[str] = set()
    for idx, row in enumerate(detections, start=1):
        if not isinstance(row, dict):
            continue
        object_id = row.get("object_id", idx)
        text = str(object_id).strip()
        if normalize_ids:
            import re

            match = re.search(r"(\d+)$", text)
            text = match.group(1) if match else text
        ids.add(text)
    return ids


async def run_scene_graph_evaluation(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting scene graph evaluation phase")
    stage_metrics = StageMetrics(stage="scene_graph_evaluation")

    gt_graphs = load_json(
        run.run_dir / config.paths.ground_truth_scene_graph_file, default={}
    )
    pred_graphs = load_json(
        run.run_dir / config.paths.draft_scene_graph_file, default={}
    )
    detections = load_json(run.run_dir / config.paths.detections_file, default={})
    context_rot = load_json(run.run_dir / config.paths.context_rot_file, default={})
    vocabulary = load_json(run.run_dir / config.paths.vocabulary_final_file, default={})

    if not isinstance(gt_graphs, dict) or not gt_graphs:
        raise RuntimeError(
            f"Ground truth file {run.run_dir / config.paths.ground_truth_scene_graph_file} is empty or invalid."
        )
    if not isinstance(pred_graphs, dict) or not pred_graphs:
        raise RuntimeError(
            f"Prediction file {run.run_dir / config.paths.draft_scene_graph_file} is empty or invalid."
        )

    aliases = _manifest_aliases(config, run)
    gt_items = _canonicalize_payload_keys(gt_graphs, aliases)
    pred_items = _canonicalize_payload_keys(pred_graphs, aliases)
    det_items = _canonicalize_payload_keys(
        detections if isinstance(detections, dict) else {}, aliases
    )

    keys = sorted(set(gt_items.keys()) | set(pred_items.keys()) | set(det_items.keys()))
    run.logger.info("Evaluation keyspace size=%d", len(keys))

    per_image: dict[str, dict] = {}
    eval_detections: dict[str, list[dict]] = {}
    missing_counts = {
        "missing_ground_truth": 0,
        "missing_prediction": 0,
        "missing_both": 0,
    }

    for key in keys:
        t0 = perf_counter()
        gt_payload = gt_items.get(key)
        pred_payload = pred_items.get(key)
        det_payload = det_items.get(key)

        gt_missing = gt_payload is None
        pred_missing = pred_payload is None

        if config.evaluation.missing_policy == "skip" and (gt_missing or pred_missing):
            if gt_missing and pred_missing:
                missing_counts["missing_both"] += 1
                stage_metrics.record_skipped("missing_both")
            elif gt_missing:
                missing_counts["missing_ground_truth"] += 1
                stage_metrics.record_skipped("missing_ground_truth")
            else:
                missing_counts["missing_prediction"] += 1
                stage_metrics.record_skipped("missing_prediction")
            continue
        if gt_missing and pred_missing:
            missing_counts["missing_both"] += 1
        elif gt_missing:
            missing_counts["missing_ground_truth"] += 1
        elif pred_missing:
            missing_counts["missing_prediction"] += 1

        if gt_payload is None:
            gt_payload = {"relationships": []}
        if pred_payload is None:
            pred_payload = {"relationships": []}

        row = evaluate_graph_pair(
            gt_payload,
            pred_payload,
            normalize_ids=config.evaluation.normalize_ids,
            normalize_relations=config.evaluation.normalize_relations,
            compute_ged=config.evaluation.compute_ged,
        )
        if config.evaluation.compute_per_predicate:
            row["per_predicate"] = per_predicate_counts(
                gt_payload,
                pred_payload,
                normalize_ids=config.evaluation.normalize_ids,
                normalize_relations=config.evaluation.normalize_relations,
            )
        row["diagnostics"] = graph_diagnostics(
            gt_payload,
            pred_payload,
            valid_object_ids=_valid_ids(
                det_payload, normalize_ids=config.evaluation.normalize_ids
            ),
            vocabulary=vocabulary if isinstance(vocabulary, dict) else {},
            normalize_ids=config.evaluation.normalize_ids,
            normalize_relations=config.evaluation.normalize_relations,
        )
        per_image[key] = row
        if isinstance(det_payload, list):
            eval_detections[key] = det_payload
        else:
            eval_detections[key] = []
        stage_metrics.record_ok(perf_counter() - t0)

    summary = summarize_per_image(
        per_image,
        include_per_predicate=config.evaluation.compute_per_predicate,
    )
    summary["missing"] = missing_counts
    if config.evaluation.bootstrap_rounds:
        summary["bootstrap_ci95"] = {
            "strict_triplet_f1": bootstrap_metric_ci(
                per_image,
                metric_group="strict_triplet",
                rounds=config.evaluation.bootstrap_rounds,
            ),
            "attribute_f1": bootstrap_metric_ci(
                per_image,
                metric_group="attribute",
                rounds=config.evaluation.bootstrap_rounds,
            ),
            "pair_f1": bootstrap_metric_ci(
                per_image,
                metric_group="pair",
                rounds=config.evaluation.bootstrap_rounds,
            ),
        }
    save_json(run.run_dir / config.paths.scene_graph_metrics_per_image_file, per_image)
    save_json(run.run_dir / config.paths.scene_graph_metrics_summary_file, summary)

    potency_payload = {}
    if config.evaluation.compute_potency:
        potency_per_image, potency_summary = compute_image_potency(
            detections=eval_detections,
            gt_graphs={key: gt_items.get(key) for key in per_image},
            pred_graphs={key: pred_items.get(key) for key in per_image},
            normalize_ids=config.evaluation.normalize_ids,
            normalize_relations=config.evaluation.normalize_relations,
        )
        potency_payload = {"per_image": potency_per_image, "summary": potency_summary}
        save_json(
            run.run_dir / config.paths.image_potency_metrics_file, potency_payload
        )

    sensitivity = {
        "vocabulary": build_vocab_sensitivity_curve(
            context_rot if isinstance(context_rot, dict) else {}
        ),
        "prompt": build_prompt_sensitivity_table(
            runs_root=config.paths.output_root / "runs"
        ),
    }
    save_json(run.run_dir / config.paths.sensitivity_metrics_file, sensitivity)

    stage_metrics.finish()
    save_json(
        run.run_dir / "metrics_scene_graph_evaluation_stage.json",
        stage_metrics.to_dict(),
    )

    run.logger.info(
        "Evaluation summary images=%d strict_f1=%.4f attr_f1=%.4f pair_f1=%.4f",
        summary.get("images_evaluated", 0),
        summary.get("strict_triplet_micro", {}).get("f1", 0.0),
        summary.get("attribute_micro", {}).get("f1", 0.0),
        summary.get("pair_micro", {}).get("f1", 0.0),
    )
    if potency_payload:
        run.logger.info(
            "Potency summary object_mean=%.3f pair_potential_mean=%.3f",
            potency_payload["summary"].get("object_count", {}).get("mean", 0.0),
            potency_payload["summary"].get("pair_potential", {}).get("mean", 0.0),
        )

    return {
        "per_image": per_image,
        "summary": summary,
        "potency": potency_payload,
        "sensitivity": sensitivity,
    }
