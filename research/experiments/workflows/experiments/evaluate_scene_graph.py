from pathlib import Path
from time import perf_counter

from research.experiments.config.models import ExperimentConfig
from research.experiments.eval import build_prompt_sensitivity_table
from research.experiments.eval import build_vocab_sensitivity_curve
from research.experiments.eval import compute_image_potency
from research.experiments.eval import evaluate_graph_pair
from research.experiments.eval import per_predicate_counts
from research.experiments.eval import summarize_per_image
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json


def _build_key_index(
    items: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    by_full = {str(Path(key)): value for key, value in items.items()}
    by_name: dict[str, object] = {}
    collisions: set[str] = set()
    for key, value in by_full.items():
        name = Path(key).name
        if name in by_name and by_name[name] is not value:
            collisions.add(name)
        else:
            by_name[name] = value
    for name in collisions:
        by_name.pop(name, None)
    return by_full, by_name


def _lookup(
    index_full: dict[str, object], index_name: dict[str, object], key: str
) -> object | None:
    full_key = str(Path(key))
    if full_key in index_full:
        return index_full[full_key]
    return index_name.get(Path(full_key).name)


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

    if not isinstance(gt_graphs, dict) or not gt_graphs:
        raise RuntimeError(
            f"Ground truth file {run.run_dir / config.paths.ground_truth_scene_graph_file} is empty or invalid."
        )
    if not isinstance(pred_graphs, dict) or not pred_graphs:
        raise RuntimeError(
            f"Prediction file {run.run_dir / config.paths.draft_scene_graph_file} is empty or invalid."
        )

    gt_full, gt_name = _build_key_index(gt_graphs)
    pred_full, pred_name = _build_key_index(pred_graphs)
    det_full, det_name = _build_key_index(
        detections if isinstance(detections, dict) else {}
    )

    keys = sorted(set(gt_full.keys()) | set(pred_full.keys()) | set(det_full.keys()))
    run.logger.info("Evaluation keyspace size=%d", len(keys))

    per_image: dict[str, dict] = {}
    eval_detections: dict[str, list[dict]] = {}

    for key in keys:
        t0 = perf_counter()
        gt_payload = _lookup(gt_full, gt_name, key)
        pred_payload = _lookup(pred_full, pred_name, key)
        det_payload = _lookup(det_full, det_name, key)

        gt_missing = gt_payload is None
        pred_missing = pred_payload is None

        if config.evaluation.missing_policy == "skip" and (gt_missing or pred_missing):
            if gt_missing and pred_missing:
                stage_metrics.record_skipped("missing_both")
            elif gt_missing:
                stage_metrics.record_skipped("missing_ground_truth")
            else:
                stage_metrics.record_skipped("missing_prediction")
            continue

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
    save_json(run.run_dir / config.paths.scene_graph_metrics_per_image_file, per_image)
    save_json(run.run_dir / config.paths.scene_graph_metrics_summary_file, summary)

    potency_payload = {}
    if config.evaluation.compute_potency:
        potency_per_image, potency_summary = compute_image_potency(
            detections=eval_detections,
            gt_graphs={key: _lookup(gt_full, gt_name, key) for key in per_image},
            pred_graphs={key: _lookup(pred_full, pred_name, key) for key in per_image},
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
