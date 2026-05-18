import numpy as np

from .normalization import canonicalize_edges
from .normalization import split_unary_binary


def _safe_div(num: float, den: float) -> float:
    """Divide two numbers and return 0.0 for an empty denominator."""
    return float(num / den) if den else 0.0


def _stats(values: list[float]) -> dict[str, float]:
    """Summarize a numeric vector for potency reports.

    Args:
        values: Numeric observations across images.

    Returns:
        Count, mean, variance, standard deviation, min, max, and selected
        percentiles. Empty inputs return zeros for every field.
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "var": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def compute_image_potency(
    *,
    detections: dict[str, list[dict]],
    gt_graphs: dict[str, object],
    pred_graphs: dict[str, object],
    normalize_ids: bool = True,
    normalize_relations: bool = True,
) -> tuple[dict[str, dict], dict]:
    """Compute relation and attribute density diagnostics per image.

    Args:
        detections: Mapping from image key to detection rows. Rows with an
            object_id contribute to object count.
        gt_graphs: Mapping from image key to ground-truth graph payload.
        pred_graphs: Mapping from image key to predicted graph payload.
        normalize_ids: Normalize object IDs before graph counting.
        normalize_relations: Normalize relation and attribute labels.

    Returns:
        Tuple of per-image potency rows and a summary dictionary. Per-image rows
        include object count, ordered-pair potential, GT/pred relation counts,
        GT/pred attribute counts, relation density, effective potency, and
        attribute potency. The summary contains descriptive statistics for the
        main potency fields.
    """
    per_image: dict[str, dict] = {}

    for image_path, det_rows in detections.items():
        n_objects = sum(row.get("object_id") is not None for row in det_rows)
        pair_potential = max(0, n_objects * (n_objects - 1))

        gt_edges = canonicalize_edges(
            gt_graphs.get(image_path),
            normalize_ids=normalize_ids,
            normalize_relations=normalize_relations,
        )
        pred_edges = canonicalize_edges(
            pred_graphs.get(image_path),
            normalize_ids=normalize_ids,
            normalize_relations=normalize_relations,
        )

        gt_unary, gt_binary = split_unary_binary(gt_edges)
        pred_unary, pred_binary = split_unary_binary(pred_edges)

        n_rel_gt = len(gt_binary)
        n_rel_pred = len(pred_binary)
        n_attr_gt = len(gt_unary)
        n_attr_pred = len(pred_unary)

        rel_density_gt = _safe_div(n_rel_gt, pair_potential)
        rel_density_pred = _safe_div(n_rel_pred, pair_potential)

        per_image[image_path] = {
            "n_objects": n_objects,
            "pair_potential": pair_potential,
            "n_rel_gt": n_rel_gt,
            "n_rel_pred": n_rel_pred,
            "n_attr_gt": n_attr_gt,
            "n_attr_pred": n_attr_pred,
            "relation_density_gt": rel_density_gt,
            "relation_density_pred": rel_density_pred,
            "effective_potency_gt": rel_density_gt,
            "effective_potency_pred": rel_density_pred,
            "attribute_potency_gt": _safe_div(n_attr_gt, n_objects),
            "attribute_potency_pred": _safe_div(n_attr_pred, n_objects),
        }

    summary = {
        "images": len(per_image),
        "object_count": _stats([v["n_objects"] for v in per_image.values()]),
        "pair_potential": _stats([v["pair_potential"] for v in per_image.values()]),
        "relation_density_gt": _stats(
            [v["relation_density_gt"] for v in per_image.values()]
        ),
        "relation_density_pred": _stats(
            [v["relation_density_pred"] for v in per_image.values()]
        ),
        "attribute_potency_gt": _stats(
            [v["attribute_potency_gt"] for v in per_image.values()]
        ),
        "attribute_potency_pred": _stats(
            [v["attribute_potency_pred"] for v in per_image.values()]
        ),
    }

    return per_image, summary
