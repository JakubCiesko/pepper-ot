from statistics import mean
from statistics import pstdev
from statistics import pvariance

from .normalization import canonicalize_edges
from .normalization import split_unary_binary


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _stats(values: list[float]) -> dict[str, float]:
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
    return {
        "count": len(values),
        "mean": mean(values),
        "var": pvariance(values) if len(values) > 1 else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p25": _percentile(values, 0.25),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
    }


def compute_image_potency(
    *,
    detections: dict[str, list[dict]],
    gt_graphs: dict[str, object],
    pred_graphs: dict[str, object],
    normalize_ids: bool = True,
    normalize_relations: bool = True,
) -> tuple[dict[str, dict], dict]:
    per_image: dict[str, dict] = {}

    for image_path, det_rows in detections.items():
        n_objects = len([row for row in det_rows if row.get("object_id") is not None])
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

        per_image[image_path] = {
            "n_objects": n_objects,
            "pair_potential": pair_potential,
            "n_rel_gt": n_rel_gt,
            "n_rel_pred": n_rel_pred,
            "n_attr_gt": n_attr_gt,
            "n_attr_pred": n_attr_pred,
            "relation_density_gt": _safe_div(n_rel_gt, pair_potential),
            "relation_density_pred": _safe_div(n_rel_pred, pair_potential),
            "effective_potency_gt": _safe_div(n_rel_gt, pair_potential),
            "effective_potency_pred": _safe_div(n_rel_pred, pair_potential),
            "attribute_potency_gt": _safe_div(n_attr_gt, n_objects),
            "attribute_potency_pred": _safe_div(n_attr_pred, n_objects),
        }

    object_counts = [float(row["n_objects"]) for row in per_image.values()]
    pair_potentials = [float(row["pair_potential"]) for row in per_image.values()]
    rel_density_gt = [float(row["relation_density_gt"]) for row in per_image.values()]
    rel_density_pred = [
        float(row["relation_density_pred"]) for row in per_image.values()
    ]
    attr_potency_gt = [float(row["attribute_potency_gt"]) for row in per_image.values()]
    attr_potency_pred = [
        float(row["attribute_potency_pred"]) for row in per_image.values()
    ]

    summary = {
        "images": len(per_image),
        "object_count": _stats(object_counts),
        "pair_potential": _stats(pair_potentials),
        "relation_density_gt": _stats(rel_density_gt),
        "relation_density_pred": _stats(rel_density_pred),
        "attribute_potency_gt": _stats(attr_potency_gt),
        "attribute_potency_pred": _stats(attr_potency_pred),
    }
    return per_image, summary
