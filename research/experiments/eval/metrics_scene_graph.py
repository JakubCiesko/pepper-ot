from collections import Counter
from dataclasses import dataclass
import random

from .normalization import CanonicalEdge
from .normalization import canonicalize_edges
from .normalization import normalize_node
from .normalization import normalize_relation
from .normalization import split_unary_binary


def _safe_div(num: float, den: float) -> float:
    """Divide two numbers and return 0.0 for an empty denominator."""
    return (num / den) if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Compute precision, recall, and F1 from true/false positive counts.

    Args:
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.

    Returns:
        Dictionary containing tp, fp, fn, precision, recall, and f1.
    """
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def _counter_prf(gt: Counter, pred: Counter) -> dict[str, float | int]:
    """Compute PRF for multisets represented as Counters.

    Args:
        gt: Ground-truth item counts.
        pred: Predicted item counts.

    Returns:
        Precision/recall/F1 dictionary where duplicates are counted by multiset
        intersection and difference.
    """
    tp = int(sum((gt & pred).values()))
    fp = int(sum((pred - gt).values()))
    fn = int(sum((gt - pred).values()))
    return _prf(tp, fp, fn)


def _normalized_ged(
    gt_edges: set[CanonicalEdge], pred_edges: set[CanonicalEdge]
) -> float:
    """Approximate normalized graph edit distance between two edge sets.

    Args:
        gt_edges: Canonical ground-truth edges.
        pred_edges: Canonical predicted edges.

    Returns:
        Node symmetric-difference plus edge symmetric-difference divided by the
        combined graph size. This is a lightweight diagnostic proxy, not a full
        graph edit solver.
    """
    gt_nodes = {edge.sub for edge in gt_edges} | {edge.obj for edge in gt_edges}
    pred_nodes = {edge.sub for edge in pred_edges} | {edge.obj for edge in pred_edges}
    node_diff = len(gt_nodes.symmetric_difference(pred_nodes))
    edge_diff = len(gt_edges.symmetric_difference(pred_edges))
    normalizer = max(1, len(gt_nodes | pred_nodes) + len(gt_edges | pred_edges))
    return float(node_diff + edge_diff) / float(normalizer)


def evaluate_graph_pair(
    gt_payload: object,
    pred_payload: object,
    *,
    normalize_ids: bool = True,
    normalize_relations: bool = True,
    compute_ged: bool = False,
) -> dict:
    """Evaluate one predicted scene graph against one ground-truth graph.

    Args:
        gt_payload: Ground-truth graph payload in a supported relationship
            shape.
        pred_payload: Predicted graph payload in a supported relationship shape.
        normalize_ids: Normalize object IDs before matching.
        normalize_relations: Normalize relation and attribute labels before
            matching.
        compute_ged: Include normalized_ged when true.

    Returns:
        Metric dictionary containing edge counts, strict_triplet,
        binary_triplet, attribute, pair_ordered, pair_unordered,
        predicate_only, overgeneration_rate, undergeneration_rate, and optional
        normalized_ged.
    """
    gt = canonicalize_edges(
        gt_payload,
        normalize_ids=normalize_ids,
        normalize_relations=normalize_relations,
    )
    pred = canonicalize_edges(
        pred_payload,
        normalize_ids=normalize_ids,
        normalize_relations=normalize_relations,
    )

    strict = _prf(
        tp=len(gt & pred),
        fp=len(pred - gt),
        fn=len(gt - pred),
    )

    gt_unary, gt_binary = split_unary_binary(gt)
    pred_unary, pred_binary = split_unary_binary(pred)
    attribute = _prf(
        tp=len(gt_unary & pred_unary),
        fp=len(pred_unary - gt_unary),
        fn=len(gt_unary - pred_unary),
    )
    binary_strict = _prf(
        tp=len(gt_binary & pred_binary),
        fp=len(pred_binary - gt_binary),
        fn=len(gt_binary - pred_binary),
    )

    gt_ordered_pairs = {(edge.sub, edge.obj) for edge in gt_binary}
    pred_ordered_pairs = {(edge.sub, edge.obj) for edge in pred_binary}
    pair_ordered = _prf(
        tp=len(gt_ordered_pairs & pred_ordered_pairs),
        fp=len(pred_ordered_pairs - gt_ordered_pairs),
        fn=len(gt_ordered_pairs - pred_ordered_pairs),
    )

    gt_unordered_pairs = {tuple(sorted((edge.sub, edge.obj))) for edge in gt_binary}
    pred_unordered_pairs = {tuple(sorted((edge.sub, edge.obj))) for edge in pred_binary}
    pair_unordered = _prf(
        tp=len(gt_unordered_pairs & pred_unordered_pairs),
        fp=len(pred_unordered_pairs - gt_unordered_pairs),
        fn=len(gt_unordered_pairs - pred_unordered_pairs),
    )

    gt_rel_counter = Counter(edge.rel for edge in gt_binary)
    pred_rel_counter = Counter(edge.rel for edge in pred_binary)
    predicate = _counter_prf(gt_rel_counter, pred_rel_counter)

    overgeneration_rate = _safe_div(strict["fp"], len(pred))
    undergeneration_rate = _safe_div(strict["fn"], len(gt))

    out = {
        "n_gt_edges": len(gt),
        "n_pred_edges": len(pred),
        "strict_triplet": strict,
        "binary_triplet": binary_strict,
        "attribute": attribute,
        "pair_ordered": pair_ordered,
        "pair_unordered": pair_unordered,
        "predicate_only": predicate,
        "overgeneration_rate": overgeneration_rate,
        "undergeneration_rate": undergeneration_rate,
    }
    if compute_ged:
        out["normalized_ged"] = _normalized_ged(gt, pred)
    return out


def _relationship_rows(payload: object) -> list[dict]:
    """Extract raw relationship rows for diagnostics.

    Args:
        payload: None, a list of rows, a graph dictionary with relationships,
            edges, or no_label_edges, or a single {sub, rel, obj} row.

    Returns:
        List of dictionary rows with unsupported shapes filtered out.
    """
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


def graph_diagnostics(
    gt_payload: object,
    pred_payload: object,
    *,
    valid_object_ids: set[str],
    vocabulary: dict | None = None,
    normalize_ids: bool = True,
    normalize_relations: bool = True,
) -> dict[str, int]:
    """Compute non-PRF graph quality diagnostics for one prediction.

    Args:
        gt_payload: Ground-truth graph payload.
        pred_payload: Predicted graph payload.
        valid_object_ids: Object IDs allowed by detection results for the image.
        vocabulary: Optional vocabulary containing allowed predicates and
            attributes.
        normalize_ids: Normalize IDs before checking references.
        normalize_relations: Normalize labels before vocabulary checks.

    Returns:
        Counts for invalid object references, hallucinated object IDs,
        duplicate relations, out-of-vocabulary predicates and attributes, and
        direction errors where the reverse edge exists in ground truth.
    """
    vocabulary = vocabulary or {}
    allowed_predicates = {
        normalize_relation(item, normalize_relations=normalize_relations)
        for item in vocabulary.get("predicates", [])
    }
    allowed_attributes = {
        normalize_relation(item, normalize_relations=normalize_relations)
        for item in vocabulary.get("attributes", [])
    }
    pred_rows = _relationship_rows(pred_payload)
    pred_edges = [
        CanonicalEdge(
            sub=normalize_node(row.get("sub"), normalize_ids=normalize_ids),
            rel=normalize_relation(
                row.get("rel"), normalize_relations=normalize_relations
            ),
            obj=normalize_node(row.get("obj"), normalize_ids=normalize_ids),
        )
        for row in pred_rows
        if all(key in row for key in ("sub", "rel", "obj"))
    ]
    gt_edges = canonicalize_edges(
        gt_payload,
        normalize_ids=normalize_ids,
        normalize_relations=normalize_relations,
    )
    pred_counter = Counter(pred_edges)
    duplicates = sum(count - 1 for count in pred_counter.values() if count > 1)
    invalid_ids = 0
    hallucinated_objects: set[str] = set()
    oov_predicates = 0
    oov_attributes = 0
    direction_errors = 0
    gt_binary = {edge for edge in gt_edges if edge.sub != edge.obj}

    for edge in pred_edges:
        for node in (edge.sub, edge.obj):
            if valid_object_ids and node not in valid_object_ids:
                invalid_ids += 1
                hallucinated_objects.add(node)
        if edge.sub == edge.obj:
            if allowed_attributes and edge.rel not in allowed_attributes:
                oov_attributes += 1
        elif allowed_predicates and edge.rel not in allowed_predicates:
            oov_predicates += 1
        if (
            edge.sub != edge.obj
            and edge not in gt_binary
            and CanonicalEdge(edge.obj, edge.rel, edge.sub) in gt_binary
        ):
            direction_errors += 1

    return {
        "invalid_object_id_refs": invalid_ids,
        "hallucinated_object_count": len(hallucinated_objects),
        "duplicate_relations": duplicates,
        "oov_predicates": oov_predicates,
        "oov_attributes": oov_attributes,
        "direction_errors": direction_errors,
    }


@dataclass
class PredicateStats:
    """Mutable accumulator for per-predicate TP/FP/FN counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int, fp: int, fn: int) -> None:
        """Add counts from one image or batch into the accumulator."""
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def to_dict(self) -> dict:
        """Return accumulated counts with precision, recall, and F1."""
        out = _prf(self.tp, self.fp, self.fn)
        return out


def summarize_per_image(
    per_image: dict[str, dict],
    *,
    include_per_predicate: bool = True,
) -> dict:
    """Aggregate per-image scene graph metrics into a run-level summary.

    Args:
        per_image: Mapping from image key to evaluate_graph_pair output,
            optionally extended with diagnostics and per_predicate counts.
        include_per_predicate: Whether to aggregate per-predicate counts and
            compute predicate_macro_f1.

    Returns:
        Summary dictionary with image count, total ground-truth/predicted edges,
        micro PRF groups, optional normalized_ged_avg, optional diagnostics, and
        optional per-predicate metrics.
    """
    total_gt = 0
    total_pred = 0

    strict_tp = strict_fp = strict_fn = 0
    binary_tp = binary_fp = binary_fn = 0
    pair_ordered_tp = pair_ordered_fp = pair_ordered_fn = 0
    pair_unordered_tp = pair_unordered_fp = pair_unordered_fn = 0
    attr_tp = attr_fp = attr_fn = 0
    pred_tp = pred_fp = pred_fn = 0
    ged_values: list[float] = []
    diagnostic_totals: Counter[str] = Counter()

    predicate_stats: dict[str, PredicateStats] = {}

    for row in per_image.values():
        total_gt += int(row.get("n_gt_edges", 0))
        total_pred += int(row.get("n_pred_edges", 0))

        strict_row = row.get("strict_triplet", {})
        binary_row = row.get("binary_triplet", {})
        pair_ordered_row = row.get("pair_ordered", {})
        pair_unordered_row = row.get("pair_unordered", {})
        attr_row = row.get("attribute", {})
        pred_row = row.get("predicate_only", {})

        strict_tp += int(strict_row.get("tp", 0))
        strict_fp += int(strict_row.get("fp", 0))
        strict_fn += int(strict_row.get("fn", 0))

        binary_tp += int(binary_row.get("tp", 0))
        binary_fp += int(binary_row.get("fp", 0))
        binary_fn += int(binary_row.get("fn", 0))

        pair_ordered_tp += int(pair_ordered_row.get("tp", 0))
        pair_ordered_fp += int(pair_ordered_row.get("fp", 0))
        pair_ordered_fn += int(pair_ordered_row.get("fn", 0))

        pair_unordered_tp += int(pair_unordered_row.get("tp", 0))
        pair_unordered_fp += int(pair_unordered_row.get("fp", 0))
        pair_unordered_fn += int(pair_unordered_row.get("fn", 0))

        attr_tp += int(attr_row.get("tp", 0))
        attr_fp += int(attr_row.get("fp", 0))
        attr_fn += int(attr_row.get("fn", 0))

        pred_tp += int(pred_row.get("tp", 0))
        pred_fp += int(pred_row.get("fp", 0))
        pred_fn += int(pred_row.get("fn", 0))

        if "normalized_ged" in row:
            ged_values.append(float(row["normalized_ged"]))
        for key, value in row.get("diagnostics", {}).items():
            diagnostic_totals[key] += int(value)

        if include_per_predicate:
            for predicate, counts in row.get("per_predicate", {}).items():
                stats = predicate_stats.setdefault(predicate, PredicateStats())
                stats.add(
                    int(counts.get("tp", 0)),
                    int(counts.get("fp", 0)),
                    int(counts.get("fn", 0)),
                )

    strict = _prf(strict_tp, strict_fp, strict_fn)
    binary = _prf(binary_tp, binary_fp, binary_fn)
    pair_ordered = _prf(pair_ordered_tp, pair_ordered_fp, pair_ordered_fn)
    pair_unordered = _prf(pair_unordered_tp, pair_unordered_fp, pair_unordered_fn)
    attribute = _prf(attr_tp, attr_fp, attr_fn)
    predicate_only = _prf(pred_tp, pred_fp, pred_fn)

    summary = {
        "images_evaluated": len(per_image),
        "edges_total_gt": total_gt,
        "edges_total_pred": total_pred,
        "strict_triplet_micro": strict,
        "binary_triplet_micro": binary,
        "pair_ordered_micro": pair_ordered,
        "pair_unordered_micro": pair_unordered,
        "attribute_micro": attribute,
        "predicate_only_micro": predicate_only,
    }
    if ged_values:
        summary["normalized_ged_avg"] = sum(ged_values) / len(ged_values)

    if include_per_predicate and predicate_stats:
        per_predicate = {
            name: stats.to_dict() for name, stats in predicate_stats.items()
        }
        macro_f1 = sum(item["f1"] for item in per_predicate.values()) / len(
            per_predicate
        )
        summary["per_predicate"] = per_predicate
        summary["predicate_macro_f1"] = macro_f1
    if diagnostic_totals:
        summary["diagnostics"] = dict(diagnostic_totals)

    return summary


def bootstrap_metric_ci(
    per_image: dict[str, dict],
    *,
    metric_group: str,
    metric_name: str = "f1",
    rounds: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate a bootstrap confidence interval for one metric group.

    Args:
        per_image: Mapping from image key to per-image metric rows.
        metric_group: Metric group to resample, such as strict_triplet or
            attribute.
        metric_name: Metric key inside the group, usually f1.
        rounds: Number of bootstrap resampling rounds.
        seed: Random seed for deterministic intervals.

    Returns:
        Dictionary with mean, ci95_low, and ci95_high. Returns an empty
        dictionary when there are no rows or rounds is not positive.
    """
    rows = list(per_image.values())
    if not rows or rounds <= 0:
        return {}
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(rounds):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        tp = sum(int(row.get(metric_group, {}).get("tp", 0)) for row in sample)
        fp = sum(int(row.get(metric_group, {}).get("fp", 0)) for row in sample)
        fn = sum(int(row.get(metric_group, {}).get("fn", 0)) for row in sample)
        values.append(float(_prf(tp, fp, fn)[metric_name]))
    values.sort()
    lo_idx = max(0, int(0.025 * (len(values) - 1)))
    hi_idx = min(len(values) - 1, int(0.975 * (len(values) - 1)))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": values[lo_idx],
        "ci95_high": values[hi_idx],
    }


def per_predicate_counts(
    gt_payload: object,
    pred_payload: object,
    *,
    normalize_ids: bool = True,
    normalize_relations: bool = True,
) -> dict[str, dict[str, int]]:
    """Compute TP/FP/FN counts for each relation or attribute label.

    Args:
        gt_payload: Ground-truth graph payload.
        pred_payload: Predicted graph payload.
        normalize_ids: Normalize object IDs before matching.
        normalize_relations: Normalize labels before grouping by predicate.

    Returns:
        Mapping from predicate or attribute label to tp, fp, and fn counts.
    """
    gt = canonicalize_edges(
        gt_payload,
        normalize_ids=normalize_ids,
        normalize_relations=normalize_relations,
    )
    pred = canonicalize_edges(
        pred_payload,
        normalize_ids=normalize_ids,
        normalize_relations=normalize_relations,
    )

    predicates = {edge.rel for edge in gt | pred}
    out: dict[str, dict[str, int]] = {}
    for predicate in predicates:
        gt_subset = {edge for edge in gt if edge.rel == predicate}
        pred_subset = {edge for edge in pred if edge.rel == predicate}
        out[predicate] = {
            "tp": len(gt_subset & pred_subset),
            "fp": len(pred_subset - gt_subset),
            "fn": len(gt_subset - pred_subset),
        }
    return out
