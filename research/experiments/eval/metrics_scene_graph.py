from collections import Counter
from dataclasses import dataclass

from .normalization import CanonicalEdge
from .normalization import canonicalize_edges
from .normalization import split_unary_binary


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
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
    tp = int(sum((gt & pred).values()))
    fp = int(sum((pred - gt).values()))
    fn = int(sum((gt - pred).values()))
    return _prf(tp, fp, fn)


def _normalized_ged(
    gt_edges: set[CanonicalEdge], pred_edges: set[CanonicalEdge]
) -> float:
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

    gt_pairs = {(edge.sub, edge.obj) for edge in gt_binary}
    pred_pairs = {(edge.sub, edge.obj) for edge in pred_binary}
    pair = _prf(
        tp=len(gt_pairs & pred_pairs),
        fp=len(pred_pairs - gt_pairs),
        fn=len(gt_pairs - pred_pairs),
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
        "pair": pair,
        "predicate_only": predicate,
        "overgeneration_rate": overgeneration_rate,
        "undergeneration_rate": undergeneration_rate,
    }
    if compute_ged:
        out["normalized_ged"] = _normalized_ged(gt, pred)
    return out


@dataclass
class PredicateStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def to_dict(self) -> dict:
        out = _prf(self.tp, self.fp, self.fn)
        return out


def summarize_per_image(
    per_image: dict[str, dict],
    *,
    include_per_predicate: bool = True,
) -> dict:
    total_gt = 0
    total_pred = 0

    strict_tp = strict_fp = strict_fn = 0
    binary_tp = binary_fp = binary_fn = 0
    pair_tp = pair_fp = pair_fn = 0
    attr_tp = attr_fp = attr_fn = 0
    pred_tp = pred_fp = pred_fn = 0
    ged_values: list[float] = []

    predicate_stats: dict[str, PredicateStats] = {}

    for row in per_image.values():
        total_gt += int(row.get("n_gt_edges", 0))
        total_pred += int(row.get("n_pred_edges", 0))

        strict_row = row.get("strict_triplet", {})
        binary_row = row.get("binary_triplet", {})
        pair_row = row.get("pair", {})
        attr_row = row.get("attribute", {})
        pred_row = row.get("predicate_only", {})

        strict_tp += int(strict_row.get("tp", 0))
        strict_fp += int(strict_row.get("fp", 0))
        strict_fn += int(strict_row.get("fn", 0))

        binary_tp += int(binary_row.get("tp", 0))
        binary_fp += int(binary_row.get("fp", 0))
        binary_fn += int(binary_row.get("fn", 0))

        pair_tp += int(pair_row.get("tp", 0))
        pair_fp += int(pair_row.get("fp", 0))
        pair_fn += int(pair_row.get("fn", 0))

        attr_tp += int(attr_row.get("tp", 0))
        attr_fp += int(attr_row.get("fp", 0))
        attr_fn += int(attr_row.get("fn", 0))

        pred_tp += int(pred_row.get("tp", 0))
        pred_fp += int(pred_row.get("fp", 0))
        pred_fn += int(pred_row.get("fn", 0))

        if "normalized_ged" in row:
            ged_values.append(float(row["normalized_ged"]))

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
    pair = _prf(pair_tp, pair_fp, pair_fn)
    attribute = _prf(attr_tp, attr_fp, attr_fn)
    predicate_only = _prf(pred_tp, pred_fp, pred_fn)

    summary = {
        "images_evaluated": len(per_image),
        "edges_total_gt": total_gt,
        "edges_total_pred": total_pred,
        "strict_triplet_micro": strict,
        "binary_triplet_micro": binary,
        "pair_micro": pair,
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

    return summary


def per_predicate_counts(
    gt_payload: object,
    pred_payload: object,
    *,
    normalize_ids: bool = True,
    normalize_relations: bool = True,
) -> dict[str, dict[str, int]]:
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
