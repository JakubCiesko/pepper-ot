#!/usr/bin/env python3
"""Report scene graph over/under-generation against human annotations.

Standalone thesis helper. It compares one or more experiment run directories
against a ground-truth scene graph JSON and reports:
- whether each run predicts fewer or more edges than the human annotation
- the same split into binary relations and unary attributes
- labels that are most over-generated, under-generated, and best classified
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Edge:
    sub: str
    rel: str
    obj: str


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_node(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)$", text)
    return match.group(1) if match else text


def normalize_relation(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def relationship_rows(payload: object) -> list[dict]:
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


def canonical_edges(payload: object) -> set[Edge]:
    out: set[Edge] = set()
    for row in relationship_rows(payload):
        if not all(key in row for key in ("sub", "rel", "obj")):
            continue
        sub = normalize_node(row.get("sub"))
        rel = normalize_relation(row.get("rel"))
        obj = normalize_node(row.get("obj"))
        if sub and rel and obj:
            out.add(Edge(sub, rel, obj))
    return out


def split_edges(edges: set[Edge]) -> tuple[set[Edge], set[Edge]]:
    attrs = {edge for edge in edges if edge.sub == edge.obj}
    rels = {edge for edge in edges if edge.sub != edge.obj}
    return attrs, rels


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def match_payload_by_key(items: dict, key: str, basename_index: dict[str, str]):
    if key in items:
        return items[key]
    resolved = str(Path(key).resolve())
    if resolved in items:
        return items[resolved]
    basename = Path(key).name
    indexed = basename_index.get(basename)
    if indexed is not None:
        return items.get(indexed)
    return None


def build_basename_index(items: dict) -> dict[str, str]:
    out = {}
    for key in items:
        basename = Path(str(key)).name
        if basename not in out:
            out[basename] = key
    return out


def safe_ratio(num: int, den: int) -> float:
    return num / den if den else math.inf if num else 0.0


def add_label_counts(
    label_counts: dict[tuple[str, str], Counter],
    *,
    kind: str,
    gt_edges: set[Edge],
    pred_edges: set[Edge],
) -> None:
    labels = {edge.rel for edge in gt_edges | pred_edges}
    for label in labels:
        gt_subset = {edge for edge in gt_edges if edge.rel == label}
        pred_subset = {edge for edge in pred_edges if edge.rel == label}
        counter = label_counts.setdefault((kind, label), Counter())
        counter["tp"] += len(gt_subset & pred_subset)
        counter["fp"] += len(pred_subset - gt_subset)
        counter["fn"] += len(gt_subset - pred_subset)
        counter["gt"] += len(gt_subset)
        counter["pred"] += len(pred_subset)


def evaluate_run(
    run_dir: Path, gt_items: dict
) -> tuple[dict, dict[tuple[str, str], Counter]]:
    pred_path = run_dir / "draft_scene_graph.json"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    pred_items = load_json(pred_path)
    pred_basename_index = build_basename_index(pred_items)

    totals = Counter()
    label_counts: dict[tuple[str, str], Counter] = {}
    image_count = 0

    for key, gt_payload in gt_items.items():
        image_count += 1
        pred_payload = match_payload_by_key(pred_items, key, pred_basename_index)
        gt_edges = canonical_edges(gt_payload)
        pred_edges = canonical_edges(pred_payload)
        gt_attrs, gt_rels = split_edges(gt_edges)
        pred_attrs, pred_rels = split_edges(pred_edges)

        totals["gt_edges"] += len(gt_edges)
        totals["pred_edges"] += len(pred_edges)
        totals["gt_attributes"] += len(gt_attrs)
        totals["pred_attributes"] += len(pred_attrs)
        totals["gt_relations"] += len(gt_rels)
        totals["pred_relations"] += len(pred_rels)

        totals["strict_tp"] += len(gt_edges & pred_edges)
        totals["strict_fp"] += len(pred_edges - gt_edges)
        totals["strict_fn"] += len(gt_edges - pred_edges)
        totals["attribute_tp"] += len(gt_attrs & pred_attrs)
        totals["attribute_fp"] += len(pred_attrs - gt_attrs)
        totals["attribute_fn"] += len(gt_attrs - pred_attrs)
        totals["relation_tp"] += len(gt_rels & pred_rels)
        totals["relation_fp"] += len(pred_rels - gt_rels)
        totals["relation_fn"] += len(gt_rels - pred_rels)

        add_label_counts(
            label_counts, kind="attribute", gt_edges=gt_attrs, pred_edges=pred_attrs
        )
        add_label_counts(
            label_counts, kind="relation", gt_edges=gt_rels, pred_edges=pred_rels
        )

    strict = prf(totals["strict_tp"], totals["strict_fp"], totals["strict_fn"])
    attrs = prf(totals["attribute_tp"], totals["attribute_fp"], totals["attribute_fn"])
    rels = prf(totals["relation_tp"], totals["relation_fp"], totals["relation_fn"])
    summary = {
        "run": run_dir.name,
        "images": image_count,
        "gt_edges": totals["gt_edges"],
        "pred_edges": totals["pred_edges"],
        "edge_delta": totals["pred_edges"] - totals["gt_edges"],
        "edge_ratio": safe_ratio(totals["pred_edges"], totals["gt_edges"]),
        "gt_attributes": totals["gt_attributes"],
        "pred_attributes": totals["pred_attributes"],
        "attribute_delta": totals["pred_attributes"] - totals["gt_attributes"],
        "attribute_ratio": safe_ratio(
            totals["pred_attributes"], totals["gt_attributes"]
        ),
        "gt_relations": totals["gt_relations"],
        "pred_relations": totals["pred_relations"],
        "relation_delta": totals["pred_relations"] - totals["gt_relations"],
        "relation_ratio": safe_ratio(totals["pred_relations"], totals["gt_relations"]),
        "strict_precision": strict["precision"],
        "strict_recall": strict["recall"],
        "strict_f1": strict["f1"],
        "attribute_precision": attrs["precision"],
        "attribute_recall": attrs["recall"],
        "attribute_f1": attrs["f1"],
        "relation_precision": rels["precision"],
        "relation_recall": rels["recall"],
        "relation_f1": rels["f1"],
    }
    return summary, label_counts


def label_rows(label_counts: dict[tuple[str, str], Counter]) -> list[dict]:
    rows = []
    for (kind, label), counts in sorted(label_counts.items()):
        gt = int(counts["gt"])
        pred = int(counts["pred"])
        metrics = prf(int(counts["tp"]), int(counts["fp"]), int(counts["fn"]))
        rows.append(
            {
                "kind": kind,
                "label": label,
                "gt": gt,
                "pred": pred,
                "delta": pred - gt,
                "ratio": safe_ratio(pred, gt),
                **metrics,
            }
        )
    return rows


def finite_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def top_over_under_best(
    rows: list[dict], *, kind: str, min_gt: int, limit: int
) -> tuple[list[dict], list[dict], list[dict]]:
    filtered = [row for row in rows if row["kind"] == kind and row["gt"] >= min_gt]
    over = sorted(
        [row for row in filtered if row["delta"] > 0],
        key=lambda row: (row["delta"], row["pred"]),
        reverse=True,
    )[:limit]
    under = sorted(
        [row for row in filtered if row["delta"] < 0],
        key=lambda row: (row["delta"], -row["gt"]),
    )[:limit]
    best = sorted(
        filtered,
        key=lambda row: (row["f1"], row["gt"], row["precision"], row["recall"]),
        reverse=True,
    )[:limit]
    return over, under, best


def format_label_rows(rows: list[dict]) -> list[list[object]]:
    return [
        [
            row["label"],
            row["gt"],
            row["pred"],
            row["delta"],
            finite_ratio(float(row["ratio"])),
            f"{float(row['precision']):.2f}",
            f"{float(row['recall']):.2f}",
            f"{float(row['f1']):.2f}",
        ]
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    run_summaries: list[dict],
    aggregate_labels: list[dict],
    *,
    min_gt: int,
    limit: int,
) -> str:
    lines = ["# Scene Graph Generation Bias Report", ""]
    lines.append("## Run-Level Generation")
    lines.append(
        markdown_table(
            [
                "Run",
                "GT",
                "Pred",
                "Ratio",
                "Attr ratio",
                "Rel ratio",
                "Strict P",
                "Strict R",
                "Strict F1",
            ],
            [
                [
                    row["run"],
                    row["gt_edges"],
                    row["pred_edges"],
                    finite_ratio(float(row["edge_ratio"])),
                    finite_ratio(float(row["attribute_ratio"])),
                    finite_ratio(float(row["relation_ratio"])),
                    f"{float(row['strict_precision']):.2f}",
                    f"{float(row['strict_recall']):.2f}",
                    f"{float(row['strict_f1']):.2f}",
                ]
                for row in run_summaries
            ],
        )
    )

    for kind, title in (("relation", "Relations"), ("attribute", "Attributes")):
        over, under, best = top_over_under_best(
            aggregate_labels, kind=kind, min_gt=min_gt, limit=limit
        )
        lines.extend(["", f"## Most Over-Generated {title}"])
        lines.append(
            markdown_table(
                ["Label", "GT", "Pred", "Delta", "Ratio", "P", "R", "F1"],
                format_label_rows(over),
            )
        )
        lines.extend(["", f"## Most Under-Generated {title}"])
        lines.append(
            markdown_table(
                ["Label", "GT", "Pred", "Delta", "Ratio", "P", "R", "F1"],
                format_label_rows(under),
            )
        )
        lines.extend(["", f"## Best Classified {title}"])
        lines.append(
            markdown_table(
                ["Label", "GT", "Pred", "Delta", "Ratio", "P", "R", "F1"],
                format_label_rows(best),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("research/artifacts/experiments/vocab_presentation_effect/runs"),
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("data/human_labeled_completed.json"),
    )
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--min-gt", type=int, default=5)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research/artifacts/reports/sgg_generation_bias"),
    )
    args = parser.parse_args()

    gt_items = load_json(args.gt)
    if not isinstance(gt_items, dict) or not gt_items:
        raise SystemExit(f"Ground truth is empty or invalid: {args.gt}")

    run_dirs = [
        path
        for path in sorted(args.runs_dir.glob(args.run_glob))
        if path.is_dir() and (path / "draft_scene_graph.json").exists()
    ]
    if not run_dirs:
        raise SystemExit(
            f"No run directories with draft_scene_graph.json under {args.runs_dir}"
        )

    run_summaries = []
    aggregate_counts: dict[tuple[str, str], Counter] = {}
    per_run_label_rows = []
    for run_dir in run_dirs:
        summary, counts = evaluate_run(run_dir, gt_items)
        run_summaries.append(summary)
        for key, counter in counts.items():
            aggregate_counts.setdefault(key, Counter()).update(counter)
        per_run_label_rows.extend([{"run": run_dir.name, **row} for row in label_rows(counts)])

    aggregate_label_rows = label_rows(aggregate_counts)
    report = build_report(
        run_summaries,
        aggregate_label_rows,
        min_gt=args.min_gt,
        limit=args.limit,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sgg_generation_bias_report.md").write_text(
        report, encoding="utf-8"
    )
    write_csv(args.out_dir / "run_generation_summary.csv", run_summaries)
    write_csv(args.out_dir / "aggregate_label_generation.csv", aggregate_label_rows)
    write_csv(args.out_dir / "per_run_label_generation.csv", per_run_label_rows)

    print(report)
    print(f"Wrote report files to {args.out_dir}")


if __name__ == "__main__":
    main()
