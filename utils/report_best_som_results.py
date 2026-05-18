#!/usr/bin/env python3
"""Summarize best-SoM scene graph experiment results.

This is a local reporting helper. It reads existing metric JSON files and prints:
- best runs by strict triplet F1
- best run per model family
- average metrics per SoM setting
- strongest predicates/attributes for a selected run
- easiest/hardest images for a selected run
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

MODEL_PREFIXES = ["geminiPRO", "geminiR", "gemini", "gpt5", "gpt"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def f1(summary: dict, group: str) -> float:
    return float(summary.get(group, {}).get("f1", 0.0))


def split_run_name(name: str) -> tuple[str, str]:
    for prefix in MODEL_PREFIXES:
        marker = prefix + "_"
        if name.startswith(marker):
            return prefix, name[len(marker) :]
    return name.split("_", 1)[0], name


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(runs_dir.glob("*/metrics_scene_graph_summary.json")):
        run_name = summary_path.parent.name
        family, setting = split_run_name(run_name)
        summary = load_json(summary_path)
        rows.append(
            {
                "run": run_name,
                "family": family,
                "setting": setting,
                "images": int(summary.get("images_evaluated", 0)),
                "strict": f1(summary, "strict_triplet_micro"),
                "attribute": f1(summary, "attribute_micro"),
                "pair_ordered": f1(summary, "pair_ordered_micro"),
                "pair_unordered": f1(summary, "pair_unordered_micro"),
                "summary": summary,
                "run_dir": summary_path.parent,
            }
        )
    return rows


def print_metric_row(row: dict) -> None:
    print(
        f"{row['run']:<36} "
        f"strict={row['strict']:.4f} "
        f"attr={row['attribute']:.4f} "
        f"pair_ord={row['pair_ordered']:.4f} "
        f"pair_unord={row['pair_unordered']:.4f} "
        f"n={row['images']}"
    )


def print_top_runs(rows: list[dict], limit: int) -> None:
    print("\nTop runs by strict triplet F1")
    for row in sorted(rows, key=lambda item: item["strict"], reverse=True)[:limit]:
        print_metric_row(row)


def print_best_per_family(rows: list[dict]) -> None:
    print("\nBest run per model family")

    families: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        families[row["family"]].append(row)

    for family in sorted(families):
        best = max(families[family], key=lambda item: item["strict"])
        print(f"{family:<10} {best['setting']:<24} ", end="")
        print_metric_row(best)


def print_setting_averages(rows: list[dict]) -> None:
    print("\nAverage by SoM setting")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["setting"]].append(row)
    averaged = []
    for setting, items in groups.items():
        averaged.append(
            (
                setting,
                len(items),
                sum(item["strict"] for item in items) / len(items),
                sum(item["attribute"] for item in items) / len(items),
                sum(item["pair_ordered"] for item in items) / len(items),
                sum(item["pair_unordered"] for item in items) / len(items),
            )
        )
    for setting, n, strict, attr, pair_ordered, pair_unordered in sorted(
        averaged, key=lambda item: item[2], reverse=True
    ):
        print(
            f"{setting:<24} "
            f"n={n} "
            f"strict={strict:.4f} "
            f"attr={attr:.4f} "
            f"pair_ord={pair_ordered:.4f} "
            f"pair_unord={pair_unordered:.4f}"
        )


def load_attrs(vocab_path: Path | None) -> set[str]:
    if vocab_path is None or not vocab_path.exists():
        return set()
    vocab = load_json(vocab_path)
    return set(vocab.get("attributes", []))


def print_terms(row: dict, attrs: set[str], limit: int) -> None:
    per_predicate = row["summary"].get("per_predicate", {}) or {}
    items = []
    for term, metrics in per_predicate.items():
        support = int(metrics.get("tp", 0)) + int(metrics.get("fn", 0))
        if support <= 0:
            continue
        kind = "attribute" if term in attrs else "predicate"
        items.append(
            (
                kind,
                term,
                support,
                float(metrics.get("f1", 0.0)),
                float(metrics.get("precision", 0.0)),
                float(metrics.get("recall", 0.0)),
            )
        )
    for kind in ("predicate", "attribute"):
        print(f"\nTop {kind}s for {row['run']}")
        filtered = [item for item in items if item[0] == kind and item[2] >= 5]
        for _, term, support, score, precision, recall in sorted(
            filtered, key=lambda item: item[3], reverse=True
        )[:limit]:
            print(
                f"{term:<18} "
                f"f1={score:.3f} "
                f"precision={precision:.3f} "
                f"recall={recall:.3f} "
                f"support={support}"
            )


def print_images(row: dict, limit: int) -> None:
    per_image_path = row["run_dir"] / "metrics_scene_graph_per_image.json"
    if not per_image_path.exists():
        return

    per_image = load_json(per_image_path)
    image_rows = []

    for image_path, metrics in per_image.items():
        image_rows.append(
            (
                image_path,
                float(metrics.get("strict_triplet", {}).get("f1", 0.0)),
                float(metrics.get("attribute", {}).get("f1", 0.0)),
                float(metrics.get("pair_unordered", {}).get("f1", 0.0)),
                int(metrics.get("n_gt_edges", 0)),
                int(metrics.get("n_pred_edges", 0)),
            )
        )

    print(f"\nEasiest images for {row['run']}")

    for image_path, strict, attr, pair_unordered, n_gt, n_pred in sorted(
        image_rows, key=lambda item: item[1], reverse=True
    )[:limit]:
        print(
            f"{Path(image_path).name} "
            f"strict={strict:.3f} "
            f"attr={attr:.3f} "
            f"pair_unord={pair_unordered:.3f} "
            f"gt={n_gt} "
            f"pred={n_pred}"
        )

    print(f"\nHardest images for {row['run']}")

    for image_path, strict, attr, pair_unordered, n_gt, n_pred in sorted(
        image_rows, key=lambda item: item[1]
    )[:limit]:
        print(
            f"{Path(image_path).name} "
            f"strict={strict:.3f} "
            f"attr={attr:.3f} "
            f"pair_unord={pair_unordered:.3f} "
            f"gt={n_gt} "
            f"pred={n_pred}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("research/artifacts/experiments/best_som/runs"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(
            "research/artifacts/human_eval/vocab/eval_draft_sgg_frozen_vocab_v1/vocabulary_final.json"
        ),
    )
    parser.add_argument("--focus-run", default="geminiR_id_labels_bbox")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = load_rows(args.runs_dir)
    if not rows:
        raise SystemExit(f"No metric summaries found under {args.runs_dir}")

    print_top_runs(rows, args.limit)
    print_best_per_family(rows)
    print_setting_averages(rows)

    focus = next((row for row in rows if row["run"] == args.focus_run), None)
    if focus is None:
        focus = max(rows, key=lambda item: item["strict"])
    attrs = load_attrs(args.vocab)
    print_terms(focus, attrs, args.limit)
    print_images(focus, min(args.limit, 8))


if __name__ == "__main__":
    main()
