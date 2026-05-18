#!/usr/bin/env python3
"""Plot vocabulary-size/context-rot scene graph results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

METRICS = [
    ("strict_triplet_micro", "Strict triplet F1"),
    ("attribute_micro", "Attribute F1"),
    ("pair_ordered_micro", "Ordered pair F1"),
    ("pair_unordered_micro", "Unordered pair F1"),
]
COLORS = {
    "strict_triplet_micro": "#315C9E",
    "attribute_micro": "#59A14F",
    "pair_ordered_micro": "#F28E2B",
    "pair_unordered_micro": "#E15759",
}
COMPARISON_COLORS = {
    "semantic": "#315C9E",
    "frequency": "#59A14F",
    "random": "#E15759",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=220)


def metric_f1(summary: dict, metric: str) -> float:
    value = summary.get(metric, {})
    if isinstance(value, dict):
        return float(value.get("f1", 0.0))
    return 0.0


def parse_rows(context_rot: dict) -> list[dict]:
    rows = []
    for key, value in context_rot.items():
        if not isinstance(value, dict):
            continue
        sliced = value.get("slice", {})
        summary = value.get("metrics_summary", {})
        if not isinstance(sliced, dict) or not isinstance(summary, dict):
            continue
        name = str(sliced.get("name") or key)
        size = int(sliced.get("size") or 0)
        if not size:
            match = re.search(r"(\d+)", name)
            size = int(match.group(1)) if match else 0
        row = {
            "name": name,
            "vocab_size": size,
            "relationship_count_avg": float(value.get("relationship_count_avg", 0.0)),
        }
        for metric, _ in METRICS:
            row[metric] = metric_f1(summary, metric)
        rows.append(row)
    rows.sort(key=lambda item: item["vocab_size"])
    return rows


def plot_f1_scree(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    x = [row["vocab_size"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for metric, label in METRICS:
        y = [row[metric] for row in rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=COLORS[metric],
            label=label,
        )
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.17), ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "context_rot_f1_scree")
    plt.close(fig)


def plot_strict_and_edges(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    x = [row["vocab_size"] for row in rows]
    strict = [row["strict_triplet_micro"] for row in rows]
    edges = [row["relationship_count_avg"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(10.8, 6.2))
    ax1.plot(
        x,
        strict,
        color=COLORS["strict_triplet_micro"],
        marker="o",
        linewidth=2.0,
        label="Strict triplet F1",
    )
    ax1.set_xlabel("Vocabulary size")
    ax1.set_ylabel("Strict triplet F1")
    ax1.set_ylim(0.0, max(0.5, max(strict) + 0.1 if strict else 0.5))
    ax1.grid(axis="y", alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        edges,
        color="#555555",
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="Predicted edges/image",
    )
    ax2.set_ylabel("Predicted edges per image")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False)
    ax1.set_xticks(x)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "context_rot_strict_f1_edges")
    plt.close(fig)


def strategy_label(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "semantic" in name:
        return "Semantic mapping"
    if "frequency" in name:
        return "Frequency mapping"
    if "random" in name:
        return "Random mapping"
    return run_dir.name.replace("_", " ")


def strategy_color(label: str) -> str:
    key = label.split()[0].lower()
    return COMPARISON_COLORS.get(key, "#555555")


def load_run_rows(run_dir: Path) -> list[dict]:
    context_rot_path = run_dir / "context_rot.json"
    if not context_rot_path.exists():
        return []
    return parse_rows(load_json(context_rot_path))


def plot_strict_comparison(
    run_rows: list[tuple[Path, list[dict]]], out_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    for run_dir, rows in run_rows:
        x = [row["vocab_size"] for row in rows]
        y = [row["strict_triplet_micro"] for row in rows]
        label = strategy_label(run_dir)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=strategy_color(label),
            label=label,
        )
    all_sizes = sorted({row["vocab_size"] for _, rows in run_rows for row in rows})
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Strict triplet F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(all_sizes)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir, "context_rot_strict_f1_compare")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run",
        type=Path,
        help="Run directory containing context_rot.json.",
    )
    source.add_argument(
        "--runs-root",
        type=Path,
        help="Directory containing multiple run directories with context_rot.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("thesis-source/pics"),
    )
    args = parser.parse_args()

    setup_matplotlib()
    if args.run:
        context_rot_path = args.run / "context_rot.json"
        rows = parse_rows(load_json(context_rot_path))
        if not rows:
            raise SystemExit(f"No context-rot metric rows found in {context_rot_path}")
        plot_f1_scree(rows, args.out_dir)
        plot_strict_and_edges(rows, args.out_dir)
    else:
        run_rows = []
        for run_dir in sorted(args.runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            rows = load_run_rows(run_dir)
            if rows:
                run_rows.append((run_dir, rows))
        if not run_rows:
            raise SystemExit(f"No context-rot runs found in {args.runs_root}")
        plot_strict_comparison(run_rows, args.out_dir)
    print(f"Wrote context-rot plots to {args.out_dir}")


if __name__ == "__main__":
    main()
