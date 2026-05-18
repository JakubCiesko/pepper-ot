#!/usr/bin/env python3
"""Plot raw-image vs SoM scene graph prompting experiment results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = ["geminiR", "gpt5"]
MODEL_LABELS = {
    "geminiR": "gemini-robotics-er-1.6-preview",
    "gpt5": "gpt-5.5-2026-04-23",
}
MODEL_COLORS = {
    "geminiR": "#59A14F",
    "gpt5": "#E15759",
}
CONDITION_ORDER = [
    "raw_objects_vocab",
    "raw_objects_vocab_caption",
    "som_objects_vocab",
    "som_objects_vocab_caption",
]
CONDITION_LABELS = {
    "raw_objects_vocab": "Raw\nobjects\nvocab",
    "raw_objects_vocab_caption": "Raw\nobjects\nvocab\ncaption",
    "som_objects_vocab": "SoM\nobjects\nvocab",
    "som_objects_vocab_caption": "SoM\nobjects\nvocab\ncaption",
}
METRICS = [
    ("strict", "Strict triplet F1"),
    ("attribute", "Attribute F1"),
    ("pair_ordered", "Ordered pair F1"),
    ("pair_unordered", "Unordered pair F1"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_name(name: str) -> tuple[str, str]:
    for model in sorted(MODEL_ORDER, key=len, reverse=True):
        marker = model + "_"
        if name.startswith(marker):
            return model, name[len(marker) :]
    raise ValueError(f"Unexpected run name: {name}")


def f1(summary: dict, group: str) -> float:
    return float(summary.get(group, {}).get("f1", 0.0))


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    missing = []
    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            run_name = f"{model}_{condition}"
            summary_path = runs_dir / run_name / "metrics_scene_graph_summary.json"
            if not summary_path.exists():
                missing.append(str(summary_path))
                continue
            parsed_model, parsed_condition = split_name(run_name)
            summary = load_json(summary_path)
            rows.append(
                {
                    "run": run_name,
                    "model": parsed_model,
                    "condition": parsed_condition,
                    "strict": f1(summary, "strict_triplet_micro"),
                    "attribute": f1(summary, "attribute_micro"),
                    "pair_ordered": f1(summary, "pair_ordered_micro"),
                    "pair_unordered": f1(summary, "pair_unordered_micro"),
                }
            )
    if missing:
        details = "\n".join(missing)
        raise SystemExit(f"Missing expected metrics files:\n{details}")
    return rows


def metric_lookup(rows: list[dict], metric: str) -> dict[tuple[str, str], float]:
    return {
        (row["model"], row["condition"]): float(row[metric])
        for row in rows
    }


def condition_averages(rows: list[dict], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(float(row[metric]))
    return {
        condition: sum(values) / len(values)
        for condition, values in grouped.items()
        if values
    }


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


def plot_grouped_strict(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = metric_lookup(rows, "strict")
    avg = condition_averages(rows, "strict")
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    for i, model in enumerate(MODEL_ORDER):
        offsets = x + (i - (len(MODEL_ORDER) - 1) / 2.0) * width
        values = [lookup[(model, condition)] for condition in CONDITION_ORDER]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.92,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)

    avg_values = [avg[condition] for condition in CONDITION_ORDER]
    ax.plot(
        x,
        avg_values,
        color="#111111",
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Mean across models",
        zorder=4,
    )

    ax.set_ylabel("Strict triplet F1")
    ax.set_xlabel("Prompting condition")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITION_ORDER])
    ax.set_ylim(0.0, max(0.46, max(avg_values) + 0.08))
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.17))
    fig.tight_layout()
    save_figure(fig, out_dir, "som_vs_no_som_strict_f1_grouped")
    plt.close(fig)


def plot_metric_panels(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(CONDITION_ORDER), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), sharex=True)
    axes_flat = list(axes.ravel())

    for ax, (metric, title) in zip(axes_flat, METRICS, strict=True):
        avg = condition_averages(rows, metric)
        values = [avg[condition] for condition in CONDITION_ORDER]
        best_idx = int(np.nanargmax(values))
        colors = ["#B8C4D6"] * len(CONDITION_ORDER)
        colors[best_idx] = "#315C9E"
        bars = ax.bar(x, values, color=colors, width=0.70)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
        ax.set_title(title)
        ax.set_ylim(0.0, max(0.76, max(values) + 0.08))
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[-2:]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[condition] for condition in CONDITION_ORDER]
        )
    for ax in axes_flat[:2]:
        ax.tick_params(labelbottom=False)

    fig.suptitle("Mean scene graph metrics by prompting condition", y=0.995)
    fig.tight_layout()
    save_figure(fig, out_dir, "som_vs_no_som_metric_panels")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("research/artifacts/experiments/som_prompting_effect/runs"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("thesis-source/pics"),
    )
    args = parser.parse_args()

    rows = load_rows(args.runs_dir)
    if not rows:
        raise SystemExit(f"No metrics found under {args.runs_dir}")

    setup_matplotlib()
    plot_grouped_strict(rows, args.out_dir)
    plot_metric_panels(rows, args.out_dir)
    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
