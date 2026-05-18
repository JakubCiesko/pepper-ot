#!/usr/bin/env python3
"""Plot best-SoM scene graph experiment results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

MODEL_ORDER = [
    "gemini",
    "geminiR",
    #    "geminiPRO",
    "gpt",
    "gpt5",
]
MODEL_LABELS = {
    "gemini": "gemini-3.1-flash-lite",
    "geminiR": "gemini-robotics-er-1.6-preview",
    #    "geminiPRO": "Gemini Pro",
    "gpt": "gpt-5-nano-2025-08-07",
    "gpt5": "gpt-5.5-2026-04-23",
}
MODEL_COLORS = {
    "gemini": "#4E79A7",
    "geminiR": "#59A14F",
    #   "geminiPRO": "#B07AA1",
    "gpt": "#F28E2B",
    "gpt5": "#E15759",
}
SETTING_ORDER = [
    "id",
    "id_bbox",
    "id_labels",
    "id_labels_bbox",
    "id_mask",
    "id_bbox_mask",
    "id_labels_mask",
    "id_labels_bbox_mask",
]
SETTING_LABELS = {
    "id": "ID",
    "id_bbox": "ID\nbbox",
    "id_labels": "ID\nlabels",
    "id_labels_bbox": "ID\nlabels\nbbox",
    "id_mask": "ID\nmask",
    "id_bbox_mask": "ID\nbbox\nmask",
    "id_labels_mask": "ID\nlabels\nmask",
    "id_labels_bbox_mask": "ID\nlabels\nbbox\nmask",
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
    return name.split("_", 1)[0], name


def f1(summary: dict, group: str) -> float:
    return float(summary.get(group, {}).get("f1", 0.0))


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(runs_dir.glob("*/metrics_scene_graph_summary.json")):
        if "geminiPRO" in str(summary_path.parent):
            continue
        run_name = summary_path.parent.name
        model, setting = split_name(run_name)
        summary = load_json(summary_path)
        rows.append(
            {
                "run": run_name,
                "model": model,
                "setting": setting,
                "strict": f1(summary, "strict_triplet_micro"),
                "attribute": f1(summary, "attribute_micro"),
                "pair_ordered": f1(summary, "pair_ordered_micro"),
                "pair_unordered": f1(summary, "pair_unordered_micro"),
            }
        )
    return rows


def available_models(rows: list[dict]) -> list[str]:
    found = {row["model"] for row in rows}
    ordered = [model for model in MODEL_ORDER if model in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def available_settings(rows: list[dict]) -> list[str]:
    found = {row["setting"] for row in rows}
    ordered = [setting for setting in SETTING_ORDER if setting in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def metric_lookup(rows: list[dict], metric: str) -> dict[tuple[str, str], float]:
    return {(row["model"], row["setting"]): float(row[metric]) for row in rows}


def setting_averages(rows: list[dict], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(float(row[metric]))
    return {
        setting: sum(values) / len(values)
        for setting, values in grouped.items()
        if values
    }


def setup_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
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

    models = available_models(rows)
    settings = available_settings(rows)
    lookup = metric_lookup(rows, "strict")
    avg = setting_averages(rows, "strict")

    x = np.arange(len(settings), dtype=float)
    width = min(0.14, 0.72 / max(1, len(models)))

    fig, ax = plt.subplots(figsize=(10.8, 7.5))
    for i, model in enumerate(models):
        offsets = x + (i - (len(models) - 1) / 2.0) * width
        values = [lookup.get((model, setting), np.nan) for setting in settings]
        ax.bar(
            offsets,
            values,
            width=width,
            color=MODEL_COLORS.get(model, "#777777"),
            label=MODEL_LABELS.get(model, model),
            alpha=0.92,
        )

    avg_values = [avg.get(setting, np.nan) for setting in settings]
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
    ax.set_xlabel("SoM rendering")
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS.get(setting, setting) for setting in settings])
    ax.set_ylim(0.0, max(0.42, np.nanmax(avg_values) + 0.08))
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.20))
    fig.tight_layout()
    save_figure(fig, out_dir, "best_som_strict_f1_grouped")
    plt.close(fig)


def plot_metric_panels(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    settings = available_settings(rows)
    x = np.arange(len(settings), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    axes_flat = list(axes.ravel())

    for ax, (metric, title) in zip(axes_flat, METRICS, strict=True):
        avg = setting_averages(rows, metric)
        values = [avg.get(setting, np.nan) for setting in settings]
        best_idx = int(np.nanargmax(values))
        colors = ["#B8C4D6"] * len(settings)
        colors[best_idx] = "#315C9E"
        ax.bar(x, values, color=colors, width=0.72)
        ax.set_title(title)
        ax.set_ylim(0.0, max(0.7, np.nanmax(values) + 0.08))
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for idx, value in enumerate(values):
            if not np.isnan(value):
                ax.text(
                    idx,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

    for ax in axes_flat[-2:]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SETTING_LABELS.get(setting, setting) for setting in settings],
            rotation=0,
        )
    for ax in axes_flat[:2]:
        ax.tick_params(labelbottom=False)

    fig.suptitle("Mean scene graph metrics by SoM rendering", y=0.995, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "best_som_metric_panels")
    plt.close(fig)


def plot_model_heatmap(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = available_models(rows)
    settings = available_settings(rows)
    lookup = metric_lookup(rows, "strict")
    matrix = np.full((len(models), len(settings)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, setting in enumerate(settings):
            if (model, setting) in lookup:
                matrix[i, j] = lookup[(model, setting)]

    fig, ax = plt.subplots(figsize=(10.8, 3.9))
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#F2F2F2")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.15, vmax=0.36)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(
        [
            SETTING_LABELS.get(setting, setting).replace("\n", " ")
            for setting in settings
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax.set_title("Strict triplet F1 by model and SoM rendering")
    for i in range(len(models)):
        for j in range(len(settings)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Strict triplet F1")
    fig.tight_layout()
    save_figure(fig, out_dir, "best_som_strict_f1_heatmap")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("research/artifacts/experiments/best_som/runs"),
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
    plot_model_heatmap(rows, args.out_dir)
    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
