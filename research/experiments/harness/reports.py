from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research.experiments.io import load_json
from research.experiments.io import save_json


def _metric(summary: dict[str, Any], group: str, key: str = "f1") -> float:
    value = summary.get(group, {})
    if isinstance(value, dict):
        return float(value.get(key, 0.0))
    return 0.0


def _read_run_row(run_dir: Path) -> dict[str, Any]:
    metadata = load_json(run_dir / "run_metadata.json", default={})
    matrix = load_json(run_dir / "matrix_result.json", default={})
    summary = load_json(run_dir / "metrics_scene_graph_summary.json", default={})
    draft_metrics = load_json(run_dir / "metrics_draft_scene_graph.json", default={})
    context_rot = load_json(run_dir / "context_rot.json", default={})

    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    row = {
        "run_id": metadata.get("run_id", run_dir.name),
        "variant": matrix.get("variant", config.get("experiment_id", run_dir.name)),
        "ok": matrix.get("ok", True),
        "provider": config.get("draft_sgg_model", {}).get("provider", ""),
        "model_id": config.get("draft_sgg_model", {}).get("model_id", ""),
        "use_som_image": config.get("draft_scene_graph", {}).get("use_som_image", ""),
        "images_evaluated": summary.get("images_evaluated", 0),
        "strict_triplet_f1": _metric(summary, "strict_triplet_micro"),
        "binary_triplet_f1": _metric(summary, "binary_triplet_micro"),
        "pair_f1": _metric(summary, "pair_micro"),
        "attribute_f1": _metric(summary, "attribute_micro"),
        "predicate_only_f1": _metric(summary, "predicate_only_micro"),
        "draft_ok": draft_metrics.get("ok", 0),
        "draft_failed": draft_metrics.get("failed", 0),
        "draft_duration_s": draft_metrics.get("duration_s", 0.0),
        "context_rot_points": len(context_rot) if isinstance(context_rot, dict) else 0,
    }
    return row


def aggregate_runs(runs_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    rows = [_read_run_row(run_dir) for run_dir in run_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json(out_dir / "metrics_summary.json", rows)
    write_report(out_dir, rows)
    _write_basic_plots(out_dir, rows)
    return rows


def write_report(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    lines = [
        "# Experiment Report",
        "",
        f"Runs aggregated: {len(rows)}",
        "",
        "| Variant | Model | SoM | Images | Strict F1 | Pair F1 | Attr F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {model} | {som} | {images} | {strict:.3f} | {pair:.3f} | {attr:.3f} |".format(
                variant=row.get("variant", ""),
                model=row.get("model_id", ""),
                som=row.get("use_som_image", ""),
                images=row.get("images_evaluated", 0),
                strict=float(row.get("strict_triplet_f1", 0.0)),
                pair=float(row.get("pair_f1", 0.0)),
                attr=float(row.get("attribute_f1", 0.0)),
            )
        )
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_basic_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [str(row.get("variant", row.get("run_id", ""))) for row in rows]
    x = list(range(len(labels)))

    for metric, filename, ylabel in [
        ("strict_triplet_f1", "strict_triplet_f1.pdf", "Strict triplet F1"),
        ("pair_f1", "pair_f1.pdf", "Pair F1"),
        ("attribute_f1", "attribute_f1.pdf", "Attribute F1"),
    ]:
        values = [float(row.get(metric, 0.0)) for row in rows]
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.7), 4))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        fig.savefig(out_dir / filename.replace(".pdf", ".png"), dpi=180)
        plt.close(fig)

    durations = [
        float(row.get("draft_duration_s", 0.0)) / max(1, int(row.get("draft_ok", 0)))
        for row in rows
    ]
    if any(durations):
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.7), 4))
        ax.bar(x, durations)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Draft SGG seconds per successful image")
        fig.tight_layout()
        fig.savefig(out_dir / "draft_latency_proxy.pdf")
        fig.savefig(out_dir / "draft_latency_proxy.png", dpi=180)
        plt.close(fig)
