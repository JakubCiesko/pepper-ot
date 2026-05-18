from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.harness.manifest import load_manifest
from research.experiments.io import save_json


async def run_pipeline_batch(
    *,
    server_config: Path,
    manifest: Path,
    out_dir: Path,
    preset: str = "full",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the server perception pipeline over every image in a manifest.

    Args:
        server_config: Server AppConfig YAML loaded before pipeline creation.
        manifest: Manifest JSONL containing image_id and image_path rows.
        out_dir: Output directory for batch artifacts.
        preset: Pipeline controls preset applied to the server config.
        limit: Optional maximum number of manifest rows to process.

    Returns:
        Dictionary with per_image rows and a summary. Successful per-image rows
        include pipeline metrics, executed stages, detections, optional scene
        graph, and duration; failed rows include an error string.

    Side Effects:
        Imports the server app, builds an in-process perception pipeline, reads
        image files, and writes pipeline_batch_per_image.json plus
        pipeline_batch_summary.json.
    """
    ensure_server_app_importable()
    from app.core.pipeline_factory import build_perception_pipeline
    from app.schemas.config import AppConfig

    cfg = AppConfig.load(str(server_config))
    cfg.pipeline_controls.preset = preset
    cfg.pipeline_controls = cfg.pipeline_controls.model_validate(
        cfg.pipeline_controls.model_dump()
    )
    pipeline = build_perception_pipeline(cfg)
    rows = load_manifest(manifest)
    if limit is not None:
        rows = rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, Any] = {}
    for row in rows:
        t0 = perf_counter()
        try:
            with Image.open(row.image_path) as img:
                image = img.convert("RGB")
            result = await pipeline.process(image, robot_metadata=None)
            per_image[row.image_id] = {
                "image_path": row.image_path,
                "ok": True,
                "metrics": result.metrics,
                "executed_stages": result.executed_stages,
                "detections": [det.model_dump() for det in result.detections],
                "scene_graph": (
                    result.scene_graph.model_dump()
                    if result.scene_graph is not None
                    else None
                ),
                "duration_s": perf_counter() - t0,
            }
        except Exception as exc:
            per_image[row.image_id] = {
                "image_path": row.image_path,
                "ok": False,
                "error": str(exc),
                "duration_s": perf_counter() - t0,
            }

    ok_rows = [row for row in per_image.values() if row.get("ok")]
    summary = {
        "preset": preset,
        "images": len(per_image),
        "ok": len(ok_rows),
        "failed": len(per_image) - len(ok_rows),
    }
    for key in [
        "caption_time",
        "detection_time",
        "memory_update_time",
        "som_image_paint_time",
        "scene_graph_generation_time",
        "qa_generation_time",
        "scene_graph_memory_update_time",
        "wall_processing_time",
    ]:
        values = [
            float(row.get("metrics", {}).get(key, 0.0))
            for row in ok_rows
            if key in row.get("metrics", {})
        ]
        if values:
            values = sorted(values)
            summary[f"{key}_p50"] = values[len(values) // 2]
            summary[f"{key}_p95"] = values[
                min(len(values) - 1, int(len(values) * 0.95))
            ]

    save_json(out_dir / "pipeline_batch_per_image.json", per_image)
    save_json(out_dir / "pipeline_batch_summary.json", summary)
    return {"per_image": per_image, "summary": summary}


def run_pipeline_batch_sync(**kwargs) -> dict[str, Any]:
    """Synchronous wrapper around run_pipeline_batch for CLI-style callers."""
    return asyncio.run(run_pipeline_batch(**kwargs))
