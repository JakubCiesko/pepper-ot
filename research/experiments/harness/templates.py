from __future__ import annotations

from pathlib import Path

from research.experiments.io import load_json
from research.experiments.io import save_json


def write_ground_truth_template(
    run_dir: Path, out: Path | None = None, *, prefill_draft: bool = False
) -> Path:
    """Create a human-editable ground-truth scene graph template.

    Args:
        run_dir: Run directory containing detections.json and optionally
            draft_scene_graph.json.
        out: Optional destination path. Defaults to
            ground_truth_scene_graph.template.json in run_dir.
        prefill_draft: When true, copy draft relationships into each template
            row; otherwise start with empty relationship lists.

    Returns:
        Path to the written template JSON.

    Side Effects:
        Writes a JSON object keyed by image, with detected objects,
        relationships, annotation_source, and a short annotator comment.
    """
    detections = load_json(run_dir / "detections.json", default={})
    drafts = load_json(run_dir / "draft_scene_graph.json", default={})
    if out is None:
        out = run_dir / "ground_truth_scene_graph.template.json"

    keys = sorted(set(detections.keys()) | set(drafts.keys()))
    payload: dict[str, dict] = {}
    for key in keys:
        detected = detections.get(key, [])
        draft = drafts.get(key, {})
        payload[key] = {
            "objects": [
                {
                    "id": row.get("object_id", idx),
                    "label": row.get("label"),
                    "bbox": row.get("bbox"),
                }
                for idx, row in enumerate(detected, start=1)
            ],
            "relationships": draft.get("relationships", []) if prefill_draft else [],
            "annotation_source": (
                "draft_prefilled_template" if prefill_draft else "blank_template"
            ),
            "comment": "Annotate ground truth relationships over the listed object IDs.",
        }
    save_json(out, payload)
    return out
