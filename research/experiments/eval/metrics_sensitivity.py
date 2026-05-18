from hashlib import md5
from pathlib import Path
import re

from research.experiments.io import load_json


def build_vocab_sensitivity_curve(context_rot_results: dict) -> dict:
    """Build a vocabulary-size sensitivity curve from context-rot results.

    Args:
        context_rot_results: Parsed context_rot.json mapping level keys to
            context-rot statistics.

    Returns:
        Dictionary with a sorted points list. Each point contains vocabulary
        size, image count, average relationship count, and any available F1
        metrics extracted from the level summary.
    """
    points: list[dict] = []
    for key, value in context_rot_results.items():
        if not key.startswith("vocab_") or not isinstance(value, dict):
            continue
        match = re.search(r"vocab_(\d+)|size_(\d+)", key)
        if not match:
            continue
        vocab_size = int(next(group for group in match.groups() if group))
        point = {
            "vocab_size": vocab_size,
            "images": int(value.get("images", 0)),
            "relationship_count_avg": float(value.get("relationship_count_avg", 0.0)),
        }
        summary = value.get("metrics_summary", {})
        if "triplet_f1_avg" in value:
            point["triplet_f1_avg"] = float(value.get("triplet_f1_avg", 0.0))
        elif isinstance(summary, dict):
            point["triplet_f1_avg"] = float(
                summary.get("strict_triplet_micro", {}).get("f1", 0.0)
            )
        if "attribute_f1_avg" in value:
            point["attribute_f1_avg"] = float(value.get("attribute_f1_avg", 0.0))
        elif isinstance(summary, dict):
            point["attribute_f1_avg"] = float(
                summary.get("attribute_micro", {}).get("f1", 0.0)
            )
        if "pair_ordered_f1_avg" in value:
            point["pair_ordered_f1_avg"] = float(value.get("pair_ordered_f1_avg", 0.0))
        elif isinstance(summary, dict):
            point["pair_ordered_f1_avg"] = float(
                summary.get("pair_ordered_micro", {}).get("f1", 0.0)
            )
        if "pair_unordered_f1_avg" in value:
            point["pair_unordered_f1_avg"] = float(
                value.get("pair_unordered_f1_avg", 0.0)
            )
        elif isinstance(summary, dict):
            point["pair_unordered_f1_avg"] = float(
                summary.get("pair_unordered_micro", {}).get("f1", 0.0)
            )
        points.append(point)
    points.sort(key=lambda item: item["vocab_size"])
    return {"points": points}


def _prompt_fingerprint(system_prompt: str, user_prompt: str) -> str:
    """Create a stable short identifier for a system/user prompt pair."""
    payload = f"{system_prompt}\n---\n{user_prompt}".encode()
    return md5(payload).hexdigest()[:12]


def build_prompt_sensitivity_table(*, runs_root: Path) -> dict:
    """Collect prompt and model sensitivity rows across completed runs.

    Args:
        runs_root: Directory containing experiment run subdirectories with
            run_metadata.json and optional metrics_scene_graph_summary.json.

    Returns:
        Dictionary with rows. Each row records run ID, prompt fingerprint,
        provider, model ID, and available scene-graph F1 metrics. Missing
        runs_root returns an empty rows list.
    """
    rows: list[dict] = []
    if not runs_root.exists():
        return {"rows": rows}

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        metadata = load_json(run_dir / "run_metadata.json", default={})
        config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
        draft = config.get("draft_scene_graph", {}) if isinstance(config, dict) else {}
        model = config.get("draft_sgg_model", {}) if isinstance(config, dict) else {}
        summary = load_json(run_dir / "metrics_scene_graph_summary.json", default={})
        if not draft:
            continue

        system_prompt = str(draft.get("system_prompt", ""))
        user_prompt = str(draft.get("user_prompt_template", ""))
        prompt_id = _prompt_fingerprint(system_prompt, user_prompt)
        rows.append(
            {
                "run_id": metadata.get("run_id", run_dir.name),
                "prompt_id": prompt_id,
                "provider": model.get("provider"),
                "model_id": model.get("model_id"),
                "strict_triplet_f1_micro": (
                    summary.get("strict_triplet_micro", {}).get("f1")
                    if isinstance(summary, dict)
                    else None
                ),
                "attribute_f1_micro": (
                    summary.get("attribute_micro", {}).get("f1")
                    if isinstance(summary, dict)
                    else None
                ),
                "pair_ordered_f1_micro": (
                    summary.get("pair_ordered_micro", {}).get("f1")
                    if isinstance(summary, dict)
                    else None
                ),
                "pair_unordered_f1_micro": (
                    summary.get("pair_unordered_micro", {}).get("f1")
                    if isinstance(summary, dict)
                    else None
                ),
            }
        )
    return {"rows": rows}
