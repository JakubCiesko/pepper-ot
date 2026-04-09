from hashlib import md5
from pathlib import Path

from research.experiments.io import load_json


def build_vocab_sensitivity_curve(context_rot_results: dict) -> dict:
    points: list[dict] = []
    for key, value in context_rot_results.items():
        if not key.startswith("vocab_") or not isinstance(value, dict):
            continue
        try:
            vocab_size = int(key.split("_", maxsplit=1)[1])
        except Exception:
            continue
        point = {
            "vocab_size": vocab_size,
            "images": int(value.get("images", 0)),
            "relationship_count_avg": float(value.get("relationship_count_avg", 0.0)),
        }
        if "triplet_f1_avg" in value:
            point["triplet_f1_avg"] = float(value.get("triplet_f1_avg", 0.0))
        if "attribute_f1_avg" in value:
            point["attribute_f1_avg"] = float(value.get("attribute_f1_avg", 0.0))
        if "pair_f1_avg" in value:
            point["pair_f1_avg"] = float(value.get("pair_f1_avg", 0.0))
        points.append(point)
    points.sort(key=lambda item: item["vocab_size"])
    return {"points": points}


def _prompt_fingerprint(system_prompt: str, user_prompt: str) -> str:
    payload = f"{system_prompt}\n---\n{user_prompt}".encode()
    return md5(payload).hexdigest()[:12]


def build_prompt_sensitivity_table(*, runs_root: Path) -> dict:
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
                "pair_f1_micro": (
                    summary.get("pair_micro", {}).get("f1")
                    if isinstance(summary, dict)
                    else None
                ),
            }
        )
    return {"rows": rows}
