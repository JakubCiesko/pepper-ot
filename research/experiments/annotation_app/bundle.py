from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from research.experiments.io import load_json
from research.experiments.io import save_json


@dataclass(frozen=True)
class BundleImage:
    image_path: str
    raw_image_uri: str
    som_image_uri: str | None
    caption: str
    objects: list[dict[str, Any]]
    vocabulary: dict[str, list[str]]
    relationships: list[dict[str, str]]


def _safe_name(idx: int, path: str, prefix: str) -> str:
    base = Path(path).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return f"{idx:05d}_{prefix}_{base}"


def _json_for_html(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    text = text.replace("</", "<\\/")
    text = text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return text


def _collect_objects(item: dict[str, Any], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects = item.get("objects")
    if isinstance(objects, list) and objects:
        return objects
    collected: list[dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        collected.append(
            {
                "id": det.get("object_id", idx),
                "label": det.get("label"),
                "bbox": det.get("bbox"),
                "confidence": det.get("confidence"),
            }
        )
    return collected


def _collect_relationships(item: dict[str, Any]) -> list[dict[str, str]]:
    relationships = item.get("relationships")
    if not isinstance(relationships, list):
        return []
    out: list[dict[str, str]] = []
    for row in relationships:
        if not isinstance(row, dict):
            continue
        sub = row.get("sub")
        rel = row.get("rel")
        obj = row.get("obj")
        if sub is None or rel is None or obj is None:
            continue
        out.append({"sub": str(sub), "rel": str(rel), "obj": str(obj)})
    return out


def _collect_vocabulary(item: dict[str, Any], run_vocab: dict[str, Any]) -> dict[str, list[str]]:
    vocab = item.get("vocabulary")
    if isinstance(vocab, dict):
        predicates = [str(x) for x in vocab.get("predicates", []) if str(x).strip()]
        attributes = [str(x) for x in vocab.get("attributes", []) if str(x).strip()]
        return {"predicates": predicates, "attributes": attributes}
    predicates = [str(x) for x in run_vocab.get("predicates", []) if str(x).strip()]
    attributes = [str(x) for x in run_vocab.get("attributes", []) if str(x).strip()]
    return {"predicates": predicates, "attributes": attributes}


def build_annotation_bundle(run_dir: Path, out_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(run_dir / "run_metadata.json", default={})
    detections = load_json(run_dir / "detections.json", default={})
    descriptions = load_json(run_dir / "descriptions.json", default={})
    drafts = load_json(run_dir / "draft_scene_graph.json", default={})
    run_vocab = load_json(run_dir / "vocabulary_final.json", default={})

    raw_assets = out_dir / "assets" / "raw"
    som_assets = out_dir / "assets" / "som"
    raw_assets.mkdir(parents=True, exist_ok=True)
    som_assets.mkdir(parents=True, exist_ok=True)

    keys = sorted(set(detections.keys()) | set(descriptions.keys()) | set(drafts.keys()))
    items: list[dict[str, Any]] = []
    for idx, key in enumerate(keys, start=1):
        draft = drafts.get(key, {})
        detection_rows = detections.get(key, [])
        raw_image_path = Path(str(draft.get("image_path") or key))
        if not raw_image_path.is_absolute():
            raw_image_path = (run_dir / raw_image_path).resolve()
        if not raw_image_path.exists():
            raw_image_path = Path(str(key))
        som_image_path = draft.get("som_image_path")
        som_resolved = None
        if som_image_path:
            som_resolved = Path(str(som_image_path))
            if not som_resolved.is_absolute():
                som_resolved = (run_dir / som_resolved).resolve()
            if not som_resolved.exists():
                som_resolved = None
        if som_resolved is None:
            candidate = run_dir / "som_images_draft" / f"som_{raw_image_path.name}"
            if candidate.exists():
                som_resolved = candidate.resolve()

        raw_uri = None
        if raw_image_path.exists():
            raw_name = _safe_name(idx, key, "raw")
            raw_target = raw_assets / raw_name
            shutil.copy2(raw_image_path, raw_target)
            raw_uri = str(raw_target.relative_to(out_dir)).replace("\\", "/")

        som_uri = None
        if som_resolved is not None:
            som_name = _safe_name(idx, key, "som")
            som_target = som_assets / som_name
            shutil.copy2(som_resolved, som_target)
            som_uri = str(som_target.relative_to(out_dir)).replace("\\", "/")

        item = {
            "key": key,
            "image_path": key,
            "raw_image_uri": raw_uri,
            "som_image_uri": som_uri,
            "caption": str(draft.get("caption") or descriptions.get(key, {}).get("text") or ""),
            "objects": _collect_objects(draft, detection_rows),
            "vocabulary": _collect_vocabulary(draft, run_vocab),
            "draft_relationships": _collect_relationships(draft),
            "source_run_id": metadata.get("run_id"),
        }
        items.append(item)

    bundle = {
        "bundle_id": out_dir.name,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_run": metadata,
        "items": items,
    }

    save_json(out_dir / "bundle.json", bundle)
    (out_dir / "app.js").write_text(
        (Path(__file__).parent / "static" / "app.js").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "style.css").write_text(
        (Path(__file__).parent / "static" / "style.css").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    template = (Path(__file__).parent / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("{{BUNDLE_JSON}}", _json_for_html(bundle))
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return out_dir


def _extract_annotation_payload(payload: Any) -> dict[str, dict[str, list[dict[str, str]]]]:
    if isinstance(payload, dict) and "annotations" in payload and isinstance(
        payload["annotations"], dict
    ):
        payload = payload["annotations"]
    if not isinstance(payload, dict):
        raise ValueError("Annotation export must be a JSON object.")

    normalized: dict[str, dict[str, list[dict[str, str]]]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        relationships = value.get("relationships")
        if not isinstance(relationships, list):
            continue
        cleaned: list[dict[str, str]] = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            sub = rel.get("sub")
            relation = rel.get("rel")
            obj = rel.get("obj")
            if sub is None or relation is None or obj is None:
                continue
            cleaned.append({"sub": str(sub), "rel": str(relation), "obj": str(obj)})
        normalized[str(key)] = {"relationships": cleaned}
    return normalized


def import_annotation_export(run_dir: Path, annotations_path: Path) -> Path:
    run_dir = run_dir.resolve()
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    normalized = _extract_annotation_payload(annotations)
    out_path = run_dir / "ground_truth_scene_graph.json"
    save_json(out_path, normalized)
    return out_path
