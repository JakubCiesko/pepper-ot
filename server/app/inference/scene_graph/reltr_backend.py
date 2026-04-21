import asyncio
import logging
from pathlib import Path
import re
import threading
import tempfile
from typing import Any

from PIL import Image

from app.inference.scene_graph.reltr_predictor import VG_REL_CLASSES_ATTRIBUTEABLE
from app.inference.scene_graph.reltr_predictor import RelTRModel
from app.inference.scene_graph.reltr_predictor import predict_image
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.inference.types import SceneGraphEdge
from app.schemas.config import SGGRelTRConfig

logger = logging.getLogger(__name__)


# TODO: needs to be killed and started each time..
class RelTRSceneGraphGenerator:
    def __init__(self, config: SGGRelTRConfig):
        self.config = config
        self._attributeable_predicates = {
            str(p).strip().lower().replace(" ", "_")
            for p in VG_REL_CLASSES_ATTRIBUTEABLE
            if str(p).strip()
        }
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.project_root = Path(__file__).resolve().parents[4]
        self.repo_root = self.project_root / "reltr"
        self.state_dir = self.project_root / "server" / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.model = RelTRModel(
            self.repo_root, self.checkpoint_path, self.config.device
        )
        self._model_lock = threading.Lock()

    def update_runtime(self, config: SGGRelTRConfig):
        self.config = config

    def _get_bbox_to_id_mapping(
        self, objects: list[dict[str, Any]]
    ) -> dict[int, list[float]]:
        reltr_id_to_box: dict[int, list[float]] = {}
        for obj in objects:
            try:
                obj_id = int(obj.get("id"))
                bbox = obj.get("bbox") or []
                if len(bbox) != 4:
                    continue
                reltr_id_to_box[obj_id] = [float(v) for v in bbox]
            except Exception:
                continue
        return reltr_id_to_box

    def _filter_results(
        self,
        relationships: list[dict[str, Any]],
        reltr_id_to_box: dict[int, list[float]],
        id_to_label: dict[str, str],
        det_with_ids,
    ):
        filtered_no_label_edges: list[SceneGraphEdge] = []
        filtered_edges: list[SceneGraphEdge] = []
        seen: set[tuple[str, str, str]] = set()
        kept_binary = 0
        kept_unary_from_sub = 0
        kept_unary_from_obj = 0
        dropped_unmatched = 0
        dropped_invalid = 0

        for rel in relationships:
            logger.debug("REL: %s", rel)
            rel_name = self._normalize_token(str(rel.get("rel") or ""))
            if not rel_name:
                dropped_invalid += 1
                continue

            sub_raw = str(rel.get("sub") or "")
            obj_raw = str(rel.get("obj") or "")
            sub_label, sub_reltr_id = self._split_reltr_token(sub_raw)
            obj_label, obj_reltr_id = self._split_reltr_token(obj_raw)
            if sub_reltr_id is None or obj_reltr_id is None:
                dropped_invalid += 1
                continue
            sub_box = reltr_id_to_box.get(sub_reltr_id)
            obj_box = reltr_id_to_box.get(obj_reltr_id)
            if sub_box is None or obj_box is None:
                dropped_invalid += 1
                continue

            sub_track_id, _ = self._best_match_to_detection_id(sub_box, det_with_ids)
            obj_track_id, _ = self._best_match_to_detection_id(obj_box, det_with_ids)
            # TODO: test this properly, not having the last condition proved to make weird shit
            if (
                sub_track_id is not None
                and obj_track_id is not None
                and sub_track_id != obj_track_id
            ):
                sub_str = str(sub_track_id)
                obj_str = str(obj_track_id)
                key = (sub_str, rel_name, obj_str)
                if key in seen:
                    continue
                seen.add(key)
                filtered_no_label_edges.append(
                    SceneGraphEdge(sub=sub_str, rel=rel_name, obj=obj_str)
                )
                filtered_edges.append(
                    SceneGraphEdge(
                        sub=f"{id_to_label.get(sub_str, 'object')}_{sub_str}",
                        rel=rel_name,
                        obj=f"{id_to_label.get(obj_str, 'object')}_{obj_str}",
                    )
                )
                kept_binary += 1
                continue

            if rel_name in self._attributeable_predicates:
                if sub_track_id is not None:
                    sub_str = str(sub_track_id)
                    unary_rel = self._normalize_token(f"{rel_name}_{obj_label}")
                    if unary_rel:
                        key = (sub_str, unary_rel, sub_str)
                        if key not in seen:
                            seen.add(key)
                            filtered_no_label_edges.append(
                                SceneGraphEdge(sub=sub_str, rel=unary_rel, obj=sub_str)
                            )
                            filtered_edges.append(
                                SceneGraphEdge(
                                    sub=f"{id_to_label.get(sub_str, 'object')}_{sub_str}",
                                    rel=unary_rel,
                                    obj=f"{id_to_label.get(sub_str, 'object')}_{sub_str}",
                                )
                            )
                            kept_unary_from_sub += 1
                            continue
                if obj_track_id is not None:
                    obj_str = str(obj_track_id)
                    unary_rel = self._normalize_token(f"{rel_name}_{sub_label}")
                    if unary_rel:
                        key = (obj_str, unary_rel, obj_str)
                        if key not in seen:
                            seen.add(key)
                            filtered_no_label_edges.append(
                                SceneGraphEdge(sub=obj_str, rel=unary_rel, obj=obj_str)
                            )
                            filtered_edges.append(
                                SceneGraphEdge(
                                    sub=f"{id_to_label.get(obj_str, 'object')}_{obj_str}",
                                    rel=unary_rel,
                                    obj=f"{id_to_label.get(obj_str, 'object')}_{obj_str}",
                                )
                            )
                            kept_unary_from_obj += 1
                            continue

            dropped_unmatched += 1

        dropped = dropped_invalid + dropped_unmatched
        logger.info(
            "RelTR merge summary binary=%d unary_sub=%d unary_obj=%d dropped_unmatched=%d dropped_invalid=%d",
            kept_binary,
            kept_unary_from_sub,
            kept_unary_from_obj,
            dropped_unmatched,
            dropped_invalid,
        )

        logger.info(
            "RelTR scene graph edges kept=%d dropped=%d (iou_threshold=%.2f)",
            len(filtered_no_label_edges),
            dropped,
            self.config.iou_match_threshold,
        )
        return filtered_edges, filtered_no_label_edges

    def _process_reltr_output(
        self, reltr_output: dict[str, Any], det_with_ids: list[InferenceDetectionObject]
    ) -> SceneGraph:
        objects = reltr_output.get("objects") or []
        relationships = reltr_output.get("relationships") or []
        if not objects or not relationships:
            return SceneGraph(raw=reltr_output)

        reltr_id_to_box = self._get_bbox_to_id_mapping(objects)
        id_to_label = {
            str(d.object_id): (d.label or "object").strip() for d in det_with_ids
        }

        filtered_edges, filtered_no_label_edges = self._filter_results(
            relationships, reltr_id_to_box, id_to_label, det_with_ids
        )
        logger.info(
            "FIltereddedges=%s, nonlabeledges=%s",
            filtered_edges,
            filtered_no_label_edges,
        )
        return SceneGraph(
            edges=filtered_edges,
            no_label_edges=filtered_no_label_edges,
            raw=reltr_output,
        )

    async def generate(
        self, image: Image.Image | None, detections: list[InferenceDetectionObject]
    ) -> SceneGraph:
        return await asyncio.to_thread(self.generate_sync, image, detections)

    def generate_sync(
        self, image: Image.Image | None, detections: list[InferenceDetectionObject]
    ) -> SceneGraph:
        if not self.config.enabled:
            return SceneGraph()
        if image is None:
            logger.info("RelTR scene graph skipped: image is None")
            return SceneGraph()
        if not self.config.checkpoint_path:
            logger.warning("RelTR enabled but checkpoint_path is not configured")
            return SceneGraph()
        det_with_ids = [d for d in detections if d.object_id is not None]
        if not det_with_ids:
            logger.info(
                "RelTR scene graph skipped: no tracked detections with object_id"
            )
            return SceneGraph()

        with self._model_lock:
            if self.model.model is None:
                logger.warning(
                    "RelTR enabled but model is not yet configured, building model from scratch, device=%s, checkpoint path=%s",
                    self.model._device,
                    self.model.checkpoint_path,
                )
                self.model.build_model()
            reltr_output = self._run_reltr(image)
        return self._process_reltr_output(reltr_output, det_with_ids)

    def _run_reltr(self, image: Image.Image) -> dict[str, Any]:

        with tempfile.NamedTemporaryFile(
            dir=self.state_dir, prefix="reltr_input_", suffix=".jpg", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        image.convert("RGB").save(tmp_path, format="JPEG")
        output = {"objects": [], "relationships": []}
        try:
            pred = predict_image(
                model=self.model,
                image_path=tmp_path,
                device=self.config.device,
                threshold=self.config.threshold,
                topk=self.config.topk,
            )
            output = pred.to_dict()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete temporary RelTR image: %s", tmp_path)
        return output

    @staticmethod
    def _extract_tail_int(token: str) -> int | None:
        match = re.search(r"(\d+)$", token.strip())
        if not match:
            return None
        return int(match.group(1))

    @classmethod
    def _split_reltr_token(cls, token: str) -> tuple[str, int | None]:
        stripped = token.strip()
        reltr_id = cls._extract_tail_int(stripped)
        if reltr_id is None:
            return cls._normalize_token(stripped), None
        label = re.sub(r"_\d+$", "", stripped)
        return cls._normalize_token(label), reltr_id

    @staticmethod
    def _normalize_token(token: str) -> str:
        x = token.strip().lower().replace(" ", "_").replace("-", "_")
        x = re.sub(r"[^a-z0-9_]+", "", x)
        x = re.sub(r"_+", "_", x).strip("_")
        return x

    def _best_match_to_detection_id(
        self, reltr_box: list[float], detections: list[InferenceDetectionObject]
    ) -> tuple[int | str | None, float]:
        best_iou = 0.0
        best_id: int | str | None = None
        for det in detections:
            iou = self._bbox_iou(reltr_box, [float(v) for v in det.bbox])
            if iou > best_iou:
                best_iou = iou
                best_id = det.object_id
        if best_iou < float(self.config.iou_match_threshold):
            return None, best_iou
        return best_id, best_iou

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
