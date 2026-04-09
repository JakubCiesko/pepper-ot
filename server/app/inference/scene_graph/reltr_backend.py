import logging
from pathlib import Path
import re
import tempfile
from typing import Any

from PIL import Image

from app.inference.scene_graph.reltr_predictor import predict_image
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.inference.types import SceneGraphEdge
from app.schemas.config import SGGRelTRConfig

logger = logging.getLogger(__name__)


class RelTRSceneGraphGenerator:
    def __init__(self, config: SGGRelTRConfig):
        self.config = config

    def update_runtime(self, config: SGGRelTRConfig):
        self.config = config

    async def generate(
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

        reltr_output = self._run_reltr(image)
        objects = reltr_output.get("objects") or []
        relationships = reltr_output.get("relationships") or []
        if not objects or not relationships:
            return SceneGraph(raw=reltr_output)

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

        id_to_label = {
            str(d.object_id): (d.label or "object").strip() for d in det_with_ids
        }
        filtered_no_label_edges: list[SceneGraphEdge] = []
        filtered_edges: list[SceneGraphEdge] = []
        dropped = 0

        for rel in relationships:
            logger.info("REL: %s", rel)
            rel_name = str(rel.get("rel") or "").strip().lower().replace(" ", "_")
            if not rel_name:
                dropped += 1
                continue

            sub_raw = str(rel.get("sub") or "")
            obj_raw = str(rel.get("obj") or "")
            sub_reltr_id = self._extract_tail_int(sub_raw)
            obj_reltr_id = self._extract_tail_int(obj_raw)
            if sub_reltr_id is None or obj_reltr_id is None:
                dropped += 1
                continue
            sub_box = reltr_id_to_box.get(sub_reltr_id)
            obj_box = reltr_id_to_box.get(obj_reltr_id)
            if sub_box is None or obj_box is None:
                dropped += 1
                continue

            sub_track_id = self._match_to_detection_id(sub_box, det_with_ids)
            obj_track_id = self._match_to_detection_id(obj_box, det_with_ids)
            if sub_track_id is None or obj_track_id is None:
                dropped += 1
                continue

            sub_str = str(sub_track_id)
            obj_str = str(obj_track_id)
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

        logger.info(
            "RelTR scene graph edges kept=%d dropped=%d (iou_threshold=%.2f)",
            len(filtered_no_label_edges),
            dropped,
            self.config.iou_match_threshold,
        )
        return SceneGraph(
            edges=filtered_edges,
            no_label_edges=filtered_no_label_edges,
            raw=reltr_output,
        )

    def _run_reltr(self, image: Image.Image) -> dict[str, Any]:
        checkpoint_path = Path(self.config.checkpoint_path)
        project_root = Path(__file__).resolve().parents[4]
        repo_root = project_root / "reltr"
        state_dir = project_root / "server" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=state_dir, prefix="reltr_input_", suffix=".jpg", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        image.convert("RGB").save(tmp_path, format="JPEG")
        try:
            pred = predict_image(
                repo_root=repo_root,
                checkpoint_path=checkpoint_path,
                image_path=tmp_path,
                dataset=self.config.dataset,
                device=self.config.device,
                threshold=self.config.threshold,
                topk=self.config.topk,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete temporary RelTR image: %s", tmp_path)
        return pred.to_dict()

    @staticmethod
    def _extract_tail_int(token: str) -> int | None:
        match = re.search(r"(\d+)$", token.strip())
        if not match:
            return None
        return int(match.group(1))

    def _match_to_detection_id(
        self, reltr_box: list[float], detections: list[InferenceDetectionObject]
    ) -> int | str | None:
        best_iou = 0.0
        best_id: int | str | None = None
        for det in detections:
            iou = self._bbox_iou(reltr_box, [float(v) for v in det.bbox])
            if iou > best_iou:
                best_iou = iou
                best_id = det.object_id
        if best_iou < float(self.config.iou_match_threshold):
            return None
        return best_id

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
