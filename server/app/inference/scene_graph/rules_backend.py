from dataclasses import dataclass
import math

from app.inference.types import DetectionObject
from app.inference.types import SceneGraph
from app.inference.types import SceneGraphEdge
from app.schemas.config import SGGRule
from app.schemas.config import SGGRulesConfig


@dataclass
class RuleBasedSceneGraphBackend:
    rules_config: SGGRulesConfig

    def generate(self, detections: list[DetectionObject]) -> SceneGraph:
        if not self.rules_config.enabled:
            return SceneGraph()
        dets = [d for d in detections if d.object_id is not None]
        if not dets:
            return SceneGraph()

        centers = {d.object_id: _center(d.bbox) for d in dets}
        edges: list[SceneGraphEdge] = []
        for rule in self.rules_config.rule_list:
            edges.extend(self._apply_rule(rule, dets, centers))
        return SceneGraph.from_list([e.__dict__ for e in edges])

    def _apply_rule(
        self,
        rule: SGGRule,
        dets: list[DetectionObject],
        centers: dict[int, tuple[float, float]],
    ) -> list[SceneGraphEdge]:
        edges: list[SceneGraphEdge] = []
        for sub in dets:
            for obj in dets:
                if sub.object_id == obj.object_id:
                    continue
                if not _passes_constraints(rule, sub.label, obj.label):
                    continue
                if self._match(rule, sub, obj, centers):
                    edges.append(
                        SceneGraphEdge(
                            sub=str(sub.object_id),
                            rel=rule.predicate,
                            obj=str(obj.object_id),
                        )
                    )
        return edges

    def _match(
        self,
        rule: SGGRule,
        sub: DetectionObject,
        obj: DetectionObject,
        centers: dict[int, tuple[float, float]],
    ) -> bool:
        sx, sy = centers[sub.object_id]
        ox, oy = centers[obj.object_id]
        thresholds = rule.thresholds or {}

        match rule.type:
            case "spatial" | "space":
                dist = math.hypot(sx - ox, sy - oy)
                return _range_check(
                    dist,
                    thresholds,
                    min_key="min_distance",
                    max_key="max_distance",
                    fallback_key="center_distance_px",
                )
            case "directional" | "direction":
                axis = thresholds.get("axis", "x")
                delta = (sx - ox) if axis == "x" else (sy - oy)
                return _range_check(
                    delta,
                    thresholds,
                    min_key="min_delta",
                    max_key="max_delta",
                    fallback_key="threshold",
                )
            case "overlap":
                iou = _iou(sub.bbox, obj.bbox)
                return _range_check(
                    iou,
                    thresholds,
                    min_key="min_iou",
                    max_key="max_iou",
                    fallback_key="threshold",
                )
            case "containment" | "contain":
                inside = _inside_ratio(sub.bbox, obj.bbox)
                return _range_check(
                    inside,
                    thresholds,
                    min_key="min_inside",
                    max_key="max_inside",
                    fallback_key="threshold",
                )
            case "label_pair":
                return True
            case _:
                return False


def _passes_constraints(rule: SGGRule, sub_label: str, obj_label: str) -> bool:
    c = rule.constraints
    if c is None:
        return True
    if c.labels_any and sub_label not in c.labels_any and obj_label not in c.labels_any:
        return False
    if c.subject_labels and sub_label not in c.subject_labels:
        return False
    return not (c.object_labels and obj_label not in c.object_labels)


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _inside_ratio(inner: list[float], outer: list[float]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_inner = (ix2 - ix1) * (iy2 - iy1)
    return inter / area_inner if area_inner > 0 else 0.0


def _range_check(
    value: float,
    thresholds: dict,
    *,
    min_key: str,
    max_key: str,
    fallback_key: str,
) -> bool:
    min_val = thresholds.get(min_key)
    max_val = thresholds.get(max_key)

    if min_val is None and max_val is None and fallback_key in thresholds:
        min_val = thresholds.get(fallback_key)

    if min_val is not None and value < float(min_val):
        return False
    return not (max_val is not None and value > float(max_val))
