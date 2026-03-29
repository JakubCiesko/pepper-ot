import colorsys
from dataclasses import dataclass
import math

import fast_colorthief
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.inference.types import SceneGraphEdge
from app.schemas.config import SGGRule
from app.schemas.config import SGGRulesConfig


@dataclass
class RuleBasedSceneGraphBackend:
    rules_config: SGGRulesConfig

    def generate(
        self, image: Image.Image | None, detections: list[InferenceDetectionObject]
    ) -> SceneGraph:
        if not self.rules_config.enabled:
            return SceneGraph()
        dets = [d for d in detections if d.object_id is not None]
        if not dets:
            return SceneGraph()
        det_id_to_label = {d.object_id: d.label for d in detections}
        centers = {d.object_id: _center(d.bbox) for d in dets}
        no_label_edges: list[SceneGraphEdge] = []
        # rules
        for rule in self.rules_config.rule_list:
            no_label_edges.extend(self._apply_rule(rule, dets, centers))
        # colors
        if image is not None:
            if isinstance(image, Image.Image):
                # PIL's internal conversion is highly optimized in C
                img_np = np.asarray(image.convert("RGBA"))
            else:
                img_np = np.asarray(image)
                # Pad NumPy arrays to 4 channels if they arrive as 3-channel (RGB/BGR)
                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    alpha_channel = np.full(
                        (img_np.shape[0], img_np.shape[1], 1), 255, dtype=img_np.dtype
                    )
                    img_np = np.concatenate((img_np, alpha_channel), axis=2)
                # Handle edge case: grayscale images (1 channel)
                elif img_np.ndim == 2:
                    img_np = np.stack(
                        (img_np, img_np, img_np, np.full_like(img_np, 255)), axis=2
                    )
            img_h, img_w = img_np.shape[:2]
            for d in detections:
                x1, y1, x2, y2 = map(int, d.bbox)
                x1, x2 = max(0, x1), min(img_w, x2)
                y1, y2 = max(0, y1), min(img_h, y2)

                w, h = x2 - x1, y2 - y1
                # filter nonsense
                if w <= 0 or h <= 0:
                    continue
                # skip small bboxes TODO:HARDCODED VALUES OUT, or make relative:
                min_edge_ratio = min(w, h) / min(img_w, img_h)
                if min_edge_ratio < 0.05:
                    continue
                # if min(w, h) < 120:
                #     continue

                crop_np = img_np[y1:y2, x1:x2]
                try:
                    if not crop_np.flags["C_CONTIGUOUS"]:
                        crop_np = np.ascontiguousarray(crop_np)
                    color_rel = _extract_color(crop_np)
                    if color_rel:
                        no_label_edges.append(
                            SceneGraphEdge(
                                sub=d.object_id,
                                rel=color_rel,
                                obj=d.object_id,
                            )
                        )
                except Exception:
                    continue

        label_edges = [
            SceneGraphEdge(
                sub=f"{det_id_to_label[edge.sub]}_{edge.sub}",
                rel=edge.rel,
                obj=f"{det_id_to_label[edge.obj]}_{edge.obj}",
            )
            for edge in no_label_edges
        ]
        return SceneGraph(no_label_edges=no_label_edges, edges=label_edges, raw="")

    def _apply_rule(
        self,
        rule: SGGRule,
        dets: list[InferenceDetectionObject],
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
                            sub=sub.object_id,
                            rel=rule.predicate,
                            obj=obj.object_id,
                        )
                    )
        return edges

    def _match(
        self,
        rule: SGGRule,
        sub: InferenceDetectionObject,
        obj: InferenceDetectionObject,
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
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _inside_ratio(inner: list[float], outer: list[float]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
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


def _center_crop(img, ratio=0.5):
    h, w = img.shape[:2]
    dh, dw = int(h * ratio), int(w * ratio)
    y1 = (h - dh) // 2
    x1 = (w - dw) // 2
    return img[y1 : y1 + dh, x1 : x1 + dw]


def _extract_color(image_np: NDArray, conf_threshold: float = 0.5) -> str | None:
    # TODO: test, vibe coded
    image_np = _center_crop(image_np)
    palette = fast_colorthief.get_palette(image_np, quality=5, color_count=5)
    if not palette:
        return None
    votes = {}
    for rgb in palette:
        c = _rgb_to_bucket(rgb)
        votes[c] = votes.get(c, 0) + 1
    color, count = max(votes.items(), key=lambda kv: kv[1])
    confidence = count / max(1, len(palette))
    if confidence >= conf_threshold:
        return color
    return None


def _rgb_to_bucket(rgb: tuple[int, int, int]) -> str:
    r8, g8, b8 = rgb
    r, g, b = r8 / 255.0, g8 / 255.0, b8 / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360.0

    # Achromatic handling first (most important for robustness)
    if v <= 0.14:
        return "is_black"
    if s <= 0.10 and v >= 0.90:
        return "is_white"
    if s <= 0.16:
        # light vs dark gray split (optional; keeps richer vocabulary)
        return "is_light_gray" if v >= 0.60 else "is_gray"

    # Brown as dark orange range
    if 15 <= h_deg < 45 and 0.20 <= v <= 0.75:
        return "is_brown"

    # Hue buckets (chromatic)
    if h_deg >= 345 or h_deg < 12:
        return "is_red"
    if 12 <= h_deg < 25:
        return "is_orange"
    if 25 <= h_deg < 50:
        return "is_yellow"
    if 50 <= h_deg < 75:
        return "is_lime"
    if 75 <= h_deg < 160:
        return "is_green"
    if 160 <= h_deg < 190:
        return "is_cyan"
    if 190 <= h_deg < 230:
        return "is_blue"
    if 230 <= h_deg < 255:
        return "is_navy"
    if 255 <= h_deg < 285:
        return "is_purple"
    if 285 <= h_deg < 330:
        return "is_magenta"
    if 330 <= h_deg < 345:
        return "is_pink"

    # Fallback
    return "is_gray"
