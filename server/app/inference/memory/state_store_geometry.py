import math

from PIL import Image

from app.inference.types import DetectionObject
from app.schemas.robot import RobotMetadata


class SceneMemoryStoreGeometryMixin:
    @staticmethod
    def compute_bearing(
        det: DetectionObject,
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ) -> tuple[float, float] | None:
        if robot_metadata is None:
            return None
        if robot_metadata.camera_hfov is None or robot_metadata.camera_vfov is None:
            return None
        if len(det.bbox) != 4:
            return None
        width = robot_metadata.image_width or image.width
        height = robot_metadata.image_height or image.height
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        yaw_rel = (0.5 - cx / width) * math.radians(robot_metadata.camera_hfov)
        pitch_rel = (0.5 - cy / height) * math.radians(robot_metadata.camera_vfov)
        base_yaw = (robot_metadata.body_yaw or 0.0) + robot_metadata.head_yaw
        base_pitch = robot_metadata.head_pitch
        return base_yaw + yaw_rel, base_pitch + pitch_rel

    def fuse_people_perception(
        self,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
        fusion_config,
    ) -> list[DetectionObject]:
        if robot_metadata is None or not robot_metadata.people:
            return detections

        width = robot_metadata.image_width or image.width
        height = robot_metadata.image_height or image.height
        hfov = robot_metadata.camera_hfov
        vfov = robot_metadata.camera_vfov
        if hfov is None or vfov is None or width <= 0 or height <= 0:
            return detections

        persons = [
            d for d in detections if d.label == "person" and d.object_id is not None
        ]
        others = [d for d in detections if d.label != "person" or d.object_id is None]

        match_thresh = getattr(fusion_config, "person_bbox_match_threshold_px", 10.0)
        base_px = getattr(fusion_config, "estimated_person_bbox_base_px", 80.0)
        min_px = getattr(fusion_config, "estimated_person_bbox_min_px", 40.0)
        max_px = getattr(fusion_config, "estimated_person_bbox_max_px", 200.0)

        base_yaw = (robot_metadata.body_yaw or 0.0) + robot_metadata.head_yaw
        base_pitch = robot_metadata.head_pitch

        def to_pixel(yaw, pitch):
            x = (0.5 - (yaw - base_yaw) / math.radians(hfov)) * width
            y = (0.5 - (pitch - base_pitch) / math.radians(vfov)) * height
            return x, y

        fused = []
        used_ids = set()
        for person in robot_metadata.people:
            px, py = to_pixel(person.yaw, person.pitch)
            matched = None
            for det in persons:
                if len(det.bbox) != 4:
                    continue
                x1, y1, x2, y2 = det.bbox
                if (x1 - match_thresh) <= px <= (x2 + match_thresh) and (
                    y1 - match_thresh
                ) <= py <= (y2 + match_thresh):
                    matched = det
                    break
            if matched:
                matched.confidence = max(matched.confidence, 1.0)
                used_ids.add(matched.object_id)
                fused.append(matched)
                continue

            scale = base_px / max(person.distance, 0.3)
            size = max(min(scale, max_px), min_px)
            x1 = max(0.0, px - size / 2)
            y1 = max(0.0, py - size / 2)
            x2 = min(width, px + size / 2)
            y2 = min(height, py + size / 2)
            if x2 <= x1 or y2 <= y1:
                continue

            det = DetectionObject(
                class_id=-1,
                label="person",
                confidence=1.0,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                object_id=None,
            )
            fused.append(det)

        remaining = [p for p in persons if p.object_id not in used_ids]
        return fused + remaining + others
