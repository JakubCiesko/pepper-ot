import logging
import math

from PIL import Image

from app.inference.types import InferenceDetectionObject
from app.schemas.robot import PersonMetadata
from app.schemas.robot import RobotMetadata

logger = logging.getLogger(__name__)


class SceneMemoryStoreGeometryMixin:
    @staticmethod
    def _image_geometry(
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ) -> tuple[int, int, float, float] | None:
        if robot_metadata is None:
            return None
        if robot_metadata.camera_hfov is None or robot_metadata.camera_vfov is None:
            return None
        width = int(robot_metadata.image_width or image.width)
        height = int(robot_metadata.image_height or image.height)
        if width <= 0 or height <= 0:
            return None
        return (
            width,
            height,
            float(robot_metadata.camera_hfov),
            float(robot_metadata.camera_vfov),
        )

    @staticmethod
    def _base_angles(robot_metadata: RobotMetadata) -> tuple[float, float]:
        return (
            robot_metadata.body_yaw or 0.0
        ) + robot_metadata.head_yaw, robot_metadata.head_pitch

    def compute_bearing(
        self,
        det: InferenceDetectionObject,
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ) -> tuple[float, float] | None:
        geometry = self._image_geometry(robot_metadata, image)
        if geometry is None or len(det.bbox) != 4 or robot_metadata is None:
            return None
        width, height, hfov, vfov = geometry
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        yaw_rel = (0.5 - cx / width) * hfov
        pitch_rel = (0.5 - cy / height) * vfov
        base_yaw, base_pitch = self._base_angles(robot_metadata)
        return base_yaw + yaw_rel, base_pitch + pitch_rel

    def project_robot_person_to_pixel(
        self,
        person: PersonMetadata,
        robot_metadata: RobotMetadata,
        image: Image.Image,
    ) -> tuple[float, float] | None:
        geometry = self._image_geometry(robot_metadata, image)
        if geometry is None:
            return None
        width, height, hfov, vfov = geometry
        base_yaw, base_pitch = self._base_angles(robot_metadata)
        x = (0.5 - (person.yaw - base_yaw) / hfov) * width
        y = (0.5 - (person.pitch - base_pitch) / vfov) * height
        return x, y

    @staticmethod
    def angular_difference(
        detection_bearing: tuple[float, float] | None,
        person: PersonMetadata,
    ) -> tuple[float, float] | None:
        if detection_bearing is None:
            return None
        det_yaw, det_pitch = detection_bearing
        return abs(det_yaw - person.yaw), abs(det_pitch - person.pitch)

    @staticmethod
    def angular_similarity(
        dyaw: float,
        dpitch: float,
        *,
        sigma_yaw: float = 0.12,
        sigma_pitch: float = 0.08,
    ) -> float:
        angular_distance = math.sqrt(
            (dyaw / max(sigma_yaw, 1e-6)) ** 2 + (dpitch / max(sigma_pitch, 1e-6)) ** 2
        )
        return math.exp(-0.5 * angular_distance**2)

    @staticmethod
    def _bbox_contains_point(
        det: InferenceDetectionObject,
        px: float,
        py: float,
        margin: float,
    ) -> bool:
        if len(det.bbox) != 4:
            return False
        x1, y1, x2, y2 = det.bbox
        return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (
            y2 + margin
        )

    @staticmethod
    def distance_size_consistency(
        person: PersonMetadata,
        det: InferenceDetectionObject,
        *,
        base_px: float,
    ) -> float:
        if len(det.bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = det.bbox
        actual_size = max(x2 - x1, y2 - y1)
        predicted_size = base_px / max(person.distance, 0.3)
        if predicted_size <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(actual_size - predicted_size) / predicted_size)

    def candidate_person_detections_for_robot_person(
        self,
        person: PersonMetadata,
        persons: list[InferenceDetectionObject],
        robot_metadata: RobotMetadata,
        image: Image.Image,
        *,
        pixel_margin: float,
        yaw_threshold: float,
        pitch_threshold: float,
    ) -> list[InferenceDetectionObject]:
        projected = self.project_robot_person_to_pixel(person, robot_metadata, image)
        projected_candidates: list[InferenceDetectionObject] = []
        if projected is not None:
            px, py = projected
            projected_candidates = [
                det
                for det in persons
                if self._bbox_contains_point(det, px, py, pixel_margin)
            ]
            if projected_candidates:
                return projected_candidates

        candidates: list[InferenceDetectionObject] = []
        for det in persons:
            bearing = self.compute_bearing(det, robot_metadata, image)
            diff = self.angular_difference(bearing, person)
            if diff is None:
                continue
            dyaw, dpitch = diff
            if dyaw <= yaw_threshold and dpitch <= pitch_threshold:
                candidates.append(det)
        return candidates

    def score_robot_person_match(
        self,
        person: PersonMetadata,
        det: InferenceDetectionObject,
        robot_metadata: RobotMetadata,
        image: Image.Image,
        *,
        base_px: float,
    ) -> tuple[float, dict[str, float]] | None:
        if det.object_id is None:
            return None
        bearing = self.compute_bearing(det, robot_metadata, image)
        diff = self.angular_difference(bearing, person)
        if diff is None:
            return None
        dyaw, dpitch = diff
        angular = self.angular_similarity(dyaw, dpitch)
        previous_binding_bonus = (
            1.0 if self.get_bound_server_object_id(person.id) == det.object_id else 0.0
        )
        track = (
            self.tracks.get(det.object_id) if isinstance(det.object_id, int) else None
        )
        tracker_bonus = 0.0 if track is None else min(track.hits / 5.0, 1.0)
        distance_bonus = self.distance_size_consistency(person, det, base_px=base_px)
        score = (
            0.55 * angular
            + 0.25 * previous_binding_bonus
            + 0.15 * tracker_bonus
            + 0.05 * distance_bonus
        )
        breakdown = {
            "angular": angular,
            "previous_binding_bonus": previous_binding_bonus,
            "tracker_bonus": tracker_bonus,
            "distance_bonus": distance_bonus,
            "dyaw": dyaw,
            "dpitch": dpitch,
        }
        return score, breakdown

    def assign_robot_people_to_detections(
        self,
        persons: list[InferenceDetectionObject],
        robot_people: list[PersonMetadata],
        robot_metadata: RobotMetadata,
        image: Image.Image,
        fusion_config,
    ) -> tuple[
        list[tuple[PersonMetadata, InferenceDetectionObject, float]],
        list[PersonMetadata],
    ]:
        pixel_margin = getattr(fusion_config, "person_bbox_match_threshold_px", 10.0)
        base_px = getattr(fusion_config, "estimated_person_bbox_base_px", 80.0)
        yaw_threshold = getattr(fusion_config, "angular_yaw_threshold_rad", 0.20)
        pitch_threshold = getattr(fusion_config, "angular_pitch_threshold_rad", 0.15)

        scored_pairs: list[tuple[float, PersonMetadata, InferenceDetectionObject]] = []
        for person in robot_people:
            candidates = self.candidate_person_detections_for_robot_person(
                person,
                persons,
                robot_metadata,
                image,
                pixel_margin=pixel_margin,
                yaw_threshold=yaw_threshold,
                pitch_threshold=pitch_threshold,
            )
            for det in candidates:
                scored = self.score_robot_person_match(
                    person,
                    det,
                    robot_metadata,
                    image,
                    base_px=base_px,
                )
                if scored is None:
                    continue
                score, breakdown = scored
                logger.info(
                    "Pepper person candidate pepper_id=%s det_id=%s score=%.3f breakdown=%s",
                    person.id,
                    det.object_id,
                    score,
                    breakdown,
                )
                scored_pairs.append((score, person, det))

        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        matched_pepper_ids: set[int] = set()
        matched_detection_ids: set[int] = set()
        matches: list[tuple[PersonMetadata, InferenceDetectionObject, float]] = []
        for score, person, det in scored_pairs:
            det_identity = id(det)
            if person.id in matched_pepper_ids or det_identity in matched_detection_ids:
                continue
            matches.append((person, det, score))
            matched_pepper_ids.add(person.id)
            matched_detection_ids.add(det_identity)

        unmatched_people = [
            person for person in robot_people if person.id not in matched_pepper_ids
        ]
        return matches, unmatched_people

    def create_synthetic_person_detection(
        self,
        person: PersonMetadata,
        robot_metadata: RobotMetadata,
        image: Image.Image,
        fusion_config,
    ) -> InferenceDetectionObject | None:
        projected = self.project_robot_person_to_pixel(person, robot_metadata, image)
        geometry = self._image_geometry(robot_metadata, image)
        if projected is None or geometry is None:
            return None
        width, height, _, _ = geometry
        base_px = getattr(fusion_config, "estimated_person_bbox_base_px", 80.0)
        min_px = getattr(fusion_config, "estimated_person_bbox_min_px", 40.0)
        max_px = getattr(fusion_config, "estimated_person_bbox_max_px", 200.0)
        confidence = getattr(fusion_config, "synthetic_person_confidence", 0.65)

        px, py = projected
        scale = base_px / max(person.distance, 0.3)
        size = max(min(scale, max_px), min_px)
        x1 = max(0.0, px - size / 2)
        y1 = max(0.0, py - size / 2)
        x2 = min(width, px + size / 2)
        y2 = min(height, py + size / 2)
        if x2 <= x1 or y2 <= y1:
            return None
        return InferenceDetectionObject(
            class_id=-1,
            label="person",
            confidence=confidence,
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            object_id=None,
        )

    def fuse_people_perception(
        self,
        detections: list[InferenceDetectionObject],
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
        fusion_config,
    ) -> list[InferenceDetectionObject]:
        self.clear_frame_pepper_state()
        logger.info("Running fusion of detected people and robot-detected people")
        if robot_metadata is None or not robot_metadata.people:
            logger.info(
                "No people provided by the robot, ending fusion. Keeping original detections only"
            )
            return detections

        geometry = self._image_geometry(robot_metadata, image)
        if geometry is None:
            logger.info(
                "No information on horizontal or vertical field of view, or invalid width/height"
            )
            return detections

        persons = [
            d for d in detections if d.label == "person" and d.object_id is not None
        ]
        others = [d for d in detections if d.label != "person" or d.object_id is None]
        logger.info(
            "Server detection comprised of %d people and %d other detected objects",
            len(persons),
            len(others),
        )
        logger.info(
            "Robot metadata contain %d detected people", len(robot_metadata.people)
        )

        matches, unmatched_people = self.assign_robot_people_to_detections(
            persons,
            robot_metadata.people,
            robot_metadata,
            image,
            fusion_config,
        )

        matched_person_detections: list[InferenceDetectionObject] = []
        matched_detection_ids: set[int] = set()
        min_confidence = getattr(fusion_config, "matched_person_min_confidence", 0.85)
        timestamp = robot_metadata.timestamp

        for person, det, score in matches:
            matched_detection_ids.add(id(det))
            det.confidence = max(det.confidence, min_confidence)
            if isinstance(det.object_id, int):
                self.upsert_pepper_binding(
                    person.id,
                    det.object_id,
                    confidence=score,
                    timestamp=timestamp,
                )
            logger.info(
                "Pepper person matched pepper_id=%s det_id=%s det_bbox=%s score=%.3f",
                person.id,
                det.object_id,
                det.bbox,
                score,
            )
            matched_person_detections.append(det)

        synthetic_persons: list[InferenceDetectionObject] = []
        for person in unmatched_people:
            det = self.create_synthetic_person_detection(
                person,
                robot_metadata,
                image,
                fusion_config,
            )
            if det is None:
                logger.info(
                    "Skipping synthetic Pepper person pepper_id=%s because projected bbox was invalid",
                    person.id,
                )
                continue
            logger.info(
                "No server detection matching robot-detected person, creating artificial person detection pepper_id=%s det=%s",
                person.id,
                det.model_dump(),
            )
            self.note_pending_synthetic_pepper_detection(det, person.id)
            synthetic_persons.append(det)

        remaining = [p for p in persons if id(p) not in matched_detection_ids]
        return matched_person_detections + synthetic_persons + remaining + others
