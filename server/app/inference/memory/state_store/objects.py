import time

from PIL import Image

from app.inference.types import InferenceDetectionObject
from app.schemas.robot import RobotMetadata
from app.schemas.scene import TrackedObjectState


class SceneMemoryStoreObjectsMixin:
    def insert_object(self, obj: TrackedObjectState):
        if obj.id in self.objects_state:
            raise ValueError(f"Object with id={obj.id} already exists")
        self.objects_state[obj.id] = obj.model_copy(deep=True)
        self.next_id = max(self.next_id, obj.id + 1)

    def patch_object(self, object_id: int, updates: dict) -> TrackedObjectState:
        current = self.objects_state.get(object_id)
        if current is None:
            raise KeyError(f"Object with id={object_id} does not exist")
        for field in (
            "label",
            "status",
            "source",
            "attributes",
            "pepper_person_id",
            "robot_distance",
            "robot_engagement_zone",
            "robot_last_seen_ts",
            "bearing_yaw",
            "bearing_pitch",
            "frame_id",
            "scan_id",
            "first_seen",
            "last_seen",
            "hits",
            "bbox",
        ):
            if field in updates:
                setattr(current, field, updates[field])

        track = self.tracks.get(object_id)
        if track is not None:
            if "label" in updates:
                track.label = current.label
            if "bbox" in updates:
                track.bbox = current.bbox
            if "last_seen" in updates:
                track.last_seen = current.last_seen
            if "first_seen" in updates:
                track.first_seen = current.first_seen
            if "hits" in updates:
                track.hits = current.hits
        return current

    def delete_object(self, object_id: int, cascade_relations: bool = True) -> bool:
        removed = self.objects_state.pop(object_id, None)
        if removed is None:
            return False
        self.tracks.pop(object_id, None)
        if cascade_relations:
            keys_to_remove = [
                key
                for key in self.relations_state
                if key[0] == object_id or key[2] == object_id
            ]
            for key in keys_to_remove:
                self.relations_state.pop(key, None)
        return True

    def update_objects_from_detections(
        self,
        detections: list[InferenceDetectionObject],
        robot_metadata: RobotMetadata | None,
        image: Image.Image,
    ):
        now = time.time()
        people_by_id = self.people_by_id(robot_metadata)
        social_people_by_id = self.social_people_by_id(robot_metadata)
        for det in detections:
            if det.object_id is None:
                continue
            bearing = self.compute_bearing(det, robot_metadata, image)
            current = self.objects_state.get(det.object_id)
            if current is None:
                current = TrackedObjectState(
                    id=det.object_id,
                    label=det.label,
                    status="active",
                    source="tracked",
                    attributes=[],
                    pepper_person_id=None,
                    robot_distance=None,
                    robot_engagement_zone=None,
                    robot_last_seen_ts=None,
                    bearing_yaw=bearing[0] if bearing else None,
                    bearing_pitch=bearing[1] if bearing else None,
                    frame_id=robot_metadata.frame_id if robot_metadata else None,
                    scan_id=robot_metadata.scan_id if robot_metadata else None,
                    first_seen=now,
                    last_seen=now,
                    hits=1,
                    bbox=det.bbox,
                )
                self.objects_state[det.object_id] = current
            else:
                current.label = det.label
                current.bbox = det.bbox
                if bearing:
                    current.bearing_yaw, current.bearing_pitch = bearing
                if robot_metadata:
                    current.frame_id = robot_metadata.frame_id or current.frame_id
                    current.scan_id = robot_metadata.scan_id or current.scan_id
                current.last_seen = now
                current.hits += 1

            if det.label != "person" or not isinstance(det.object_id, int):
                continue

            pepper_person_id = self._frame_server_to_pepper.get(det.object_id)
            if pepper_person_id is None:
                continue
            self.update_person_robot_fields(
                current,
                pepper_person_id=pepper_person_id,
                robot_person=people_by_id.get(pepper_person_id),
                social_person=social_people_by_id.get(pepper_person_id),
                fallback_timestamp=robot_metadata.timestamp if robot_metadata else None,
            )
