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
        for det in detections:
            if det.object_id is None:
                continue
            bearing = self.compute_bearing(det, robot_metadata, image)
            current = self.objects_state.get(det.object_id)
            if current is None:
                self.objects_state[det.object_id] = TrackedObjectState(
                    id=det.object_id,
                    label=det.label,
                    status="active",
                    source="tracked",
                    attributes=[],
                    bearing_yaw=bearing[0] if bearing else None,
                    bearing_pitch=bearing[1] if bearing else None,
                    frame_id=robot_metadata.frame_id if robot_metadata else None,
                    scan_id=robot_metadata.scan_id if robot_metadata else None,
                    first_seen=now,
                    last_seen=now,
                    hits=1,
                    bbox=det.bbox,
                )
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
