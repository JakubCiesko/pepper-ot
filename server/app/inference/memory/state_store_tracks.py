import time

from app.inference.types import InferenceDetectionObject
from app.inference.types import TrackedObject


class SceneMemoryStoreTracksMixin:
    def drop_dormant_tracks(
        self,
        unmatched_track_indices: list[int],
        active_tracks: list[TrackedObject],
        max_dormant_frames: int,
    ):
        if max_dormant_frames <= 0:
            return
        for t_idx in unmatched_track_indices:
            if t_idx < 0 or t_idx >= len(active_tracks):
                continue
            track = active_tracks[t_idx]
            if track.frames_since_seen > max_dormant_frames:
                self.tracks.pop(track.id, None)

    def set_limits(self, max_age_seconds: int, max_objects: int, max_relations: int):
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be > 0")
        if max_objects <= 0:
            raise ValueError("max_objects must be > 0")
        if max_relations <= 0:
            raise ValueError("max_relations must be > 0")
        self.memory_max_age_seconds = max_age_seconds
        self.memory_max_objects = max_objects
        self.memory_max_relations = max_relations

    def reset(self):
        self.tracks.clear()
        self.objects_state.clear()
        self.relations_state.clear()
        self.next_id = 1

    def create_track(self, det: InferenceDetectionObject, embedding) -> int:
        object_id = self.next_id
        self.tracks[object_id] = TrackedObject(
            id=object_id,
            label=det.label,
            embedding=embedding,
            bbox=det.bbox,
            confidence=det.confidence,
        )
        self.next_id += 1
        return object_id

    def age_unmatched_tracks(
        self, unmatched_track_indices: list[int], active_tracks: list[TrackedObject]
    ):
        for t_idx in unmatched_track_indices:
            if t_idx < 0 or t_idx >= len(active_tracks):
                continue
            track = active_tracks[t_idx]
            track.frames_since_seen += 1

    def prune_memory(self):
        now = time.time()
        cutoff = now - self.memory_max_age_seconds

        stale_object_ids = {
            obj_id
            for obj_id, obj in self.objects_state.items()
            if obj.last_seen < cutoff
        }

        if len(self.objects_state) > self.memory_max_objects:
            sorted_objs = sorted(self.objects_state.values(), key=lambda o: o.last_seen)
            overflow = len(self.objects_state) - self.memory_max_objects
            stale_object_ids.update(obj.id for obj in sorted_objs[:overflow])

        for obj_id in stale_object_ids:
            self.objects_state.pop(obj_id, None)

        stale_relation_keys = []
        for key, rel in self.relations_state.items():
            if rel.last_seen < cutoff:
                stale_relation_keys.append(key)
                continue
            if key[0] in stale_object_ids or key[2] in stale_object_ids:
                stale_relation_keys.append(key)
        for key in stale_relation_keys:
            self.relations_state.pop(key, None)

        if len(self.relations_state) > self.memory_max_relations:
            sorted_rels = sorted(
                self.relations_state.values(), key=lambda r: r.last_seen
            )
            overflow = len(self.relations_state) - self.memory_max_relations
            for rel in sorted_rels[:overflow]:
                key = (rel.subject_id, rel.predicate, rel.object_id)
                self.relations_state.pop(key, None)

        stale_track_ids = {
            track_id
            for track_id, track in self.tracks.items()
            if track.last_seen < cutoff
        }
        stale_track_ids.update(
            track_id for track_id in self.tracks if track_id not in self.objects_state
        )
        if len(self.tracks) > self.memory_max_objects:
            sorted_tracks = sorted(self.tracks.values(), key=lambda t: t.last_seen)
            overflow = len(self.tracks) - self.memory_max_objects
            stale_track_ids.update(track.id for track in sorted_tracks[:overflow])
        for track_id in stale_track_ids:
            self.tracks.pop(track_id, None)

    def snapshot(self) -> list[dict]:
        return [
            {
                "id": track.id,
                "label": track.label,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "last_seen": track.last_seen,
                "first_seen": track.first_seen,
                "hits": track.hits,
            }
            for track in self.tracks.values()
        ]
