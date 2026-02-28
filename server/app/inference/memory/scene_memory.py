import logging
import threading

from PIL import Image

from app.inference.memory.state_store import SceneMemoryStore
from app.inference.tracking.associator import Associator
from app.inference.tracking.embeddings import FeatureExtractor
from app.inference.types import DetectionObject
from app.inference.types import SceneGraph
from app.schemas.robot import RobotMetadata
from app.schemas.scene import SceneState

logger = logging.getLogger(__file__)


class SceneMemory:
    """Manages the lifecycle of objects in the robot's world."""

    def __init__(
        self,
        memory_max_age_seconds: int = 60,
        memory_max_objects: int = 200,
        memory_max_relations: int = 500,
    ):
        logger.info("Initializing SceneMemory")
        self.store = SceneMemoryStore(
            memory_max_age_seconds=memory_max_age_seconds,
            memory_max_objects=memory_max_objects,
            memory_max_relations=memory_max_relations,
        )
        self._lock = threading.Lock()

        # Dependencies
        self.extractor = FeatureExtractor()
        self.associator = Associator()
        logger.info("SceneMemory initialized")

    @property
    def tracks(self):
        return self.store.tracks

    @property
    def objects_state(self):
        return self.store.objects_state

    @property
    def relations_state(self):
        return self.store.relations_state

    @property
    def next_id(self) -> int:
        return self.store.next_id

    @next_id.setter
    def next_id(self, value: int):
        self.store.next_id = value

    def update(
        self,
        image: Image.Image,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None = None,
        fusion_config=None,
    ) -> list[DetectionObject]:
        """
        Main pipeline step:
        1. Extract embeddings
        2. Associate detections with existing tracks
        3. Update IDs, fuse optional people metadata, and update persistent state
        """
        logger.info(f"Updating SceneMemory with {len(detections)} detections")
        if not detections:
            logger.info("No detections provided, pruning memory and returning.")
            with self._lock:
                self.store.prune_memory()
            return detections

        with self._lock:
            try:
                logger.debug("Extracting detection embeddings")
                embeddings = self.extractor.extract(image, detections)
            except Exception as exc:
                logger.exception(f"Embedding extraction failed: {exc}")
                return detections

            active_tracks_list = list(self.store.tracks.values())
            try:
                matches, un_tracks, un_dets = self.associator.match(
                    active_tracks_list, detections, embeddings
                )
            except Exception as exc:
                logger.exception(f"Track association failed: {exc}")
                matches, un_tracks, un_dets = (
                    [],
                    list(range(len(active_tracks_list))),
                    list(range(len(detections))),
                )

            logger.debug(f"Updating {len(matches)} matched tracks")
            for t_idx, d_idx in matches:
                if (
                    t_idx < 0
                    or d_idx < 0
                    or t_idx >= len(active_tracks_list)
                    or d_idx >= len(detections)
                    or d_idx >= len(embeddings)
                ):
                    logger.debug(
                        f"Skipping invalid match indices: track={t_idx}, det={d_idx}"
                    )
                    continue
                track = active_tracks_list[t_idx]
                det = detections[d_idx]
                emb = embeddings[d_idx]
                track.update(det, emb)
                det.object_id = track.id

            self.store.age_unmatched_tracks(un_tracks, active_tracks_list)

            logger.debug(f"Creating {len(un_dets)} new tracks")
            for d_idx in un_dets:
                if d_idx < 0 or d_idx >= len(detections) or d_idx >= len(embeddings):
                    logger.debug(
                        f"Skipping unmatched detection index out of bounds: {d_idx}"
                    )
                    continue
                det = detections[d_idx]
                emb = embeddings[d_idx]
                det.object_id = self.store.create_track(det, emb)

            try:
                fused_detections = self.store.fuse_people_perception(
                    detections, robot_metadata, image, fusion_config
                )
            except Exception as exc:
                logger.exception(f"People perception fusion failed: {exc}")
                fused_detections = detections

            try:
                self.store.update_objects_from_detections(
                    fused_detections, robot_metadata, image
                )
                self.store.prune_memory()
            except Exception as exc:
                logger.exception(f"State update/pruning failed: {exc}")

        # Return fused detections so downstream receives people-fusion output.
        return fused_detections

    def prune_memory(self):
        with self._lock:
            self.store.prune_memory()

    def set_limits(self, max_age_seconds: int, max_objects: int, max_relations: int):
        with self._lock:
            self.store.set_limits(max_age_seconds, max_objects, max_relations)

    def reset(self):
        with self._lock:
            self.store.reset()

    def update_scene_graph(self, scene_graph: SceneGraph):
        if scene_graph is None:
            return
        with self._lock:
            self.store.update_scene_graph(scene_graph)

    def scene_state(self) -> SceneState:
        with self._lock:
            logger.info("Returning scene state from memory")
            return self.store.scene_state()

    def upsert_scene_state(self, state: SceneState):
        """Merge external SceneState into memory (for manual injections)."""
        with self._lock:
            self.store.upsert_scene_state(state)

    def snapshot(self) -> list[dict]:
        """Return a lightweight view of the current tracked objects."""
        with self._lock:
            return self.store.snapshot()
