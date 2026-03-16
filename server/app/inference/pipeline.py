from contextlib import asynccontextmanager
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from app.inference.scene_graph.service import SceneGraphService
from app.inference.types import DetectionObject
from app.inference.types import PipelineResult
from app.inference.types import SceneGraph
from app.schemas.config import PipelineControls
from app.schemas.config import VisConfig
from app.schemas.robot import RobotMetadata

logger = logging.getLogger(__name__)


# TODO: think whether to have stage_status
@asynccontextmanager
async def timer(step_name: str, metrics: dict[str, float]):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    duration = t1 - t0
    metrics[step_name] = duration
    logger.info("%s took %f seconds", step_name, duration)


class PerceptionPipeline:
    def __init__(
        self,
        detector: Any,
        memory: Any,
        painter: Any,
        scene_graph_service: SceneGraphService,
        fusion_config,
        vis_config: VisConfig,
        pipeline_controls: PipelineControls,
    ):
        self.detector = detector
        self.memory = memory
        self.painter = painter
        self.scene_graph_service = scene_graph_service
        self.fusion_config = fusion_config
        self.vis_config = vis_config
        self.pipeline_controls = pipeline_controls

    def set_detection_threshold(self, threshold: float):
        logger.info("Setting detection threshold to: %.2f", threshold)
        self.detector.threshold = threshold

    async def process(
        self, image: Image.Image, robot_metadata: RobotMetadata | None = None
    ) -> PipelineResult:
        controls = self.pipeline_controls
        metrics: dict[str, float | str] = {}
        # TODO: drop stage_status completely
        stage_status: dict[str, dict[str, float | str]] = {}
        executed_stages: list[str] = []
        logger.info("Processing image with robot metadata=%s", robot_metadata)

        raw_detections = await self._run_detection(
            image, controls, metrics, stage_status, executed_stages
        )
        tracked_detections = await self._run_tracking(
            image,
            raw_detections,
            robot_metadata,
            controls,
            metrics,
            stage_status,
            executed_stages,
        )
        som_image = await self._run_som_paint(
            image, tracked_detections, controls, metrics, stage_status, executed_stages
        )
        # fallback
        if som_image is None:
            logger.info("SoM image is None, fallback to som_image=image")
            som_image = image

        scene_graph = await self._run_scene_graph(
            image,
            som_image,
            tracked_detections,
            controls,
            metrics,
            stage_status,
            executed_stages,
        )

        await self._run_scene_memory_update(
            scene_graph, controls, metrics, stage_status, executed_stages
        )

        total = sum(
            float(value)
            for key, value in metrics.items()
            if key.endswith("_time") and isinstance(value, (int, float))
        )
        metrics["total_processing"] = total

        return PipelineResult(
            raw_image=image,
            som_image=som_image,
            detections=tracked_detections,
            scene_graph=scene_graph,
            metrics=metrics,
            executed_stages=executed_stages,
        )

    async def _run_detection(
        self,
        image: Image.Image,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        stage_status: dict[str, dict[str, float | str]],
        executed_stages: list[str],
    ) -> list[DetectionObject]:
        if not controls.detect:
            stage_status["detect"] = {"status": "skipped", "reason": "disabled"}
            return []

        async with timer("detection_time", metrics):
            detections = self.detector.detect(image)
        stage_status["detect"] = {
            "status": "executed",
            "duration": float(metrics.get("detection_time", 0.0)),
        }
        executed_stages.append("detect")
        logger.info("Detected %d detections", len(detections))
        return detections

    async def _run_tracking(
        self,
        image: Image.Image,
        detections: list[DetectionObject],
        robot_metadata: RobotMetadata | None,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        stage_status: dict[str, dict[str, float | str]],
        executed_stages: list[str],
    ) -> list[DetectionObject]:
        if not controls.track_memory:
            for idx, det in enumerate(detections, start=1):
                det.object_id = idx
            stage_status["track_memory"] = {
                "status": "skipped",
                "reason": "disabled; assigned frame-local IDs",
            }
            return detections

        if not detections:
            stage_status["track_memory"] = {
                "status": "skipped",
                "reason": "no detections",
            }
            return detections

        async with timer("memory_update_time", metrics):
            tracked = self.memory.update(
                image, detections, robot_metadata, self.fusion_config
            )
        stage_status["track_memory"] = {
            "status": "executed",
            "duration": float(metrics.get("memory_update_time", 0.0)),
        }
        executed_stages.append("track_memory")
        logger.info("%d tracked detections after memory update", len(tracked))
        return tracked

    async def _run_som_paint(
        self,
        image: Image.Image,
        detections: list[DetectionObject],
        controls: PipelineControls,
        metrics: dict[str, float | str],
        stage_status: dict[str, dict[str, float | str]],
        executed_stages: list[str],
    ) -> np.ndarray | None:
        if not controls.paint_som:
            stage_status["paint_som"] = {"status": "skipped", "reason": "disabled"}
            return None
        if not controls.detect:
            stage_status["paint_som"] = {
                "status": "skipped",
                "reason": "detection disabled",
            }
            return None

        async with timer("som_image_paint_time", metrics):
            image_np = np.array(image)
            som_image = self.painter.paint(
                image_np,
                detections,
                bbox=self.vis_config.show_bbox,
                mask=self.vis_config.show_mask,
                polygon=self.vis_config.show_polygon,
                class_names=self.vis_config.show_labels,
            )
        stage_status["paint_som"] = {
            "status": "executed",
            "duration": float(metrics.get("som_image_paint_time", 0.0)),
        }
        executed_stages.append("paint_som")
        return som_image

    async def _run_scene_graph(
        self,
        image: Image.Image,
        som_image: np.ndarray | None,
        detections: list[DetectionObject],
        controls: PipelineControls,
        metrics: dict[str, float | str],
        stage_status: dict[str, dict[str, float | str]],
        executed_stages: list[str],
    ) -> SceneGraph | None:
        if not controls.scene_graph:
            stage_status["scene_graph"] = {"status": "skipped", "reason": "disabled"}
            return None

        # Direct-image mode: no detection and no SoM.
        if not controls.detect and som_image is None:
            route = "direct_image"
        else:
            route = "som_image" if som_image is not None else "raw_image"

        async with timer("scene_graph_generation_time", metrics):
            scene_graph = await self.scene_graph_service.generate(
                detections,
                som_image=som_image,
                raw_image=image,
            )
        stage_status["scene_graph"] = {
            "status": "executed",
            "duration": float(metrics.get("scene_graph_generation_time", 0.0)),
            "reason": route,
        }
        executed_stages.append("scene_graph")
        return scene_graph

    async def _run_scene_memory_update(
        self,
        scene_graph: SceneGraph | None,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        stage_status: dict[str, dict[str, float | str]],
        executed_stages: list[str],
    ) -> None:
        if not controls.update_scene_memory:
            stage_status["update_scene_memory"] = {
                "status": "skipped",
                "reason": "disabled",
            }
            return
        if scene_graph is None:
            stage_status["update_scene_memory"] = {
                "status": "skipped",
                "reason": "scene_graph unavailable",
            }
            return

        async with timer("scene_graph_memory_update_time", metrics):
            self.memory.update_scene_graph(scene_graph)
        stage_status["update_scene_memory"] = {
            "status": "executed",
            "duration": float(metrics.get("scene_graph_memory_update_time", 0.0)),
        }
        executed_stages.append("update_scene_memory")
