import asyncio
from contextlib import asynccontextmanager
import logging
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from app.inference.caption.service import CaptionInferenceService
from app.inference.qa.service import SceneQAGenerationService
from app.inference.scene_graph.service import SceneGraphService
from app.inference.types import InferenceDetectionObject
from app.inference.types import PipelineResult
from app.inference.types import SceneGraph
from app.schemas.config import PipelineControls
from app.schemas.config import VisConfig
from app.schemas.robot import RobotMetadata
from app.schemas.scene import SceneCaptionState
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def stage_timer(step_name: str, metrics: dict[str, float | str]):
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
        caption_service: CaptionInferenceService | None,
        qa_service: SceneQAGenerationService | None,
        fusion_config,
        vis_config: VisConfig,
        pipeline_controls: PipelineControls,
    ):
        self.detector = detector
        self.memory = memory
        self.painter = painter
        self.scene_graph_service = scene_graph_service
        self.caption_service = caption_service
        self.qa_service = qa_service
        self.fusion_config = fusion_config
        self.vis_config = vis_config
        self.pipeline_controls = pipeline_controls
        self._detector_lock = threading.Lock()

    # SINGLE SOURCE OF TRUTH
    async def process(
        self, image: Image.Image, robot_metadata: RobotMetadata | None = None
    ) -> PipelineResult:
        if self.pipeline_controls.parallel_execution:
            return await self.process_in_parallel(image, robot_metadata)
        return await self.process_sequentially(image, robot_metadata)

    async def process_sequentially(
        self, image: Image.Image, robot_metadata: RobotMetadata | None = None
    ) -> PipelineResult:
        wall_start = time.perf_counter()
        controls = self.pipeline_controls
        metrics: dict[str, float | str] = {}
        caption_text: str | None = None
        caption_provider: str | None = None
        caption_model_id: str | None = None

        executed_stages: list[str] = []
        logger.info("Processing image with robot metadata=%s", robot_metadata)
        caption_text, caption_provider, caption_model_id = await self._run_caption(
            image, controls, metrics, executed_stages
        )

        raw_detections = await self._run_detection(
            image, controls, metrics, executed_stages
        )
        tracked_detections = await self._run_tracking(
            image,
            raw_detections,
            robot_metadata,
            controls,
            metrics,
            executed_stages,
        )
        som_image = await self._render_som_overlay(
            image, tracked_detections, controls, metrics, executed_stages
        )
        # fallback
        if som_image is None:
            logger.info("SoM image is None, fallback to som_image=image")
            som_image = image

        scene_state = self.memory.scene_state() if self.memory else None
        scene_graph = await self._run_scene_graph(
            image,
            som_image,
            tracked_detections,
            controls,
            metrics,
            executed_stages,
            caption_text,
            scene_state=scene_state,
        )
        qa_pairs = await self._run_qa_generation(
            scene_graph=scene_graph,
            detections=tracked_detections,
            caption_text=caption_text,
            controls=controls,
            metrics=metrics,
            executed_stages=executed_stages,
        )

        await self._run_caption_memory_update(
            caption_text,
            caption_provider,
            caption_model_id,
            robot_metadata,
            metrics,
            executed_stages,
        )

        await self._update_scene_memory_from_graph(
            scene_graph, controls, metrics, executed_stages
        )

        self._finalize_metrics(metrics, wall_start)

        return PipelineResult(
            raw_image=image,
            som_image=som_image,
            detections=tracked_detections,
            scene_graph=scene_graph,
            caption=caption_text,
            caption_provider=caption_provider,
            caption_model_id=caption_model_id,
            qa_pairs=qa_pairs,
            metrics=metrics,
            executed_stages=executed_stages,
        )

    async def process_in_parallel(
        self, image: Image.Image, robot_metadata: RobotMetadata | None = None
    ) -> PipelineResult:
        wall_start = time.perf_counter()
        controls = self.pipeline_controls
        metrics: dict[str, float | str] = {}
        logger.info(
            "Processing image with robot metadata=%s parallel_execution=true",
            robot_metadata,
        )

        caption_image = (
            self._copy_image_for_parallel(image) if controls.caption else image
        )
        detection_image = (
            self._copy_image_for_parallel(image) if controls.detect else image
        )
        caption_task = asyncio.create_task(
            self._run_caption_stage(caption_image, controls, metrics)
        )
        detection_task = asyncio.create_task(
            self._run_detection_stage(
                detection_image, controls, metrics, offload_to_thread=True
            )
        )

        try:
            detection_executed, raw_detections = await detection_task
            post_detection_stages: list[str] = []
            tracked_detections = await self._run_tracking(
                image,
                raw_detections,
                robot_metadata,
                controls,
                metrics,
                post_detection_stages,
            )
            som_image = await self._render_som_overlay(
                image, tracked_detections, controls, metrics, post_detection_stages
            )
            if som_image is None:
                logger.info("SoM image is None, fallback to som_image=image")
                som_image = image

            caption_executed, caption_text, caption_provider, caption_model_id = (
                await caption_task
            )
        except Exception:
            if not caption_task.done():
                caption_task.cancel()
            raise
        executed_stages: list[str] = []
        if caption_executed:
            executed_stages.append("caption")
        if detection_executed:
            executed_stages.append("detect")
        executed_stages.extend(post_detection_stages)

        scene_state = self.memory.scene_state() if self.memory else None
        scene_graph = await self._run_scene_graph(
            image,
            som_image,
            tracked_detections,
            controls,
            metrics,
            executed_stages,
            caption_text,
            scene_state=scene_state,
        )
        qa_pairs = await self._run_qa_generation(
            scene_graph=scene_graph,
            detections=tracked_detections,
            caption_text=caption_text,
            controls=controls,
            metrics=metrics,
            executed_stages=executed_stages,
        )

        await self._run_caption_memory_update(
            caption_text,
            caption_provider,
            caption_model_id,
            robot_metadata,
            metrics,
            executed_stages,
        )

        await self._update_scene_memory_from_graph(
            scene_graph, controls, metrics, executed_stages
        )
        self._finalize_metrics(metrics, wall_start)

        return PipelineResult(
            raw_image=image,
            som_image=som_image,
            detections=tracked_detections,
            scene_graph=scene_graph,
            caption=caption_text,
            caption_provider=caption_provider,
            caption_model_id=caption_model_id,
            qa_pairs=qa_pairs,
            metrics=metrics,
            executed_stages=executed_stages,
        )

    @staticmethod
    def _copy_image_for_parallel(image: Image.Image) -> Image.Image:
        return image.copy() if isinstance(image, Image.Image) else image

    def _detect_threadsafe(self, image: Image.Image) -> list[InferenceDetectionObject]:
        with self._detector_lock:
            return self.detector.detect(image)

    @staticmethod
    def _finalize_metrics(metrics: dict[str, float | str], wall_start: float) -> None:
        total = sum(
            float(value)
            for key, value in metrics.items()
            if key.endswith("_time") and isinstance(value, (int, float))
        )
        metrics["total_processing"] = total
        metrics["wall_processing_time"] = time.perf_counter() - wall_start

    async def _run_caption_stage(
        self,
        image: Image.Image,
        controls: PipelineControls,
        metrics: dict[str, float | str],
    ) -> tuple[bool, str | None, str | None, str | None]:
        if not controls.caption:
            return False, None, None, None
        if self.caption_service is None:
            logger.warning(
                "Pipeline caption stage enabled but caption_service is not configured"
            )
            return False, None, None, None

        try:
            async with stage_timer("caption_time", metrics):
                result = await self.caption_service.caption_image(image)
            return True, result.text, result.provider, result.model_id
        except Exception as exc:
            logger.warning("Caption stage failed, continuing pipeline: %s", exc)
            return False, None, None, None

    async def _run_caption(
        self,
        image: Image.Image,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> tuple[str | None, str | None, str | None]:
        executed, text, provider, model_id = await self._run_caption_stage(
            image, controls, metrics
        )
        if executed:
            executed_stages.append("caption")
        return text, provider, model_id

    async def _run_caption_memory_update(
        self,
        caption_text: str | None,
        caption_provider: str | None,
        caption_model_id: str | None,
        robot_metadata: RobotMetadata | None,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> None:
        if not caption_text:
            return
        if self.memory is None:
            return
        if not hasattr(self.memory, "upsert_caption"):
            return

        now = time.time()
        caption_state = SceneCaptionState(
            id=str(uuid4()),
            text=caption_text,
            provider=caption_provider,
            model_id=caption_model_id,
            source="pipeline_caption",
            frame_id=robot_metadata.frame_id if robot_metadata else None,
            scan_id=robot_metadata.scan_id if robot_metadata else None,
            first_seen=now,
            last_seen=now,
            count=1,
        )
        async with stage_timer("caption_memory_update_time", metrics):
            self.memory.upsert_caption(caption_state)
        executed_stages.append("update_caption_memory")

    async def _run_detection(
        self,
        image: Image.Image,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> list[InferenceDetectionObject]:
        executed, detections = await self._run_detection_stage(
            image, controls, metrics, offload_to_thread=False
        )
        if executed:
            executed_stages.append("detect")
        return detections

    async def _run_detection_stage(
        self,
        image: Image.Image,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        *,
        offload_to_thread: bool,
    ) -> tuple[bool, list[InferenceDetectionObject]]:
        if not controls.detect:
            return False, []

        async with stage_timer("detection_time", metrics):
            if offload_to_thread:
                detections = await asyncio.to_thread(self._detect_threadsafe, image)
            else:
                detections = self.detector.detect(image)
        logger.info("Detected %d detections", len(detections))
        return True, detections

    async def _run_tracking(
        self,
        image: Image.Image,
        detections: list[InferenceDetectionObject],
        robot_metadata: RobotMetadata | None,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> list[InferenceDetectionObject]:
        if not controls.track_memory:
            for idx, det in enumerate(detections, start=1):
                det.object_id = idx
            return detections

        if not detections:
            return detections

        async with stage_timer("memory_update_time", metrics):
            tracked = self.memory.update(
                image, detections, robot_metadata, self.fusion_config
            )
        executed_stages.append("track_memory")
        logger.info("%d tracked detections after memory update", len(tracked))
        return tracked

    async def _render_som_overlay(
        self,
        image: Image.Image,
        detections: list[InferenceDetectionObject],
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> np.ndarray | None:
        if not controls.paint_som:
            return None
        if not controls.detect:
            return None

        async with stage_timer("som_image_paint_time", metrics):
            image_np = np.array(image)
            som_image = self.painter.paint(
                image_np,
                detections,
                bbox=self.vis_config.show_bbox,
                mask=self.vis_config.show_mask,
                polygon=self.vis_config.show_polygon,
                class_names=self.vis_config.show_labels,
            )
        executed_stages.append("paint_som")
        return som_image

    async def _run_scene_graph(
        self,
        image: Image.Image,
        som_image: np.ndarray | None,
        detections: list[InferenceDetectionObject],
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
        caption_text: str | None,
        scene_state: SceneState | None = None,
    ) -> SceneGraph | None:
        if not controls.scene_graph:
            return None

        async with stage_timer("scene_graph_generation_time", metrics):
            scene_graph = await self.scene_graph_service.generate(
                detections,
                som_image=som_image,
                raw_image=image,
                caption_text=caption_text,
                scene_state=scene_state,
            )
        executed_stages.append("scene_graph")
        return scene_graph

    async def _update_scene_memory_from_graph(
        self,
        scene_graph: SceneGraph | None,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> None:
        if not controls.update_scene_memory:
            return
        if scene_graph is None:
            return

        async with stage_timer("scene_graph_memory_update_time", metrics):
            self.memory.update_scene_graph(scene_graph)
        executed_stages.append("update_scene_memory")

    async def _run_qa_generation(
        self,
        *,
        scene_graph: SceneGraph | None,
        detections: list[InferenceDetectionObject],
        caption_text: str | None,
        controls: PipelineControls,
        metrics: dict[str, float | str],
        executed_stages: list[str],
    ) -> list[dict[str, str]]:
        if not controls.qa_generation:
            return []
        if self.qa_service is None:
            logger.warning(
                "Pipeline QA generation stage enabled but qa_service is not configured"
            )
            return []
        if scene_graph is None:
            return []

        try:
            async with stage_timer("qa_generation_time", metrics):
                qa_pairs = await self.qa_service.generate(
                    scene_graph=scene_graph,
                    detections=detections,
                    caption_text=caption_text,
                )
            if qa_pairs:
                executed_stages.append("qa_generation")
            return qa_pairs
        except Exception as exc:
            logger.warning("QA generation stage failed, continuing pipeline: %s", exc)
            return []
