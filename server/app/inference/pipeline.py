import logging

import numpy as np
from PIL import Image

from app.inference.detection.service import DetectionService
from app.inference.memory.scene_memory import SceneMemory
from app.inference.scene_graph.generation import SceneGraphGenerator
from app.inference.scene_graph.rules import RuleBasedSceneGraph
from app.inference.scene_graph.som import SoMPainter
from app.inference.types import PipelineResult
from app.schemas.config import VisConfig
from app.schemas.robot import RobotMetadata

logger = logging.getLogger(__name__)


class VisualPipeline:
    def __init__(
        self,
        detector: DetectionService,
        memory: SceneMemory,
        painter: SoMPainter,
        sgg: SceneGraphGenerator,
        rules_sgg: RuleBasedSceneGraph,
        sgg_mode: str,
        fusion_config,
        vis_config: VisConfig,
    ):
        self.detector = detector
        self.memory = memory
        self.painter = painter
        self.sgg = sgg
        self.rules_sgg = rules_sgg
        self.sgg_mode = sgg_mode
        self.fusion_config = fusion_config
        self.vis_config = vis_config

    def set_detection_threshold(self, threshold: float):
        logger.info(f"Setting detection threshold to: {threshold}")
        self.detector.threshold = threshold

    async def process(
        self, image: Image.Image, robot_metadata: RobotMetadata | None = None
    ) -> PipelineResult:
        """
        Runs the full See-Track-Understand loop.
        """
        # 1. DETECT (Get raw boxes, no IDs yet)
        logger.info(f"Processing image with metadata={robot_metadata}")
        raw_detections = self.detector.detect(image)
        logger.info(f"Detected {len(raw_detections)} detections")

        # 2. MEMORY (Assign Persistent IDs via ReID)
        # This modifies the detection objects in-place or returns new ones
        logger.info("Updating memory...")
        tracked_detections = self.memory.update(
            image, raw_detections, robot_metadata, self.fusion_config
        )
        logger.info(f"{len(tracked_detections)} Tracked detections after memory update")
        # 3. PAINT (Draw Set-of-Mark tags using the IDs)
        # We need numpy for the painter, but we keep PIL for the VLM if needed
        logger.info("Painting SoM over Image for VLM SGG")
        image_np = np.array(image)
        som_image = self.painter.paint(
            image_np,
            tracked_detections,
            bbox=self.vis_config.show_bbox,
            mask=self.vis_config.show_mask,
            polygon=self.vis_config.show_polygon,
            class_names=self.vis_config.show_labels,
        )

        # 4. UNDERSTAND (Generate Scene Graph from SoM Image)
        # We pass the tagged image so the VLM can reference "Object 1"
        logger.info("Running SceneGraph generation with SoM Image...")
        match self.sgg_mode:
            case "rules":
                rules_graph = self.rules_sgg.generate(tracked_detections)
                scene_graph = rules_graph
            case "vlm":
                scene_graph = await self.sgg.generate(som_image)
            case _:
                vlm_graph = await self.sgg.generate(som_image)
                rules_graph = self.rules_sgg.generate(tracked_detections)
                scene_graph = vlm_graph + rules_graph
        logger.info("Updating memory with generated scene graph")
        self.memory.update_scene_graph(scene_graph)

        return PipelineResult(
            raw_image=image,
            som_image=som_image,
            detections=tracked_detections,
            scene_graph=scene_graph,
        )
