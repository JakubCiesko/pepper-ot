import numpy as np
from PIL import Image

from .detection.service import DetectionService
from .memory.scene_memory import SceneMemory
from .scene_graph.generation import SceneGraphGenerator
from .scene_graph.som import SoMPainter
from .types import PipelineResult


class VisualPipeline:
    def __init__(
        self,
        detector: DetectionService,
        memory: SceneMemory,
        painter: SoMPainter,
        sgg: SceneGraphGenerator,
    ):
        self.detector = detector
        self.memory = memory
        self.painter = painter
        self.sgg = sgg

    async def process(self, image: Image.Image) -> PipelineResult:
        """
        Runs the full See-Track-Understand loop.
        """
        # 1. DETECT (Get raw boxes, no IDs yet)
        raw_detections = self.detector.detect(image)

        # 2. MEMORY (Assign Persistent IDs via ReID)
        # This modifies the detection objects in-place or returns new ones
        tracked_detections = self.memory.update(image, raw_detections)

        # 3. PAINT (Draw Set-of-Mark tags using the IDs)
        # We need numpy for the painter, but we keep PIL for the VLM if needed
        image_np = np.array(image)
        som_image = self.painter.paint(
            image_np,
            tracked_detections,
            bbox=True,
            mask=False,  # Turn on if you want segmentation masks
            polygon=False,
            class_names=True,  # Helps the VLM know "Object 1" is a "cup"
        )

        # 4. UNDERSTAND (Generate Scene Graph from SoM Image)
        # We pass the tagged image so the VLM can reference "Object 1"
        scene_graph = await self.sgg.generate(som_image, verbose=False)

        return PipelineResult(
            raw_image=image,
            som_image=som_image,
            detections=tracked_detections,
            scene_graph=scene_graph,
        )
