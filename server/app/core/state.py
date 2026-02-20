from dataclasses import dataclass
import logging
from pathlib import Path

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.service import DetectionService
from app.inference.memory.scene_memory import SceneMemory
from app.inference.pipeline import VisualPipeline
from app.inference.scene_graph.generation import OntologyConfig
from app.inference.scene_graph.generation import SceneGraphGenerator
from app.inference.scene_graph.som import SoMPainter
from app.inference.scene_graph.vlm import LLMLabelerConfig
from app.inference.scene_graph.vlm import VLMBackend
from app.schemas.config import AppConfig
from app.services.chat import ChatService

logger = logging.getLogger(__name__)


@dataclass
class MLState:
    config: AppConfig | None = None
    pipeline: VisualPipeline | None = None
    chat_service: object | None = None
    initialized: bool = False

    async def initialize(self, config_path: str | None = None):
        if self.initialized:
            return

        self.config = AppConfig.load(config_path or "config.yaml")
        logger.info("Loaded config")

        detection_backend = DetectionModelType(self.config.detection.backend)
        model_path = (
            Path(self.config.detection.weights_path)
            if self.config.detection.weights_path
            else None
        )
        detector = DetectionService(
            model_name=detection_backend,
            model_path=model_path,
            threshold=self.config.detection.confidence_threshold,
        )

        memory = SceneMemory()
        painter = SoMPainter()

        llm_config = LLMLabelerConfig(
            backend=VLMBackend(self.config.understanding.backend),
            model_id=self.config.understanding.model_id,
            temperature=self.config.understanding.inference.get("temperature", 0.0),
            max_tokens=self.config.understanding.inference.get("max_tokens", 512),
            backend_kwargs=self.config.understanding.inference.get(
                "backend_kwargs", {}
            ),
        )
        ontology = OntologyConfig(
            predicates=self.config.understanding.ontology.predicates
        )
        sgg = SceneGraphGenerator(llm_config, ontology)

        self.pipeline = VisualPipeline(
            detector=detector,
            memory=memory,
            painter=painter,
            sgg=sgg,
        )

        self.chat_service = ChatService(self.config.understanding, memory)

        self.initialized = True
        logger.info("MLState initialized")

    async def reload_pipeline(self):
        self.initialized = False
        self.pipeline = None
        await self.initialize()


ml_state = MLState()
