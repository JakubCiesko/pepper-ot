from dataclasses import dataclass
import logging
from pathlib import Path

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.service import DetectionService
from app.inference.memory.scene_memory import SceneMemory
from app.inference.pipeline import VisualPipeline
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
        await self.apply_config(self.config)

        self.initialized = True
        logger.info("MLState initialized")

    async def reload_pipeline(self):
        self.initialized = False
        self.pipeline = None
        await self.initialize()

    async def apply_config(self, config: AppConfig):
        self.config = config

        base_dir = (
            self.config._config_path.parent
            if self.config._config_path is not None
            else Path.cwd()
        )

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
        painter = SoMPainter(
            line_thickness=self.config.visualization.line_thickness,
            color_lookup=self.config.visualization.color_lookup,
            mask_opacity=self.config.visualization.mask_opacity,
        )

        llm_config = LLMLabelerConfig(
            backend=VLMBackend(self.config.understanding.backend),
            model_id=self.config.understanding.model_id,
            temperature=self.config.understanding.inference.get("temperature", 0.0),
            max_tokens=self.config.understanding.inference.get("max_tokens", 512),
            backend_kwargs=self.config.understanding.inference.get(
                "backend_kwargs", {}
            ),
        )
        vlm_system_prompt = self.config.understanding.system_prompt.resolve(base_dir)
        vlm_user_prompt = (
            self.config.understanding.user_prompt.resolve(base_dir)
            if self.config.understanding.user_prompt is not None
            else None
        )
        predicates, objects = self.config.understanding.ontology.resolve(base_dir)
        sgg = SceneGraphGenerator(
            llm_config,
            predicates=predicates,
            objects=objects,
            system_prompt=vlm_system_prompt,
            user_prompt=vlm_user_prompt,
        )

        self.pipeline = VisualPipeline(
            detector=detector,
            memory=memory,
            painter=painter,
            sgg=sgg,
            vis_config=self.config.visualization,
        )

        chat_system_prompt = self.config.chat.system_prompt.resolve(base_dir)
        chat_context_template = (
            self.config.chat.context_template.resolve(base_dir)
            if self.config.chat.context_template is not None
            else None
        )
        self.chat_service = ChatService(
            self.config.chat,
            memory,
            system_prompt=chat_system_prompt,
            context_template=chat_context_template,
        )


ml_state = MLState()
