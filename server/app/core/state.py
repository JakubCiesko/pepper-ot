from dataclasses import dataclass
import logging
from pathlib import Path

from app.core.storage import load_last_state
from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.service import DetectionService
from app.inference.memory.scene_memory import SceneMemory
from app.inference.pipeline import VisualPipeline
from app.inference.scene_graph.rules_backend import RuleBasedSceneGraphBackend
from app.inference.scene_graph.service import SceneGraphService
from app.inference.scene_graph.som import SoMPainter
from app.inference.scene_graph.vlm_backend import VLMSceneGraphBackend
from app.schemas.config import AppConfig
from app.services.chat import ChatService

logger = logging.getLogger(__name__)


@dataclass
class MLState:
    config: AppConfig | None = None
    pipeline: VisualPipeline | None = None
    chat_service: object | None = None
    initialized: bool = False
    last_state: dict | None = None

    async def initialize(self, config_path: str | None = None):
        logger.info("Initializing ML App State")
        if self.initialized:
            logger.info("ML App State already initialized. Returning.")
            return

        pth = config_path or "config.yaml"
        logger.info(f"Loading ML App State config from {pth}")
        self.config = AppConfig.load(pth)
        logger.info("Loaded config")
        if self.config.storage.persist_last_state:
            base_dir = (
                self.config._config_path.parent
                if self.config._config_path is not None
                else Path.cwd()
            )
            state_path = base_dir / self.config.storage.last_state_path
            self.last_state = load_last_state(state_path)
        await self.apply_config(self.config)

        self.initialized = True
        logger.info("MLState initialized")

    async def reload_pipeline(self):
        logger.info("Reloading ML App State. Starting Initialization.")
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
            device=self.config.detection.device,
            threshold=self.config.detection.confidence_threshold,
            ontology=self.config.detection.resolve_ontology(base_dir),
        )

        memory = SceneMemory(
            memory_max_age_seconds=self.config.tracking.memory_max_age_seconds,
            memory_max_objects=self.config.tracking.memory_max_objects,
            memory_max_relations=self.config.tracking.memory_max_relations,
            association_config=self.config.tracking.association,
            feature_extraction_config=self.config.tracking.feature_extraction,
        )
        painter = SoMPainter(
            line_thickness=self.config.visualization.line_thickness,
            color_lookup=self.config.visualization.color_lookup,
            mask_opacity=self.config.visualization.mask_opacity,
        )
        vlm_system_prompt = self.config.scene_graph.vlm.system_prompt.resolve(base_dir)
        vlm_user_prompt = (
            self.config.scene_graph.vlm.user_prompt.resolve(base_dir)
            if self.config.scene_graph.vlm.user_prompt is not None
            else None
        )
        predicates, objects = self.config.scene_graph.vlm.ontology.resolve(base_dir)
        vlm_backend = VLMSceneGraphBackend(
            self.config.scene_graph.vlm,
            predicates=predicates,
            objects=objects,
            system_prompt=vlm_system_prompt,
            user_prompt=vlm_user_prompt,
        )
        rule_backend = RuleBasedSceneGraphBackend(self.config.scene_graph.rules)
        scene_graph_service = SceneGraphService(
            mode=self.config.scene_graph.mode,
            vlm_backend=vlm_backend,
            rule_backend=rule_backend,
        )

        logger.info("Initializing VisualPipeline inference engine.")
        self.pipeline = VisualPipeline(
            detector=detector,
            memory=memory,
            painter=painter,
            scene_graph_service=scene_graph_service,
            fusion_config=self.config.fusion,
            vis_config=self.config.visualization,
        )

        chat_system_prompt = self.config.chat.system_prompt.resolve(base_dir)
        chat_context_template = (
            self.config.chat.context_template.resolve(base_dir)
            if self.config.chat.context_template is not None
            else None
        )
        logger.info(
            f"Initializing ChatService with chat_system_prompt {chat_system_prompt} and chat_context_template {chat_context_template}"
        )
        self.chat_service = ChatService(
            self.config.chat,
            memory,
            system_prompt=chat_system_prompt,
            context_template=chat_context_template,
        )


ml_state = MLState()
