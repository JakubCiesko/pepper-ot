from dataclasses import dataclass
import logging
from pathlib import Path

from app.core.infra.storage import load_last_state
from app.core.pipeline_factory import build_visual_pipeline
from app.core.runtime.worker_manager import WorkerManager
from app.core.runtime.worker_types import StopReason
from app.inference.memory.chat_memory_proxy import EmptyChatMemory
from app.inference.memory.chat_memory_proxy import WorkerChatMemoryProxy
from app.inference.pipeline import VisualPipeline
from app.orchestration.caption_service import CaptionService
from app.orchestration.chat_service import ChatService
from app.orchestration.conversation_service import ConversationService
from app.schemas.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class MLState:
    config: AppConfig | None = None
    pipeline: VisualPipeline | None = None
    worker_manager: WorkerManager | None = None
    chat_service: object | None = None
    conversation_service: object | None = None
    caption_service: object | None = None
    initialized: bool = False
    last_state: dict | None = None
    config_version: int = 0

    async def initialize(self, config_path: str | None = None):
        logger.info("Initializing ML App State")
        if self.initialized:
            logger.info("ML App State already initialized. Returning.")
            return

        pth = config_path or "config.yaml"
        logger.info(f"Loading ML App State config from {pth}")
        self.config = AppConfig.load(pth)
        self.config_version = 0
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
        if self.worker_manager is not None:
            await self.worker_manager.close()
            self.worker_manager = None
        await self.initialize()

    async def apply_config(self, config: AppConfig):
        self.config = config

        base_dir = (
            self.config._config_path.parent
            if self.config._config_path is not None
            else Path.cwd()
        )
        self.config_version += 1

        if self.worker_manager is None:
            self.worker_manager = WorkerManager(self.config)
            await self.worker_manager.start_monitor()
        else:
            await self.worker_manager.update_config(self.config)

        if self.config.worker.enabled:
            logger.info("Worker mode enabled: skipping in-process VisualPipeline build")
            self.pipeline = None
            if self.worker_manager:
                await self.worker_manager.start_monitor()
                await self.worker_manager.hard_reload(self.config, self.config_version)
        else:
            logger.info("Initializing in-process VisualPipeline inference engine.")
            self.pipeline = build_visual_pipeline(self.config)
            if self.worker_manager:
                await self.worker_manager.stop_monitor()
                await self.worker_manager.stop(StopReason.MANUAL)

        chat_system_prompt = self.config.chat.system_prompt.resolve(base_dir)
        chat_context_template = (
            self.config.chat.context_template.resolve(base_dir)
            if self.config.chat.context_template is not None
            else None
        )
        logger.info(
            f"Initializing ChatService with chat_system_prompt {chat_system_prompt} and chat_context_template {chat_context_template}"
        )
        if self.pipeline is not None:
            chat_memory = self.pipeline.memory
        elif self.config.worker.enabled and self.worker_manager is not None:
            chat_memory = WorkerChatMemoryProxy(self.worker_manager)
        else:
            chat_memory = EmptyChatMemory()
        self.chat_service = ChatService(
            self.config.chat,
            chat_memory,
            system_prompt=chat_system_prompt,
            context_template=chat_context_template,
        )
        self.conversation_service = ConversationService(max_messages=10)

        caption_system_prompt = self.config.caption.system_prompt.resolve(base_dir)
        caption_user_prompt = (
            self.config.caption.user_prompt.resolve(base_dir)
            if self.config.caption.user_prompt is not None
            else None
        )
        self.caption_service = CaptionService(
            self,
            self.config.caption,
            system_prompt=caption_system_prompt,
            user_prompt=caption_user_prompt,
        )


ml_state = MLState()
