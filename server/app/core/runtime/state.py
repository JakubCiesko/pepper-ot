from dataclasses import dataclass
import logging
from pathlib import Path

from app.core.infra.storage import load_last_state
from app.core.pipeline_factory import build_perception_pipeline
from app.core.runtime.worker_client.manager import WorkerManager
from app.core.runtime.worker_client.types import StopReason
from app.inference.memory.chat_memory_proxy import EmptyChatMemory
from app.inference.memory.chat_memory_proxy import WorkerChatMemoryProxy
from app.inference.pipeline import PerceptionPipeline
from app.orchestration.caption_service import CaptionService
from app.orchestration.chat_service import ChatService
from app.orchestration.conversation_service import ConversationService
from app.schemas.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    config: AppConfig | None = None
    pipeline: PerceptionPipeline | None = None
    worker_manager: WorkerManager | None = None
    chat_service: ChatService | None = None
    conversation_service: ConversationService | None = None
    caption_service: CaptionService | None = None
    initialized: bool = False
    last_state: dict | None = None
    config_version: int = 0

    async def initialize(self, config_path: str | None = None):
        logger.info("Initializing App State")
        if self.initialized:
            logger.info("App State already initialized. Returning.")
            return

        pth = config_path or "config.yaml"
        logger.info("Loading App State config from %s", pth)
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
        logger.info("AppState initialized")

    async def reload_pipeline(self):
        logger.info("Reloading App State. Starting Initialization.")
        self.initialized = False
        self.pipeline = None
        if self.worker_manager is not None:
            await self.worker_manager.close()
            self.worker_manager = None
        await self.initialize()

    async def apply_config(self, config: AppConfig):
        self.config = config
        self.config_version += 1
        await self._ensure_worker_manager()
        await self._apply_runtime_mode()
        base_dir = self._resolve_base_dir()
        self._initialize_chat_components(base_dir)
        self._initialize_caption_component(base_dir)

    def _resolve_base_dir(self) -> Path:
        assert self.config is not None
        return (
            self.config._config_path.parent
            if self.config._config_path is not None
            else Path.cwd()
        )

    async def _ensure_worker_manager(self):
        assert self.config is not None
        if self.worker_manager is None:
            self.worker_manager = WorkerManager(self.config)
            await self.worker_manager.start_monitor()
            return
        await self.worker_manager.update_config(self.config)

    async def _apply_runtime_mode(self):
        assert self.config is not None
        if self.config.worker.enabled:
            logger.info(
                "Worker mode enabled: skipping in-process PerceptionPipeline build"
            )
            self.pipeline = None
            if self.worker_manager:
                await self.worker_manager.start_monitor()
                await self.worker_manager.hard_reload(self.config, self.config_version)
            return

        logger.info("Initializing in-process PerceptionPipeline inference engine.")
        self.pipeline = build_perception_pipeline(self.config)
        if self.worker_manager:
            await self.worker_manager.stop_monitor()
            await self.worker_manager.stop(StopReason.MANUAL)

    def _build_chat_memory_adapter(self):
        assert self.config is not None
        if self.pipeline is not None:
            return self.pipeline.memory
        if self.config.worker.enabled and self.worker_manager is not None:
            return WorkerChatMemoryProxy(self.worker_manager)
        return EmptyChatMemory()

    def _initialize_chat_components(self, base_dir: Path):
        assert self.config is not None
        chat_system_prompt = self.config.chat.system_prompt.resolve(base_dir)
        chat_context_template = (
            self.config.chat.context_template.resolve(base_dir)
            if self.config.chat.context_template is not None
            else None
        )
        logger.info(
            "Initializing ChatService with chat_system_prompt "
            "%s and chat_context_template %s",
            chat_system_prompt,
            chat_context_template,
        )
        chat_memory = self._build_chat_memory_adapter()
        self.chat_service = ChatService(
            self.config.chat,
            chat_memory,
            system_prompt=chat_system_prompt,
            context_template=chat_context_template,
        )
        if self.conversation_service is None:
            self.conversation_service = ConversationService(max_messages=10)

    def _initialize_caption_component(self, base_dir: Path):
        assert self.config is not None
        caption_system_prompt = self.config.caption.system_prompt.resolve(base_dir)
        caption_user_prompt = (
            self.config.caption.user_prompt.resolve(base_dir)
            if self.config.caption.user_prompt is not None
            else None
        )
        if self.caption_service is None:
            self.caption_service = CaptionService(
                self,
                self.config.caption,
                system_prompt=caption_system_prompt,
                user_prompt=caption_user_prompt,
            )
            return
        self.caption_service.update_runtime(
            self.config.caption,
            system_prompt=caption_system_prompt,
            user_prompt=caption_user_prompt,
            rebuild_client=True,
        )


app_state = AppState()
