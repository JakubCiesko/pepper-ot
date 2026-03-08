import inspect
import logging
from typing import Any

from app.inference.memory.scene_memory import SceneMemory
from app.schemas.config import ChatConfig
from app.schemas.scene import SceneState
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        config: ChatConfig,
        memory: SceneMemory | Any,
        system_prompt: str,
        context_template: str | None = None,
    ):
        self.memory = memory
        self.system_prompt = system_prompt
        self.context_template = context_template
        self.llm = LLMClient(config)

    async def compose_prompt(self):
        world_context = await self._build_context_string()
        system_definition_prompt = self.system_prompt
        context_definition_prompt = (
            self.context_template.format(context=world_context)
            if self.context_template
            else "Context:\n{context}"
        )
        system_prompt = f"{system_definition_prompt}\n{context_definition_prompt}"
        return system_prompt

    async def chat(self, user_query: str) -> str:
        system_prompt = await self.compose_prompt()
        logger.info(f"System prompt for LLM Chat: {system_prompt}")
        logger.info(f"User prompt for LLM Chat: {user_query}")
        return await self.llm.generate_text(system_prompt, user_query)

    async def _get_scene_state(self) -> SceneState:
        state_or_awaitable = self.memory.scene_state()
        if inspect.isawaitable(state_or_awaitable):
            return await state_or_awaitable
        return state_or_awaitable

    async def _build_context_string(self) -> str:
        # Iterate over self.memory.tracks and format text
        state = await self._get_scene_state()
        if not state.objects:
            return "You see nothing."

        object_lines = []
        for obj in state.objects:
            attrs = ", ".join(obj.attributes) if obj.attributes else "no attributes"
            object_lines.append(f"- ID {obj.id}: {obj.label} ({attrs})")

        relation_lines = [
            f"- {rel.subject_id} {rel.predicate} {rel.object_id}"
            for rel in state.relationships
        ]

        parts = ["Objects:"] + object_lines
        if relation_lines:
            parts += ["Relationships:"] + relation_lines

        return "\n".join(parts)
