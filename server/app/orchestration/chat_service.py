import inspect
import logging

from app.inference.memory.scene_memory import SceneMemory
from app.providers.llm_client import LLMClient
from app.schemas.config import ChatConfig
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        config: ChatConfig,
        memory: SceneMemory,
        system_prompt: str,
        context_template: str | None = None,
    ):
        self.memory = memory
        self.system_prompt = system_prompt
        self.context_template = context_template
        self.llm = LLMClient(config)

    async def _get_scene_state(self) -> SceneState:
        state_or_awaitable = self.memory.scene_state()
        if inspect.isawaitable(state_or_awaitable):
            return await state_or_awaitable
        return state_or_awaitable

    async def _build_context_string(self) -> str:
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

    async def compose_prompt(self) -> str:
        world_context = await self._build_context_string()
        context_text = (
            self.context_template.format(context=world_context)
            if self.context_template
            else f"Context:\n{world_context}"
        )
        return f"{self.system_prompt}\n{context_text}"

    @staticmethod
    def _format_history(history: list[tuple[str, str]] | None) -> str:
        if not history:
            return ""
        lines = []
        for role, text in history:
            role_name = "User" if role == "user" else "Pepper"
            lines.append(f"{role_name}: {text}")
        return "\n".join(lines)

    async def chat(
        self,
        user_query: str,
        *,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> str:
        system_prompt = await self.compose_prompt()
        logger.debug("Chat request received")
        history_text = self._format_history(conversation_history)
        if history_text:
            user_prompt = (
                "Conversation so far:\n"
                f"{history_text}\n\n"
                "Current user message:\n"
                f"{user_query}"
            )
        else:
            user_prompt = user_query
        return await self.llm.generate_text(system_prompt, user_prompt)
