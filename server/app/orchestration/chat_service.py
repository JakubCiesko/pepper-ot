import inspect
import logging
from typing import Any

from app.inference.memory.scene_memory import SceneMemory
from app.providers.llm_client import LLMClient
from app.schemas.config import ChatConfig
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


class _SafeTemplateDict(dict[str, Any]):
    def __missing__(self, key: str):
        return "{" + key + "}"


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

        obj_id_to_label = {obj.id: obj.label for obj in state.objects}

        object_lines = []
        for obj in state.objects:
            attrs = ", ".join(obj.attributes) if obj.attributes else "no attributes"
            object_lines.append(f"- ID {obj.id}: {obj.label} ({attrs})")
        # TODO: what if memory fkcsup ? need to filter only presnet in obj_id_to_label?
        relation_lines = [
            f"- {obj_id_to_label[rel.subject_id]}_{rel.subject_id} {rel.predicate} {obj_id_to_label[rel.object_id]}_{rel.object_id}"
            for rel in state.relationships
        ]

        parts = ["Objects:"] + object_lines
        if relation_lines:
            parts += ["Relationships:"] + relation_lines
        return "\n".join(parts)

    async def _latest_caption(self) -> str:
        state = await self._get_scene_state()
        captions = sorted(state.captions, key=lambda c: c.last_seen, reverse=True)
        if not captions:
            return "No caption available."
        return captions[0].text

    async def _recent_captions(self, limit: int = 5) -> str:
        state = await self._get_scene_state()
        captions = sorted(state.captions, key=lambda c: c.last_seen, reverse=True)[
            :limit
        ]
        if not captions:
            return "No recent captions."
        lines = [f"- {caption.text}" for caption in captions if caption.text]
        return "\n".join(lines) if lines else "No recent captions."

    async def compose_prompt(self) -> str:
        world_context = await self._build_context_string()
        latest_caption = await self._latest_caption()
        captions_recent = await self._recent_captions()
        if self.context_template:
            template_values = _SafeTemplateDict(
                context=world_context,
                caption=latest_caption,
                captions_recent=captions_recent,
            )
            context_text = self.context_template.format_map(template_values)
        else:
            context_text = (
                f"Context:\n{world_context}\n\n"
                f"Latest Caption:\n{latest_caption}\n\n"
                f"Recent Captions:\n{captions_recent}"
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
        logger.debug("Chat request received, system prompt: %s", system_prompt)
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
