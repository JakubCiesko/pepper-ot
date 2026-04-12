import inspect
import logging

from app.core.prompting.renderer import PromptRenderContext
from app.core.prompting.renderer import render_prompt_template
from app.inference.memory.scene_memory import SceneMemory
from app.providers.llm.client import LLMClient
from app.schemas.config import ChatConfig
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        config: ChatConfig,
        memory: SceneMemory,
        system_prompt: str,
    ):
        self.memory = memory
        self.system_prompt = system_prompt
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

        ordered_objects = sorted(
            state.objects, key=self._object_salience_key, reverse=True
        )  # ordered by social-saliency, biggest first (maybe the opposite is better?)
        obj_id_to_label = {obj.id: obj.label for obj in ordered_objects}

        object_lines = []
        for obj in ordered_objects:
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

    @staticmethod
    def _person_social_salience(attributes: set[str], obj) -> float:
        score = 0.0
        if obj.label == "person":
            score += 10.0
        if "is_waving" in attributes:
            score += 40.0
        if "is_looking_at_robot" in attributes:
            score += 30.0
        if "is_sitting" in attributes:
            score += 10.0
        if any(attribute.startswith("engagement_zone_1") for attribute in attributes):
            score += 20.0
        elif any(attribute.startswith("engagement_zone_2") for attribute in attributes):
            score += 10.0
        if obj.robot_distance is not None:
            score += max(0.0, 10.0 - (obj.robot_distance * 5.0))
        return score

    @classmethod
    def _object_salience_key(cls, obj) -> tuple[float, float, int]:
        attributes = set(obj.attributes or [])
        return (
            cls._person_social_salience(attributes, obj),
            obj.last_seen,
            obj.hits,
        )

    async def compose_prompt(self, base: str) -> str:
        world_context = await self._build_context_string()
        latest_caption = await self._latest_caption()
        captions_recent = await self._recent_captions()
        render_context = PromptRenderContext(
            context=world_context,
            caption=latest_caption,
            captions_recent=captions_recent,
        )
        rendered = render_prompt_template(base, render_context)
        return rendered or base

    @staticmethod
    def _format_history(history: list[tuple[str, str]] | None) -> str:
        if not history:
            return ""
        lines = []
        for role, text in history:
            role_name = "User" if role == "user" else "Assistant"
            lines.append(f"{role_name}: {text}")
        return "\n".join(lines)

    async def chat(
        self,
        user_query: str,
        *,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> str:
        system_prompt = await self.compose_prompt(self.system_prompt)
        user_query = await self.compose_prompt(user_query)
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
