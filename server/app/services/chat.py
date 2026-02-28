import logging

from app.inference.memory.scene_memory import SceneMemory
from app.schemas.config import ChatConfig
from app.services.llm_client import LLMClient

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

    def compose_prompt(self):
        world_context = self._build_context_string()
        system_definition_prompt = self.system_prompt
        context_definition_prompt = (
            self.context_template.format(context=world_context)
            if self.context_template
            else "Context:\n{context}"
        )
        system_prompt = f"{system_definition_prompt}\n{context_definition_prompt}"
        return system_prompt

    async def chat(self, user_query: str) -> str:
        system_prompt = self.compose_prompt()
        logger.info(f"System prompt for LLM Chat: {system_prompt}")
        logger.info(f"User prompt for LLM Chat: {user_query}")
        return await self.llm.generate_text(system_prompt, user_query)

    def _build_context_string(self) -> str:
        # Iterate over self.memory.tracks and format text
        state = self.memory.scene_state()
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
