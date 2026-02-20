from app.inference.memory.scene_memory import SceneMemory
from app.schemas.config import ChatConfig
from app.services.llm_client import LLMClient


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

    async def chat(self, user_query: str) -> str:
        # 1. READ: Get current world state from the shared memory
        # (This is the "WorldState" logic we discussed)
        world_context = self._build_context_string()

        # 2. PROMPT: Create the system prompt
        if self.context_template:
            try:
                # {context} needs to be in prompt
                system_prompt = self.context_template.format(context=world_context)
            except Exception:
                system_prompt = f"{self.system_prompt}\nContext:\n{world_context}"
        else:
            system_prompt = f"{self.system_prompt}\nContext:\n{world_context}"

        # 3. GENERATE
        return await self.llm.generate_text(system_prompt, user_query)

    def _build_context_string(self) -> str:
        # Iterate over self.memory.tracks and format text
        if not self.memory.tracks:
            return "You see nothing."

        lines = [f"- {t.label} (ID: {t.id})" for t in self.memory.tracks.values()]
        # Add spatial/relation info here...
        return "\n".join(lines)
