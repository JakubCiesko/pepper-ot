from ..inference.memory import SceneMemory
from ..models.config import UnderstandingConfig
from .llm_client import LLMClient


class ChatService:
    def __init__(self, config: UnderstandingConfig, memory: SceneMemory):
        self.memory = memory
        self.llm = LLMClient(config)  # Your interface to GPT-4o

    async def chat(self, user_query: str) -> str:
        # 1. READ: Get current world state from the shared memory
        # (This is the "WorldState" logic we discussed)
        world_context = self._build_context_string()

        # 2. PROMPT: Create the system prompt
        system_prompt = (
            "You are Pepper. Answer based on what you see.\n"
            f"Context:\n{world_context}"
        )

        # 3. GENERATE
        return await self.llm.generate(system_prompt, user_query)

    def _build_context_string(self) -> str:
        # Iterate over self.memory.tracks and format text
        if not self.memory.tracks:
            return "You see nothing."

        lines = ["- {t.label} (ID: {t.id})" for t in self.memory.tracks.values()]
        # Add spatial/relation info here...
        return "\n".join(lines)
