import logging
import os

from openai import AsyncOpenAI

from ..models.config import UnderstandingConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    A unified client for Text Generation (Chat).
    Supports OpenAI (GPT-4) and Local LLMs (via OpenAI-compatible API like vLLM/Ollama).
    """

    def __init__(self, config: UnderstandingConfig):
        self.config = config

        # Determine API Key and Base URL
        # If backend is 'local', we assume an OpenAI-compatible local server (e.g., localhost:8000)
        # If backend is 'openai', we use the real OpenAI API.

        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        base_url = os.getenv(
            "LLM_BASE_URL", None
        )  # e.g., "http://localhost:11434/v1" for Ollama

        if config.backend == "openai":
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            # For "local" or "local_4bit", we assume a local inference server
            # or we default back to OpenAI if no local URL is set.
            # You can customize this to load HuggingFace models directly if you prefer,
            # but keeping it API-based is cleaner for the server.
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a text response based on the system and user prompts.
        """
        try:
            # Extract inference params with defaults
            max_tokens = self.config.inference.get("max_tokens", 512)
            temperature = self.config.inference.get("temperature", 0.7)

            response = await self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content
            return content if content else "I'm not sure what to say."

        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return "I am having trouble connecting to my language center right now."
