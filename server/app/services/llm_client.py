import logging
import os

from openai import AsyncOpenAI

from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    A unified client for Text Generation (Chat).
    Supports OpenAI (GPT-4) and Local LLMs (via OpenAI-compatible API like vLLM/Ollama).
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        base_url = os.getenv(
            "LLM_BASE_URL", None
        )  # e.g., "http://localhost:11434/v1" for Ollama

        if config.backend == "openai":
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            # TODO: Add proper local
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a text response based on the system and user prompts.
        """
        try:
            # Extract inference params with defaults.
            # Keep device support in config for local backends.
            inference = dict(self.config.inference or {})
            max_tokens = inference.pop("max_tokens", 512)
            temperature = inference.pop("temperature", 0.7)
            backend_kwargs = inference.pop("backend_kwargs", {})
            extra_body = backend_kwargs.get("extra_body", {})
            if self.config.device is not None:
                extra_body.setdefault("device", self.config.device)
            if extra_body:
                backend_kwargs["extra_body"] = extra_body
            if self.config.backend == "openai":
                # neutralize for now...
                backend_kwargs = {}

            response = await self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **backend_kwargs,
            )

            content = response.choices[0].message.content
            return content if content else "I'm not sure what to say."

        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return "I am having trouble connecting to my language center right now."
