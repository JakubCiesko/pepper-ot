"""Facade re-export for VLM clients.

Implementation was split into app.providers.vlm.* modules.
"""

from app.providers.vlm import BaseVLMClient
from app.providers.vlm import GeminiVLMClient
from app.providers.vlm import Local4BitVLMClient
from app.providers.vlm import LocalHFVLMClient
from app.providers.vlm import OpenAIVLMClient
from app.providers.vlm import build_vlm_client

__all__ = [
    "BaseVLMClient",
    "build_vlm_client",
    "OpenAIVLMClient",
    "GeminiVLMClient",
    "LocalHFVLMClient",
    "Local4BitVLMClient",
]
