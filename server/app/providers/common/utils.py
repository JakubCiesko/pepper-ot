# TODO: Rework this whole fucking mess
from typing import Any

ProviderName = str


def provider_capability_matrix() -> dict[str, Any]:
    return {
        "structured_output_support": {
            "openai": {
                "provider_native": True,
                "parse_output": True,
                "instructor": True,
            },
            "gemini": {
                "provider_native": True,
                "parse_output": True,
                "instructor": False,
            },
            "openai_compatible": {
                "provider_native": False,
                "parse_output": True,
                "instructor": True,
            },
            "local_hf": {
                "provider_native": False,
                "parse_output": True,
                "instructor": False,
            },
            "local_4bit": {
                "provider_native": False,
                "parse_output": True,
                "instructor": False,
            },
        },
        "base_url_support": {
            "openai": True,
            "openai_compatible": True,
            "gemini": False,
            "local_hf": False,
            "local_4bit": False,
        },
        "examples": {
            "openai_compatible_base_urls": [
                "http://localhost:8000/v1",
                "http://127.0.0.1:11434/v1",
            ],
            "provider_required_env_default": {
                "openai": "OPENAI_API_KEY",
                "openai_compatible": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
            },
        },
        "precedence": [
            "config defaults",
            "runtime PATCH updates",
            "per-call overrides",
        ],
    }


def normalize_call_kwargs(
    provider: ProviderName, kwargs: dict[str, Any]
) -> dict[str, Any]:
    out = dict(kwargs)

    # Common aliasing for token limits.
    if "max_completion_tokens" in out and "max_tokens" not in out:
        out["max_tokens"] = out["max_completion_tokens"]
    if (
        "max_output_tokens" in out
        and "max_tokens" not in out
        and provider
        in {
            "openai",
            "openai_compatible",
            "local_hf",
            "local_4bit",
        }
    ):
        out["max_tokens"] = out["max_output_tokens"]
    if "max_tokens" in out and "max_output_tokens" not in out and provider == "gemini":
        out["max_output_tokens"] = out["max_tokens"]
    return out


def normalize_openai_parse_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    if "max_tokens" in out and "max_output_tokens" not in out:
        out["max_output_tokens"] = out.pop("max_tokens")
    return out
