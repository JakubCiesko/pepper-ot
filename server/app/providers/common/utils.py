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


def _expect_type(
    where: str, key: str, value: Any, allowed_types: tuple[type, ...]
) -> None:
    if not isinstance(value, allowed_types):
        allowed = ", ".join(t.__name__ for t in allowed_types)
        raise ValueError(f"{where}.{key} must be of type: {allowed}")


# def validate_client_init_kwargs(
#     provider: ProviderName, kwargs: dict[str, Any], where: str
# ) -> None:
#     if not isinstance(kwargs, dict):
#         raise ValueError(f"{where} must be a JSON object")

#     known: dict[str, tuple[type, ...]] = {
#         "api_key": (str,),
#         "base_url": (str,),
#         "timeout": (int, float),
#         "max_retries": (int,),
#         "organization": (str,),
#         "project": (str,),
#         "http_options": (dict,),
#         "device_map": (str, dict),
#         "torch_dtype": (str,),
#         "trust_remote_code": (bool,),
#         "tokenizer_kwargs": (dict,),
#         "processor_kwargs": (dict,),
#     }

#     for key, value in kwargs.items():
#         if key in known:
#             _expect_type(where, key, value, known[key])


# def validate_call_kwargs(
#     provider: ProviderName, kwargs: dict[str, Any], where: str
# ) -> None:
#     if not isinstance(kwargs, dict):
#         raise ValueError(f"{where} must be a JSON object")

#     common_known: dict[str, tuple[type, ...]] = {
#         "temperature": (int, float),
#         "top_p": (int, float),
#         "top_k": (int,),
#         "max_tokens": (int,),
#         "max_output_tokens": (int,),
#         "max_completion_tokens": (int,),
#         "max_new_tokens": (int,),
#         "do_sample": (bool,),
#         "response_format": (dict,),
#         "generate_content_config": (dict,),
#         "presence_penalty": (int, float),
#         "frequency_penalty": (int, float),
#         "reasoning": (dict,),
#         "seed": (int,),
#         "stop": (str, list),
#     }

#     for key, value in kwargs.items():
#         if key in common_known:
#             _expect_type(where, key, value, common_known[key])

#         if (
#             key == "stop"
#             and isinstance(value, list)
#             and not all(isinstance(item, str) for item in value)
#         ):
#             raise ValueError(f"{where}.stop list must contain only strings")

#     if provider == "gemini" and "response_format" in kwargs:
#         raise ValueError(
#             f"{where}.response_format is OpenAI-style and unsupported for provider=gemini; use generate_content_config"
#         )
