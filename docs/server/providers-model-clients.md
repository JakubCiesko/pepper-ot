# Providers and Model Clients

## Files Covered

- `app/providers/llm/base.py`
- `app/providers/llm/client.py`
- `app/providers/llm/openai_llm.py`
- `app/providers/llm/gemini_llm.py`
- `app/providers/llm/hf_llm.py`
- `app/providers/vlm/base.py`
- `app/providers/vlm/factory.py`
- `app/providers/vlm/openai_vlm.py`
- `app/providers/vlm/gemini_vlm.py`
- `app/providers/vlm/local_hf_vlm.py`
- `app/providers/caption/client.py`
- `app/providers/translation/google_trans.py`
- `app/providers/common/io.py`
- `app/providers/common/runtime_setup.py`
- `app/providers/common/utils.py`

## Design Goal

The provider layer hides model-family differences while still exposing provider-specific configuration when needed.

The rest of the application should mostly talk to:
- `LLMClient`
- `BaseVLMClient` implementations via factory
- `CaptionClient`
- `TranslationService`

## Text LLM stack

### Base classes

- `LLMResponse`
- `BaseTextProvider`

### Unified entry point

`LLMClient` wraps provider selection and runtime updates.

Supported provider values:
- `openai`
- `gemini`
- `openai_compatible`
- `local_hf`
- `local_4bit`

### Provider implementations

- `OpenAITextProvider`
- `GeminiTextProvider`
- `LocalHFTextProvider`

## VLM stack

### Base class

- `BaseVLMClient`

### Factory

`build_vlm_client(config)` in `factory.py` resolves provider implementation.

### Implementations

- `OpenAIVLMClient`
- `GeminiVLMClient`
- `LocalHFVLMClient`
- `Local4BitVLMClient`

## Caption stack

`CaptionClient` is the main caption facade.

There is also `LocalBLIPCaptionClient` for local BLIP usage.

This stack is used by:
- the dedicated caption endpoint
- the perception pipeline caption stage
- crop fallback captioning in object chat
- worker caption path

## Translation stack

Implemented in `translation/google_trans.py`.

Key pieces:
- `TranslationService`
- language inversion helpers
- output-language enforcement helper

This layer is used by both `/chat` and `/vision_chat`.

## Structured output helpers

Implemented in `providers/common/io.py`.

Important helpers:
- extract plain text from provider responses
- extract JSON blocks from raw text
- parse structured text against schema
- validate parsed outputs
- convert schema to JSON schema
- resolve structured output strategy

Structured output modes supported by config include:
- `provider_native`
- `parse_output`
- `instructor`

## Runtime setup helpers

Implemented in `providers/common/runtime_setup.py`.

Purpose:
- resolve API keys from env
- build OpenAI async client kwargs
- build Gemini client kwargs

## Provider capability helpers

Implemented in `providers/common/utils.py`.

Used for:
- provider capability matrix
- kwargs normalization
- kwargs validation
- parsing/normalizing OpenAI-compatible options

This matters because dashboard config editing writes raw JSON kwargs that must still be validated before provider construction.

## Where to Tweak What

### Change provider family
- update config provider/model/base_url/api_key_env
- may require hard reload

### Change generation behavior
- update `call_kwargs`
- hot update in many cases

### Change client construction behavior
- update `client_init_kwargs`
- often hard reload

### Change structured parsing behavior
- update `structured_output`
- inspect `common/io.py`

### Add a new provider
- create provider implementation
- update capability matrix
- update factory/client selection logic
- update dashboard provider options if needed
