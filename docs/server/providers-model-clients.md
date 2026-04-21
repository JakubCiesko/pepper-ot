# Providers and Model Clients

Provider clients are the boundary between the server and model backends. They normalize prompts, call kwargs, structured-output behavior, credentials, and local model loading.

## Main Files

Text LLM:

- `server/app/providers/llm/client.py`
- `server/app/providers/llm/base.py`
- `server/app/providers/llm/openai_llm.py`
- `server/app/providers/llm/gemini_llm.py`
- `server/app/providers/llm/hf_llm.py`

VLM:

- `server/app/providers/vlm/base.py`
- `server/app/providers/vlm/factory.py`
- `server/app/providers/vlm/openai_vlm.py`
- `server/app/providers/vlm/gemini_vlm.py`
- `server/app/providers/vlm/local_hf_vlm.py`

Caption:

- `server/app/providers/caption/client.py`

Common provider utilities:

- `server/app/providers/common/io.py`
- `server/app/providers/common/runtime_setup.py`
- `server/app/providers/common/utils.py`

Translation:

- `server/app/providers/translation/google_trans.py`
- `server/app/providers/translation/vocabulary.py`
- `server/app/providers/translation/lexicons/*.json`
- `server/app/providers/translation/lexicons_user/*.json`

## LLMClient

File: `server/app/providers/llm/client.py`

`LLMClient` is provider-agnostic text generation. It supports:

- `openai`
- `openai_compatible`
- `gemini`
- `local_hf`
- `local_4bit`

It exposes:

- `generate(system_prompt, user_prompt, output_schema=None, call_overrides=None)`
- `generate_text(system_prompt, user_prompt)`
- `generate_structured(system_prompt, user_prompt, output_schema=...)`
- `update_runtime(config, rebuild_client=True)`

`generate_text` catches provider exceptions and returns a safe fallback string. `generate_structured` requires `response.parsed` and raises when structured parsing fails.

## Provider Capability Matrix

File: `server/app/providers/common/utils.py`

`provider_capability_matrix()` reports structured output support and base URL support to the dashboard.

Structured output support:

- `openai`: provider native, parse output, instructor.
- `gemini`: provider native, parse output.
- `openai_compatible`: parse output, instructor.
- `local_hf`: parse output.
- `local_4bit`: parse output.

The matrix is used by `resolve_structured_mode` and dashboard capability hints.

## Structured Output Resolution

File: `server/app/providers/common/io.py`

Structured mode values:

- `provider_native`
- `parse_output`
- `instructor`

`resolve_structured_mode(config, output_schema, supports_native_structured, provider_name)` checks the configured mode against the capability matrix. Unsupported native/instructor modes fall back to `parse_output` with a warning.

### Provider Native

For OpenAI, provider-native structured output uses `client.responses.parse` with `text_format=output_schema`.

For Gemini, provider-native structured output sets `response_mime_type=application/json` and `response_json_schema` when schema conversion succeeds.

### Instructor

OpenAI text and VLM providers create `instructor.from_openai(..., mode=instructor.Mode.TOOLS)` clients. In instructor mode they call `chat.completions.create(..., response_model=output_schema)`.

For `openai_compatible`, instructor support depends on whether the target OpenAI-compatible server actually handles the tool/schema request correctly. The server advertises instructor as supported but falls back to parse output on failure.

### Parse Output

Parse-output mode asks the model for JSON where possible and parses JSON blocks from text using Pydantic `TypeAdapter`. For OpenAI-compatible chat completions, `response_format={"type":"json_object"}` is set when an output schema is present.

If strict mode is true and no JSON is found or validation fails, an exception is raised. If strict is false, parser failures can yield `None`.

## OpenAI Text Provider

File: `server/app/providers/llm/openai_llm.py`

`OpenAITextProvider` uses `AsyncOpenAI`. For `provider=openai`, native structured output is allowed. For `provider=openai_compatible`, native structured output is disabled by the capability matrix, but instructor and parse-output are available.

Generation path:

1. Merge config `call_kwargs` with call overrides.
2. Normalize token kwargs.
3. Resolve structured mode.
4. Try provider-native if selected.
5. Try instructor if selected.
6. Fall back to parse-output/plain chat completions.
7. Return `LLMResponse(text, parsed, raw)`.

## Gemini Text Provider

File: `server/app/providers/llm/gemini_llm.py`

Gemini text generation flattens system/user messages into one combined prompt. Native structured mode uses Gemini `GenerateContentConfig` JSON schema support. Parse-output mode uses JSON mime type and then local parser.

## Local HF Text Provider

File: `server/app/providers/llm/hf_llm.py`

Loads `AutoTokenizer` and `AutoModelForCausalLM`. It constructs a simple text prompt from role messages and decodes generated text. Structured output is parse-output only.

Important config keys:

- `model_id`
- `device`
- `client_init_kwargs`
- `call_kwargs.max_new_tokens` or `max_tokens`
- `temperature`, `top_p`, `do_sample`

## VLM Factory

File: `server/app/providers/vlm/factory.py`

Builds a `BaseVLMClient` from an LLM-style config. VLM providers share structured output config with chat providers.

Supported providers mirror text clients:

- OpenAI / OpenAI-compatible VLM
- Gemini VLM
- local HF VLM
- local 4-bit HF VLM

## OpenAI VLM Provider

File: `server/app/providers/vlm/openai_vlm.py`

Prepares two request formats:

- native Responses API input for provider-native structured output
- standard chat-completions messages with `image_url` for instructor/parse/plain output

It supports provider-native, instructor, and parse-output paths similarly to text OpenAI provider.

## Gemini VLM Provider

File: `server/app/providers/vlm/gemini_vlm.py`

Builds Gemini content parts with user prompt plus image bytes. System prompt is placed into `GenerateContentConfig.system_instruction`. Structured output uses Gemini JSON schema or JSON mime type depending on mode.

## Local HF VLM Provider

File: `server/app/providers/vlm/local_hf_vlm.py`

Loads `AutoModelForImageTextToText` and `AutoProcessor`. It tries processor chat template rendering first and falls back to a plain prompt when template rendering fails or `local_vlm_hints.prompt_template_style=plain`.

Local hints:

- `prompt_template_style`: `auto`, `chatml`, `plain`
- `image_token_strategy`: `auto`, `single`, `multi`

Structured output is parse-output only.

## CaptionClient

File: `server/app/providers/caption/client.py`

`CaptionClient` selects:

- `LocalBLIPCaptionClient` when `provider=local_hf` and model id contains `blip-image-captioning`.
- otherwise a normal VLM client from `build_vlm_client`.

The BLIP client is prefix-conditioned, not instruction-chat based. It intentionally avoids feeding long system prompts into BLIP. For prompted mode, if the prompt is empty, too question-like, or too long, it uses `a photo of` as prefix. For unconditional mode, it sends no prefix.

## TranslationService

File: `server/app/providers/translation/google_trans.py`

Wraps `googletrans.Translator` for language detection, translation, and enforcement.

Global helpers:

- `english_to_czech`
- `czech_to_english`
- `enforce_output_language(text, output_language, return_languages=False)`

`enforce_output_language` supports:

- `default`: no translation
- `english`: enforce English
- `czech`: enforce Czech

The helper detects current language and only translates text whose detected language differs from the target.

## VocabularyTranslationService

File: `server/app/providers/translation/vocabulary.py`

This is token-level translation for robot/dashboard memory display. It is separate from free-text translation.

It supports Czech (`cs`) lexicons for:

- object labels
- attributes
- relations

It loads static lexicons from `lexicons/` and user-editable lexicons from `lexicons_user/`. User lexicons override static defaults.

Important methods:

- `warm_from_config(cfg, base_dir)`: collects ontology/rule terms and translates missing Czech entries.
- `translate_token(token, token_type, language)`: returns token translation or original token.
- `build_memory_display_overrides(state, language)`: builds object/relation override maps for memory summary rendering.
- `replace_user_map(...)`: used by dashboard translation patching.

## Runtime Updates

Provider runtime updates are split by service:

- `LLMClient.update_runtime` can rebuild provider or update config reference.
- Chat hot reload updates prompts and `llm.update_runtime(new.chat)`.
- Caption hot reload updates prompts and caption client runtime.
- VLM scene graph hot reload updates backend config, prompts, ontology, and client runtime without rebuild unless hard reload required.

Provider/model/base-url changes are usually hard reload because existing client/model objects may not be safely mutated.

## Where To Change Things

- Add a provider: add config literal, provider class, factory/client branch, capability matrix, dashboard select option.
- Change structured-output behavior: `providers/common/io.py`, provider classes, capability matrix.
- Change OpenAI-compatible behavior: `openai_llm.py`, `openai_vlm.py`, capability matrix.
- Change BLIP caption prompt handling: `providers/caption/client.py`.
- Change language enforcement: `providers/translation/google_trans.py`.
- Change memory display vocabulary: `providers/translation/vocabulary.py` and lexicon JSON files.
