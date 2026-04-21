# Server Communication

All HTTP communication is centralized in `client/app/scripts/pepper_client/core/transport.py`. The robot client should not call `requests` directly elsewhere.

`PepperServerTransport` wraps a `requests.Session`, attaches `Accept: application/json`, builds request payloads, applies timeouts, verifies TLS according to config, parses JSON, logs compact responses, and raises client-specific exceptions.

## Error Model

Transport exceptions are converted into domain errors from `pepper_client.utils.error_policy`:

| Low-level condition | Raised exception | Spoken by `TurnManager` as |
|---|---|---|
| `requests.Timeout` | `ServerTimeoutError` | Server timeout fallback. |
| Other `requests.RequestException` | `ServerUnavailableError` | Server unavailable fallback. |
| HTTP status `>=400` | `ServerUnavailableError` | Server unavailable fallback. |
| Response body is not JSON | `MalformedResponseError` | Malformed response fallback. |
| Required response fields missing | `MalformedResponseError` | Malformed response fallback. |

`TurnManager._run_guarded` catches these exceptions and speaks localized fallback messages.

## Base URL And TLS

`update_config` reads `server.base_url` and `server.verify_tls`. The base URL is stripped of its trailing slash and prepended to every configured path.

If the robot or local dev environment uses ngrok or internal HTTPS and certificate validation fails, `server.verify_tls` controls whether `requests` verifies certificates. Disabling verification can be useful for experiments, but it also hides real security and deployment problems.

## Multipart Image Uploads

Image endpoints use multipart form data. The client encodes captured frames as JPEG bytes and sends them as `image/jpeg`.

### `caption(...)`

Server route: configured `server.caption_path`, default `/api/v1/caption`.

Form fields:

| Field | Value |
|---|---|
| `file` | `capture.jpg`, JPEG bytes, MIME `image/jpeg`. |
| `metadata` | JSON string built by `MetadataBuilder`. |
| `run_detect` | String `true` or `false`. |
| `publish` | String `true` or `false`. |
| `prompt` | Optional prompt override. |
| `language` | Optional server language, usually `english` or `czech`. |

Validation:

- Response must be a JSON object.
- Response must contain a non-empty `caption` field.

Used by:

- `TurnManager._run_look`.

Server docs:

- [`../server/api-reference.md`](../server/api-reference.md)
- [`../server/orchestration-and-conversations.md`](../server/orchestration-and-conversations.md)

### `detect(...)`

Server route: configured `server.detect_path`, default `/api/v1/detect`.

Form fields:

| Field | Value |
|---|---|
| `file` | `capture.jpg`, JPEG bytes, MIME `image/jpeg`. |
| `metadata` | JSON string built by `MetadataBuilder`. |
| `publish` | String `true` or `false`. |

Validation:

- Response must be a JSON object.
- Response must contain `objects` as a list.

Used by:

- `TurnManager._run_scan_sequential`.
- `TurnManager._refresh_visual_context`.

### `panorama_detect(...)`

Server route: configured `server.detect_panorama_path`, default `/api/v1/detect/panorama`.

This endpoint sends multiple captures in one request.

Multipart fields:

| Field | Multiplicity | Value |
|---|---:|---|
| `files` | repeated | `capture_<index>.jpg`, JPEG bytes, MIME `image/jpeg`. |
| `metadata` | repeated | JSON metadata string for the corresponding image. |
| `publish` | once | String `true` or `false`. |
| `resize_image` | once | String `true` or `false`. Client currently passes `true`. |
| `stick_together` | once | String `true` or `false` from config. |

Validation:

- Response must be a JSON object.
- Response must contain `objects` as a list.

Used by:

- `TurnManager._run_scan_panorama`.

Server docs:

- [`../server/architecture-overview.md`](../server/architecture-overview.md)
- [`../server/detection-tracking-and-fusion.md`](../server/detection-tracking-and-fusion.md)

## JSON Endpoints

### `chat(...)`

Server route: configured `server.chat_path`, default `/api/v1/chat`.

Payload fields:

| Field | When sent | Meaning |
|---|---|---|
| `query` | always | User/model query text. |
| `chat_id` | when available | Existing conversation id from `SessionStore`. |
| `language` | when provided | Normalized output language, `english` or `czech`. |
| `mode` | when provided | Chat mode such as `general` or `object`. |
| `object_label` | object chat only | Label captured from `memory_objects`. |
| `model_facing_language` | when configured | Optional server-side model-facing language override. |

Validation:

- Response must be a JSON object.
- Response must contain `sentence`.
- Response must contain `chat_id`.

Wrapper methods:

- `chat_general(query, chat_id=None, language=None)` sends `mode="general"`.
- `chat_object(object_label, query, chat_id=None, language=None)` sends `mode="object"` and `object_label`.

Used by:

- `_run_ask`.
- `_run_object_ask`.
- `_run_cached_answer` fallback.
- `_run_scan` summary path.

Server docs:

- [`../server/orchestration-and-conversations.md`](../server/orchestration-and-conversations.md)

### `memory_summary(...)`

Server route: configured `server.memory_summary_path`, default `/api/v1/memory/summary`.

Request method: `GET`.

Query parameters:

| Parameter | Value |
|---|---|
| `render_limit` | Integer limit for rendered object crops/graph detail. |
| `lang` | Optional `en` or `cs`, derived from runtime language. |

Validation:

- Response must be a JSON object.

Expected fields consumed by the client:

- `labels`.
- `label_counts`.
- `scene_graph`.
- `graph_svg`.
- `pregenerated_qa` may be present but `TurnManager._run_show_memory` currently also calls `pregenerate_qa` explicitly.

Used by:

- Dynamic concept refresh.
- Tablet memory display.
- Show memory flow.

Server docs:

- [`../server/scene-memory-and-state.md`](../server/scene-memory-and-state.md)

### `reset_memory()`

Server route: configured `server.memory_reset_path`, default `/api/v1/memory/reset`.

Request method: `POST`.

If the configured path has no query string, the client appends `?confirm=true`.

Validation:

- Response must be a JSON object with `ok` truthy.

Used by:

- `TurnManager._run_reset_memory`.

### `pregenerate_qa(...)`

Server route: configured `server.pregenerate_qa_path`, default `/api/v1/chat/pregenerate_qa`.

Payload fields:

| Field | Value |
|---|---|
| `requested_number_of_pairs` | Integer from `tablet.pregenerated_questions_count`. |
| `output_language` | Optional `english` or `czech`. |

Validation:

- Response must be a JSON object.
- Response must contain `pregenerated_qa` as a list.

Used by:

- `TurnManager._run_show_memory`.

Returned pairs are stored by `SessionStore.update_after_pregenerated_qa` and become:

- Dynamic concept entries in `memory_cached_questions`.
- Exact question-to-answer cache entries.
- Tablet Q/A buttons.

Server docs:

- [`../server/orchestration-and-conversations.md`](../server/orchestration-and-conversations.md)
- [`../server/api-reference.md`](../server/api-reference.md)

### `reset_conversation(chat_id)`

Server route: hard-coded `/api/v1/chat/conversations/{chat_id}/reset`.

This path is not currently configurable in `client_config.json`.

Used by:

- `PepperGroundedClient.resetConversation`.
- `TurnManager._run_reset_memory` when an old chat id exists.

If `chat_id` is empty, the method returns `{"ok": True, "skipped": True}` without an HTTP request.

## Language Normalization

`_normalize_output_language` accepts:

- `en`, `english` -> `english`.
- `cs`, `cz`, `czech`, `czc` -> `czech`.
- Anything else -> `english`.

This normalization is used for chat and Q/A payloads.

`memory_summary` uses `speech_policy.language_code`, producing `en` or `cs` for the `lang` query parameter.

## Logging

`_request` logs:

- HTTP method and full URL before request.
- Error body excerpt for HTTP errors.
- First 240 characters of parsed JSON response.

`pepper_client.utils.logging.safe_json` safely stringifies response data for logs.

## Where To Add A New Server Call

Add new server calls only to `PepperServerTransport`.

A good method should:

1. Read path and timeout from config when appropriate.
2. Normalize language or booleans using existing helpers.
3. Validate response shape immediately.
4. Raise `MalformedResponseError` for contract mismatch.
5. Let `_request` convert network and HTTP errors.
6. Keep `TurnManager` code focused on workflow logic, not raw HTTP details.
