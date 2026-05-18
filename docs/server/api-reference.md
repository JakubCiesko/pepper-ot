# API Reference

This document lists current public API routes, dashboard routes, and internal worker routes. Public API routes are mounted under `/api/v1`.

## Detection

### `POST /api/v1/detect`

File: `server/app/api/v1/detect.py`

Multipart form fields:

- `file`: required image upload.
- `metadata`: optional JSON string matching `RobotMetadata`.
- `publish`: bool, default `true`.
- `resize_image`: bool, default `true`.

Behavior:

- Reads image bytes.
- Optionally resizes image.
- Parses/normalizes robot metadata.
- Runs `DetectService.process`.
- Publishes dashboard event when `publish=true`.
- Returns `DetectionResponse`.

Response fields:

- `id`
- `objects`
- `timestamp`
- `image_width`
- `image_height`
- `caption`
- `caption_provider`
- `caption_model_id`

The public HTTP response is intentionally compact. When `publish=true`, the dashboard WebSocket payload carries the richer runtime state: base64 image, objects, scene graph, QA pairs, caption metadata, memory snapshot, metrics, and executed stage names. If you need pipeline metrics from an HTTP workflow, use the dashboard event payload, memory/debug endpoints, or the research `pipeline-batch` command rather than expecting them in `DetectionResponse`.

### `POST /api/v1/detect/panorama`

Multipart form fields:

- `files`: list of image uploads.
- `metadata`: optional list of JSON strings.
- `publish`: bool, default `true`.
- `resize_image`: bool, default `true`.
- `stick_together`: bool, default `true`.

Behavior:

- `stick_together=true`: stitch images horizontally, merge metadata, run one detect pipeline.
- `stick_together=false`: process images independently and combine object lists in response.

Use this route for robot scan sweeps.

When images are stitched, `RobotMetadata.merge_robot_metadata_for_panorama` adjusts the frame geometry so person yaw values line up with the panorama coordinate frame. When images are processed independently, the returned HTTP response merges only object lists; richer per-frame graph, metric, and memory detail is not preserved in that compact response.

## Caption

### `POST /api/v1/caption`

File: `server/app/api/v1/caption.py`

Multipart form fields:

- `file`: required image upload.
- `metadata`: optional JSON string.
- `prompt`: optional prompt override.
- `run_detect`: bool, default `true`.
- `publish`: bool.
- `language`: optional output language.
- `resize_image`: bool.

Behavior:

- Returns a fast caption.
- Enforces output language when requested/configured.
- If `run_detect=true`, starts a background full detect pipeline using the same detect orchestration path.

Response includes caption text, provider, model id, whether detect was started, detect request id, and timestamp.

## Text Chat

### `POST /api/v1/chat`

File: `server/app/api/v1/chat.py`

JSON body: `ChatRequest`.

Important fields:

- `query`
- `chat_id`
- `conversation_id`
- `language`
- `input_language`
- `output_language`
- `model_facing_language`
- `mode`: `general` or `object`
- `object_label`
- `max_instances`
- `max_crop_fallbacks`

Behavior:

- Resolves output/model-facing language.
- Stores original/model-facing conversation messages.
- Builds model-facing history.
- Runs general chat or object chat.
- Enforces assistant output language.
- Broadcasts dashboard chat message.

Response: `ChatResponse` with response text, chat id, conversation id, metadata, timestamp.

### `GET /api/v1/chat/conversations`

Query:

- `limit`: max conversations to list.

Returns conversation summaries from process memory.

### `GET /api/v1/chat/conversations/{chat_id}`

Returns messages for a chat id.

### `POST /api/v1/chat/conversations/{chat_id}/reset`

Clears a conversation but keeps the id usable.

### `DELETE /api/v1/chat/conversations/{chat_id}`

Deletes a conversation from process memory.

## QA Pool

### `POST /api/v1/chat/pregenerate_qa`

JSON body: `PregeneratedQARequest`.

Fields:

- `language`
- `input_language`
- `output_language`
- `requested_number_of_pairs`
- `force_generation`

Behavior:

- Reads current QA pool.
- Returns pairs in requested language.
- If pool is empty and `force_generation=true`, generates pairs from current memory text description and inserts them.

Response: `PregeneratedQAResponse` with `pregenerated_qa` list and metadata.

### `GET /api/v1/chat/pregenerated_qa_pool`

Returns full bilingual QA pool items for dashboard editing.

### `PUT /api/v1/chat/pregenerated_qa_pool`

JSON body: `PregeneratedQAPoolUpdateRequest`.

Replaces QA pool with bilingual items.

## Vision Chat

### `POST /api/v1/vision_chat`

File: `server/app/api/v1/vision_chat.py`

Multipart form:

- `file`: required image upload.
- form fields from `VisionChatFormRequest`, including query/chat/language fields.

Behavior:

- Uses conversation history and current image.
- Calls VLM image backend via runtime adapter.
- Stores original/model-facing messages.
- Enforces output language.

## Memory

### `GET /api/v1/memory`

Returns full `SceneState`.

### `GET /api/v1/memory/summary`

Query:

- `render_limit`: default 5, limited by renderer cap.
- `lang`: `en`, `english`, `cs`, or `czech`.
- `force_generation`: bool, default false.

Returns `MemorySummary` including labels, counts, scene graph, SVG, and pregenerated QA.

If `force_generation=true`, the route generates QA only if the QA pool is empty.

### `GET /api/v1/memory/object/{object_id}/crop`

Returns base64 last crop for a tracked object, or `null` image if absent.

### `GET /api/v1/memory/objects`

Query:

- `label`
- `min_hits`
- `skip`
- `limit`
- `sort_by`: `last_seen`, `first_seen`, or `hits`

Returns object page and timestamp.

### `GET /api/v1/memory/relations`

Query filters:

- `subject_id`
- `subject_label`
- `predicate`
- `object_id`
- `object_label`
- `skip`
- `limit`

Returns relationship page and timestamp.

### `POST /api/v1/memory/upsert`

Body: `SceneState`.

Merges objects, relationships, and captions into memory.

### `POST /api/v1/memory/reset?confirm=true`

Clears scene memory and QA pool. Without `confirm=true`, returns validation error.

### Object CRUD

- `POST /api/v1/memory/object`
- `PATCH /api/v1/memory/object/{object_id}`
- `DELETE /api/v1/memory/object/{object_id}` with `cascade_relations=true|false`

### Relation CRUD

- `POST /api/v1/memory/relation`
- `PATCH /api/v1/memory/relation`
- `DELETE /api/v1/memory/relation?subject_id=...&predicate=...&object_id=...`

## Config

### `GET /api/v1/config`

Returns:

- active config
- saved config
- active resolved config
- translation maps
- behavior contracts

### `PATCH /api/v1/config`

Body: partial config patch.

Deep-merges, validates, applies hot or hard reload behavior, and returns diff info. Also supports translation patch payloads.

### `POST /api/v1/config/save`

Writes active config to YAML.

### `POST /api/v1/config/reload`

Reloads YAML from disk and applies it.

### `POST /api/v1/config/upload`

Uploads YAML, validates path safety, and applies it.

### `GET /api/v1/config/download`

Downloads active or saved YAML.

## State

### `GET /api/v1/state`

Returns current last state used by dashboard live panel. This may include persisted state loaded at startup and/or last published detection state.

## Worker Control

### `GET /api/v1/worker/status`

Returns worker manager status.

### `POST /api/v1/worker/warmup`

Starts/warmups worker pipeline.

### `POST /api/v1/worker/stop`

Stops worker process.

## Dashboard Routes

### `GET /dashboard`

Renders dashboard HTML.

### `WebSocket /dashboard/events`

Live dashboard event stream.

### `GET /dashboard/config/get_models`

Returns detection backend enum values.

### `POST /dashboard/chat_message`

Broadcasts a chat message event to dashboard clients.

## Internal Worker Routes

Internal routes are mounted by `server/app/worker/routes.py` in the child worker process.

Routes:

- `GET /internal/health`
- `GET /internal/status`
- `POST /internal/config/reload`
- `POST /internal/config/hot_update`
- `POST /internal/warmup`
- `POST /internal/detect`
- `POST /internal/caption`
- `POST /internal/vision_chat`
- `POST /internal/shutdown`
- `GET /internal/memory`
- `GET /internal/memory/summary`
- `GET /internal/memory/object/{object_id}/crop`
- `GET /internal/memory/objects`
- `GET /internal/memory/relations`
- `POST /internal/memory/upsert`
- `POST /internal/memory/reset`
- object CRUD mirror routes
- relation CRUD mirror routes

These routes are only for API-process to worker-process communication.
