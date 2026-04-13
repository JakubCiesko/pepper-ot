# API Reference

## Files Covered

- `app/api/v1/router.py`
- `app/api/v1/detect.py`
- `app/api/v1/caption.py`
- `app/api/v1/chat.py`
- `app/api/v1/vision_chat.py`
- `app/api/v1/config.py`
- `app/api/v1/memory.py`
- `app/api/v1/worker.py`
- `app/api/v1/image_utils.py`
- `app/api/v1/memory_route_utils.py`

All public routes live under `/api/v1`.

## Detection Routes

### `POST /api/v1/detect`

Defined in `detect.py`.

Purpose:
- Main image processing entry point.

Inputs:
- image upload
- robot metadata JSON
- publish flag
- debug/panorama-related form flags depending on client use

Behavior:
- Parses robot metadata into `RobotMetadata`.
- Routes to in-process or worker runtime.
- Returns normalized detection objects and optional caption.
- When publishing, broadcasts websocket payload and may persist last state.

Response model:
- `DetectionResponse`

### `POST /api/v1/detect/panorama`

Purpose:
- Build panorama from multiple frames and run detect flow on the stitched result.

Depends on:
- `image_utils.create_panorama()`

## Caption Route

### `POST /api/v1/caption`

Purpose:
- Caption image independently of the full detection pipeline.

Response model:
- `CaptionResponse`

Use this when testing caption quality or prompt changes without invoking the full pipeline.

## Chat Routes

### `POST /api/v1/chat`

Purpose:
- Main grounded text chat endpoint.

Request model:
- `ChatRequest`

Important request fields:
- `query`
- `chat_id`
- `language`
- `mode`
- `object_label`
- `max_instances`
- `max_crop_fallbacks`

Modes:
- `general`
- `object`

Behavior:
- Ensures a conversation exists.
- Translates input to configured output/model language when needed.
- Stores user message.
- Broadcasts user message to websocket subscribers.
- Builds conversation history for model prompting.
- Calls either `ChatService.chat()` or `ChatService.object_chat()`.
- Translates output back if needed.
- Stores assistant message and broadcasts it.
- Returns `ChatResponse` with extra metadata.

Metadata includes:
- provider/model info
- input/output language info
- requested/resolved object label
- matched object IDs and counts
- crop fallback usage

### `GET /api/v1/chat/conversations`

Purpose:
- List recent conversation sessions.

### `GET /api/v1/chat/conversations/{chat_id}`

Purpose:
- Return serialized conversation content.

### `POST /api/v1/chat/conversations/{chat_id}/reset`

Purpose:
- Reset a single conversation.

### `DELETE /api/v1/chat/conversations/{chat_id}`

Purpose:
- Delete a conversation entirely.

## Vision Chat Route

### `POST /api/v1/vision_chat`

Purpose:
- Direct VLM chat on image input, separate from scene-memory-grounded chat.

Request model:
- `VisionChatFormRequest`

Key form fields:
- image file
- `query`
- `chat_id`
- `language`
- `system_prompt`
- `resize_image`

Behavior:
- Maintains conversation state like text chat.
- Optionally resizes image before VLM call.
- Builds history-aware prompt.
- Routes directly to runtime adapter `vision_chat()`.
- Stores assistant response in conversation history.

Response model:
- `VisionChatResponse`

## Config Routes

### `GET /api/v1/config`

Returns:
- active config
- saved config
- resolved active config
- behavior contracts and hard reload field list

### `GET /api/v1/state`

Returns latest dashboard/live payload snapshot.

### `PATCH /api/v1/config`

Purpose:
- Apply partial config patch.

Behavior:
- Merge patch into current config.
- Compute diff.
- If hard changes exist, rebuild runtime via `app_state.apply_config()`.
- Otherwise apply hot changes in place.

### `POST /api/v1/config/save`

Purpose:
- Atomically write current runtime config to disk.

### `POST /api/v1/config/reload`

Purpose:
- Reload config from disk and rebuild runtime.

### `POST /api/v1/config/upload`

Purpose:
- Replace runtime config from uploaded YAML.

### `GET /api/v1/config/download`

Query param:
- `source=saved` to download saved file instead of active runtime state.

## Memory Routes

### `GET /api/v1/memory`
- full `SceneState`

### `GET /api/v1/memory/objects`
Filters:
- `label`
- `min_hits`
- `skip`
- `limit`
- `sort_by = last_seen | first_seen | hits`

### `GET /api/v1/memory/relations`
Filters:
- subject/object IDs
- subject/object labels
- predicate
- pagination

### `POST /api/v1/memory/upsert`
- merge external `SceneState` into memory

### `POST /api/v1/memory/reset`
- requires `confirm=true`

### `POST /api/v1/memory/object`
- create tracked object manually

### `PATCH /api/v1/memory/object/{object_id}`
- patch tracked object fields

### `DELETE /api/v1/memory/object/{object_id}`
- optional cascade relation deletion

### `POST /api/v1/memory/relation`
- create relation manually

### `PATCH /api/v1/memory/relation`
- patch relation metadata

### `DELETE /api/v1/memory/relation`
- delete by `(subject_id, predicate, object_id)`

All memory mutations route through `run_memory_action()` so errors become HTTP-safe and successful changes trigger dashboard broadcasts.

## Worker Routes

### `GET /api/v1/worker/status`
- current worker status snapshot

### `POST /api/v1/worker/warmup`
- force worker startup and warmup

### `POST /api/v1/worker/stop`
- stop worker process

## Utility Helpers

### `image_utils.py`

Contains helpers for:
- saving debug images
- resizing uploaded images
- building panorama images

### `memory_route_utils.py`

Provides common wrapper for memory operations so route handlers stay thin and broadcast behavior stays consistent.

## When to Change API Layer vs Lower Layers

Change route files when:
- input form/query shape changes
- endpoint semantics change
- auth/validation/error mapping changes

Change orchestration/inference when:
- model behavior changes
- memory semantics change
- runtime selection changes
- broadcast payload content changes
