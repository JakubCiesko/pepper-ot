# Orchestration and Conversations

The orchestration layer sits between FastAPI route handlers and low-level inference/runtime components. It keeps API handlers thin and centralizes cross-cutting concerns such as publishing, storage, language enforcement, conversation history, and QA pool ingestion.

## Main Files

API routes:

- `server/app/api/v1/chat.py`
- `server/app/api/v1/vision_chat.py`
- `server/app/api/v1/caption.py`
- `server/app/api/v1/detect.py`
- `server/app/api/v1/memory.py`

Services:

- `server/app/orchestration/services/chat.py`
- `server/app/orchestration/services/conversation.py`
- `server/app/orchestration/services/caption.py`
- `server/app/orchestration/services/detection.py`
- `server/app/orchestration/services/memory.py`
- `server/app/orchestration/services/memory_graph_render.py`
- `server/app/orchestration/services/qa_pool.py`

Runtime adapters:

- `server/app/orchestration/adapters/runtime.py`

## Runtime Adapters

`orchestration/adapters/runtime.py` hides in-process versus worker execution.

Adapters provide methods such as:

- `detect(image_bytes, robot_metadata)`
- `caption(image_bytes, prompt_override)`
- `vision_chat(image_bytes, user_prompt, system_prompt)`
- `scene_state()`
- `get_track_crop(object_id)`
- memory CRUD methods

This lets services and routes use one interface regardless of runtime mode.

## DetectService

File: `server/app/orchestration/services/detection.py`

This is API-level detection orchestration, not the low-level detector model service.

Responsibilities:

- Parse and normalize robot metadata JSON.
- Normalize angle units when metadata appears to be degrees.
- Fill default robot metadata when absent.
- Resolve runtime adapter.
- Call detect runtime.
- Convert runtime payload into `DetectionResponse`.
- Build dashboard WebSocket payload.
- Ingest pipeline-generated QA pairs into `QAPoolService`.
- Persist last state when configured.
- Publish dashboard events when requested.

QA ingestion happens only for memory-updating detect flows. It checks executed stages such as tracking and memory updates before adding pairs to the pool.

## CaptionService

File: `server/app/orchestration/services/caption.py`

This service handles the public caption endpoint behavior.

Responsibilities:

- Lazily create or update `CaptionClient`.
- Resolve caption user prompt from override, config prompt, mode, and max word cap.
- Run caption locally or through worker internal caption route.
- Enforce requested output language or `system.output_language` fallback.
- Optionally start background full detect pipeline when `run_detect=true`.

The background detect path uses `DetectService.process`, so it can update memory and QA pool like a normal detect request.

## ChatService

File: `server/app/orchestration/services/chat.py`

`ChatService` owns text-only grounded LLM behavior.

It stores:

- `LLMClient`
- memory adapter/proxy
- general system prompt
- general user prompt template
- object-specific system prompt
- object-specific user prompt template

### Scene Context

General chat builds scene context from memory:

- objects grouped by label and id
- attributes per object
- relationships
- latest caption
- recent captions

System prompt templates support placeholders such as:

- `{context}`
- `{caption}`
- `{captions_recent}`
- `{caption_recent}`
- `{predicates}`

General user prompt template supports placeholders such as:

- `{query}`
- `{history}`
- `{context}`
- `{caption}`
- `{captions_recent}`

### Object Chat

`object_chat` narrows the context to instances matching `object_label`.

Matching supports exact and loose labels. The service sorts matching objects by salience. Salience currently boosts people/animals/robot/cats/dogs, waving, looking at robot, sitting, engagement zone, closeness, hit count, and recency.

For each matched object it gathers:

- object id
- label
- bbox
- hits
- attributes
- robot distance and engagement fields
- incoming/outgoing relationships
- crop fallback descriptions when configured and needed

Object prompt templates support placeholders such as:

- `{query}`
- `{object_label}`
- `{resolved_label}`
- `{matched_ids}`
- `{matched_count}`
- `{history}`
- `{object_context}`
- `{scene_context}`

`max_instances` and `max_crop_fallbacks` can be `None` for no limit.

### Structured Chat

`chat_structured` calls `LLMClient.generate_structured` with a supplied Pydantic schema. It is used by forced QA generation fallback endpoints.

## ConversationService

File: `server/app/orchestration/services/conversation.py`

Conversation state is process-memory. It stores messages per chat id.

Each message stores:

- `id`
- `role`
- `text_original`
- `text_model`
- `language_original`
- `language_model`
- `translation_applied`
- `timestamp`
- `metadata`

Important methods:

- `add_message`
- `history`
- `prompt_history_model`
- `list_conversations`
- `reset`
- `delete`

`prompt_history_model` builds model-facing history and can exclude the latest user message because the current query is passed separately.

Default conversation id is `-1` when the client/robot does not provide a chat id.

## Text Chat Endpoint

File: `server/app/api/v1/chat.py`

`POST /api/v1/chat` supports `ChatMode.GENERAL` and `ChatMode.OBJECT`. `RELATION` and `ATTRIBUTE` enum values exist as placeholders but currently fall through to general behavior.

Language flow:

1. Output language is resolved from `output_language`, `language`, or config `system.output_language`.
2. Model-facing language is resolved from `model_facing_language` or output language.
3. User query is enforced into model-facing language.
4. Original and model-facing user text are stored.
5. Model-facing history is built.
6. Chat service runs selected mode.
7. Assistant output is enforced into output language.
8. Original/model-facing assistant text is stored.
9. Dashboard chat event is broadcast.

This design lets the admin decide whether the model should receive Czech or English by config/request. The server does not assume every model processes English best.

## Vision Chat Endpoint

File: `server/app/api/v1/vision_chat.py`

Vision chat is multipart image plus query. It uses the same conversation storage pattern as text chat but calls the VLM image backend through runtime adapter.

It keeps current-frame image context and text history. It does not currently rewrite the VLM system prompt beyond the route-specific behavior already implemented.

## QA Pool Service

File: `server/app/orchestration/services/qa_pool.py`

`QAPoolService` stores bilingual generated Q/A items in process memory.

Features:

- thread-safe `RLock`
- max entry cap
- dedup by normalized English question
- replacement/move-to-newest for duplicates
- source/frame/scan metadata
- English canonical storage
- Czech lazy translation and caching
- full snapshot for dashboard JSON editing
- replace-all update for dashboard saving
- clear on memory reset

The pool is not persisted across server restarts.

## QA Routes

File: `server/app/api/v1/chat.py`

Routes:

- `POST /api/v1/chat/pregenerate_qa`
- `GET /api/v1/chat/pregenerated_qa_pool`
- `PUT /api/v1/chat/pregenerated_qa_pool`

`POST /chat/pregenerate_qa` primarily reads the pool. If the pool is empty and `force_generation=true`, it can generate pairs from current memory text description and insert them.

`GET /chat/pregenerated_qa_pool` returns bilingual full pool items for dashboard editing.

`PUT /chat/pregenerated_qa_pool` replaces pool content with user-provided bilingual items.

## Memory Summary QA

File: `server/app/api/v1/memory.py`

`GET /api/v1/memory/summary` attaches `pregenerated_qa` from the QA pool in requested language. With `force_generation=true`, it can force generation only when the pool is empty.

## Dashboard Broadcasts

Services use `ws_manager.broadcast` to send:

- detection events
- memory events
- chat message events

The dashboard frontend modules subscribe through `/dashboard/events` and update live panels.

## Where To Change Things

- Change chat language behavior: `api/v1/chat.py`, `providers/translation/google_trans.py`, `ConversationService` metadata.
- Add a chat mode: `ChatMode`, `ChatRequest`, route `match`, and methods in `ChatService`.
- Change object-chat context: `ChatService.object_chat`.
- Change caption background detect: `CaptionService.caption_with_optional_detect`.
- Change QA pool semantics: `QAPoolService`, chat QA routes, memory summary route.
- Change publish payload: `DetectService.process`, dashboard live/memory JS.
