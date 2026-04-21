# Dashboard and Operator UI

The dashboard is a server-hosted operator UI for monitoring live perception, editing config, inspecting memory, chatting, editing QA pool entries, and tuning runtime settings.

## Main Files

Backend:

- `server/app/dashboard.py`
- `server/app/core/infra/ws_manager.py`

Templates:

- `server/app/static/templates/dashboard.html`
- `server/app/static/templates/dashboard/pages/live.html`
- `server/app/static/templates/dashboard/pages/detection.html`
- `server/app/static/templates/dashboard/pages/som.html`
- `server/app/static/templates/dashboard/pages/scene.html`
- `server/app/static/templates/dashboard/pages/runtime.html`
- `server/app/static/templates/dashboard/pages/memory-settings.html`
- `server/app/static/templates/dashboard/pages/chat.html`
- `server/app/static/templates/dashboard/pages/caption.html`
- `server/app/static/templates/dashboard/pages/qa-pregeneration.html`
- `server/app/static/templates/dashboard/pages/translations.html`
- `server/app/static/templates/dashboard/pages/storage.html`

JavaScript:

- `server/app/static/js/dashboard/app.js`
- `server/app/static/js/dashboard/core/http.js`
- `server/app/static/js/dashboard/core/notifications.js`
- `server/app/static/js/dashboard/core/ws.js`
- `server/app/static/js/dashboard/features/config/index.js`
- `server/app/static/js/dashboard/features/live/index.js`
- `server/app/static/js/dashboard/features/memory/*`
- `server/app/static/js/dashboard/features/conversation/index.js`
- `server/app/static/js/dashboard/features/scene_graph/index.js`
- `server/app/static/js/dashboard/features/qa/index.js`
- `server/app/static/js/dashboard/features/ui_shell/*`

Styles/assets:

- `server/app/static/css/style.css`
- `server/app/static/pepper_icon.png`

## Backend Routes

File: `server/app/dashboard.py`

Routes:

- `GET /dashboard`: renders `dashboard.html`.
- `WebSocket /dashboard/events`: websocket for live events.
- `GET /dashboard/config/get_models`: returns detection backend enum values.
- `POST /dashboard/chat_message`: manually broadcasts a chat message event.

## WebSocket Events

`ws_manager.broadcast` sends JSON text to all active dashboard clients.

Common event types:

- `detection`: latest detect/pipeline payload.
- `memory`: current scene memory payload.
- `chat_message`: conversation message.

The frontend dispatches events from `dashboard/app.js` to feature modules.

## Dashboard Layout

`dashboard.html` composes all page partials and loads dashboard JS modules. The UI shell modules handle sidebar, tabs, theme, and navigation.

Pages are hidden/shown as panels rather than separate routes.

## Live Page

Template: `pages/live.html`

JS: `features/live/index.js`

The Live page displays:

- architecture/system explanation
- latest processed image
- detections summary
- metrics
- captions
- scene graph carousel
- memory panel integration
- uploaded-image processing controls

It can upload an image to `/api/v1/detect` and maintains a carousel of recent frames.

## Detection Page

Template: `pages/detection.html`

JS: `features/config/index.js`

Controls:

- detector backend
- detector device
- confidence threshold slider/input
- NMS enabled checkbox
- NMS type select
- NMS IoU threshold slider/input
- object ontology textarea
- robot fusion numeric settings

The NMS controls map to:

- `detection.run_nms_post_filter`
- `detection.nms_type`
- `detection.nms_iou_threshold`

The robot fusion controls map to `fusion.*` fields.

## SoM Page

Template: `pages/som.html`

Controls visualization:

- bbox
- mask
- polygon
- labels
- line thickness
- mask opacity
- color lookup
- mask backend
- mask backend device

These map to `visualization.*` config.

## Scene Graph Page

Template: `pages/scene.html`

Controls:

- SGG backend checkboxes: VLM, Rules, RelTR
- Scene graph backend parallelism checkbox
- VLM system/user prompts
- VLM provider/model/base URL/API key/device
- VLM structured output mode/strict/schema
- local VLM hints
- VLM client/call kwargs
- predicate ontology
- rules JSON array
- RelTR checkpoint/device/threshold/topk/IoU match threshold

Backend checkboxes map to:

- `scene_graph.vlm.enabled`
- `scene_graph.rules.enabled`
- `scene_graph.reltr.enabled`
- `scene_graph.parallel_execution`

The parallelism checkbox starts enabled graph backends concurrently and merges their outputs deterministically. It should be treated as a latency/VRAM tradeoff knob.

## Runtime Page

Template: `pages/runtime.html`

Controls:

- pipeline preset
- individual pipeline stage checkboxes
- pipeline parallelism checkbox
- worker enabled
- worker host/port
- idle/startup/request/shutdown timeouts
- startup queue
- healthcheck interval
- restart circuit-breaker settings
- auto warmup

Pipeline controls include QA generation as a first-class stage.

The pipeline parallelism checkbox maps to `pipeline_controls.parallel_execution`. It overlaps caption and detection where safe, while tracking, SoM painting, scene graph generation, QA generation, and memory updates remain ordered.

## Memory Settings Page

Template: `pages/memory-settings.html`

Controls:

- max dormant frames
- association visual/geometry weights and match threshold
- feature extraction model/device/target size/resampling
- max memory age
- max objects
- max relations
- max captions
- caption max age

These map to `tracking.*` config.

## Chat Page

Template: `pages/chat.html`

Controls:

- general system prompt
- general user prompt template
- object system prompt
- object user prompt template
- chat device/provider/model/base URL/API key
- chat structured output mode and strict flag
- chat client init kwargs
- chat call kwargs

The conversation panel also supports text chat and vision chat from the dashboard using current active frame snapshots.

## Caption Page

Template: `pages/caption.html`

Controls:

- caption system/user prompts
- caption mode
- max words
- provider/device/model/base URL/API key
- client init kwargs
- call kwargs

## QA Pregeneration Page

Template: `pages/qa-pregeneration.html`

JS: `features/qa/index.js`

Displays and edits the bilingual QA pool.

Controls:

- auto pairs per frame (`qa_generation.pairs_per_update`)
- pool max entries (`qa_generation.pool_max_entries`)
- refresh pool
- save pool JSON
- reset editor
- force generate if empty

The JSON editor expects items with:

- `question_en`
- `answer_en`
- `question_cs`
- `answer_cs`

It uses:

- `GET /api/v1/chat/pregenerated_qa_pool`
- `PUT /api/v1/chat/pregenerated_qa_pool`
- `POST /api/v1/chat/pregenerate_qa`

## Translations Page

Template: `pages/translations.html`

Lets the operator edit user vocabulary translation maps for Czech:

- labels
- attributes
- relations

These are persisted in `providers/translation/lexicons_user` by the vocabulary translation service.

## Storage Page

Template: `pages/storage.html`

Controls:

- persist last state
- store image in last state
- last state path

## Config JS

File: `server/app/static/js/dashboard/features/config/index.js`

This is the central config UI module. It:

- fetches `/api/v1/config`
- populates all config controls
- validates JSON textareas
- syncs sliders and numeric inputs
- derives pipeline preset state
- displays structured-output capability hints
- builds PATCH payloads
- applies/saves/reloads/downloads/uploads config

If a new config field is added and needs dashboard control, this is usually where load/save wiring goes.

## Memory UI JS

Files under `features/memory/` handle memory panel DOM references, parsers, API calls, render functions, and actions. They display memory object/relationship/caption state and memory graph summaries.

## Conversation JS

File: `features/conversation/index.js`

Supports dashboard chat panel:

- text route `/api/v1/chat`
- vision route `/api/v1/vision_chat`
- loading latest conversation
- new conversation reset
- WebSocket chat message updates

## Where To Change Things

- Add a dashboard page: template partial, sidebar/tab registration, JS module import/init.
- Add a config control: template element, DOM ref in config JS, populate/load logic, buildPatch logic, event handling.
- Change live detection rendering: `features/live/index.js`.
- Change memory rendering: `features/memory/render.js` and related modules.
- Change QA pool editor: `features/qa/index.js` and `qa-pregeneration.html`.
- Change websocket event handling: `dashboard/app.js` and `core/ws.js`.
