# Server Architecture Overview

The server is a FastAPI application with three major responsibilities:

1. Run perception over robot or dashboard-uploaded images.
2. Maintain a persistent scene memory that can be queried by chat, dashboard, and robot clients.
3. Expose runtime controls through API and dashboard without requiring code changes for common model/pipeline tuning.

The entrypoint is `server/app/main.py`. It creates the FastAPI app, mounts `/static`, includes `/api/v1` routes, includes the dashboard router, and initializes global `app_state` during lifespan startup.

## Architectural Shape

The code is intentionally layered:

```text
HTTP route
  -> orchestration service
  -> runtime adapter
  -> in-process pipeline or worker RPC
  -> inference services and scene memory
```

Routes should stay thin. They parse request transport details, validate Pydantic request models, and call orchestration services. Orchestration services own API-level behavior: choosing runtime mode, formatting responses, broadcasting WebSocket events, persisting last state, and updating process-level services such as the QA pool or conversation history. The inference package owns model work and memory mutation. The worker package mirrors inference operations behind internal HTTP so the public API does not need separate worker-specific behavior.

## Top-Level Runtime Pieces

### FastAPI App

Files:

- `server/app/main.py`
- `server/app/api/v1/router.py`
- `server/app/dashboard.py`

`main.py` configures logging, loads `AppState`, optionally starts ngrok, and shuts down the worker manager on server shutdown. Public APIs are mounted under `/api/v1`. The dashboard is mounted at `/dashboard` and uses `/dashboard/events` WebSocket.

### AppState

File: `server/app/core/runtime/state.py`

`AppState` owns the process-level runtime objects:

- active `AppConfig`
- config version
- optional in-process `PerceptionPipeline`
- optional `WorkerManager`
- `ChatService`
- `ConversationService`
- `CaptionService`
- `QAPoolService`
- last persisted state payload

This is the dependency root for API service classes. API handlers instantiate lightweight orchestration services and pass `app_state` into them.

The most important invariant is that `AppState.pipeline` exists only in in-process mode. In worker mode the authoritative pipeline and scene memory are inside `WorkerRuntime`; API code must go through a runtime adapter or memory proxy instead of touching `app_state.pipeline`.

### Runtime Adapter Layer

File: `server/app/orchestration/adapters/runtime.py`

API services do not call the pipeline directly. They call a runtime adapter:

- `InProcessRuntimeAdapter` runs `PerceptionPipeline` directly in the API process.
- `WorkerRuntimeAdapter` forwards requests to the worker process through `WorkerManager`.
- `WorkerInternalRuntimeAdapter` exposes the same memory/runtime shape inside the worker process.

This abstraction lets memory, detect, caption, and chat code stay mostly the same whether worker mode is enabled.

Runtime adapters normalize the process boundary. In-process methods can return Python objects directly. Worker methods must serialize images, metadata, detections, graph edges, memory state, metrics, and stage names through Pydantic RPC models. If a new pipeline output must be visible in API responses, it has to cross this adapter layer and the worker RPC contract.

### Worker Process

Files:

- `server/app/core/runtime/worker_client/manager.py`
- `server/app/core/runtime/worker_client/manager_process.py`
- `server/app/core/runtime/worker_client/manager_monitor.py`
- `server/app/core/runtime/worker_client/manager_rpc.py`
- `server/app/worker/main.py`
- `server/app/worker/runtime.py`
- `server/app/worker/routes.py`

Worker mode starts a separate FastAPI app serving `/internal/*` routes. Heavy GPU objects live in the worker process. The API process sends image bytes and metadata over internal HTTP. The worker supports health/status, warmup, hot config update, hard config reload through restart, detect, caption, vision chat, memory state access, crops, and memory CRUD.

## Request Flow: Detect

Primary files:

- `server/app/api/v1/detect.py`
- `server/app/orchestration/services/detection.py`
- `server/app/orchestration/adapters/runtime.py`
- `server/app/inference/pipeline.py`

Flow:

1. The public `/api/v1/detect` route accepts multipart `file`, `metadata`, `publish`, and `resize_image` fields.
2. Image bytes may be resized by `api/v1/image_utils.py` before inference.
3. If metadata is present, `DetectService.parse_metadata` validates `RobotMetadata`, normalizes angle units, and falls back to safe defaults when metadata is absent.
4. `DetectService.process` resolves the runtime adapter.
5. The adapter either runs `PerceptionPipeline.process` locally or calls `WorkerManager.detect`.
6. The returned runtime payload is normalized into `DetectionResponse`, WebSocket payload, memory payload, and QA pool ingestion.
7. If `publish=true`, dashboard clients receive a `detection` event and then memory state is available for the dashboard memory panel.
8. If `storage.persist_last_state=true`, last-state persistence code saves the latest payload.

Ownership detail:

- `api/v1/detect.py` owns multipart parsing, optional image resize, panorama stitching, and response model binding.
- `DetectService` owns metadata validation, runtime selection, response normalization, QA-pool ingestion, WebSocket publishing, and last-state persistence.
- `PerceptionPipeline` owns stage execution and returns a `PipelineResult`; it does not know about public HTTP responses or dashboard clients.

## Request Flow: Panorama Detect

File: `server/app/api/v1/detect.py`

`POST /api/v1/detect/panorama` accepts multiple image files and a list of metadata JSON strings.

There are two modes:

- `stick_together=true`: images are stitched horizontally, metadata is merged using `RobotMetadata.merge_robot_metadata_for_panorama`, and one detect pipeline is run on the panorama.
- `stick_together=false`: each image is processed separately and the endpoint returns a combined object list. This mode is useful when you want per-frame geometry and do not want panorama stitching artifacts.

Metadata merging assumes images are ordered left-to-right in the same order as files. It sums horizontal FOVs, averages vertical FOVs, shifts Pepper person yaw values into the panorama coordinate frame, and carries social metadata forward.

## Request Flow: Caption

Files:

- `server/app/api/v1/caption.py`
- `server/app/orchestration/services/caption.py`
- `server/app/providers/caption/client.py`

`POST /api/v1/caption` returns a fast caption. It can also trigger a full detect pipeline in the background with `run_detect=true`. That background detect path is intentionally the same `DetectService.process` path used by `/detect`, so memory, scene graph, and QA generation can still happen when configured.

## Request Flow: Chat

Files:

- `server/app/api/v1/chat.py`
- `server/app/orchestration/services/chat.py`
- `server/app/orchestration/services/conversation.py`
- `server/app/providers/translation/google_trans.py`

`POST /api/v1/chat` is the main text chat route. It supports `mode=general` and `mode=object`. The endpoint:

1. Resolves output language from request fields or `config.system.output_language`.
2. Resolves model-facing language from request or output language.
3. Enforces the user query into model-facing language.
4. Stores the original and model-facing user text in `ConversationService`.
5. Builds model-facing history from previous model-facing messages.
6. Calls `ChatService.chat` or `ChatService.object_chat`.
7. Enforces the assistant response into the requested output language.
8. Stores original/model-facing assistant text.
9. Broadcasts the message to dashboard clients.

The design preserves original text for UI/debugging and model-facing text for consistent prompt history.

Object chat uses scene memory as grounding. In in-process mode the chat service can read memory from the pipeline. In worker mode it reads through `WorkerChatMemoryProxy`, because the worker process owns the current `SceneMemory`.

## Request Flow: Vision Chat

Files:

- `server/app/api/v1/vision_chat.py`
- `server/app/orchestration/adapters/runtime.py`
- `server/app/worker/runtime.py`

`POST /api/v1/vision_chat` accepts an image and a text query. It shares `ConversationService` history semantics with text chat, but the underlying answer comes from the scene-graph VLM backend client rather than the text-only chat LLM. The route keeps the current VLM prompt behavior separate from the normal text chat system prompt.

## Scene Memory

Core files:

- `server/app/inference/memory/scene_memory.py`
- `server/app/inference/memory/state_store/store.py`
- `server/app/inference/memory/state_store/*.py`
- `server/app/orchestration/services/memory.py`

Scene memory stores:

- active and dormant tracks
- object state records
- relationship state records
- caption state records
- Pepper-person to server-object bindings
- last crop bytes per tracked object

Memory is updated by the pipeline after tracking and scene graph generation. It can also be manually edited through memory CRUD API endpoints.

Memory has two roles. During inference it is an identity and grounding store: it assigns persistent IDs, keeps crops and embeddings, carries robot/social metadata, and stores graph facts. During chat/dashboard use it is a queryable state snapshot: memory routes and summaries expose objects, relationships, captions, crops, and graph visualization.

## Scene Graph

Core files:

- `server/app/inference/scene_graph/service.py`
- `server/app/inference/scene_graph/rules_backend.py`
- `server/app/inference/scene_graph/reltr_backend.py`
- `server/app/inference/scene_graph/vlm_backend.py`
- `server/app/inference/scene_graph/som.py`

Scene graph generation is backend-compositional. Any combination of `rules`, `reltr`, and `vlm` can run for a frame. Their `SceneGraph` objects are added together and deduplicated. After backend merge, robot-derived object attributes stored in scene memory are injected into the graph for current objects.

## QA Generation

Files:

- `server/app/inference/qa/service.py`
- `server/app/orchestration/services/qa_pool.py`
- `server/app/api/v1/chat.py`
- `server/app/api/v1/memory.py`

QA generation is a pipeline stage after scene graph generation. It generates English factual question-answer pairs from graph triples and caption context. `DetectService` ingests generated pairs into `QAPoolService`, which stores bilingual items and lazily translates Czech text. The pool is exposed through chat QA endpoints and memory summaries.

## Dashboard

Files:

- `server/app/static/templates/dashboard.html`
- `server/app/static/templates/dashboard/pages/*.html`
- `server/app/static/js/dashboard/app.js`
- `server/app/static/js/dashboard/features/*`

The dashboard is a modular operator UI. It receives WebSocket events for live frames, memory changes, and chat messages. It also patches config through `/api/v1/config`, edits memory through memory CRUD APIs, displays QA pool JSON, and shows memory graph SVG summaries.

## Configuration Philosophy

The config system intentionally separates:

- model/backend changes that require hard reload or worker restart
- runtime knobs that can be hot-applied to existing services
- per-call request overrides such as caption prompt or chat language

The source of truth is `server/config.yaml`, validated by `server/app/schemas/config.py`. Runtime patch logic lives under `server/app/core/config` and `server/app/core/config/mutations`.

When changing config, decide whether the field can be hot-applied to already constructed services. Prompt text, thresholds, pipeline toggles, visualization switches, and some runtime limits can usually be patched. Model/provider/backend changes usually require a hard reload because live model instances have to be rebuilt, and in worker mode that means restarting the worker process.
