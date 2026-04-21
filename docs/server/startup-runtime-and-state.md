# Startup, Runtime, and Shared State

This document explains how the FastAPI server starts, how global runtime state is initialized, and how in-process versus worker runtime mode is selected.

## Entry Point

File: `server/app/main.py`

Startup sequence:

1. Configure colorized logging with `setup_logging()`.
2. Create `FastAPI(title="Pepper Object Detection Server")`.
3. Mount static assets at `/static` from `app/static`.
4. Include public API router under `/api/v1`.
5. Include dashboard router under `/dashboard`.
6. During lifespan startup call `await app_state.initialize("./config.yaml")`.
7. If `USE_NGROK=True`, create an ngrok tunnel and update `SERVER_SETTINGS.BASE_URL`.
8. During lifespan shutdown, close the worker manager if present and kill ngrok tunnels if enabled.

## AppState

File: `server/app/core/runtime/state.py`

`AppState` is a dataclass used as the process-level dependency root. It contains:

- `config`: current `AppConfig`.
- `pipeline`: in-process `PerceptionPipeline`, or `None` in worker mode.
- `worker_manager`: `WorkerManager` responsible for child worker process lifecycle.
- `chat_service`: process-level `ChatService` configured from `config.chat`.
- `conversation_service`: in-memory conversation store used by `/chat` and `/vision_chat`.
- `caption_service`: API-level caption orchestration service.
- `qa_pool_service`: bilingual process-memory pool of generated scene Q/A pairs.
- `initialized`: startup guard.
- `last_state`: optional persisted latest dashboard state loaded on startup.
- `config_version`: monotonically increasing runtime config version.

`AppState.initialize()` loads config, optionally restores persisted last state, then delegates to `apply_config()`.

## Config Application

`AppState.apply_config(config)` does the following:

1. Stores the new config and increments `config_version`.
2. Ensures a `WorkerManager` exists and starts its monitor loop.
3. Applies runtime mode:
   - if `config.worker.enabled=true`, skip in-process pipeline build, hard-reload worker manager, and keep `pipeline=None`.
   - if `config.worker.enabled=false`, build an in-process `PerceptionPipeline` and stop the worker monitor/process.
4. Warms vocabulary translations from config ontology and rule terms.
5. Initializes or updates `ChatService` and `ConversationService`.
6. Initializes or updates `CaptionService`.
7. Initializes or updates `QAPoolService` max size.
8. Optionally warm-starts worker if `worker.auto_warmup_on_startup=true`.

## Runtime Mode Selection

### In-Process Mode

In-process mode is active when `config.worker.enabled=false`.

`AppState.pipeline` is a real `PerceptionPipeline` built by `server/app/core/pipeline_factory.py`. API requests run model inference in the same process as FastAPI.

Use in-process mode when debugging code paths or avoiding child-process orchestration. It is less useful for VRAM lifecycle control because heavy model objects remain in the API process.

### Worker Mode

Worker mode is active when `config.worker.enabled=true`.

`AppState.pipeline=None`. Heavy inference runs in a child FastAPI process started by `WorkerManager`. API code accesses it through `WorkerRuntimeAdapter` and internal HTTP routes under `/internal/*`.

Worker mode is useful because:

- GPU-heavy models live in a separable process.
- Worker can idle-shutdown to release VRAM.
- Hard config changes can restart only the worker process.
- API process remains responsive while worker lifecycle changes.

## Service Initialization Details

### Chat Components

`_initialize_chat_components(base_dir)` resolves prompt sources from config:

- `chat.system_prompt`
- `chat.user_prompt`
- `chat.object_system_prompt`
- `chat.object_user_prompt`

It builds the chat memory adapter:

- in-process pipeline memory when `pipeline` exists
- `WorkerChatMemoryProxy` when worker mode is enabled
- `EmptyChatMemory` fallback otherwise

Then it creates `ChatService`. `ConversationService(max_messages=10)` is created once and kept across config hot updates.

### Caption Component

`_initialize_caption_component(base_dir)` resolves caption prompts and creates or updates `CaptionService`. In worker mode, it does not rebuild local caption clients unnecessarily; the worker handles its own caption client.

### QA Pool Component

`_initialize_qa_pool_component()` creates `QAPoolService(max_entries=config.qa_generation.pool_max_entries)` or updates the existing max size. The pool is process-memory only. It is cleared by memory reset routes.

## Last-State Persistence

Files:

- `server/app/core/infra/storage.py`
- `server/app/core/runtime/state.py`
- `server/app/orchestration/services/detection.py`

If `storage.persist_last_state=true`, startup tries to read `storage.last_state_path`. The load helper also resolves an external `image_path` into base64 image data when possible.

During detect publishing, the detection orchestration layer can persist the latest state payload. If `storage.store_image=true`, image content is saved separately and the JSON references it.

## WebSocket Manager

File: `server/app/core/infra/ws_manager.py`

`ConnectionManager` tracks dashboard WebSocket connections and broadcasts JSON messages. It removes dead connections opportunistically when sends fail.

Events currently include:

- `detection`: latest detection/pipeline result payload.
- `memory`: current memory state payload.
- `chat_message`: canonical user/assistant conversation message.

## Important Runtime Invariants

- `AppState.config_version` increments on full config application and hot patch application.
- In worker mode, `AppState.pipeline` should be `None`; code that needs memory should use a runtime adapter or chat memory proxy.
- `ConversationService` is process memory and is not persisted across server restarts.
- `QAPoolService` is process memory and is not persisted across server restarts.
- Worker internal memory is the source of truth in worker mode.
- Dashboard memory and chat displays are eventually consistent through WebSocket events and API refreshes.
