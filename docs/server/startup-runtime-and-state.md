# Startup, Runtime, and State

## Files Covered

- `app/main.py`
- `app/core/runtime/state.py`
- `app/core/pipeline_factory.py`
- `app/core/infra/storage.py`
- `app/core/infra/ws_manager.py`
- `app/dashboard.py`
- `app/orchestration/adapters/runtime.py`

## `app/main.py`

### `ServerSettings`

Environment-backed settings:
- `BASE_URL`: defaults to `http://localhost:8000`
- `USE_NGROK`: parsed from env var `USE_NGROK == "True"`

### Startup flow

- Logging is configured through `setup_logging()` using `colorlog`.
- Global `app_state.initialize("./config.yaml")` runs in FastAPI lifespan startup.
- If `USE_NGROK` is enabled, ngrok tunnel is opened and `BASE_URL` is updated.
- Static files are mounted at `/static` from `app/static`.
- API router is mounted at `/api/v1`.
- Dashboard router is mounted without extra prefix.

### Shutdown flow

- Worker manager is closed if it exists.
- Ngrok is killed if enabled.

## `app/core/runtime/state.py`

This file is the real application container.

`AppState` owns at least these runtime concerns:
- active `AppConfig`
- config version counter
- perception pipeline
- chat service
- caption service
- conversation service
- worker manager
- last published dashboard state

### Why `AppState` matters

Every route ultimately depends on `app_state`. If you are changing initialization behavior, switching providers, or reworking runtime boundaries, this is one of the first files to inspect.

### Typical responsibilities

- initialize runtime from config
- rebuild runtime on hard config changes
- hold references to assembled services
- choose whether worker mode is active

## `app/core/pipeline_factory.py`

This is the dependency assembly point for perception.

It builds and wires together:
- detector
- scene memory
- SoM painter
- scene graph service
- caption service
- `PerceptionPipeline`

If you want to swap a detector, change feature extraction defaults, alter scene graph backend creation, or add a new pipeline dependency, this is the safest construction point.

## Runtime adapter selection

Implemented in `app/orchestration/adapters/runtime.py`.

### Adapters

- `InProcessRuntimeAdapter`
- `WorkerRuntimeAdapter`
- `WorkerInternalRuntimeAdapter`

### Selection rule

`resolve_runtime_adapter(state)` uses worker mode when all of these are true:
- config exists
- `config.worker.enabled == true`
- `state.worker_manager is not None`

Otherwise it uses in-process execution.

### Why this matters

Most API and memory services deliberately target the adapter interface, not the raw pipeline. That keeps route logic mostly independent from process placement.

## Websocket manager

Implemented in `app/core/infra/ws_manager.py`.

### Responsibility

- hold active websocket connections
- broadcast live detection/chat/memory updates to dashboard clients

### Broadcast message types used in practice

- `detection`
- `chat_message`
- memory update payloads emitted by memory service
- dashboard-specific status messages

If live UI seems stale while HTTP routes still work, inspect websocket manager usage and broadcasting call sites first.

## Last-state persistence

Implemented in `app/core/infra/storage.py`.

### Functions

- `load_last_state(path)`
- `save_last_state(path, payload)`
- `save_last_image(path, image_b64)`
- async wrappers for the same

### Behavior

If storage persistence is enabled, the latest published detection payload is saved to disk. The image may be stripped from JSON and stored alongside as a `.jpg`, depending on config.

## Dashboard bootstrap

Implemented in `app/dashboard.py`.

### Routes/handlers

- `dashboard(request)` returns main HTML shell
- `dashboard_ws(websocket)` exposes websocket channel
- `list_models()` exposes model listing helper endpoint
- `dashboard_chat_message(payload)` provides dashboard-side message route

This file is small but important: it is the bridge between backend runtime events and the operator UI.

## Safe Changes

- logging configuration
- ngrok enable/disable behavior
- storage persistence path logic
- runtime adapter routing policy

## Risky Changes

- mutating `AppState` responsibilities without updating route consumers
- changing what detection payload shape is stored as `last_state`
- changing websocket message format without updating dashboard JS
