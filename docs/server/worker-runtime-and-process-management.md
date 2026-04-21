# Worker Runtime and Process Management

Worker mode isolates heavy inference in a child FastAPI process. The public API process remains the orchestrator, dashboard server, config manager, and chat/conversation owner.

## Main Files

Worker client in API process:

- `server/app/core/runtime/worker_client/manager.py`
- `server/app/core/runtime/worker_client/manager_process.py`
- `server/app/core/runtime/worker_client/manager_monitor.py`
- `server/app/core/runtime/worker_client/manager_rpc.py`
- `server/app/core/runtime/worker_client/rpc.py`
- `server/app/core/runtime/worker_client/types.py`
- `server/app/core/runtime/worker_client/errors.py`

Worker process:

- `server/app/worker/main.py`
- `server/app/worker/runtime.py`
- `server/app/worker/routes.py`

Runtime adapter:

- `server/app/orchestration/adapters/runtime.py`

## Why Worker Mode Exists

Worker mode exists to keep GPU-heavy model objects out of the API process. Benefits:

- child process can be killed to release VRAM
- hard model/provider changes restart only the worker
- API process can keep dashboard/config/chat alive
- worker can lazy-start and idle-stop
- internal API gives clear process boundary for detect/memory operations

## WorkerManager

File: `server/app/core/runtime/worker_client/manager.py`

`WorkerManager` combines three mixins:

- `WorkerProcessMixin`: start/stop child process.
- `WorkerMonitorMixin`: lazy start, idle shutdown, circuit breaker, monitor loop.
- `WorkerRPCMixin`: internal HTTP helper.

State fields include:

- worker state
- subprocess handle
- lifecycle lock
- startup event/waiter count
- inflight request count
- last active timestamp
- restart counters
- idle kill/crash counters
- circuit breaker timestamp
- config version
- HTTPX async client

## Worker States

File: `server/app/core/runtime/worker_client/types.py`

States:

- `STOPPED`
- `STARTING`
- `READY`
- `BUSY`
- `STOPPING`
- `FAILED`

Restart reasons:

- `LAZY_START`
- `CONFIG_RELOAD`
- `MANUAL_WARMUP`

Stop reasons:

- `IDLE`
- `MANUAL`
- `SHUTDOWN`
- `CONFIG_RELOAD`
- `FAILURE`

## Startup Flow

`WorkerManager.ensure_started(reason)`:

1. Rejects if worker mode disabled.
2. Checks circuit breaker.
3. If worker already ready/busy, returns.
4. If worker is starting, waits on startup event subject to queue limit and startup timeout.
5. Otherwise acquires lifecycle lock and starts worker.

`_start_worker(reason)`:

1. Sets state `STARTING`.
2. Applies restart-window/circuit-breaker accounting.
3. Starts `python -m uvicorn app.worker.main:app --host ... --port ...` with cwd at server root.
4. Creates stdout/stderr forwarding tasks.
5. Waits for `/internal/health`.
6. Posts `/internal/config/reload` with full config and config version.
7. Optionally posts `/internal/warmup`.
8. Sets state `READY`.

## Shutdown Flow

`_stop_unlocked(reason)`:

1. If already stopped, cleans stream tasks and exits.
2. Sets `STOPPING`.
3. Attempts graceful `/internal/shutdown`.
4. Waits up to `shutdown_grace_seconds`.
5. Terminates or kills if needed.
6. Cancels stream forwarders.
7. Clears process handle and sets `STOPPED`.

## Monitor Loop

File: `manager_monitor.py`

The monitor loop periodically:

- detects worker crashes
- updates failure counters
- stops idle ready worker after `idle_timeout_seconds`
- obeys `idle_check_interval_seconds`

The circuit breaker opens when restarts exceed `restart_max_attempts` within `restart_window_seconds`, and remains open for `circuit_breaker_cooldown_seconds`.

## Internal RPC Types

File: `server/app/core/runtime/worker_client/rpc.py`

Important models:

- `DetectRPCRequest`: request id, config version, base64 image, optional robot metadata.
- `DetectRPCResponse`: ok flag, image output, objects, scene graph, QA pairs, caption metadata, memory, metrics, executed stages, image dimensions.
- `WorkerConfigRPCRequest`: full config dict.
- `WorkerStatusResponse`: state, pid, uptime, inflight count, counters, last error.

## Public Worker Control API

File: `server/app/api/v1/worker.py`

Routes:

- `GET /api/v1/worker/status`
- `POST /api/v1/worker/warmup`
- `POST /api/v1/worker/stop`

These act on `AppState.worker_manager`.

## Internal Worker Routes

File: `server/app/worker/routes.py`

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
- memory mirror routes under `/internal/memory...`

These routes are not meant for external clients. They are the private API between API process and worker process.

## WorkerRuntime

File: `server/app/worker/runtime.py`

`WorkerRuntime` owns the worker-local runtime:

- `config`
- `pipeline`
- worker state
- startup/last-active/inflight/error/config version
- lock
- caption client and prompts

Important methods:

- `apply_config(cfg, version, rebuild=True)`
- `ensure_pipeline()`
- `warmup()`
- `detect(image_b64, robot_metadata)`
- `caption(image_b64, prompt_override)`
- `vision_chat(image_b64, user_prompt, system_prompt)`
- memory state/CRUD/crop methods
- `status()`

The worker builds its pipeline lazily on first warmup/detect/memory operation.

## Hot Config in Worker Mode

Public config PATCH calls `WorkerManager.apply_hot_config(new_config, version)`. If worker is running, manager posts `/internal/config/hot_update`.

The worker route:

1. Validates config dict into `AppConfig`.
2. Calls `runtime.apply_config(cfg, version, rebuild=False)`.
3. Applies pipeline runtime updates if pipeline exists.
4. Applies scene graph runtime updates if pipeline exists.
5. Updates caption runtime if caption client exists.

Hard config changes cause worker stop through `WorkerManager.hard_reload`. The next request or warmup starts a fresh worker with full config.

## Memory in Worker Mode

When worker mode is enabled, the authoritative `SceneMemory` is inside `WorkerRuntime.pipeline.memory`.

Public memory routes use `MemoryService(WorkerRuntimeAdapter)`, which forwards memory calls through `WorkerManager.request(...)` to internal worker routes.

`ChatService` receives `WorkerChatMemoryProxy`, so text chat can still build context from worker memory.

## Detect in Worker Mode

Public detect path:

1. `DetectService.process` calls `WorkerRuntimeAdapter.detect`.
2. Adapter calls `WorkerManager.detect`.
3. Manager posts `/internal/detect` with base64 image and metadata.
4. Worker decodes image, runs `pipeline.process`, encodes SoM output image, returns objects/graph/QA/caption/memory/metrics/stages.
5. API process ingests QA pairs into its own `QAPoolService` and publishes dashboard events.

## Common Failure Modes

- Worker disabled but adapter requested: `WorkerUnavailableError`.
- Too many requests waiting during startup: `WorkerQueueFullError`.
- Startup health timeout: `WorkerStartupTimeoutError`.
- Restart thrashing: `WorkerCircuitOpenError`.
- Invalid worker response shape: `WorkerProtocolError`.

## Where To Change Things

- Change worker lifecycle behavior: `manager_process.py` and `manager_monitor.py`.
- Change internal request protocol: `worker_client/rpc.py`, `WorkerManager`, `worker/routes.py`, `worker/runtime.py`.
- Add pipeline output crossing worker boundary: `PipelineResult`, `WorkerRuntime.detect`, `DetectRPCResponse`, `WorkerManager.detect`, runtime adapter normalization, detection orchestration payload.
- Add memory operation in worker mode: public memory route/service, runtime adapter method, internal worker route, `WorkerRuntime` method.
