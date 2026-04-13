# Server Documentation Index

This directory documents the current `server/` codebase as it exists now.

The goal of this docs set is not to restate file names. It is to explain how the server actually works, where state lives, how requests move through the stack, what can be changed hot vs hard, which models are involved, and where to tweak behavior safely.

## What This Docs Set Covers

- FastAPI bootstrap, lifespan, state initialization, ngrok, static mounts
- Runtime state and dependency assembly
- Configuration schema, `config.yaml`, mutation rules, hot reload vs hard reload
- Public API routes under `/api/v1`
- Orchestration services and runtime adapters
- Inference pipeline stages and execution controls
- Detection backends, tracking, ReID, Pepper fusion, and SoM rendering
- Scene memory, scene graph generation, and grounded dialogue context
- LLM/VLM/caption/translation providers
- External worker process management and internal worker runtime
- Dashboard frontend modules, websocket updates, and operator controls
- Prompts, ontologies, persisted state, tests, scripts, and support assets
- An exhaustive grouped file inventory for the server tree

## Recommended Reading Paths

### If you want the big picture first

1. [architecture-overview.md](./architecture-overview.md)
2. [startup-runtime-and-state.md](./startup-runtime-and-state.md)
3. [configuration-and-reload.md](./configuration-and-reload.md)
4. [api-reference.md](./api-reference.md)
5. [inference-pipeline.md](./inference-pipeline.md)

### If you want to change model behavior

1. [configuration-and-reload.md](./configuration-and-reload.md)
2. [providers-model-clients.md](./providers-model-clients.md)
3. [detection-tracking-and-fusion.md](./detection-tracking-and-fusion.md)
4. [scene-graph-and-grounding.md](./scene-graph-and-grounding.md)

### If you want to change dialogue behavior

1. [orchestration-and-conversations.md](./orchestration-and-conversations.md)
2. [scene-memory-and-state.md](./scene-memory-and-state.md)
3. [providers-model-clients.md](./providers-model-clients.md)
4. [api-reference.md](./api-reference.md)

### If you want to change worker/process behavior

1. [worker-runtime-and-process-management.md](./worker-runtime-and-process-management.md)
2. [startup-runtime-and-state.md](./startup-runtime-and-state.md)
3. [configuration-and-reload.md](./configuration-and-reload.md)

### If you want to change the dashboard/operator surface

1. [dashboard-and-operator-ui.md](./dashboard-and-operator-ui.md)
2. [api-reference.md](./api-reference.md)
3. [configuration-and-reload.md](./configuration-and-reload.md)

## Document Map

- [architecture-overview.md](./architecture-overview.md)
  - End-to-end system shape, request flow, and subsystem boundaries.
- [startup-runtime-and-state.md](./startup-runtime-and-state.md)
  - `main.py`, `AppState`, pipeline assembly, storage, websocket manager, runtime adapter selection.
- [configuration-and-reload.md](./configuration-and-reload.md)
  - `config.yaml`, `schemas/config.py`, config manager, reload rules, mutation groups, runtime update rules.
- [api-reference.md](./api-reference.md)
  - Public HTTP routes, request/response models, side effects, publish/broadcast behavior.
- [orchestration-and-conversations.md](./orchestration-and-conversations.md)
  - Chat orchestration, caption orchestration, detection orchestration, memory service, conversation service.
- [inference-pipeline.md](./inference-pipeline.md)
  - `PerceptionPipeline`, stage ordering, execution metrics, stage enable/disable semantics.
- [detection-tracking-and-fusion.md](./detection-tracking-and-fusion.md)
  - Detector registry, detector implementations, associator, embedding extraction, Pepper-specific person fusion.
- [scene-memory-and-state.md](./scene-memory-and-state.md)
  - `SceneMemory`, state store mixins, object/relation/caption state, manual memory editing semantics.
- [scene-graph-and-grounding.md](./scene-graph-and-grounding.md)
  - Rule SGG, VLM SGG, RelTR path, SoM painter, robot metadata augmentation.
- [providers-model-clients.md](./providers-model-clients.md)
  - LLM/VLM/caption/translation client stack, structured output handling, provider-specific concerns.
- [data-models-and-contracts.md](./data-models-and-contracts.md)
  - Shared Pydantic models, inference data types, and worker contracts.
- [worker-runtime-and-process-management.md](./worker-runtime-and-process-management.md)
  - Worker manager, RPC schema, worker app, worker runtime, internal routes, idle shutdown and restart behavior.
- [dashboard-and-operator-ui.md](./dashboard-and-operator-ui.md)
  - Dashboard HTML/JS module layout, websocket message consumption, config editor, live view, scene graph panel.
- [assets-tests-and-support-files.md](./assets-tests-and-support-files.md)
  - Prompts, ontologies, persisted state, scripts, model weights, mocks, tests.
- [file-inventory.md](./file-inventory.md)
  - Exhaustive grouped inventory of the server tree with one-line file roles.

## Core Source Roots

- `server/app/api/v1`: public HTTP surface
- `server/app/core`: bootstrap, config mutation, runtime state, infra helpers
- `server/app/orchestration`: service layer and runtime adapters
- `server/app/inference`: detection, tracking, memory, scene graph, pipeline
- `server/app/providers`: model/provider abstraction layer
- `server/app/schemas`: shared Pydantic request/response/config/state models
- `server/app/worker`: separate worker process runtime
- `server/app/static`: dashboard UI
- `server/tests`: regression and contract tests

## Fast Orientation Cheat Sheet

- The single global runtime container is `app.core.runtime.state.app_state`.
- The server can run inference in-process or via `WorkerManager` + `server/app/worker`.
- The main detection route is `/api/v1/detect`.
- The main chat route is `/api/v1/chat`.
- The main operator surface is `/dashboard` plus `/dashboard/ws`.
- The config source of truth is `server/config.yaml` validated by `server/app/schemas/config.py`.
- The central perception execution entry point is `app.inference.pipeline.PerceptionPipeline.process()`.
- Persistent conversational grounding depends on `app.inference.memory.scene_memory.SceneMemory`.
- Scene graph generation is selected by `scene_graph.mode` and delegated through `SceneGraphService`.
