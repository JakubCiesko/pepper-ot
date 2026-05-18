# Pepper Server Documentation

This directory documents the current server implementation under `server/app`. It is meant as a navigation and maintenance guide for people changing the perception pipeline, dashboard, worker runtime, memory system, chat endpoints, or provider integrations.

The server is a FastAPI application that receives images and robot metadata, runs a configurable perception pipeline, stores scene memory, generates grounded scene graphs and Q/A pairs, serves a dashboard, and exposes robot-facing chat and memory APIs.

Robot-side client documentation lives in [`../robot-client/index.md`](../robot-client/index.md). Use it together with these server docs when changing QiChat grammar, Pepper metadata payloads, tablet memory display, or robot-facing API contracts.

## Mental Model

Think of the server as two planes:

- **Control plane:** FastAPI routes, dashboard, config manager, worker manager, conversation state, QA pool, WebSocket broadcasts, and persistence helpers. This plane decides what should happen and exposes the public API.
- **Inference plane:** detection, tracking, scene memory, captioning, Set-of-Mark rendering, scene graph generation, and QA generation. This plane consumes images and robot metadata and produces structured perception state.

In in-process mode both planes live in the API process. In worker mode the control plane remains in the API process, while the inference plane lives in a child FastAPI worker process. The public API behavior should be the same in both modes; only ownership and transport change.

There are three important state containers:

- `AppState`: process-level dependency root for config, services, worker manager, and optional in-process pipeline.
- `SceneMemory`: persistent visual/social memory of objects, tracks, relationships, captions, and crops. In worker mode this lives inside the worker pipeline.
- `QAPoolService` and `ConversationService`: API-process memory for generated scene Q/A pairs and chat history. These are not persisted across server restarts.

## Start Here

Read these documents in this order if you are new to the server:

1. [`architecture-overview.md`](architecture-overview.md) explains the full system shape and request flow.
2. [`startup-runtime-and-state.md`](startup-runtime-and-state.md) explains `AppState`, startup, worker mode, shared services, and persisted state.
3. [`configuration-and-reload.md`](configuration-and-reload.md) explains `config.yaml`, Pydantic config models, dashboard config patches, hot reload, and hard reload.
4. [`inference-pipeline.md`](inference-pipeline.md) explains the actual frame pipeline stage by stage.
5. [`detection-tracking-and-fusion.md`](detection-tracking-and-fusion.md) explains object detection, NMS, embeddings, persistent IDs, crops, and Pepper metadata fusion.
6. [`scene-graph-and-grounding.md`](scene-graph-and-grounding.md) explains SoM rendering, rules, RelTR, VLM scene graph generation, graph merge, and robot-data enhancement.
7. [`scene-memory-and-state.md`](scene-memory-and-state.md) explains persistent scene memory, object/relationship/caption storage, memory graph SVG, crops, pruning, and manual CRUD.
8. [`orchestration-and-conversations.md`](orchestration-and-conversations.md) explains API-level services, chat modes, language enforcement, conversation history, object chat, vision chat, caption orchestration, and QA pool behavior.
9. [`providers-model-clients.md`](providers-model-clients.md) explains OpenAI/Gemini/OpenAI-compatible/local-HF text and vision clients, structured output, caption clients, and translation providers.
10. [`worker-runtime-and-process-management.md`](worker-runtime-and-process-management.md) explains the worker process, internal RPC API, warmup, idle shutdown, hot config pushes, and memory proxying.
11. [`dashboard-and-operator-ui.md`](dashboard-and-operator-ui.md) explains the dashboard templates and JavaScript feature modules.
12. [`api-reference.md`](api-reference.md) lists the public API routes, internal worker routes, request shapes, and response behavior.
13. [`data-models-and-contracts.md`](data-models-and-contracts.md) maps schemas and internal dataclasses to their usage.
14. [`file-inventory.md`](file-inventory.md) is a file-by-file locator for server code.
15. [`assets-tests-and-support-files.md`](assets-tests-and-support-files.md) documents prompts, ontology, lexicons, state, static assets, and test/support folders.

## Current Core Flow

The main image-processing path is:

1. `POST /api/v1/detect` or `POST /api/v1/detect/panorama` receives image bytes and optional `RobotMetadata` JSON.
2. `app.api.v1.detect` normalizes multipart fields, optionally resizes image bytes, and calls `orchestration.services.detection.DetectService`.
3. `DetectService` selects either the in-process runtime adapter or worker runtime adapter.
4. Runtime executes `inference.pipeline.PerceptionPipeline.process`.
5. The pipeline can run captioning, detection, tracking/memory association, SoM drawing, scene graph generation, QA generation, caption-memory update, and scene-graph-memory update depending on `pipeline_controls`.
6. A `PipelineResult` is returned and converted into API/dashboard payloads.
7. Scene memory stores persistent objects, relationships, captions, robot/social metadata attributes, and last crops.
8. Scene graph Q/A pairs generated by the pipeline are ingested into the process-level `QAPoolService`.
9. Dashboard WebSocket clients receive live detection and memory events if `publish=true`.

The image path is therefore not "endpoint calls detector". The endpoint parses transport input, the orchestration service chooses a runtime, the runtime executes the pipeline, and the orchestration service converts the runtime result into API, dashboard, memory, and QA-pool side effects.

## Important Current Design Facts

- The scene graph system no longer uses one `scene_graph.mode` string. It uses independent backend toggles: `scene_graph.rules.enabled`, `scene_graph.reltr.enabled`, and `scene_graph.vlm.enabled`.
- Scene graph outputs are merged by `SceneGraph.__add__`, which deduplicates edges in `SceneGraph.__post_init__`.
- `pipeline_controls.qa_generation` is a first-class runtime stage and requires `scene_graph=true`.
- The QA pool is process memory, not database storage. It is cleared on memory reset and can be viewed/edited through API and dashboard.
- `pipeline_controls.parallel_execution` overlaps only safe early work: captioning and detection. Tracking, SoM rendering, scene graph generation, QA generation, and memory updates stay ordered.
- Detection has optional post-filter NMS controlled by `detection.run_nms_post_filter`, `detection.nms_iou_threshold`, and `detection.nms_type`.
- SoM mask generation supports `grabcut` and `sam`. SAM prompt boxes are internally batched in fixed chunks of 4 inside `som.py` to reduce peak prompt memory.
- Worker mode is the default intended GPU-heavy mode. The API process orchestrates and delegates detect/memory work to a child FastAPI worker.
- Chat language is controlled by request fields and dashboard `system.output_language`; the server stores both original and model-facing conversation text.
- Memory summaries can return translated labels/attributes/relations for Czech display using the vocabulary translator lexicons.

## How To Read A Detection Result

A detect response combines several layers:

- **Objects:** current-frame detections, usually with persistent `object_id` when tracking is enabled.
- **Scene graph:** relation and attribute edges generated by enabled SGG backends and robot-memory enhancement.
- **Caption:** optional image caption plus provider/model metadata.
- **QA pairs:** optional generated factual Q/A pairs. The response carries them, and `DetectService` also ingests them into the API-process QA pool.
- **Memory:** current memory state after tracking and graph/caption updates.
- **Metrics and executed stages:** the most reliable way to see what actually ran for this request.

If a field is missing, first check `executed_stages`, then `pipeline_controls`, then validation dependencies in `AppConfig`.

## Where To Change Common Features

- Add/change detect endpoint behavior: `server/app/api/v1/detect.py` and `server/app/orchestration/services/detection.py`.
- Add detector config fields: `server/app/schemas/config.py`, `server/config.yaml`, config reload rule definitions, dashboard detection template, dashboard config JS.
- Change model pipeline order: `server/app/inference/pipeline.py`.
- Change tracking or identity behavior: `server/app/inference/memory/scene_memory.py`, `server/app/inference/tracking/*`, and `server/app/inference/memory/state_store/*`.
- Change robot metadata fusion: `server/app/inference/memory/state_store/geometry.py`, `social.py`, `objects.py`, and `store.py`.
- Change SoM overlays or mask backends: `server/app/inference/scene_graph/som.py`.
- Change scene graph logic: `server/app/inference/scene_graph/service.py`, `rules_backend.py`, `reltr_backend.py`, `vlm_backend.py`.
- Change chat behavior: `server/app/api/v1/chat.py`, `server/app/orchestration/services/chat.py`, `conversation.py`.
- Change object-chat crop fallback: `server/app/orchestration/services/chat.py` and memory crop plumbing.
- Change QA generation/pool behavior: `server/app/inference/qa/service.py`, `server/app/orchestration/services/qa_pool.py`, `server/app/api/v1/chat.py`, `server/app/api/v1/memory.py`.
- Change worker lifecycle: `server/app/core/runtime/worker_client/*`, `server/app/worker/runtime.py`, `server/app/worker/routes.py`.
- Change dashboard UI: `server/app/static/templates/dashboard/pages/*.html` and `server/app/static/js/dashboard/features/*`.

## Documentation Maintenance Rule

When changing a subsystem, update both its dedicated document and [`file-inventory.md`](file-inventory.md). The inventory is intentionally redundant because it lets contributors find code quickly even when they do not yet know the architecture.
