# Server File Inventory

This is a locator guide for `server/app`. It is intentionally redundant with the subsystem docs so contributors can find code quickly by file name.

## Entrypoints and Routing

### `server/app/main.py`

FastAPI application entrypoint. Configures logging, initializes `app_state`, optionally starts ngrok, mounts `/static`, includes `/api/v1` router, and includes dashboard routes.

### `server/app/dashboard.py`

Dashboard backend routes: render dashboard, manage dashboard WebSocket, list detection models, and broadcast manual chat messages.

### `server/app/api/v1/router.py`

Aggregates all public API routers under `/api/v1`.

### `server/app/api/v1/__init__.py`

Exports API router.

## Public API Routes

### `server/app/api/v1/detect.py`

Public detection endpoints. Handles multipart image upload, metadata JSON form fields, image resizing, panorama stitching/non-stitch flow, metadata patching after resize, and calls `DetectService`.

### `server/app/api/v1/caption.py`

Public caption endpoint. Handles image upload, optional metadata, prompt override, optional background detect, publish flag, language, and resize flag.

### `server/app/api/v1/chat.py`

Text chat and QA routes. Implements general/object chat dispatch, language enforcement, conversation storage, QA pool read/force-generation, QA pool get/replace, and dashboard chat broadcasts.

### `server/app/api/v1/vision_chat.py`

Vision chat route. Accepts image plus query, shares conversation service/history with text chat, and calls VLM image backend through runtime adapter.

### `server/app/api/v1/memory.py`

Public memory routes. Exposes full memory, memory summary/SVG/QA, object crops, object/relation listing, memory reset/upsert, object CRUD, and relation CRUD.

### `server/app/api/v1/memory_route_utils.py`

Converts memory service domain errors into FastAPI HTTP exceptions and optionally runs success callbacks.

### `server/app/api/v1/config.py`

Public config endpoints. Returns active/saved/resolved config and contracts, applies patches, saves/reloads/uploads/downloads YAML, and handles translation patch payloads.

### `server/app/api/v1/state.py`

Returns latest dashboard state payload, including persisted state loaded at startup if configured.

### `server/app/api/v1/worker.py`

Worker control endpoints for status, warmup, and stop.

### `server/app/api/v1/image_utils.py`

Image helper functions for resolution checking/resizing used by detect/caption endpoints.

## Schemas

### `server/app/schemas/config.py`

Top-level config models: detection, tracking, fusion, scene graph, QA generation, chat, caption, visualization, storage, worker, pipeline controls, prompt sources, ontology sources, and config validation.

### `server/app/schemas/detect.py`

Detection API response/request form schemas.

### `server/app/schemas/robot.py`

Robot metadata schemas for Pepper pose, camera FOV, geometric people, social people, and panorama metadata merging.

### `server/app/schemas/scene.py`

Scene/memory schemas: tracked object state, relationship, caption state, scene state, scene graph structured response, memory summary.

### `server/app/schemas/chat.py`

Chat modes, chat request/response, vision chat form/request schemas, pregenerated QA request/response, bilingual QA pool schemas.

## Core Runtime and Config

### `server/app/core/runtime/state.py`

Global `AppState` and initialization/config application logic. Owns config, pipeline, worker manager, chat/caption/conversation services, QA pool service, last state, and config version.

### `server/app/core/pipeline_factory.py`

Builds `PerceptionPipeline` and all inference-stage dependencies from `AppConfig`.

### `server/app/core/infra/ws_manager.py`

Dashboard WebSocket connection manager and broadcast helper.

### `server/app/core/infra/storage.py`

Last-state JSON/image load/save helpers.

### `server/app/core/prompting/renderer.py`

Simple prompt placeholder replacement with `PromptRenderContext`.

### `server/app/core/config/config_manager.py`

Loads/dumps/resolves config, returns dashboard behavior contracts, deep-merges patches, validates uploaded YAML paths, computes config diffs, and applies hot config changes.

### `server/app/core/config/mutations/components.py`

Assembles all reload rules, verifies uniqueness, computes hot/hard diff, applies hot handlers, and pushes hot config to worker.

### `server/app/core/config/mutations/reload_rule.py`

Defines `ReloadRule`, `ConfigDiff`, and hot handler groups for pipeline, QA, chat, and caption runtime updates.

### `server/app/core/config/mutations/runtime.py`

Mutates live runtime objects on hot config update: detector, pipeline controls, fusion, visualization, QA service, memory settings, scene graph backends.

### `server/app/core/config/mutations/rule_definitions/*.py`

Reload rule lists per config section: detection, caption, chat, pipeline, QA generation, scene graph, storage, tracking, visualization, worker.

## Worker Client and Worker Process

### `server/app/core/runtime/worker_client/manager.py`

Main worker manager class combining process, monitor, and RPC mixins. Provides status, config update, hot config push, hard reload, warmup, detect, and status calls.

### `server/app/core/runtime/worker_client/manager_process.py`

Starts/stops child worker uvicorn process, forwards stdout/stderr, waits for health, posts config reload, handles graceful/forced shutdown.

### `server/app/core/runtime/worker_client/manager_monitor.py`

Worker monitor loop, lazy-start waiting, idle shutdown, crash detection, startup queue limit, and circuit breaker checks.

### `server/app/core/runtime/worker_client/manager_rpc.py`

Internal HTTP request helpers for worker RPC.

### `server/app/core/runtime/worker_client/rpc.py`

Pydantic request/response models crossing the API-process to worker-process boundary.

### `server/app/core/runtime/worker_client/types.py`

Worker state enums, restart/stop reasons, and status snapshot model.

### `server/app/core/runtime/worker_client/errors.py`

Worker-specific exception types.

### `server/app/worker/main.py`

Worker FastAPI entrypoint.

### `server/app/worker/routes.py`

Internal `/internal/*` routes for health, status, config, warmup, detect, caption, vision chat, shutdown, and memory CRUD mirrors.

### `server/app/worker/runtime.py`

Worker-local runtime. Lazily builds pipeline and caption client, runs detect/caption/vision_chat, exposes memory state/CRUD/crops, and returns worker status.

## Orchestration Services and Adapters

### `server/app/orchestration/adapters/runtime.py`

In-process, worker, and worker-internal runtime adapters. Normalizes pipeline/worker outputs for API services and provides memory/crop/CRUD operations.

### `server/app/orchestration/services/detection.py`

API-level detect orchestration. Parses metadata, calls runtime adapter, builds response/publish payload, persists state, broadcasts dashboard event, and ingests pipeline QA pairs into QA pool.

### `server/app/orchestration/services/caption.py`

API-level caption orchestration. Handles caption clients, worker caption calls, language enforcement, and optional background detect.

### `server/app/orchestration/services/chat.py`

Grounded chat service. Builds scene context, prompt history, general chat prompts, object-chat narrow context, object salience, crop fallback descriptions, and structured chat calls.

### `server/app/orchestration/services/conversation.py`

In-memory conversation store with original/model-facing text fields and bounded history.

### `server/app/orchestration/services/memory.py`

API-level memory service. Wraps runtime adapter for full memory, summaries, crops, list/filter, reset, upsert, object CRUD, relation CRUD, and memory broadcasts.

### `server/app/orchestration/services/memory_graph_render.py`

Builds memory summaries, graph SVG, crop node rendering, object/attribute/relation display, and text descriptions for QA fallback generation.

### `server/app/orchestration/services/qa_pool.py`

Thread-safe in-memory bilingual QA pool with dedup, cap, snapshot, replace-all, lazy Czech translation, and source metadata.

## Inference Pipeline and Types

### `server/app/inference/pipeline.py`

Main `PerceptionPipeline` stage order and timing. Runs caption, detection, tracking, SoM, scene graph, QA generation, caption memory update, scene graph memory update.

### `server/app/inference/types.py`

Internal inference dataclasses and Pydantic objects: detection object, tracked object, scene graph edge, scene graph, pipeline result. Also contains old removal-marked classes kept in code but not central to current flow.

## Detection Inference

### `server/app/inference/detection/service.py`

Low-level detection service. Owns detector backend, threshold, NMS settings, ontology/device, `detect`, `detect_batch`, NMS helpers.

### `server/app/inference/detection/detectors.py`

Backend wrapper classes for Ultralytics, RF-DETR, and OWLv2.

### `server/app/inference/detection/model_registry.py`

Model construction/download registry for detector backends.

## Tracking and Memory Inference

### `server/app/inference/tracking/embeddings.py`

Feature extractor for detection crops and normalized embeddings. Returns crop JPEG bytes for track last-crop storage.

### `server/app/inference/tracking/associator.py`

Matches detections to active tracks using visual and geometric scoring.

### `server/app/inference/memory/scene_memory.py`

High-level memory update lifecycle and tracking/fusion orchestration.

### `server/app/inference/memory/chat_memory_proxy.py`

Memory proxy used by chat service in worker mode plus empty fallback memory.

### `server/app/inference/memory/state_store/store.py`

Main in-memory state store combining tracks, objects, relations, geometry, and social mixins. Owns Pepper bindings and frame binding maps.

### `server/app/inference/memory/state_store/tracks.py`

Track creation, aging, pruning, snapshot, and reset logic.

### `server/app/inference/memory/state_store/objects.py`

Object insert/patch/delete and update-from-detections logic.

### `server/app/inference/memory/state_store/relations.py`

Relation insert/patch/delete and scene graph memory update logic.

### `server/app/inference/memory/state_store/geometry.py`

Pixel-angle projection, angular similarity, Pepper person candidate selection/scoring, assignment, and synthetic person geometry helpers.

### `server/app/inference/memory/state_store/social.py`

Social metadata to attributes conversion and social attribute merging.

## Caption and QA Inference

### `server/app/inference/caption/service.py`

Pure inference caption service used inside `PerceptionPipeline`. No API/dashboard side effects.

### `server/app/inference/qa/service.py`

Pipeline QA generation stage. Generates English graph-grounded Q/A pairs from current scene graph, detections, and caption using structured chat LLM output.

## Scene Graph Inference

### `server/app/inference/scene_graph/service.py`

Composes enabled scene graph backends, merges/deduplicates graphs, and injects robot-derived memory attributes.

### `server/app/inference/scene_graph/rules_backend.py`

Deterministic geometry/rule/color scene graph backend.

### `server/app/inference/scene_graph/reltr_backend.py`

RelTR scene graph backend. Runs RelTR, maps predicted boxes to tracked detections by IoU, converts some predictions into binary relations or unary attributes.

### `server/app/inference/scene_graph/reltr_predictor.py`

RelTR model wrapper and Visual Genome class/relation vocabularies.

### `server/app/inference/scene_graph/vlm_backend.py`

VLM scene graph backend. Renders prompts, calls VLM client with structured schema, parses/repairs JSON, filters hallucinated ids.

### `server/app/inference/scene_graph/som.py`

Set-of-Mark rendering and mask generation. Supports boxes, labels, masks, polygons, GrabCut, and SAM3 box-prompt masks batched in chunks of 4.

## Providers

### `server/app/providers/llm/client.py`

Provider-agnostic LLM client used by chat and QA generation.

### `server/app/providers/llm/base.py`

Base text provider interface and `LLMResponse`.

### `server/app/providers/llm/openai_llm.py`

OpenAI/OpenAI-compatible text provider with provider-native, instructor, and parse-output structured paths.

### `server/app/providers/llm/gemini_llm.py`

Gemini text provider with JSON schema/mime structured output behavior.

### `server/app/providers/llm/hf_llm.py`

Local Hugging Face causal LM provider.

### `server/app/providers/vlm/base.py`

Base VLM interface.

### `server/app/providers/vlm/factory.py`

Builds VLM clients from config.

### `server/app/providers/vlm/openai_vlm.py`

OpenAI/OpenAI-compatible image client with Responses API, chat completions, instructor, and parse-output paths.

### `server/app/providers/vlm/gemini_vlm.py`

Gemini image client.

### `server/app/providers/vlm/local_hf_vlm.py`

Local Hugging Face image-text model client.

### `server/app/providers/caption/client.py`

Caption client selection. Special BLIP local client plus VLM fallback.

### `server/app/providers/common/io.py`

Structured output parsing, JSON block extraction, schema validation, mode resolution, OpenAI response text extraction.

### `server/app/providers/common/runtime_setup.py`

Provider credential/client-kwargs setup for OpenAI and Gemini clients.

### `server/app/providers/common/utils.py`

Provider capability matrix and call kwarg normalization.

### `server/app/providers/translation/google_trans.py`

Free-text language detection, translation, and output-language enforcement.

### `server/app/providers/translation/vocabulary.py`

Token-level vocabulary translation service for labels/attributes/relations and dashboard-editable Czech lexicons.

## Static Dashboard Files

### `server/app/static/templates/dashboard.html`

Root dashboard template composing page partials and JS modules.

### `server/app/static/templates/dashboard/pages/*.html`

Dashboard page partials for live, detection, SoM, scene graph, runtime, memory settings, chat, caption, QA pregeneration, translations, and storage.

### `server/app/static/js/dashboard/app.js`

Dashboard frontend bootstrap and WebSocket event dispatch.

### `server/app/static/js/dashboard/core/*.js`

Small shared frontend utilities for HTTP, notifications, and WebSocket creation.

### `server/app/static/js/dashboard/features/config/index.js`

Central config UI load/save/patch module.

### `server/app/static/js/dashboard/features/live/index.js`

Live frame rendering, detection upload, metrics, carousel, caption/graph summaries.

### `server/app/static/js/dashboard/features/memory/*.js`

Memory panel API/actions/parsing/rendering.

### `server/app/static/js/dashboard/features/conversation/index.js`

Dashboard text/vision chat panel.

### `server/app/static/js/dashboard/features/scene_graph/index.js`

Client-side scene graph visualization helpers.

### `server/app/static/js/dashboard/features/qa/index.js`

QA pool dashboard editor.

### `server/app/static/js/dashboard/features/ui_shell/*.js`

Sidebar, tabs, theme, and navigation.

## Other Support Files

### `server/app/static/css/style.css`

Dashboard styling.

### `server/app/static/pepper_icon.png`

Dashboard icon asset.

### `server/app/providers/translation/lexicons/*.json`

Static Czech vocabulary defaults.

### `server/app/providers/translation/lexicons_user/*.json`

User-editable Czech vocabulary maps written by dashboard/config translation patching.
