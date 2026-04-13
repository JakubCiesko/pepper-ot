# File Inventory

This is a grouped inventory of the current `server/` tree with practical descriptions.

## Source Code: Backend Python

### `app/main.py`
- FastAPI bootstrap, lifespan, logging, ngrok, static mount, router registration.

### `app/dashboard.py`
- Dashboard routes and websocket endpoint.

## `app/api/v1`

- `__init__.py`: package marker.
- `router.py`: combines route modules into API router.
- `detect.py`: detect and panorama-detect public endpoints.
- `caption.py`: direct caption endpoint.
- `chat.py`: grounded chat endpoints and conversation endpoints.
- `vision_chat.py`: direct image+question VLM chat endpoint.
- `config.py`: config/state read-write endpoints.
- `memory.py`: memory CRUD/read endpoints.
- `memory_route_utils.py`: shared memory route execution helper.
- `worker.py`: worker status/warmup/stop endpoints.
- `image_utils.py`: image resize/debug/panorama helpers.

## `app/core`

- `__init__.py`: package marker.
- `pipeline_factory.py`: build perception pipeline and dependencies.

### `app/core/config`

- `__init__.py`: package marker.
- `config_manager.py`: load/dump/patch/resolve config and apply hot changes.

### `app/core/config/mutations`

- `__init__.py`: exports grouped rule definitions.
- `components.py`: collects all reload rules and computes diffs.
- `reload_rule.py`: reload rule dataclasses and apply-hot groups.
- `runtime.py`: in-place pipeline/chat/caption/scene-graph runtime update helpers.

### `app/core/config/mutations/rule_definitions`

- `__init__.py`: grouped exports.
- `detection.py`: detection config reload rules.
- `caption.py`: caption config reload rules.
- `chat.py`: chat config reload rules.
- `pipeline.py`: pipeline control reload rules.
- `scene_graph.py`: scene-graph reload rules.
- `storage.py`: storage reload rules.
- `tracking.py`: tracking/memory reload rules.
- `visualization.py`: visualization reload rules.
- `worker.py`: worker reload rules.

### `app/core/infra`

- `__init__.py`: package marker.
- `storage.py`: latest-state JSON/JPG persistence helpers.
- `ws_manager.py`: websocket connection manager and broadcaster.

### `app/core/prompting`

- `__init__.py`: package marker.
- `renderer.py`: prompt template rendering using context/caption fields.

### `app/core/runtime`

- `__init__.py`: package marker.
- `state.py`: global application runtime container.

### `app/core/runtime/worker_client`

- `__init__.py`: package marker.
- `errors.py`: worker-specific error types.
- `manager.py`: top-level worker manager.
- `manager_monitor.py`: monitoring/idle/restart logic mixin.
- `manager_process.py`: subprocess lifecycle mixin.
- `manager_rpc.py`: HTTP/RPC transport mixin.
- `rpc.py`: worker request/response schemas.
- `types.py`: worker state enums and snapshots.

## `app/orchestration`

- `__init__.py`: package marker.

### `app/orchestration/adapters`

- `__init__.py`: package marker.
- `runtime.py`: runtime adapter abstraction for in-process vs worker execution.

### `app/orchestration/services`

- `__init__.py`: package marker.
- `detection.py`: detection request orchestration and persistence.
- `chat.py`: grounded dialogue composition and object chat.
- `caption.py`: caption orchestration wrapper.
- `conversation.py`: conversation state/history service.
- `memory.py`: memory service and request models for CRUD operations.

## `app/inference`

- `__init__.py`: package marker.
- `pipeline.py`: central multi-stage perception pipeline.
- `types.py`: internal inference data types.

### `app/inference/caption`

- `__init__.py`: package marker.
- `service.py`: caption inference service wrapper and result object.

### `app/inference/detection`

- `__init__.py`: package marker.
- `detectors.py`: detector backends.
- `model_registry.py`: backend/model registry.
- `service.py`: detection service wrapper.

### `app/inference/tracking`

- `__init__.py`: package marker.
- `associator.py`: track-detection matching logic.
- `embeddings.py`: ReID feature extraction and crop extraction.

### `app/inference/memory`

- `__init__.py`: package marker.
- `scene_memory.py`: high-level memory facade.
- `chat_memory_proxy.py`: worker-safe / empty chat-memory helpers.

#### `app/inference/memory/state_store`

- `__init__.py`: package marker and exports.
- `store.py`: main memory state container and Pepper binding logic.
- `objects.py`: object-state mutation helpers.
- `relations.py`: relation-state mutation helpers.
- `tracks.py`: track lifecycle helpers.
- `social.py`: Pepper social/person enrichment helpers.
- `geometry.py`: geometry and robot-relative helper logic.

### `app/inference/scene_graph`

- `__init__.py`: package marker.
- `service.py`: scene graph backend dispatch and merge.
- `rules_backend.py`: rule and color-based scene graph generation.
- `vlm_backend.py`: VLM-based scene graph generation.
- `reltr_backend.py`: RelTR adapter layer.
- `reltr_predictor.py`: RelTR model load/inference utilities.
- `som.py`: Set-of-Mark painter.

## `app/providers`

- `__init__.py`: package marker.

### `app/providers/common`

- `__init__.py`: package marker.
- `io.py`: structured output parsing and extraction helpers.
- `runtime_setup.py`: API-key/client-kwargs helpers.
- `utils.py`: capability matrix and kwargs validation.

### `app/providers/llm`

- `__init__.py`: package marker.
- `base.py`: base text provider contracts.
- `client.py`: unified LLM client facade.
- `openai_llm.py`: OpenAI-compatible text provider.
- `gemini_llm.py`: Gemini text provider.
- `hf_llm.py`: local HF text provider.

### `app/providers/vlm`

- `__init__.py`: package marker.
- `base.py`: VLM base interface.
- `factory.py`: VLM provider factory.
- `openai_vlm.py`: OpenAI VLM provider.
- `gemini_vlm.py`: Gemini VLM provider.
- `local_hf_vlm.py`: local HF VLM and 4-bit variants.

### `app/providers/caption`

- `__init__.py`: package marker.
- `client.py`: caption client facade and BLIP implementation.

### `app/providers/translation`

- `__init__.py`: package marker.
- `google_trans.py`: translation and output-language enforcement.

## `app/schemas`

- `__init__.py`: package marker.
- `config.py`: main config schema tree.
- `detect.py`: detection response/form models.
- `caption.py`: caption response/form models.
- `chat.py`: chat request/response and chat mode enum.
- `vision_chat.py`: vision chat request/response models.
- `robot.py`: Pepper metadata models.
- `scene.py`: object/relation/caption/scene state models.

## `app/worker`

- `__init__.py`: package marker.
- `main.py`: worker FastAPI app bootstrap.
- `routes.py`: internal worker router builder.
- `runtime.py`: worker-local runtime service container.

## Source Code: Frontend

### `app/static/templates`

- `dashboard.html`: dashboard shell.
- `dashboard/pages/live.html`: live processing view.
- `dashboard/pages/chat.html`: chat panel.
- `dashboard/pages/detection.html`: detection config/view page.
- `dashboard/pages/runtime.html`: runtime/worker page.
- `dashboard/pages/scene.html`: scene graph page.
- `dashboard/pages/som.html`: SoM-related page.
- `dashboard/pages/storage.html`: storage page.
- `dashboard/pages/memory-settings.html`: memory settings/editing page.
- `dashboard/pages/caption.html`: caption config/page.

### `app/static/js/dashboard/core`

- `http.js`: JSON request helpers.
- `notifications.js`: small status/toast messages.
- `ws.js`: websocket setup helper.

### `app/static/js/dashboard/features/config`

- `index.js`: giant config editor module mirroring `AppConfig`.

### `app/static/js/dashboard/features/live`

- `index.js`: live frame carousel, metrics, and scene summary UI.

### `app/static/js/dashboard/features/conversation`

- `index.js`: text chat + vision chat UI logic.

### `app/static/js/dashboard/features/memory`

- `index.js`: memory feature bootstrap.
- `actions.js`: refresh and CRUD binding.
- `api.js`: memory API calls.
- `dom_refs.js`: DOM lookup helpers.
- `parsers.js`: small parse helpers for editor inputs.
- `render.js`: memory UI rendering functions.

### `app/static/js/dashboard/features/scene_graph`

- `index.js`: graph visualization and panel init.

### `app/static/js/dashboard/features/ui_shell`

- `index.js`: UI shell bootstrap.
- `navigation.js`: page navigation.
- `tabs.js`: tab group setup.
- `sidebar.js`: sidebar open/close behavior.
- `theme.js`: theme toggle behavior.

### `app/static/js/dashboard/app.js`

- top-level dashboard JS entrypoint and websocket dispatcher.

## Config, Assets, and Data

- `config.yaml`: main runtime configuration.
- `ontology/object_detection.yaml`: detection ontology.
- `ontology/scene_generation_ontology.yaml`: scene-graph ontology.
- `prompts/chat_context.txt`: chat context prompt asset.
- `prompts/chat_object_user.txt`: object chat prompt asset.
- `prompts/chat_system.txt`: chat system prompt.
- `prompts/vlm_system.txt`: simple VLM system prompt.
- `prompts/vlm_system_complex.txt`: richer VLM system prompt.
- `prompts/vlm_user.txt`: VLM user prompt.
- `state/last_state.json`: optional persisted latest state.
- `state/last_state.jpg`: optional persisted latest rendered image.
- `app/static/css/style.css`: dashboard stylesheet.
- `app/static/pepper_icon.png`: dashboard/static image asset.

## Tests and Dev Helpers

- `tests/test_chat_language_flow.py`: translation/chat flow behavior.
- `tests/test_clients_integration.py`: provider/runtime integration checks.
- `tests/test_config_validation.py`: config schema validation.
- `tests/test_llm_contracts.py`: text provider contract checks.
- `tests/test_model_io_common.py`: structured IO helper checks.
- `tests/test_pipeline_controls.py`: pipeline control semantics.
- `tests/test_worker_config_validation.py`: worker config validation.
- `tests/test_worker_contracts.py`: worker interface contract checks.
- `tests/send_data.py`: manual request helper.
- `mock/detect.py`: mock detection helper.
- `start_server.sh`: local startup script.
- `download_models.sh`: model download helper.
- `requirements.txt`: Python dependency list.
- `setup.py`: packaging/install helper.

## Runtime Assets and Non-Code Files

- `detection_models/*.pt` and `*.pth`: detector and scene graph model weights.
- `rf-detr-*.pth`: RF-DETR checkpoints.
- `plans/*.md`: historical implementation notes.
- `plans/docs/*.html`: cached external documentation.

## Noise / Non-runtime Files

These exist in the tree but are not core runtime code:
- `.idea/*`
- `.claude/*`
- `__pycache__/*`
- compiled `.pyc` files

They do not need behavior changes unless your tooling/editor workflow depends on them.
