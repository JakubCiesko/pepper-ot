# Assets, Tests, and Support Files

## Files Covered

- `server/prompts/*`
- `server/ontology/*`
- `server/state/*`
- `server/tests/*`
- `server/mock/*`
- `server/start_server.sh`
- `server/download_models.sh`
- `server/requirements.txt`
- `server/setup.py`
- `server/detection_models/*`
- `server/rf-detr-*.pth`
- `server/plans/*`

## Prompt Assets

### Chat prompts

- `prompts/chat_system.txt`
- `prompts/chat_object_user.txt`
- `prompts/chat_context.txt`

Used by:
- `ChatService`
- dashboard config editing
- object-focused prompt construction

### VLM prompts

- `prompts/vlm_system.txt`
- `prompts/vlm_system_complex.txt`
- `prompts/vlm_user.txt`

Used by:
- scene graph VLM backend
- direct VLM prompting paths

## Ontology Files

### `ontology/object_detection.yaml`

Purpose:
- label vocabulary for detection backends that support ontology/open-vocabulary behavior

### `ontology/scene_generation_ontology.yaml`

Purpose:
- predicate and object vocabulary for scene graph generation prompts

## Persisted State Files

### `state/last_state.json`
- latest persisted dashboard/live payload when storage is enabled

### `state/last_state.jpg`
- latest persisted rendered image when `storage.store_image = true`

These files are runtime artifacts, not source of truth.

## Tests

### Contract/config tests

- `tests/test_config_validation.py`
- `tests/test_worker_config_validation.py`
- `tests/test_llm_contracts.py`
- `tests/test_worker_contracts.py`
- `tests/test_model_io_common.py`

### Runtime/integration tests

- `tests/test_pipeline_controls.py`
- `tests/test_chat_language_flow.py`
- `tests/test_clients_integration.py`

### Utilities

- `tests/send_data.py`

The test suite is useful documentation in its own right. If you change contracts or config behavior, update these tests immediately.

## Mock and helper scripts

### `mock/detect.py`
- lightweight fake detection support / debugging helper

### `start_server.sh`
- convenience launcher for local server startup

### `download_models.sh`
- helper script for fetching model weights

## Packaging and dependencies

### `requirements.txt`
- runtime Python dependencies

### `setup.py`
- packaging metadata / install helper

## Model Weight Files

Examples currently present:
- `detection_models/reltr.pth`
- `detection_models/rtdetr-x.pt`
- `detection_models/yolo11x.pt`
- `rf-detr-large-2026.pth`
- `rf-detr-medium.pth`

These are runtime assets, not code. Their presence affects available backends and startup assumptions.

## Planning Material

`server/plans/` contains historical planning notes, refactor plans, and cached external docs.

Useful for context:
- feature plans
- worker/process refactor notes
- GPU improvement notes
- cached Aldebaran docs under `plans/docs`

Not part of live runtime, but often useful when understanding why code ended up in its current shape.
