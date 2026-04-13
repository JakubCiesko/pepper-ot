# Configuration and Reload Semantics

## Files Covered

- `server/config.yaml`
- `app/schemas/config.py`
- `app/core/config/config_manager.py`
- `app/core/config/mutations/components.py`
- `app/core/config/mutations/runtime.py`
- `app/core/config/mutations/reload_rule.py`
- `app/core/config/mutations/rule_definitions/*.py`

## Source of Truth

Runtime configuration starts from `server/config.yaml` and is validated by `AppConfig` in `app/schemas/config.py`.

The code supports two update styles:
- hot updates: mutate running components in place
- hard reloads: require rebuild of pipeline and/or worker process state

## Top-Level Config Sections

### `system`

Current use:
- language/output language related behavior

### `detection`

Fields:
- `backend`: `yolo | rt_detr | rf_detr | owl_v2`
- `weights_path`
- `confidence_threshold`
- `device`
- `ontology`
- `ontology_path`

Current `config.yaml` defaults:
- backend: `rf_detr`
- threshold: `0.60`
- device: `cuda`
- ontology loaded from `ontology/object_detection.yaml`

### `tracking`

Fields:
- `max_dormant_frames`
- `memory_max_age_seconds`
- `memory_max_objects`
- `memory_max_relations`
- `memory_max_captions`
- `caption_max_age_seconds`
- `association.visual_weight`
- `association.geometry_weight`
- `association.match_threshold`
- `feature_extraction.reid_model`
- `feature_extraction.device`
- `feature_extraction.target_size`
- `feature_extraction.resampling_method`

### `scene_graph`

Fields:
- `mode`: `vlm | rules | hybrid | reltr`
- `vlm.*`
- `rules.*`
- `reltr.*`

Current `config.yaml` highlights:
- mode: `hybrid`
- VLM provider: `gemini`
- VLM model: `gemini-2.5-flash-lite`
- rules enabled: true
- reltr enabled: true

### `chat`

Fields:
- provider/model/runtime kwargs
- structured output strategy
- `system_prompt`
- `object_user_prompt`

Current config uses Gemini by default.

### `caption`

Fields:
- provider/model/runtime kwargs
- `mode`: `unconditional | prompted`
- `max_words`
- `system_prompt`
- `user_prompt`

Current config uses local BLIP captioning.

### `visualization`

Fields:
- bbox/mask/polygon/labels toggles
- line thickness
- mask opacity
- color lookup mode
- mask backend
- visualization device

### `storage`

Fields:
- `persist_last_state`
- `last_state_path`
- `store_image`

### `fusion`

Pepper-specific human fusion behavior:
- bbox match threshold
- estimated bbox sizing
- angular thresholds
- synthetic confidence values
- Pepper binding miss threshold

### `worker`

Fields:
- host/port
- startup/request/shutdown timing
- idle kill timing
- healthcheck timing
- restart policy
- circuit breaker cooldown
- auto warmup toggle

### `pipeline_controls`

Fields:
- `preset`
- `caption`
- `detect`
- `track_memory`
- `paint_som`
- `scene_graph`
- `update_scene_memory`

Preset map is implemented in `PipelineControls.preset_map()`.

## Validation Rules Worth Knowing

From `AppConfig.validate_pipeline_controls()`:
- `track_memory` requires `detect=true`
- `paint_som` requires `detect=true`
- `update_scene_memory` requires `scene_graph=true`
- `update_scene_memory` requires `track_memory=true`
- `scene_graph.mode=rules` requires detection if scene graph stage is enabled
- `scene_graph.mode=reltr` requires detection if scene graph stage is enabled

From `PromptSource`:
- exactly one of `text` or `path` must be set

From `WorkerRuntimeConfig`:
- restart backoff values must all be `> 0`

## Config Manager

`app/core/config/config_manager.py` handles:
- locating config file
- loading config
- serializing config to dict/YAML
- resolving prompt text and ontology content for display
- behavior contract reporting
- deep merge patching
- uploaded YAML parsing
- config diffing
- path safety validation
- hot config application wrapper

## Hot vs Hard Reload

The mutation system is declarative.

### Rule registry

`components.py` combines rule lists from:
- detection
- caption
- scene graph
- visualization
- storage
- worker
- tracking
- chat
- pipeline controls

### Hot update application groups

In `reload_rule.py`:
- `_apply_pipeline_group()` updates pipeline detector threshold/device/ontology, memory limits, association settings, pipeline controls, fusion config, visualization config, and scene graph runtime settings.
- `_apply_chat_group()` updates chat system prompt and text provider runtime.
- `_apply_caption_group()` updates caption prompts and caption provider runtime.

## Important Rule Outcomes

### Detection

Hard:
- backend
- weights path
- device

Hot:
- threshold
- ontology
- ontology path

### Chat

Hard:
- provider
- model id
- base url
- timeout
- api key env
- client init kwargs

Hot:
- device
- call kwargs
- structured output
- system prompt

### Caption

Hard:
- provider
- model id
- base url
- timeout
- api key env
- client init kwargs
- device

Hot:
- call kwargs
- structured output
- mode
- max words
- prompts

### Scene graph

Hard:
- VLM provider/model/base_url/api key/device/client init kwargs
- `scene_graph.mode`
- RelTR checkpoint path
- RelTR device

Hot:
- VLM call kwargs
- structured output
- structured schema
- local VLM hints
- prompts
- ontology
- rules config
- RelTR enabled/threshold/topk/iou threshold

### Tracking

Hard:
- feature extraction model
- feature extraction device

Hot:
- feature extraction target size
- feature extraction resampling
- dormant frame count
- association config
- memory limits
- caption memory limits

### Visualization

Hard:
- mask backend
- visualization device

Hot:
- full visualization object otherwise

### Storage

Hot:
- persist toggle
- last state path
- store image toggle

### Worker

Hard:
- enable/host/port
- startup/shutdown queue/health/restart/circuit-breaker parameters

Hot:
- idle timeout
- idle check interval
- request timeout

## Practical Tweak Guidance

### Safe hot tweaks during live runs

- detection threshold
- ontology list/path
- scene graph rules
- pipeline preset or stage toggles
- memory limits
- prompt text
- model call kwargs

### Tweaks that will rebuild or restart major components

- switching provider family
- switching model id
- switching detector backend
- switching VLM mode
- changing worker host/port/startup policy
- changing ReID model

## Common Failure Modes

- prompt source sets both `text` and `path`
- pipeline controls violate dependencies
- dashboard sends JSON text fields that parse incorrectly
- hot change expected, but rule is actually marked hard
- relative paths break when config path base dir is not what you expect
