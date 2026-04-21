# Configuration and Reload

Configuration is controlled by `server/config.yaml`, represented by Pydantic models in `server/app/schemas/config.py`, exposed through `/api/v1/config`, and hot/hard reloaded through `server/app/core/config`.

## Config Source of Truth

Primary files:

- `server/config.yaml`
- `server/app/schemas/config.py`
- `server/app/core/config/config_manager.py`
- `server/app/core/config/mutations/*`
- `server/app/api/v1/config.py`
- `server/app/static/js/dashboard/features/config/index.js`

`AppConfig.load(path)` reads YAML, validates it, stores `_config_path`, and returns an `AppConfig` instance.

## Main Config Sections

### `system`

A plain dict used for system-wide simple values. The important current key is:

- `output_language`: dashboard/robot configured output language. Chat and caption code use this as fallback when request language fields are absent.

### `detection`

Model and post-processing config for object detection.

Fields:

- `backend`: one of `yolo`, `rt_detr`, `rf_detr`, `owl_v2`.
- `weights_path`: optional backend-specific model path.
- `confidence_threshold`: detection confidence cutoff.
- `run_nms_post_filter`: whether to apply post-detection NMS.
- `nms_iou_threshold`: IoU threshold for NMS, `0.0` to `1.0`.
- `nms_type`: `per_class` or `general`.
- `device`: backend device string such as `cuda`, `cuda:0`, or `cpu`.
- `ontology`: inline object labels for open-vocabulary backends.
- `ontology_path`: path to YAML ontology file under allowed ontology roots.

Hot fields include threshold, NMS settings, ontology, and ontology path. Backend, weights, and device are hard reload fields.

### `tracking`

Scene memory and ReID tracking config.

Fields:

- `max_dormant_frames`: frames a track may be unmatched before being dropped.
- `memory_max_age_seconds`: object/relation TTL.
- `memory_max_objects`: object and track cap.
- `memory_max_relations`: relation cap.
- `memory_max_captions`: caption cap.
- `caption_max_age_seconds`: caption TTL.
- `association.visual_weight`, `geometry_weight`, `match_threshold`: association scoring weights.
- `feature_extraction.reid_model`, `device`, `target_size`, `resampling_method`: embedding crop model settings.

Feature extractor model/device are hard reload fields. Target size, resampling, association, and memory limits are hot fields.

### `fusion`

Pepper robot person metadata fusion config.

Fields:

- `person_bbox_match_threshold_px`: pixel margin for projected Pepper person to detection bbox matching.
- `estimated_person_bbox_base_px`: base synthetic person bbox size before distance scaling.
- `estimated_person_bbox_min_px`: lower synthetic bbox clamp.
- `estimated_person_bbox_max_px`: upper synthetic bbox clamp.
- `angular_yaw_threshold_rad`: yaw candidate threshold.
- `angular_pitch_threshold_rad`: pitch candidate threshold.
- `matched_person_min_confidence`: confidence floor for matched detections.
- `synthetic_person_confidence`: confidence for Pepper-induced synthetic detections.
- `pepper_binding_max_misses`: allowed misses before Pepper ID binding ages out.

### `scene_graph`

Scene graph config is backend-compositional. There is no current `scene_graph.mode` string.

Subsections:

- `scene_graph.vlm`: VLM backend config, prompts, ontology, structured output, schema style, local VLM hints, and `enabled` flag.
- `scene_graph.rules`: deterministic rule config and `enabled` flag.
- `scene_graph.reltr`: RelTR checkpoint/device/threshold/topk/IoU matching and `enabled` flag.

When `pipeline_controls.scene_graph=true`, at least one backend must be enabled. If rules or RelTR are enabled, detection must also be enabled because they need tracked object IDs and bboxes.

### `qa_generation`

Config for the automatic scene-graph QA stage and pool.

Fields:

- `pairs_per_update`: number of pairs requested per frame/update.
- `pool_max_entries`: max bilingual entries retained in `QAPoolService`.

Changing this is hot. It updates the QA stage runtime and the process-level QA pool max entries.

### `chat`

Text LLM config and prompt templates.

Fields inherited from `LLMConfig`:

- `provider`
- `model_id`
- `device`
- `base_url`
- `api_key_env`
- `timeout_seconds`
- `client_init_kwargs`
- `call_kwargs`
- `structured_output.mode`
- `structured_output.strict`

Prompt fields:

- `system_prompt`
- `user_prompt`
- `object_system_prompt`
- `object_user_prompt`

Provider/model/base URL/API key/init kwargs are hard reload fields. Device/call kwargs/structured output/prompts are hot fields for `ChatService`.

### `caption`

Caption provider config.

Extra fields:

- `mode`: `prompted` or `unconditional`.
- `max_words`: optional prompt suffix limit.
- `system_prompt`
- `user_prompt`

Provider/model/base URL/API key/device/client init kwargs are hard reload fields. Call kwargs, structured output, mode, max words, and prompts are hot fields.

### `visualization`

SoM overlay and mask backend config.

Fields:

- `show_bbox`
- `show_mask`
- `show_polygon`
- `show_labels`
- `line_thickness`
- `mask_opacity`
- `color_lookup`: `index`, `class`, or `track`.
- `mask_backend`: `grabcut` or `sam`.
- `device`: SAM backend device.

Changing mask backend/device is hard reload. Most overlay settings are hot.

### `storage`

Last-state persistence config.

Fields:

- `persist_last_state`
- `last_state_path`
- `store_image`

These are hot config fields.

### `worker`

Worker process lifecycle config.

Important fields:

- `enabled`
- `host`
- `port`
- `idle_timeout_seconds`
- `idle_check_interval_seconds`
- `startup_timeout_seconds`
- `request_timeout_seconds`
- `shutdown_grace_seconds`
- `max_startup_queue`
- `healthcheck_interval_seconds`
- `restart_max_attempts`
- `restart_window_seconds`
- `restart_backoff_seconds`
- `circuit_breaker_cooldown_seconds`
- `auto_warmup_on_startup`

Enabling/disabling worker or changing host/port/startup/restart settings is hard. Idle and request timeouts are hot.

### `pipeline_controls`

Controls frame-stage execution.

Fields:

- `preset`: `full`, `detect_only`, `caption_only`, `vlm_only`, `rules_only`, `minimal`, or `custom`.
- `caption`
- `detect`
- `track_memory`
- `paint_som`
- `scene_graph`
- `qa_generation`
- `update_scene_memory`

Validation rules:

- `track_memory` requires `detect`.
- `paint_som` requires `detect`.
- `update_scene_memory` requires `scene_graph`.
- `qa_generation` requires `scene_graph`.
- `update_scene_memory` requires `track_memory`.
- `scene_graph=true` requires at least one enabled backend.
- rules/RelTR scene graph backends require `detect=true` when scene graph stage runs.

## Prompt Sources

`PromptSource` allows either inline `text` or a relative `path`. Exactly one must be set. The config manager validates uploaded prompt paths to keep them under safe roots.

At runtime, prompt sources are resolved by:

- `config_manager.resolve_config()` for dashboard display.
- `AppState._initialize_chat_components()`.
- `AppState._initialize_caption_component()`.
- `pipeline_factory.build_perception_pipeline()`.
- hot reload helpers in `core/config/mutations/runtime.py`.

## Ontology Sources

`DetectionConfig.resolve_ontology(base_dir)` resolves detection object labels from inline `ontology` or `ontology_path`.

`OntologySource.resolve(base_dir)` resolves VLM predicates and optional object vocabulary from inline config and optional file path.

## Config API

File: `server/app/api/v1/config.py`

Endpoints:

- `GET /api/v1/config`: returns active config, saved config, resolved prompt/ontology config, translations, and behavior contracts.
- `PATCH /api/v1/config`: deep-merges a patch into active config, validates it, applies hot/hard semantics, and returns diff information.
- `POST /api/v1/config/save`: writes active config to YAML.
- `POST /api/v1/config/reload`: reloads saved YAML from disk and applies it.
- `POST /api/v1/config/upload`: validates uploaded YAML and applies it.
- `GET /api/v1/config/download`: downloads active or saved YAML.

The PATCH route also handles translation dictionary updates through the vocabulary translator.

## Reload Rule System

Files:

- `server/app/core/config/mutations/components.py`
- `server/app/core/config/mutations/reload_rule.py`
- `server/app/core/config/mutations/runtime.py`
- `server/app/core/config/mutations/rule_definitions/*.py`

Every reloadable field is represented by a `ReloadRule`:

- `path`: dotted config field name.
- `getter`: function used to compare old/new values.
- `mode`: `hot` or `hard`.
- `apply_hot`: optional handler to mutate runtime objects.

`components.diff_config(old, new)` returns a `ConfigDiff(hot=[...], hard=[...])`.

`components.apply_hot_changes(app_state, old, new)`:

1. Computes diff.
2. Stores new config into `AppState`.
3. Increments config version.
4. Calls each unique hot apply handler once.
5. Pushes hot config to worker if worker manager exists.

## Hot Runtime Application

`core/config/mutations/runtime.py` mutates live pipeline components:

- detector threshold, NMS settings, device, ontology
- pipeline controls
- fusion config
- visualization config
- QA service LLM config and pair count
- memory limits, max dormant frames, association, feature extraction settings
- scene graph VLM runtime, rules config, RelTR runtime

Hot updates are best-effort runtime mutations. Hard fields require worker restart or pipeline rebuild because existing model/provider objects cannot be safely mutated.

## Dashboard Config UI

Files:

- `server/app/static/templates/dashboard/pages/*.html`
- `server/app/static/js/dashboard/features/config/index.js`

The dashboard reads config from `GET /api/v1/config`, populates controls, builds PATCH payloads, and calls config endpoints. It also displays structured-output capability hints from `behavior_contracts()`.

Current dashboard config pages cover:

- detection and robot fusion
- SoM visualization
- scene graph backends/rules/RelTR/VLM
- runtime pipeline and worker controls
- memory/tracking controls
- chat and object-chat prompts/provider/structured output
- caption provider/prompts
- QA generation pair/pool settings
- vocabulary translations
- storage
