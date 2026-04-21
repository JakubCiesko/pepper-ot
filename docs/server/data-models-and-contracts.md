# Data Models and Contracts

This document maps the main Pydantic schemas, dataclasses, and payload contracts used across API, pipeline, worker, dashboard, and memory.

## Config Models

File: `server/app/schemas/config.py`

### `AppConfig`

Top-level config model. Sections:

- `system`
- `detection`
- `tracking`
- `scene_graph`
- `qa_generation`
- `chat`
- `caption`
- `visualization`
- `storage`
- `fusion`
- `worker`
- `pipeline_controls`

It also stores private `_config_path` for resolving relative prompts/ontologies.

### `DetectionConfig`

Controls detector backend and post-filtering:

- `backend`
- `weights_path`
- `confidence_threshold`
- `run_nms_post_filter`
- `nms_iou_threshold`
- `nms_type`
- `device`
- `ontology`
- `ontology_path`

### `LLMConfig`

Base provider config used by chat, caption, and VLM scene graph configs:

- `provider`
- `model_id`
- `device`
- `base_url`
- `api_key_env`
- `timeout_seconds`
- `client_init_kwargs`
- `call_kwargs`
- `structured_output`

### `StructuredOutputConfig`

- `mode`: `provider_native`, `parse_output`, or `instructor`.
- `strict`: bool.

### `ChatConfig`

Extends `LLMConfig` with:

- `system_prompt`
- `user_prompt`
- `object_system_prompt`
- `object_user_prompt`

### `CaptionConfig`

Extends `LLMConfig` with:

- `mode`: `prompted` or `unconditional`.
- `max_words`
- `system_prompt`
- `user_prompt`

Defaults to local BLIP captioning.

### `SceneGraphVLMConfig`

Extends `LLMConfig` with:

- `enabled`
- `system_prompt`
- `user_prompt`
- `ontology`
- `structured_schema`: `scene_graph` or `relationship_list`
- `local_vlm_hints`

### `SGGRulesConfig` and `SGGRule`

Rules config:

- `enabled`
- `rule_list`

Rule fields:

- `predicate`
- `type`
- `thresholds`
- `constraints`

### `SGGRelTRConfig`

RelTR config:

- `enabled`
- `checkpoint_path`
- `device`
- `threshold`
- `topk`
- `iou_match_threshold`

### `QAGenerationConfig`

- `pairs_per_update`
- `pool_max_entries`

### `PipelineControls`

Stage toggles and presets:

- `preset`
- `caption`
- `detect`
- `track_memory`
- `paint_som`
- `scene_graph`
- `qa_generation`
- `update_scene_memory`

Validation is in `AppConfig.validate_pipeline_controls`.

## Detection API Models

File: `server/app/schemas/detect.py`

### `DetectionObject`

Public object response:

- `label`
- `confidence`
- `bbox`
- `object_id`

### `DetectionResponse`

Public detection response:

- `id`
- `objects`
- `timestamp`
- `image_width`
- `image_height`
- `caption`
- `caption_provider`
- `caption_model_id`

### `DetectFormRequest`

Describes form fields:

- `metadata`
- `publish`
- `resize_image`

The current endpoint uses explicit form parameters rather than relying on this model for multipart parsing.

## Robot Metadata Models

File: `server/app/schemas/robot.py`

### `PersonMetadata`

Geometry from Pepper people perception:

- `id`
- `yaw`
- `pitch`
- `distance`

### `SocialPersonMetadata`

Social/semantic people perception:

- waving flags
- sitting flag
- looking-at-robot flag and score
- head angles
- gaze direction
- gender/code/confidence
- age/bucket/confidence
- expression scores/name/confidence
- smile score/confidence
- eyes opened
- engagement zone
- timestamp

### `RobotMetadata`

Full per-frame robot context:

- head/body pose
- camera FOV
- image dimensions
- timestamp/frame/scan/capture mode
- people list
- social people list
- battery

`merge_robot_metadata_for_panorama` creates a merged panorama metadata object for horizontally stitched scans.

## Scene and Memory Models

File: `server/app/schemas/scene.py`

### `TrackedObjectState`

Persistent object memory record. Includes detection identity, memory status, attributes, robot/person metadata fields, bearing, frame/scan ids, timings, hits, and bbox.

### `Relationship`

Persistent memory relation:

- `subject_id`
- `predicate`
- `object_id`
- `first_seen`
- `last_seen`
- `count`

### `SceneCaptionState`

Persistent caption memory record.

### `SceneState`

Full memory snapshot:

- `objects`
- `relationships`
- `captions`
- `timestamp`

### `SceneGraphRelation`

Display/API graph triple:

- `sub`
- `rel`
- `obj`

### `SceneGraphStructuredResponse`

Structured VLM response wrapper:

- `relationships: list[SceneGraphRelation]`

### `MemorySummary`

Robot/dashboard memory summary:

- `timestamp`
- `labels`
- `label_counts`
- `scene_graph`
- `graph_svg`
- `pregenerated_qa`

## Internal Inference Types

File: `server/app/inference/types.py`

### `InferenceDetectionObject`

Internal detection object used by detector, tracker, pipeline, scene graph:

- `class_id`
- `label`
- `confidence`
- `bbox`
- `object_id`

### `TrackedObject`

Internal track record with embedding and last crop:

- `id`
- `label`
- `embedding`
- `bbox`
- `confidence`
- `last_seen`
- `first_seen`
- `hits`
- `frames_since_seen`
- `last_crop`

### `SceneGraphEdge`

Dataclass triple:

- `sub`
- `rel`
- `obj`

Hashable and equality-comparable for deduplication.

### `SceneGraph`

Internal graph object:

- `edges`
- `no_label_edges`
- `raw`

Methods:

- `from_list`
- `_normalize_id`
- `deduplicate`
- `as_dict`
- `__add__`

### `PipelineResult`

Full output of one pipeline frame:

- raw image
- SoM image
- detections
- scene graph
- metrics
- executed stages
- caption metadata
- QA pairs

## Chat Models

File: `server/app/schemas/chat.py`

### `ChatMode`

Values:

- `general`
- `object`
- `relation` placeholder
- `attribute` placeholder

### `ChatRequest`

Fields:

- `query`
- `chat_id`
- `conversation_id`
- language fields
- `model_facing_language`
- `mode`
- `object_label`
- `max_instances`
- `max_crop_fallbacks`

### `ChatResponse`

Fields:

- `response`
- `chat_id`
- `conversation_id`
- `metadata`
- `timestamp`

### QA Schemas

- `PregeneratedQARequest`
- `PregeneratedQAPair`
- `PregeneratedQAPairs`
- `PregeneratedQAResponse`
- `PregeneratedQABilingualItem`
- `PregeneratedQAPoolResponse`
- `PregeneratedQAPoolUpdateRequest`

These are used by QA routes, dashboard QA tab, and robot cached-question workflows.

## Worker RPC Models

File: `server/app/core/runtime/worker_client/rpc.py`

### `WorkerRPCRequest`

- `request_id`
- `config_version`

### `WorkerRPCResponse`

- `ok`
- `error_message`
- `worker_state`
- `config_version`

### `DetectRPCRequest`

- `image_b64`
- `robot_metadata`

### `DetectRPCResponse`

- image output
- object list
- scene graph
- QA pairs
- caption metadata
- memory
- metrics
- executed stages
- dimensions

### `WorkerStatusResponse`

Worker lifecycle status exposed publicly and internally.

## Conversation Dataclasses

File: `server/app/orchestration/services/conversation.py`

### `ConversationMessage`

Stores both UI/original and model-facing text:

- role
- text_original
- text_model
- language_original
- language_model
- translation_applied
- timestamp
- metadata

### `ConversationRecord`

Contains a chat id and bounded message list.

## Provider Response Model

File: `server/app/providers/llm/base.py`

`LLMResponse` stores:

- `text`
- `parsed`
- `raw`

Used across text LLM providers.

## Contract Boundaries

- Public APIs should use schemas under `server/app/schemas` or service payload dicts documented in `api-reference.md`.
- Worker process boundary uses `worker_client/rpc.py` models.
- Dashboard websocket payloads are mostly dict contracts assembled in orchestration services and consumed by dashboard JS.
- Internal pipeline types are not necessarily stable public API contracts.

## Where To Add Fields

- Add config field: schema, YAML, reload rule, runtime update, dashboard load/save.
- Add detect response field: `DetectionResponse`, runtime adapter/worker response, detection orchestration payload, dashboard JS.
- Add memory object field: `TrackedObjectState`, store create/update/patch, memory renderer, dashboard memory UI.
- Add worker crossing field: `PipelineResult`, `DetectRPCResponse`, worker runtime, worker manager, runtime adapter, public response/publish payload if needed.
