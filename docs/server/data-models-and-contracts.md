# Data Models and Contracts

## Files Covered

- `app/schemas/config.py`
- `app/schemas/detect.py`
- `app/schemas/caption.py`
- `app/schemas/chat.py`
- `app/schemas/vision_chat.py`
- `app/schemas/robot.py`
- `app/schemas/scene.py`
- `app/inference/types.py`
- `app/core/runtime/worker_client/rpc.py`
- `app/core/runtime/worker_client/types.py`

## Purpose

This document lists the most important typed contracts in the server.

These types are the boundary objects you must keep stable when modifying behavior across:
- API routes
- orchestration
- inference pipeline
- worker IPC-like HTTP calls
- dashboard rendering

## Public API Schemas

### `schemas/detect.py`

- `DetectionObject`: single API-facing detection entry.
- `DetectionResponse`: detect route response with object list, timestamp, image dimensions, and optional caption metadata.
- `DetectFormRequest`: detect route form binding model.

### `schemas/caption.py`

- `CaptionResponse`: caption endpoint response.
- `CaptionFormRequest`: caption form binding model.

### `schemas/chat.py`

- `ChatMode`: enum for chat behavior, including general and object-focused paths.
- `ChatRequest`: main chat input.
- `ChatResponse`: chat output with `chat_id`, sentence, source object IDs, confidence, and metadata.

### `schemas/vision_chat.py`

- `VisionChatFormRequest`: multipart/form-data model for image + question routes.
- `VisionChatResponse`: answer plus provider/model metadata.

## Robot Metadata Schemas

### `schemas/robot.py`

- `PersonMetadata`: Pepper-native person localization/perception item.
- `SocialPersonMetadata`: Pepper-native social cue item.
- `RobotMetadata`: full metadata payload attached to a frame.

This model is central to Pepper fusion. If the client changes metadata shape, update this model first.

## Scene and Memory Schemas

### `schemas/scene.py`

- `SceneGraphRelation`
- `SceneGraphStructuredResponse`
- `Relationship`
- `TrackedObjectState`
- `SceneCaptionState`
- `SceneState`

### `TrackedObjectState`

This is the persistent memory representation of an object.

It carries not only visual identity but also Pepper-enriched fields such as:
- Pepper person binding
- robot distance
- engagement zone
- frame/scan identifiers
- hit count and timestamps

### `Relationship`

Used both as scene graph memory relation and manual CRUD relation model.

Identity is tuple-based in practice:
- `subject_id`
- `predicate`
- `object_id`

### `SceneCaptionState`

Captions are stored as memory entries with provider/model/source/time metadata.

## Config Schema Tree

### `schemas/config.py`

Key config model classes:
- `DetectionConfig`
- `StructuredOutputConfig`
- `LLMConfig`
- `AssociationConfig`
- `FeatureExtractionConfig`
- `TrackingConfig`
- `PromptSource`
- `OntologySource`
- `SceneGraphVLMConfig`
- `ChatConfig`
- `CaptionConfig`
- `VisConfig`
- `StorageConfig`
- `SGGRuleConstraints`
- `SGGRule`
- `SGGRulesConfig`
- `SGGRelTRConfig`
- `SceneGraphConfig`
- `FusionConfig`
- `WorkerRuntimeConfig`
- `PipelineControls`
- `AppConfig`

These models are used by:
- config load/save/patch
- dashboard config editor
- pipeline assembly
- worker config propagation

## Internal Inference Types

### `inference/types.py`

Important internal classes:
- `InferenceDetectionObject`
- `TrackedObject`
- `BoundingBox`
- `InternalDetection`
- `CameraModel`
- `FrameContext`
- `SceneGraphEdge`
- `SceneGraph`
- `PipelineResult`

### Why these matter

These are not just implementation detail classes. They are the glue between detector output, tracker state, scene graph construction, and API serialization.

Examples:
- `InferenceDetectionObject` is the normalized result after backend detection.
- `TrackedObject` carries embedding-backed track lifecycle data.
- `SceneGraph` supports backend merging and API serialization.
- `PipelineResult` is the top-level per-frame result object.

## Worker Contracts

### `worker_client/rpc.py`

- `WorkerRPCRequest`
- `WorkerRPCResponse`
- `DetectRPCRequest`
- `DetectRPCResponse`
- `WorkerConfigRPCRequest`
- `WorkerStatusResponse`

### `worker_client/types.py`

- `WorkerState`
- `RestartReason`
- `StopReason`
- `WorkerStatusSnapshot`

These types must stay aligned between:
- main process manager
- worker process routes/runtime
- dashboard worker status views

## Contract Stability Guidance

### Safe changes

- adding optional metadata fields
- adding new enum values if all consumers handle unknowns safely
- adding new config sections with dashboard support

### Breaking changes

- renaming config fields used by dashboard or reload rules
- changing object ID type/semantics
- changing `SceneState` shape without updating APIs and dashboard renderers
- changing worker response keys without updating manager parsing
