# Scene Memory and State

Scene memory is the persistent in-memory world model for the robot. It stores objects, relationships, captions, track embeddings/crops, Pepper person bindings, and metadata-derived attributes.

## Main Files

Memory runtime:

- `server/app/inference/memory/scene_memory.py`
- `server/app/inference/memory/state_store/store.py`
- `server/app/inference/memory/state_store/tracks.py`
- `server/app/inference/memory/state_store/objects.py`
- `server/app/inference/memory/state_store/relations.py`
- `server/app/inference/memory/state_store/geometry.py`
- `server/app/inference/memory/state_store/social.py`

API/service layer:

- `server/app/api/v1/memory.py`
- `server/app/api/v1/memory_route_utils.py`
- `server/app/orchestration/services/memory.py`
- `server/app/orchestration/services/memory_graph_render.py`

Schemas:

- `server/app/schemas/scene.py`

## Stored State Types

### `TrackedObjectState`

Persistent object record exposed through memory APIs.

Fields include:

- `id`
- `label`
- `status`
- `source`
- `attributes`
- `pepper_person_id`
- `robot_distance`
- `robot_engagement_zone`
- `robot_last_seen_ts`
- `bearing_yaw`
- `bearing_pitch`
- `frame_id`
- `scan_id`
- `first_seen`
- `last_seen`
- `hits`
- `bbox`

### `Relationship`

Persistent binary relation record.

Fields:

- `subject_id`
- `predicate`
- `object_id`
- `first_seen`
- `last_seen`
- `count`

### `SceneCaptionState`

Persistent caption record.

Fields:

- `id`
- `text`
- `provider`
- `model_id`
- `source`
- `frame_id`
- `scan_id`
- `first_seen`
- `last_seen`
- `count`

### `SceneState`

Snapshot containing lists of objects, relationships, captions, and timestamp.

### `MemorySummary`

Dashboard/robot-facing summary containing:

- `timestamp`
- `labels`
- `label_counts`
- `scene_graph` list of `SceneGraphRelation`
- `graph_svg`
- `pregenerated_qa`

## SceneMemory

File: `server/app/inference/memory/scene_memory.py`

`SceneMemory` is the high-level lifecycle manager. It owns:

- `SceneMemoryStore`
- `FeatureExtractor`
- `Associator`
- lock for thread-safe updates
- max dormant frame setting

Main methods:

- `update(image, detections, robot_metadata, fusion_config)`
- `prune_memory()`
- `set_limits(...)`
- `set_max_dormant_frames(...)`
- `set_association_config(...)`
- `set_feature_extraction_config(...)`
- `reset()`

The update method returns fused detections so downstream pipeline stages see any Pepper-induced synthetic people.

## SceneMemoryStore

File: `server/app/inference/memory/state_store/store.py`

The store combines mixins for tracks, geometry, social attributes, object CRUD, and relation CRUD.

Internal state:

- `tracks: dict[int, TrackedObject]`
- `next_id`
- `objects_state`
- `relations_state`
- `captions_state`
- `pepper_person_bindings`
- `_frame_server_to_pepper`
- `_pending_synthetic_pepper_by_detection`

The store is intentionally in-memory. Worker mode means this memory lives in the worker process.

## Object Memory Update

`update_objects_from_detections` creates or updates `TrackedObjectState` for every detection with an object id.

It updates bbox, label, bearing, frame id, scan id, last seen, and hit count. Person-like labels can receive Pepper/social fields if a Pepper binding exists.

Person-like labels currently include:

- `person`
- `man`
- `woman`
- `human`
- `animal`
- `child`
- `robot`
- `dog`
- `cat`

## Relationship Memory Update

`update_scene_graph(scene_graph)` reads `scene_graph.no_label_edges`.

Rules:

- If `sub == obj`, append `rel` to that object's attributes if absent.
- If `sub != obj`, create or refresh a relationship key `(sub, rel, obj)`.
- Existing relationships increment `count` and refresh `last_seen`.

This is why current scene graph backends must produce correct numeric IDs.

## Caption Memory Update

Pipeline caption-memory update inserts or refreshes `SceneCaptionState`. Caption memory supports latest caption and recent caption context for chat prompts and memory summaries.

Captions are pruned separately from objects/relations using `caption_max_age_seconds` and `memory_max_captions`.

## Pruning

File: `state_store/tracks.py`

`prune_memory()` removes:

- objects older than `memory_max_age_seconds`
- oldest objects over `memory_max_objects`
- relations older than cutoff
- relations attached to removed objects
- oldest relations over `memory_max_relations`
- tracks older than cutoff
- tracks not present in object state
- oldest tracks over max object count
- Pepper bindings pointing to removed objects
- captions older than `caption_max_age_seconds`
- oldest captions over `memory_max_captions`

## MemoryService

File: `server/app/orchestration/services/memory.py`

`MemoryService` wraps a runtime adapter and provides API-facing operations:

- get full memory
- get memory summary
- get object crop
- list objects
- list relations
- upsert full `SceneState`
- reset memory
- create/update/delete objects
- create/update/delete relations
- broadcast current memory state

It converts domain errors to HTTP errors through `memory_route_utils.run_memory_action`.

## Memory Summary and Graph SVG

File: `server/app/orchestration/services/memory_graph_render.py`

`MemoryGraphRenderService` builds compact summaries for the dashboard and robot tablet.

It produces:

- label list
- label counts
- scene graph relation list
- SVG memory graph
- text description used for forced QA generation fallback

It can render object crops as nodes when crop bytes are available. It caps rendered objects by `MAX_RENDER_OBJECTS` and by the `render_limit` query parameter.

For Czech display, `MemoryService.get_memory_summary` asks `vocabulary_translator` for object label, object attribute, and relation label overrides before rendering.

## Object Crops

Track crops are stored in `TrackedObject.last_crop`. Crop retrieval path:

1. `GET /api/v1/memory/object/{object_id}/crop`
2. `MemoryService.get_object_crop`
3. runtime adapter `get_track_crop`
4. in-process or worker memory track lookup
5. base64 encoded image bytes in response

Crops are used by memory graph rendering and can be used by object-chat fallback.

## Full Memory API

Public endpoints are documented in `api-reference.md`. The key memory routes are:

- `GET /api/v1/memory`
- `GET /api/v1/memory/summary`
- `GET /api/v1/memory/object/{object_id}/crop`
- `GET /api/v1/memory/objects`
- `GET /api/v1/memory/relations`
- `POST /api/v1/memory/upsert`
- `POST /api/v1/memory/reset`
- object CRUD routes
- relation CRUD routes

## Memory Reset

`POST /api/v1/memory/reset?confirm=true` clears runtime memory. The public API route also clears the process-level QA pool after successful memory reset.

Without `confirm=true`, reset is rejected with a validation error.

## Manual CRUD

Manual object/relation CRUD lets the dashboard or tools edit memory directly.

Object create/update validation includes:

- bbox must have four values
- `last_seen` cannot be earlier than `first_seen`
- update payload cannot be empty

Relation create/update validation includes:

- referenced object IDs must exist in store
- duplicate relationship keys are rejected
- `last_seen` cannot be earlier than `first_seen`
- update payload cannot be empty

## Worker Mode Memory

In worker mode, memory lives in `WorkerRuntime.pipeline.memory`. Public API memory routes call `WorkerRuntimeAdapter`, which forwards to internal worker routes. Dashboard code does not need to know whether memory is local or in worker.

## Where To Change Things

- Add object fields: `schemas/scene.py`, state store update/patch logic, memory renderer, dashboard memory UI.
- Add relation fields: `schemas/scene.py`, relation store CRUD, renderer/dashboard.
- Change pruning: `state_store/tracks.py`.
- Change memory summary SVG: `orchestration/services/memory_graph_render.py`.
- Change memory translation display: `providers/translation/vocabulary.py` and `MemoryService.get_memory_summary`.
- Change crop retrieval: `TrackedObject`, `SceneMemory`, runtime adapters, worker runtime/routes, memory API.
