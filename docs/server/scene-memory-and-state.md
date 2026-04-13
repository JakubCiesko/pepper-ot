# Scene Memory and State Model

## Files Covered

- `app/inference/memory/scene_memory.py`
- `app/inference/memory/chat_memory_proxy.py`
- `app/inference/memory/state_store/store.py`
- `app/inference/memory/state_store/objects.py`
- `app/inference/memory/state_store/relations.py`
- `app/inference/memory/state_store/tracks.py`
- `app/inference/memory/state_store/social.py`
- `app/inference/memory/state_store/geometry.py`
- `app/schemas/scene.py`

## Purpose

Scene memory is the dynamic world model used for grounding. It stores tracked objects, relations, captions, and Pepper-specific person bindings across frames.

Without this layer, the system would only have frame-local detections and would lose referential continuity.

## Public Surface: `SceneMemory`

### Construction knobs

- `memory_max_age_seconds`
- `memory_max_objects`
- `memory_max_relations`
- `memory_max_captions`
- `caption_max_age_seconds`
- `max_dormant_frames`
- association config
- feature extraction config

### Main methods

- `update(image, detections, robot_metadata, fusion_config)`
- `prune_memory()`
- `set_limits(...)`
- `set_max_dormant_frames(...)`
- `set_association_config(...)`
- `set_feature_extraction_config(...)`
- `reset()`
- `update_scene_graph(scene_graph)`
- `upsert_caption(caption)`
- `recent_captions(limit)`
- `scene_state()`
- `upsert_scene_state(state)`
- manual object/relation CRUD helpers

## Internal Store: `SceneMemoryStore`

This is the actual state container.

It owns:
- `tracks`
- `next_id`
- `objects_state`
- `relations_state`
- `captions_state`
- `pepper_person_bindings`

### Important derived payload

`scene_state()` returns `SceneState` with:
- list of objects
- list of relationships
- list of captions
- timestamp

## Object State

`TrackedObjectState` is the main persistent object representation.

Typical object fields include:
- `id`
- `label`
- `bbox`
- `attributes`
- `status`
- `source`
- `first_seen`
- `last_seen`
- `hits`
- Pepper-relative metadata such as engagement and bearing

## Relation State

Relations are stored keyed by:
- `(subject_id, predicate, object_id)`

Each relation tracks:
- predicate
- first/last seen
- count

This makes repeated relation observation explicit instead of replacing history blindly.

## Caption State

Captions are stored separately from object memory.

This is important because:
- captions are free-text scene summaries
- they support chat grounding
- they have separate retention limits from objects/relations

## Manual Memory Upserts

`upsert_scene_state()` merges external state into current memory.

Use this when:
- importing annotations
- manually seeding memory
- replaying experimental state

Be careful:
- IDs are preserved
- `next_id` is advanced to avoid collision

## Pruning Semantics

Memory pruning enforces:
- time-to-live
- max object count
- max relation count
- max caption count

Pruning is triggered not only during updates but also when reading scene state, so stale memory can disappear even without new frames.

## Track Crops

Crops stored alongside tracks are reused by object chat as fallback evidence.

This is a valuable feature because it lets the system describe under-specified objects even when they have little structured state.

## Chat Memory Proxy

`chat_memory_proxy.py` provides worker-safe memory access abstractions such as:
- empty/no-op memory shim
- worker-backed memory proxy behavior

Use this area if you need to decouple chat and memory more aggressively.

## Safe Tweak Points

- memory TTLs and counts
- caption retention policy
- association weights
- object attribute merge policy
- Pepper binding aging policy

## Risky Tweak Points

- `SceneState` schema fields
- relation key identity semantics
- `next_id` handling
- crop storage format
