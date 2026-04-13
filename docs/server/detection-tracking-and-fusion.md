# Detection, Tracking, and Pepper Fusion

## Files Covered

- `app/inference/detection/detectors.py`
- `app/inference/detection/model_registry.py`
- `app/inference/detection/service.py`
- `app/inference/tracking/associator.py`
- `app/inference/tracking/embeddings.py`
- `app/inference/memory/scene_memory.py`
- `app/inference/memory/state_store/tracks.py`
- `app/inference/memory/state_store/social.py`
- `app/inference/memory/state_store/geometry.py`
- `app/schemas/robot.py`
- `app/inference/types.py`

## Detection Backends

Implemented in `detectors.py`.

### Supported backend enum

`DetectionModelType` includes:
- `yolo`
- `rt_detr`
- `rf_detr`
- `owl_v2`

### Detector classes

- `BaseDetector`
- `UltralyticsDetector`
- `RoboflowDetector`
- `Owlv2Detector`

## Model registry

`DetectionModelRegistry` is responsible for loading/caching detector models and resolving backend-specific resources.

If you add a new detector family, update the registry and pipeline factory, not just detector class definitions.

## Detection output contract

The normalized object representation used downstream is `InferenceDetectionObject`.

Important fields include:
- `label`
- `confidence`
- `bbox`
- `object_id` (set later by memory/tracking)

## Tracking and ReID

Tracking logic is split across:
- `FeatureExtractor` in `embeddings.py`
- `Associator` in `associator.py`
- track state in `TrackedObject`
- `SceneMemory.update()` as the top-level coordinator

### `FeatureExtractor`

Purpose:
- generate visual embeddings for detections
- optionally return crop bytes for later fallback captioning

Configurable via:
- ReID model id
- target size
- resampling method
- device

### `Associator`

Purpose:
- match current detections to active tracks

Inputs:
- active tracks
- current detections
- embeddings

Tunable weights:
- `w_vis`
- `w_geo`
- `match_threshold`

This is the main tweak zone if identity continuity is too sticky or too unstable.

## `SceneMemory.update()`

Top-level tracking/memory update sequence:
- extract embeddings and crops
- match detections to existing tracks
- update matched tracks
- age/drop unmatched tracks
- create tracks for unmatched detections
- fuse Pepper people metadata
- create tracks for synthetic Pepper-induced detections if needed
- update persistent object state
- prune memory

## Pepper People Fusion

This is one of the most specialized parts of the codebase.

The memory store can:
- bind Pepper-native person IDs to server object IDs
- reuse those bindings across frames
- create synthetic detections when Pepper sees a person but the detector did not produce a matching box
- propagate social attributes like engagement/waving/looking-at-robot into object state

### Important concepts

- `PepperPersonBinding`
- `_frame_server_to_pepper`
- `_pending_synthetic_pepper_by_detection`
- `age_pepper_bindings()`
- `bind_pending_detection_track()`

### Fusion config knobs

Defined under `fusion` in config:
- bbox match threshold in pixels
- estimated synthetic person bbox sizing range
- angular thresholds
- confidence defaults for matched vs synthetic persons
- allowed miss count before Pepper binding is dropped

## Geometry and social enrichment

`state_store/geometry.py` and `state_store/social.py` enrich objects with:
- robot-relative distance
- bearing estimates
- engagement zone
- Pepper person ID linkage
- social attributes merged into object attributes

## Safe Tweak Points

- detection threshold
- ontology labels
- association weights
- ReID target size
- dormant frame count
- Pepper fusion confidence and miss thresholds

## Risky Tweak Points

- changing object ID assignment semantics
- changing bbox coordinate conventions
- changing detection object structure without updating scene graph, memory, and API serialization
