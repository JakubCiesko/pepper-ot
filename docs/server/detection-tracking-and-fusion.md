# Detection, Tracking, and Robot Metadata Fusion

This document explains how images become detected objects, how persistent object IDs are maintained, how crops are stored, and how Pepper robot metadata is fused into memory.

## Main Files

Detection:

- `server/app/inference/detection/service.py`
- `server/app/inference/detection/detectors.py`
- `server/app/inference/detection/model_registry.py`
- `server/app/schemas/config.py` (`DetectionConfig`)

Tracking and memory update:

- `server/app/inference/memory/scene_memory.py`
- `server/app/inference/tracking/embeddings.py`
- `server/app/inference/tracking/associator.py`
- `server/app/inference/memory/state_store/tracks.py`
- `server/app/inference/memory/state_store/objects.py`

Robot fusion:

- `server/app/schemas/robot.py`
- `server/app/inference/memory/state_store/geometry.py`
- `server/app/inference/memory/state_store/social.py`
- `server/app/inference/memory/state_store/store.py`

API orchestration:

- `server/app/api/v1/detect.py`
- `server/app/orchestration/services/detection.py`

## Detection Input and Output

The public detect endpoint accepts image bytes and optional robot metadata. The orchestration layer converts metadata JSON into `RobotMetadata` and passes the image to the runtime adapter.

The internal detection model returns `InferenceDetectionObject` records:

- `class_id`
- `label`
- `confidence`
- `bbox` as `[x1, y1, x2, y2]`
- `object_id`, initially absent until tracking assigns one

The public API response uses `DetectionObject`, which exposes label, confidence, bbox, and object_id.

## Detector Backends

`DetectionModelType` currently supports:

- `yolo`
- `rt_detr`
- `rf_detr`
- `owl_v2`

Backend model construction is handled by `model_registry.py`.

`detectors.py` contains backend wrappers:

- `BaseDetector`: common interface.
- `UltralyticsDetector`: YOLO / RT-DETR family through Ultralytics API.
- `RoboflowDetector`: RF-DETR wrapper.
- `Owlv2Detector`: Hugging Face OWLv2 open-vocabulary detector.

Open-vocabulary backends can use `DetectionConfig.ontology` or `ontology_path`.

## DetectionService

File: `server/app/inference/detection/service.py`

`DetectionService` owns:

- backend model instance
- active confidence threshold
- NMS settings
- device
- ontology

It exposes:

- `detect(image) -> list[InferenceDetectionObject]`
- `detect_batch(images) -> list[list[InferenceDetectionObject]]`

The service is constructed by `pipeline_factory.py` from current config and can be hot-updated through runtime config mutation helpers.

## Post-Detection NMS

Optional NMS is controlled by:

- `detection.run_nms_post_filter`
- `detection.nms_iou_threshold`
- `detection.nms_type`

NMS runs after backend prediction and before tracking.

NMS types:

- `per_class`: suppress overlapping boxes only within the same class/label. This is the safer default because a person and a held object can overlap without one suppressing the other.
- `general`: suppress overlapping boxes across all classes. This is more aggressive and can remove cross-class overlaps.

NMS is useful when using a lower confidence threshold and the detector emits duplicate overlapping boxes.

The service logs active detection settings and NMS before/after counts at debug level.

## Image Resizing and Metadata Geometry

File: `server/app/api/v1/detect.py`

The detect endpoint can resize image bytes before inference to control GPU load. When it resizes, it patches `RobotMetadata.image_width` and `image_height` to the resized dimensions before passing metadata to the pipeline.

This matters because robot fusion maps bbox centers to camera angles using metadata image dimensions and FOV. The server-side bbox coordinates must match metadata dimensions. The endpoint keeps them aligned by patching metadata after resizing.

## Tracking Stage

File: `server/app/inference/memory/scene_memory.py`

`SceneMemory.update(image, detections, robot_metadata, fusion_config)` performs tracking and memory mutation.

Steps:

1. Extract embeddings and crop bytes with `FeatureExtractor.extract_with_crops`.
2. Read active tracks from `SceneMemoryStore`.
3. Match detections to tracks using `Associator.match`.
4. Update matched tracks and write `det.object_id`.
5. Age unmatched tracks and drop tracks dormant for too many frames.
6. Create new tracks for unmatched detections.
7. Fuse Pepper robot people metadata.
8. Create synthetic tracks for Pepper-induced people if needed.
9. Update object states from detections.
10. Age Pepper-person bindings.
11. Prune stale objects, relationships, tracks, captions, and bindings.

## Feature Extraction and Crops

File: `server/app/inference/tracking/embeddings.py`

The feature extractor crops every detected bbox, resizes each crop to the configured ReID `target_size`, extracts normalized embeddings, and returns both embeddings and JPEG crop bytes.

The crop bytes are stored in `TrackedObject.last_crop` when a track is created or updated. Later code can retrieve these through memory crop endpoints and use them in object chat fallback or dashboard memory graph rendering.

## Associator

File: `server/app/inference/tracking/associator.py`

The associator computes matches between active tracks and current detections. It combines:

- visual similarity from embeddings
- geometry similarity from bbox/track geometry
- configured weights and match threshold

The result is:

- matched `(track_index, detection_index)` pairs
- unmatched track indices
- unmatched detection indices

## Track Store

File: `server/app/inference/memory/state_store/tracks.py`

Track operations include:

- `create_track(det, embedding, last_crop)`
- `age_unmatched_tracks`
- `drop_dormant_tracks`
- `prune_memory`
- `snapshot`
- `reset`

Pruning removes stale objects, stale relations linked to stale objects, stale tracks, stale captions, overflow by max object/relation/caption counts, and stale Pepper bindings.

## Object State Update

File: `server/app/inference/memory/state_store/objects.py`

`update_objects_from_detections` mirrors current detections into persistent `TrackedObjectState` records.

It stores:

- object id
- label
- bbox
- status
- source
- attributes
- bearing yaw/pitch
- frame id
- scan id
- first/last seen
- hit count
- Pepper person id
- robot distance
- robot engagement zone
- robot last seen timestamp

If the detection is a person-like label and has a current Pepper binding, social fields are merged into attributes.

## RobotMetadata

File: `server/app/schemas/robot.py`

`RobotMetadata` contains:

- `head_yaw`
- `head_pitch`
- `body_yaw`
- `camera_hfov`
- `camera_vfov`
- `image_width`
- `image_height`
- `timestamp`
- `frame_id`
- `scan_id`
- `capture_mode`
- `people`: geometric `PersonMetadata`
- `social_people`: social `SocialPersonMetadata`
- `battery`

`DetectService.parse_metadata` accepts JSON strings and fills a safe default when metadata is absent.

## Geometry Fusion

File: `server/app/inference/memory/state_store/geometry.py`

Geometry methods convert between pixels and robot angles.

### Detection Bearing

`compute_bearing(det, robot_metadata, image)`:

1. Reads image width/height and camera FOV from metadata.
2. Computes bbox center `(cx, cy)`.
3. Converts pixel offset to relative yaw/pitch:
   - `yaw_rel = (0.5 - cx / width) * hfov`
   - `pitch_rel = (0.5 - cy / height) * vfov`
4. Adds base robot angles:
   - `base_yaw = body_yaw + head_yaw`
   - `base_pitch = head_pitch`
5. Returns absolute `(yaw, pitch)`.

### Pepper Person Projection

`project_robot_person_to_pixel(person, robot_metadata, image)` performs the inverse mapping from Pepper-reported person yaw/pitch into image pixel coordinates.

### Match Candidate Selection

For each Pepper person, the store first tries projected pixel containment inside person-like detection bboxes with margin. If that yields candidates, it uses those. Otherwise it falls back to angular threshold candidates.

### Match Scoring

`score_robot_person_match` combines:

- angular similarity
- previous Pepper binding bonus
- tracker hit bonus
- distance/size consistency

Current weights:

- 0.55 angular
- 0.25 previous binding
- 0.15 tracker bonus
- 0.05 distance-size consistency

Candidates are sorted by score and greedily matched one-to-one.

## Pepper Bindings

File: `server/app/inference/memory/state_store/store.py`

`PepperPersonBinding` maps Pepper's person id to the server's persistent object id.

It stores:

- `pepper_person_id`
- `server_object_id`
- `first_seen`
- `last_seen`
- `confidence`
- `misses`

The store removes conflicting bindings so one Pepper person id maps to one server object id, and one server object id is not bound to multiple Pepper ids at the same time.

Bindings are aged by `pepper_binding_max_misses`.

## Synthetic Pepper Person Detections

If Pepper reports a person but the detector misses them, fusion can synthesize an `InferenceDetectionObject` with an estimated bbox.

This path uses:

- projected Pepper person pixel location
- distance-scaled bbox size
- min/max synthetic bbox clamps
- configured synthetic confidence

The synthetic detection is then embedded, assigned a track, bound to Pepper id, and inserted into memory like normal detections.

## Social Attribute Extraction

File: `server/app/inference/memory/state_store/social.py`

`SocialPersonMetadata` can produce object attributes such as:

- `is_sitting`
- `is_waving`
- `is_looking_at_robot`
- `is_looking_forward`
- `is_looking_left`
- `is_looking_right`
- `is_looking_up`
- `is_looking_down`
- `is_near`
- `is_not_far`
- `is_far`
- `is_male`
- `is_female`
- `is_child`
- `is_adult`
- `is_senior`
- `is_N_years_old`
- `has_<expression>_expression`
- `has_a_bit_<expression>_expression`
- `is_smiling`
- `has_open_eyes`
- blink-related attributes

Thresholds are currently constants in `SceneMemoryStoreSocialMixin`.

Social attributes are merged in a way that preserves non-social attributes and replaces/upserts social ones.

## Memory Attribute Injection Into Scene Graph

After graph backends run, `SceneGraphService.enhance_scene_graph_with_robot_data` adds unary graph edges for current object attributes stored in memory. This means Pepper-derived attributes can appear in scene graphs and can later feed memory summary and Q/A.

## Where To Change Things

- Add detector backend: `detectors.py`, `model_registry.py`, `DetectionModelType`, config schema/dashboard.
- Change NMS behavior: `inference/detection/service.py`.
- Change crop storage: `tracking/embeddings.py`, `types.TrackedObject`, `scene_memory.py`.
- Change identity matching: `tracking/associator.py`.
- Change memory pruning: `state_store/tracks.py`.
- Change Pepper-person matching: `state_store/geometry.py`.
- Change social attributes: `state_store/social.py`.
- Change synthetic person behavior: geometry/store mixin code and fusion config.
