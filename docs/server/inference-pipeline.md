# Inference Pipeline

The frame pipeline is implemented by `server/app/inference/pipeline.py` and constructed by `server/app/core/pipeline_factory.py`.

It takes a PIL RGB image plus optional `RobotMetadata` and returns a `PipelineResult` containing detections, SoM image, scene graph, caption, generated QA pairs, metrics, and executed stage names.

The pipeline is deliberately transport-agnostic. It does not parse HTTP requests, publish WebSocket events, persist dashboard state, or update the API-level QA pool. Those side effects happen in orchestration services. The pipeline's job is to transform one image and optional robot metadata into one structured perception result.

## Construction

File: `server/app/core/pipeline_factory.py`

`build_perception_pipeline(config)` builds:

- `DetectionService` from `inference/detection/service.py`.
- `SceneMemory` from `inference/memory/scene_memory.py`.
- `SoMPainter` from `inference/scene_graph/som.py`.
- `VLMSceneGraphGenerator` from `inference/scene_graph/vlm_backend.py`.
- `RuleSceneGraphGenerator` from `inference/scene_graph/rules_backend.py`.
- `RelTRSceneGraphGenerator` from `inference/scene_graph/reltr_backend.py`.
- `SceneGraphService` from `inference/scene_graph/service.py`.
- `CaptionInferenceService` from `inference/caption/service.py`.
- `SceneQAGenerationService` from `inference/qa/service.py`.
- `PerceptionPipeline` with pipeline controls, visualization config, and fusion config.

Prompt sources and ontologies are resolved relative to the config file directory.

## PipelineResult

Defined in `server/app/inference/types.py`.

Fields:

- `raw_image`: original PIL image passed into the pipeline.
- `som_image`: NumPy array or PIL image with Set-of-Mark overlay, or `None`.
- `detections`: list of `InferenceDetectionObject` with persistent `object_id` when tracking is enabled.
- `scene_graph`: `SceneGraph` or `None`.
- `metrics`: timing dictionary.
- `executed_stages`: list of stage names executed for this frame.
- `caption`: caption text, if caption stage ran.
- `caption_provider`: caption provider name.
- `caption_model_id`: caption model id.
- `qa_pairs`: generated English Q/A pairs from scene graph facts.

`PipelineResult` is the contract that must be preserved across both runtime modes. In worker mode it is serialized through `DetectRPCResponse`, so adding a new field requires changes in the pipeline result, worker runtime, worker RPC schema, runtime adapter normalization, detection orchestration response, and dashboard rendering when applicable.

## Data Flow Contract

| Stage | Main input | Main output | Consumer |
|---|---|---|---|
| Caption | Raw image and caption prompts | Caption text and provider metadata | Scene graph prompt context and caption memory update |
| Detection | Raw image | Frame detections | Tracking, SoM, rules/RelTR/VLM filtering |
| Tracking/memory | Detections and `RobotMetadata` | Persistent object IDs, object state, crops, robot/social attributes | Scene graph, chat memory, dashboard memory |
| SoM painting | Image and tracked detections | Marked prompt image | VLM scene graph backend and dashboard preview |
| Scene graph | Image/SoM, detections, memory state, caption | `SceneGraph` edges | QA generation and graph memory update |
| QA generation | Scene graph, detections, caption | English Q/A pairs | API-process `QAPoolService` via `DetectService` |
| Caption memory update | Caption and metadata | `SceneCaptionState` in scene memory | Memory summaries and chat grounding |
| Scene graph memory update | Scene graph no-label edges | Object attributes and relationships in scene memory | Memory summaries, chat, dashboard |

This is why object IDs matter. Detection and tracking establish IDs, SoM makes those IDs visible to the VLM, scene graph generation returns relationships over those IDs, and memory update stores graph facts by ID. If IDs are disabled or inconsistent, downstream graph memory and object chat become less reliable.

## Execution Modes

`PerceptionPipeline.process(image, robot_metadata)` is the public entrypoint. It dispatches to one of two execution modes:

- `process_sequentially(...)` when `pipeline_controls.parallel_execution=false`
- `process_in_parallel(...)` when `pipeline_controls.parallel_execution=true`

The default is sequential execution. Parallel execution is optional because its benefit depends on the configured models and hardware.

### Sequential Stage Order

Sequential mode runs stages in this order when enabled by `pipeline_controls`:

1. Caption
2. Detection
3. Tracking and memory association
4. SoM painting
5. Scene graph generation
6. QA generation
7. Caption memory update
8. Scene graph memory update
9. Total metrics finalization

The actual executed stage names are appended to `PipelineResult.executed_stages` and returned to API/dashboard payloads.

If SoM painting is disabled or fails to produce an overlay, the pipeline keeps a usable image path for scene graph generation by falling back to the original image as the prompt image. The `paint_som` stage is only recorded when the SoM rendering stage actually ran.

### Parallel Pipeline Mode

When `pipeline_controls.parallel_execution=true`, only the safe independent early stages overlap:

1. Caption task starts.
2. Detection task starts.
3. Detection is awaited first because tracking depends on detections.
4. Tracking and SoM painting run in order.
5. Caption is awaited before scene graph generation so graph backends can still receive caption context.
6. Scene graph generation, QA generation, and memory updates remain ordered.

Detection is synchronous in `DetectionService`, so the parallel path runs detector inference through `asyncio.to_thread`. Access to the detector is protected by a lock to avoid simultaneous calls into the same detector model instance across requests. The caption and detection branches receive copied PIL images so they do not read the same image object concurrently.

The `executed_stages` list remains deterministic. It records logical pipeline order rather than task completion order.

Use this mode when captioning is remote/API-backed or otherwise independent from the detector. Be careful when both caption and detection use local GPU models because overlap can increase VRAM pressure.

## Stage: Caption

Files:

- `server/app/inference/caption/service.py`
- `server/app/providers/caption/client.py`

If `pipeline_controls.caption=true`, the pipeline captions the image first. The caption can be used by scene graph generation and is later persisted into memory by caption-memory update.

Caption inference uses `CaptionInferenceService.caption_image`. It uses the configured `CaptionClient`, which chooses either BLIP-specific local captioning or a VLM provider.

Metric key: `caption_time`.

Executed stage: `caption`.

## Stage: Detection

Files:

- `server/app/inference/detection/service.py`
- `server/app/inference/detection/detectors.py`
- `server/app/inference/detection/model_registry.py`

If `pipeline_controls.detect=true`, the detector runs on the image.

Detection service applies:

1. Backend model prediction.
2. Confidence threshold at backend/model layer where supported.
3. Optional post-filter NMS using `run_nms_post_filter`, `nms_iou_threshold`, and `nms_type`.

Metric key: `detection_time`.

Executed stage: `detect`.

If detection is disabled, downstream stages that require detections should also be disabled by config validation.

## Stage: Tracking and Memory Association

Files:

- `server/app/inference/memory/scene_memory.py`
- `server/app/inference/tracking/embeddings.py`
- `server/app/inference/tracking/associator.py`
- `server/app/inference/memory/state_store/*`

If `pipeline_controls.track_memory=true`, the current detections are associated with persistent memory tracks.

The tracking stage:

1. Extracts ReID embeddings and crop bytes.
2. Matches detections against active tracks using weighted appearance/geometry association.
3. Updates matched tracks and assigns persistent object IDs.
4. Ages unmatched tracks.
5. Creates tracks for unmatched detections.
6. Fuses Pepper robot people metadata with detected people.
7. Creates synthetic Pepper-person tracks when the robot reports a person missed by the visual detector.
8. Updates object states and robot/social attributes.
9. Prunes stale memory.

Metric key: `memory_update_time`.

Executed stage: `track_memory`.

If tracking is disabled but detection runs, the pipeline assigns sequential frame-local object IDs starting from 1. These IDs are not persistent across frames.

## Stage: SoM Painting

File: `server/app/inference/scene_graph/som.py`

If `pipeline_controls.paint_som=true`, detections are rendered onto the image as Set-of-Mark overlays.

Visualization config controls:

- boxes
- masks
- polygons
- labels
- line thickness
- mask opacity
- color lookup by index/class/track
- mask backend: `grabcut` or `sam`

SAM mask prompt boxes are internally processed in fixed chunks of 4 inside `sam_bboxes_to_masks` to reduce prompt memory pressure. SAM can fall back to GrabCut if unavailable or failing.

Metric key: `som_image_paint_time`.

Executed stage: `paint_som`.

## Stage: Scene Graph Generation

Files:

- `server/app/inference/scene_graph/service.py`
- `server/app/inference/scene_graph/rules_backend.py`
- `server/app/inference/scene_graph/reltr_backend.py`
- `server/app/inference/scene_graph/vlm_backend.py`

If `pipeline_controls.scene_graph=true`, the pipeline calls `SceneGraphService.generate`.

Enabled backends are selected independently:

- rules backend when `scene_graph.rules.enabled=true`
- RelTR backend when `scene_graph.reltr.enabled=true`
- VLM backend when `scene_graph.vlm.enabled=true`

All enabled backend outputs are merged with `SceneGraph.__add__`, which deduplicates edges. After merging, `SceneGraphService.enhance_scene_graph_with_robot_data` adds current-object memory attributes derived from Pepper/social metadata.

Scene graph generation receives both label-bearing detections and current scene memory. Backends may produce label edges, id-only edges, or both. Memory update prefers id-only `no_label_edges` because translated or changed object labels should not change the identity of a relationship.

Metric key: `scene_graph_generation_time`.

Executed stage: `scene_graph`.

Scene graph backend execution has its own independent parallel flag: `scene_graph.parallel_execution`.

When `scene_graph.parallel_execution=false`, `SceneGraphService.generate_sequential` runs enabled graph backends in fixed order:

1. rules
2. RelTR
3. VLM

When `scene_graph.parallel_execution=true`, `SceneGraphService.generate_parallel` starts every enabled backend concurrently:

- Rules runs in a thread because it is synchronous CPU/image work.
- RelTR runs in a thread through `RelTRSceneGraphGenerator.generate_sync`.
- VLM runs as an async task.

Results are still merged deterministically in `rules`, `reltr`, `vlm` order. Robot-derived memory attributes are injected only after the merged graph exists.

RelTR model execution is protected by a lock, so the same RelTR model instance is not used concurrently by multiple RelTR calls. This still allows RelTR to overlap with remote VLM calls. If the VLM backend is also local GPU-backed, enabling scene graph parallelism can increase GPU memory pressure.

## Stage: QA Generation

Files:

- `server/app/inference/qa/service.py`
- `server/app/orchestration/services/qa_pool.py`
- `server/app/orchestration/services/detection.py`

If `pipeline_controls.qa_generation=true`, the pipeline generates graph-grounded English Q/A pairs after scene graph generation.

The QA stage:

1. Converts current detections and graph triples into text.
2. Calls `LLMClient.generate_structured` using `_GeneratedQAPairs` schema.
3. Normalizes non-empty question/answer pairs.
4. Deduplicates by lowercase question.
5. Returns up to `qa_generation.pairs_per_update` pairs.

Metric key: `qa_generation_time`.

Executed stage: `qa_generation`.

The pipeline itself only returns pairs. `DetectService` ingests them into the process-level `QAPoolService` when the detect flow actually updates memory.

## Stage: Caption Memory Update

If a caption was produced, it is stored as a `SceneCaptionState` in scene memory. Captions are keyed by frame or generated id, carry provider/model/source/frame/scan metadata, and are pruned by caption age and caption cap.

Caption memory does not require scene graph generation. It gives chat and summaries a natural-language observation history even when graph generation is disabled or fails.

Metric key: `caption_memory_update_time`.

Executed stage: `update_caption_memory`.

## Stage: Scene Graph Memory Update

If `pipeline_controls.update_scene_memory=true`, the current scene graph updates memory relationships and object attributes.

Rules:

- unary graph edge where `sub == obj` becomes an object attribute
- binary graph edge becomes a `Relationship(subject_id, predicate, object_id)`
- existing relationships increment `count` and refresh `last_seen`

The memory update stage does not re-run graph inference. It only projects the current `SceneGraph` into persistent memory. If `scene_graph` is disabled, `update_scene_memory` must also be disabled by config validation.

Metric key: `scene_graph_memory_update_time`.

Executed stage: `update_scene_memory`.

## Pipeline Controls and Validation

`PipelineControls` is defined in `server/app/schemas/config.py`.

Important fields:

- `parallel_execution`: overlaps caption and detection where safe.
- `caption`, `detect`, `track_memory`, `paint_som`, `scene_graph`, `qa_generation`, `update_scene_memory`: individual stage toggles.
- `preset`: convenience profile for stage toggles.

Important dependencies:

- `track_memory` requires `detect`.
- `paint_som` requires `detect`.
- `qa_generation` requires `scene_graph`.
- `update_scene_memory` requires `scene_graph` and `track_memory`.
- scene graph with rules/RelTR requires detection.

Use dashboard Runtime Orchestration controls or config YAML to enable/disable stages.

## Metrics

Each stage writes timing metrics in seconds. API/dashboard payloads include `metrics` and `executed_stages`, making it possible to see exactly which stages ran for a frame.

Common metric keys:

- `caption_time`
- `detection_time`
- `memory_update_time`
- `som_image_paint_time`
- `scene_graph_generation_time`
- `qa_generation_time`
- `caption_memory_update_time`
- `scene_graph_memory_update_time`
- `total_processing`: sum of individual `*_time` stage timings.
- `wall_processing_time`: actual end-to-end wall-clock time for the pipeline call.

In parallel modes, `total_processing` can be larger than actual elapsed time because overlapping stage durations are still summed independently. Use `wall_processing_time` when comparing latency.

## Where To Change Pipeline Behavior

- Add a new stage: `server/app/inference/pipeline.py`, `PipelineResult`, `PipelineControls`, dashboard runtime controls, config reload rules, worker RPC response if data must cross process boundary.
- Change stage order: `PerceptionPipeline.process`.
- Change stage config validation: `AppConfig.validate_pipeline_controls`.
- Change stage construction: `server/app/core/pipeline_factory.py`.
- Expose new output to API/dashboard: `orchestration/adapters/runtime.py`, `worker/runtime.py`, `worker_client/rpc.py`, `orchestration/services/detection.py`, dashboard live JS.
