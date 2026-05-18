# Scene Graph and Grounding

Scene graph generation turns tracked detections into semantic triples and object attributes. It can combine deterministic rules, RelTR predictions, VLM output over a Set-of-Mark image, and robot-derived memory attributes.

The graph is designed for grounded conversation and memory updates, not just visualization. A relation must refer to current object IDs so the server can store facts like "person_1 holding cup_2" and later answer questions about the same tracked objects.

## Main Files

- `server/app/inference/scene_graph/service.py`
- `server/app/inference/scene_graph/rules_backend.py`
- `server/app/inference/scene_graph/reltr_backend.py`
- `server/app/inference/scene_graph/reltr_predictor.py`
- `server/app/inference/scene_graph/vlm_backend.py`
- `server/app/inference/scene_graph/som.py`
- `server/app/inference/types.py` (`SceneGraph`, `SceneGraphEdge`)
- `server/app/schemas/scene.py` (`SceneGraphRelation`, structured VLM schemas)
- `server/app/schemas/config.py` (`SceneGraphConfig`)

## Current Scene Graph Model

There is no single `scene_graph.mode` field in the current code. Scene graph generation is controlled by independent backend flags:

- `scene_graph.rules.enabled`
- `scene_graph.reltr.enabled`
- `scene_graph.vlm.enabled`

`SceneGraphService.generate` runs every enabled backend and merges the results.

Backend roles:

| Backend | Best at | Typical risk |
|---|---|---|
| Rules | Deterministic geometry, configured label pairs, simple attributes such as color | Limited semantic range; only sees what rules encode |
| RelTR | Visual Genome-style visual relations from a relation model | Box-to-detection matching errors and predicate vocabulary mismatch |
| VLM | Open-ended or prompt-constrained relations over SoM object IDs | Hallucinated IDs, malformed JSON, vocabulary drift |
| Robot-data enhancement | Social/pose attributes already fused into memory | Only available for objects currently present in memory/detections |

Backend execution can be sequential or parallel:

- `scene_graph.parallel_execution=false`: enabled backends run in deterministic order.
- `scene_graph.parallel_execution=true`: enabled backends run concurrently and are merged afterward in deterministic order.

The default is `false` to avoid unexpected GPU pressure when multiple local graph backends are enabled.

## SceneGraph Data Structure

File: `server/app/inference/types.py`

`SceneGraph` stores two edge lists:

- `edges`: label-bearing references such as `person_1 holding cat_2`.
- `no_label_edges`: id-only references such as `1 holding 2`.

The id-only edge list is preferred for memory updates because object labels can change or be translated.

`SceneGraph.from_list(data)` accepts dicts with `sub`, `rel`, and `obj`, builds label edges as provided, and derives no-label edges by extracting trailing numeric ids.

`SceneGraph.__add__` merges edge lists and raw outputs. `SceneGraph.__post_init__` deduplicates edges.

Use `edges` for human-readable debugging and dashboard display. Use `no_label_edges` for memory updates and evaluation when object IDs are the stable contract. If a backend produces only label-bearing text without recoverable numeric IDs, memory update cannot reliably ground that relation.

## Service Merge Flow

File: `server/app/inference/scene_graph/service.py`

Flow:

1. Select VLM image: `som_image` if available, else raw image.
2. Log enabled backends.
3. Start with empty `SceneGraph`.
4. If rules enabled, add rules graph.
5. If RelTR enabled, await and add RelTR graph.
6. If VLM enabled and an image exists, await and add VLM graph.
7. Enhance merged graph with robot-data attributes from current scene memory.
8. Return final graph.

The merge strategy is effectively union plus deduplication.

When `scene_graph.parallel_execution=true`, the service uses `generate_parallel` instead of `generate_sequential`:

- rules backend is offloaded to a thread
- RelTR backend is offloaded to a thread through `generate_sync`
- VLM backend runs as an async task
- output merge order remains `rules`, then `reltr`, then `vlm`
- robot-data enhancement still runs once after backend merge

This is most useful when RelTR can overlap with a remote VLM call. If RelTR and VLM are both local GPU workloads, parallel execution can increase VRAM pressure and may be slower or less stable.

## Rule Backend

File: `server/app/inference/scene_graph/rules_backend.py`

The rules backend is deterministic. It uses tracked detections with object IDs and configured rules.

Supported rule types:

- `spatial` / `space`: center distance range.
- `directional` / `direction`: x/y center delta thresholds.
- `overlap`: bbox IoU range.
- `containment` / `contain`: inside-ratio range.
- `label_pair`: relation exists when label constraints pass.

Rule constraints can filter by:

- `subject_labels`
- `object_labels`
- `labels_any`

The rules backend also extracts coarse color attributes from image crops using `fast_colorthief`. Very small boxes are skipped to avoid unreliable color inference.

Rules produce both label and no-label edges.

## RelTR Backend

Files:

- `server/app/inference/scene_graph/reltr_backend.py`
- `server/app/inference/scene_graph/reltr_predictor.py`

RelTR flow:

1. Ensure RelTR backend enabled.
2. Ensure image and checkpoint exist.
3. Build RelTR model lazily if needed.
4. Save current image to a temporary file under server state.
5. Run `predict_image` with configured threshold/topk/device.
6. Map RelTR predicted boxes back to current server detections by IoU.
7. Keep binary relations when subject and object map to two different tracked detections.
8. For attributeable Visual Genome predicates, convert some unmatched subject/object predictions into unary attributes.
9. Drop invalid/unmatched relations.
10. Return a `SceneGraph` with label and no-label edges.

The backend exposes two generation paths:

- `generate_sync(...)`: synchronous implementation used when scene graph parallelism offloads RelTR to a worker thread.
- `generate(...)`: async compatibility wrapper that calls `generate_sync(...)` through `asyncio.to_thread`.

RelTR model execution is guarded by an internal lock so one `RelTRSceneGraphGenerator` instance does not run multiple model calls concurrently.

Important config:

- `scene_graph.reltr.checkpoint_path`
- `scene_graph.reltr.device`
- `scene_graph.reltr.threshold`
- `scene_graph.reltr.topk`
- `scene_graph.reltr.iou_match_threshold`

The attributeable predicate list is defined in `reltr_predictor.py` as `VG_REL_CLASSES_ATTRIBUTEABLE`.

## VLM Backend

File: `server/app/inference/scene_graph/vlm_backend.py`

VLM generation uses the configured `BaseVLMClient` and prompt templates.

Flow:

1. Serialize input image to JPEG bytes.
2. Render system prompt with `PromptRenderContext`.
3. Render user prompt or build default allowed-predicate prompt.
4. Choose output schema:
   - `SceneGraphStructuredResponse` when `structured_schema=scene_graph`.
   - `list[SceneGraphRelation]` when `structured_schema=relationship_list`.
5. Call VLM client with optional structured output schema.
6. Parse structured result, JSON raw text, or JSON block extracted from raw text.
7. If parsing fails, ask the VLM to repair JSON without resending the image.
8. Build `SceneGraph` from relation dicts.
9. Filter hallucinated relations against current detections and IDs.

Filtering keeps only relations whose normalized subject/object IDs appear in current detections. It also rebuilds label references using detection labels.

The VLM backend is where prompt design matters most. The prompt must tell the model that binary relations use different IDs and unary attributes use the same ID for `sub` and `obj`. The SoM image exists to make these IDs visually available to the model; the detection list exists so the prompt and post-filter agree on the valid ID set.

## SoM Rendering

File: `server/app/inference/scene_graph/som.py`

SoM means Set-of-Mark. It overlays detection IDs, boxes, masks, polygons, and labels onto the image. The VLM backend can use this marked image so it references object IDs visible in the scene.

The server and research package both use this idea. In the server it is part of the live perception pipeline. In research experiments it is varied systematically to compare raw images, boxes, labels, masks, and vocabulary prompts.

Config fields under `visualization` control rendering:

- `show_bbox`
- `show_mask`
- `show_polygon`
- `show_labels`
- `line_thickness`
- `mask_opacity`
- `color_lookup`
- `mask_backend`
- `device`

Color lookup modes:

- `index`: color by detection order.
- `class`: color by object class/label.
- `track`: color by persistent track id.

## Mask Backends

### GrabCut

GrabCut is the lightweight fallback mask backend. It uses bbox initialization and OpenCV GrabCut to estimate object masks.

### SAM

SAM backend lazily loads `facebook/sam3` through Hugging Face Transformers. If loading or inference fails, the code falls back to GrabCut behavior.

`sam_bboxes_to_masks` clips boxes to image bounds and now processes prompt boxes in fixed batches of 4. This is intentionally local to the function and not exposed in config. The output shape remains `(N, H, W)` bool masks. Each requested bbox gets the best predicted SAM mask by IoU; if no adequate predicted mask exists, the bbox rectangle is used as conservative fallback for that item.

## Robot-Data Enhancement

File: `server/app/inference/scene_graph/service.py`

After backend merge, `enhance_scene_graph_with_robot_data` adds unary edges for attributes already stored on current memory objects. This is how robot-derived attributes like `is_waving`, `is_sitting`, `is_near`, or `is_looking_at_robot` can enter the current graph.

It only considers objects whose ids are present in current detections. It creates label edges of the form:

```text
<object_label>_<object_id> <attribute> <object_label>_<object_id>
```

Then it converts those into a `SceneGraph` and merges it with the backend graph.

## Memory Update Semantics

File: `server/app/inference/memory/state_store/relations.py`

When scene graph memory update runs:

- unary no-label edge `id rel id` becomes object attribute `rel`
- binary no-label edge `sub rel obj` becomes or refreshes a `Relationship`

This means relation correctness depends on `no_label_edges` carrying stable numeric IDs.

Example:

```text
1 is_red 1      -> attribute "is_red" on object 1
1 holding 2     -> relationship subject=1 predicate=holding object=2
```

The memory store refreshes existing facts instead of blindly appending duplicates. Repeated observations increase relationship counts and update timestamps, which lets summaries prefer recently seen and repeatedly confirmed facts.

## Debugging Graph Problems

Use this order when a graph looks wrong:

1. Check `PipelineResult.executed_stages` and `metrics.scene_graph_generation_time` to confirm the stage ran.
2. Check current detections and object IDs; VLM and memory update can only ground IDs that exist.
3. Check whether `som_image` was produced and whether IDs are visible enough for the model.
4. Check enabled backend flags under `scene_graph.rules`, `scene_graph.reltr`, and `scene_graph.vlm`.
5. Inspect raw VLM output or RelTR raw output if enabled.
6. Check `edges` and `no_label_edges`; dashboard text may look right while memory update fails if ID extraction failed.
7. Check robot-data enhancement only after backend graph output is correct.

## Structured Output

VLM structured output mode is configured under `scene_graph.vlm.structured_output`.

Supported modes are provider-dependent:

- `provider_native`
- `parse_output`
- `instructor`

Provider capability behavior is documented in `providers-model-clients.md`.

## Where To Change Things

- Add a new SGG backend: add backend class, config schema section, pipeline factory construction, `SceneGraphService.generate` branch, dashboard controls, config reload rules.
- Change graph merge semantics: `SceneGraph.__add__` or `SceneGraphService.generate`.
- Change VLM filtering: `VLMSceneGraphGenerator.filter_hallucinated_relations`.
- Change rule vocabulary or rule types: `rules_backend.py` and `SGGRule` config.
- Change RelTR mapping/unary conversion: `reltr_backend.py`.
- Change SoM visual style or masks: `som.py` and visualization config/dashboard.
- Change robot attributes injected into graph: memory social extraction and `enhance_scene_graph_with_robot_data`.
