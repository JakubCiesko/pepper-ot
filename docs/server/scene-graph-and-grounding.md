# Scene Graph and Grounding

## Files Covered

- `app/inference/scene_graph/service.py`
- `app/inference/scene_graph/rules_backend.py`
- `app/inference/scene_graph/vlm_backend.py`
- `app/inference/scene_graph/reltr_backend.py`
- `app/inference/scene_graph/reltr_predictor.py`
- `app/inference/scene_graph/som.py`
- `app/schemas/scene.py`
- `server/ontology/scene_generation_ontology.yaml`
- `server/prompts/vlm_*.txt`

## Purpose

Scene graph generation converts detected/tracked objects into explicit relations and attributes that are better suited for grounded dialogue than raw detections alone.

The code supports four modes:
- `rules`
- `vlm`
- `reltr`
- `hybrid`

## `SceneGraphService`

This is the dispatcher.

### Inputs

- detections with stable `object_id`
- optional SoM image
- optional raw image
- optional caption text
- optional current `SceneState`

### Mode behavior

- `rules`: only rule backend
- `reltr`: only RelTR backend
- `vlm`: only VLM backend
- `hybrid`: VLM + rules + RelTR, then merged

### Robot enhancement step

After backend generation, `enhance_scene_graph_with_robot_data()` adds self-attribute style edges derived from current memory object attributes.

That means scene graph output can include robot/social metadata already attached to tracked objects, not only visual relations.

## Rule backend

Implemented in `rules_backend.py`.

### Input requirements

- detections must already have `object_id`
- optional raw image can be supplied for color inference

### Rule types supported

- spatial
- directional
- overlap
- containment
- label_pair

### Helper geometry functions

- bbox center
- IoU
- inside ratio
- threshold range check

### Color inference

The rule backend also derives color-like self-attributes from crops using palette extraction and HSV bucket mapping.

Examples of attribute outputs:
- `is_red`
- `is_blue`
- `is_white`
- `is_gray`
- `is_brown`

This is an important detail: the rules backend is not only geometric. It also enriches object attributes from crop appearance.

## VLM backend

Implemented in `vlm_backend.py`.

Responsibilities:
- render prompt with ontology and object references
- call VLM client
- request structured output if configured
- parse into scene graph form

Configuration inputs:
- provider/model/base_url/api key env
- system/user prompt text or paths
- ontology predicates/objects
- structured output mode and strictness
- local VLM hint strategy

## RelTR backend

Files:
- `reltr_backend.py`
- `reltr_predictor.py`

Responsibilities:
- load RelTR checkpoint
- run relation transformer prediction
- match predicted boxes to tracked detections using IoU threshold
- emit graph edges keyed by tracked object IDs

Key config fields:
- enabled
- checkpoint path
- device
- score threshold
- top-k
- IoU match threshold

## SoM painter

Implemented in `som.py`.

Purpose:
- render Set-of-Mark overlays so VLMs can refer to marked object regions explicitly

Inputs typically include:
- image array
- detection list
- visualization toggles such as bbox/polygon/labels/mask

This module is especially important when debugging VLM grounding quality. If grounding is weak, inspect the painted image before changing prompts.

## Graph Data Model

Core classes in `inference/types.py` and `schemas/scene.py`:
- `SceneGraphEdge`
- `SceneGraph`
- `SceneGraphRelation`
- `SceneGraphStructuredResponse`
- `Relationship`

## Merge Semantics in Hybrid Mode

Hybrid mode literally merges graph outputs from multiple backends.

This gives coverage but also means duplicate or conflicting relations are possible unless later logic collapses them.

If you observe noisy graph output, inspect merge behavior and downstream memory deduplication, not just individual backend quality.

## Safe Tweak Points

- rule thresholds
- ontology predicates and object vocabulary
- VLM prompts
- structured output mode
- RelTR threshold/top-k/IoU mapping
- SoM overlay style

## Risky Tweak Points

- relation label naming conventions
- tracked ID to relation mapping
- graph serialization contract expected by memory and dashboard
