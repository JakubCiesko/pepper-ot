# Theory And Model Rationale

This page is a compact source of truth for:

1. the background theory behind this project,
2. the concrete models and methods implemented in code,
3. the papers/ideas those choices come from,
4. why each choice is a good fit for this Pepper-based scene-aware dialogue system.

It complements:

- [System Description](system_description.md)
- [Server Docs Index](server/index.md)
- [Robot Client Docs Index](robot-client/index.md)

---

## 1) System Goal And Constraints

The system is built to let Pepper:

- perceive a real scene,
- keep persistent scene memory across captures/scans,
- answer grounded questions about what it currently perceives,
- stay responsive in interaction.

Primary engineering constraints that drive architecture:

- Pepper hardware cannot run modern VLM/LLM + detection stacks locally at useful latency.
- Robot interaction must stay robust and deterministic (speech/dialog routing must not collapse when models fail).
- Visual grounding must be explicit (object IDs and relations), not just free-form captioning.

This is why the design is split into:

- **Robot client** (embodiment, NAOqi sensing, dialog triggers, tablet UI),
- **Server** (heavy perception, memory, scene graphs, LLM/VLM generation).

Core code anchors:

- Server runtime orchestration: `server/app/inference/pipeline.py`
- Server assembly: `server/app/core/pipeline_factory.py`
- Client runtime entrypoint: `client/app/scripts/peppergroundedclient.py`
- Client turn orchestration: `client/app/scripts/pepper_client/core/turn_manager.py`

---

## 2) Theory Pillars Behind The Design

## 2.1 Grounded HRI Instead Of Pure Chat

Theory:

- LLM fluency alone is not enough in social robotics.
- Users infer “situational understanding” from fluent speech, so responses must be tied to real perception.

Fit:

- The system injects scene memory/context into chat paths and uses object/scene graph IDs for grounding.
- Client keeps deterministic trigger/routing behavior via QiChat + explicit service methods.

Related citations used in thesis:

- `kim_understanding_2024`
- `atuhurra_leveraging_2024`
- `softbank_peppers_tablet_2026`
- `softbank_alspeechrecognition_2026`

---

## 2.2 Detection Theory: Closed-Set Speed + Open-Vocabulary Flexibility

Theory:

- CNN/YOLO-style detectors are fast but closed-set.
- DETR-family reframes detection as set prediction (better compositional behavior, often better quality trade-offs).
- Open-vocabulary detectors support promptable domain terms without full retraining.

Implemented in code:

- Detector registry and wrappers: `server/app/inference/detection/detectors.py`
- Detection service orchestration: `server/app/inference/detection/service.py`
- Config schema: `server/app/schemas/config.py` (`DetectionConfig`)

Supported backends:

- `rf_detr`, `rt_detr`, `yolo`, `owl_v2`

Default runtime choice (current config):

- `rf_detr` in `server/config.yaml`

Why fit:

- RF-DETR gives strong real-time quality for server-side robotics inference.
- Optional OWLv2 keeps a path for open-vocabulary experimentation in labs.
- Backend abstraction avoids hard lock-in and lets you tune per deployment.

Key citation lineage:

- `vaswani2023attentionneed`
- `dosovitskiy2021imageworth16x16words`
- `zhao2024detrsbeatyolosrealtime`
- `robinson2026rfdetrneuralarchitecturesearch`
- `minderer_simple_2022`
- `minderer2024scalingopenvocabularyobjectdetection`
- `pmlr-v139-radford21a`
- `lin2015microsoftcococommonobjects`
- `liu_grounding_2024`
- `feng_vision-language_2025`

---

## 2.3 Tracking Theory: Re-Identification Over Discrete Viewpoint Jumps

Theory:

- Classic MOT assumptions (small frame-to-frame displacement) break in Pepper scan behavior (left/center/right captures, discontinuous views).
- Appearance embeddings + geometry weighting are more robust than IoU-only continuity for this scan pattern.

Implemented in code:

- Embedding extraction: `server/app/inference/tracking/embeddings.py`
- Association logic: `server/app/inference/tracking/associator.py`
- Memory update/fusion path: `server/app/inference/memory/state_store/`
- ReID model default from config: `nvidia/C-RADIOv4-SO400M`

Why fit:

- Preserves IDs across viewpoint shifts.
- Enables persistent memory graph and follow-up dialogue about previously seen objects.

Citation lineage:

- `wojke_simple_2017` (Deep SORT context)
- `zhang_bytetrack_2022`
- `ranzinger2026cradiov4techreport`

---

## 2.4 Scene Graph Theory: Hybrid Neuro-Symbolic Representation

Theory:

- Bounding boxes alone are weak for language reasoning.
- Scene graphs provide structured object-relation memory.
- Rules give deterministic spatial relations; learned models provide richer semantics.
- Hybrid merge avoids relying on one fragile source.

Implemented in code:

- Scene graph orchestrator: `server/app/inference/scene_graph/service.py`
- Rule backend: `server/app/inference/scene_graph/rule_backend.py`
- RelTR backend: `server/app/inference/scene_graph/reltr_backend.py`
- VLM backend: `server/app/inference/scene_graph/vlm_backend.py`
- SoM image rendering/masks: `server/app/inference/scene_graph/som.py`
- Core schema: `server/app/schemas/scene.py`

Current architectural behavior:

- Merge strategy is effectively union + dedup of enabled backends.
- Optional backend parallelization controlled by `scene_graph.parallel_execution`.

Why fit:

- Deterministic rules prevent catastrophic drift on basic spatial facts.
- RelTR gives learned relation priors from vision.
- VLM backend expands semantic relation coverage.
- SoM/object IDs enforce grounded references in generated relations.

Citation lineage:

- `johnson_image_2015`
- `cong_reltr_2023`
- `zhu_scene_2022`
- `wang_indvissgg_2025`
- `mascaro_scene_2025`
- `hughes_foundations_2024`
- `yang_set-of-mark_2023`
- `kirillov_segment_2023`
- `tam_let_2024`

---

## 2.5 Multimodal Generation Theory: Prompt-Conditioned, Structured, Grounded

Theory:

- LLM/VLM outputs must be constrained for machine use (schema/structure), but overly strict format demands can hurt reasoning quality.
- Practical systems need multiple structured-output strategies because providers differ.

Implemented in code:

- Text provider abstraction: `server/app/providers/llm/client.py`
- Text providers: `server/app/providers/llm/`
- VLM provider factory/clients: `server/app/providers/vlm/`
- Structured mode resolver: `server/app/providers/common/io.py`
- Provider capability map: `server/app/providers/common/utils.py`
- Dashboard control wiring: `server/app/static/js/dashboard/features/config/index.js`

Structured output modes present in config:

- `provider_native`, `parse_output`, `instructor`

Why fit:

- Same orchestration works across Gemini/OpenAI/OpenAI-compatible/local HF stacks.
- Allows safe degradation when provider-native schema enforcement is unavailable.

Citation lineage:

- `ouyang2022traininglanguagemodelsfollow`
- `liu_visual_2023`
- `dai_instructblip_2023`
- `driess_palm-e_2023`
- `peng_kosmos-2_2023`
- `shazeer2017outrageouslylargeneuralnetworks`

---

## 2.6 Grounded Dialogue Memory Theory: RAG-Like To CAG-Like

Theory:

- Generic free-form chat is weak for factual scene QA.
- Injecting compact, structured scene state can replace expensive retrieval in this domain.

Implemented in code:

- Chat orchestration: `server/app/orchestration/services/chat.py`
- Chat API modes: `server/app/api/v1/chat.py`
- Memory read/summary: `server/app/api/v1/memory.py`
- Inference memory core: `server/app/inference/memory/scene_memory.py`
- Scene memory store mixins: `server/app/inference/memory/state_store/`

Why fit:

- Fast, deterministic context injection from live scene memory.
- Small/medium models can answer grounded questions better with explicit context than with pure parametric recall.

Citation lineage:

- `lewis_retrieval-augmented_2020`
- `noauthor_multi-vector_nodate`
- `Chan_2025`
- `fazlollahtabar_human-robot_2025`

---

## 2.7 Dialog Management Theory: Deterministic Triggering + Generative Backends

Theory:

- Pure intent trees are brittle for open-ended visual QA.
- Pure generative routing is brittle for robot action triggers.
- A hybrid stack is preferred: deterministic rule triggers for action routing, generative models for grounded content.

Implemented in code:

- QiChat topics:
  - `client/app/pepper-grounded-client/pepper-grounded-client_enu.top`
  - `client/app/pepper-grounded-client/pepper-grounded-client_czc.top`
- Dialog adapter/dynamic concepts:
  - `client/app/scripts/pepper_client/interaction/dialog_adapter.py`
- Runtime service methods exposed to Qi:
  - `client/app/scripts/peppergroundedclient.py`

Why fit:

- Keeps robot UX predictable.
- Supports rich open QA without hand-authoring every scene-specific branch.

Citation lineage:

- `bocklisch_rasa_2017` (intent/dialog baseline context)
- plus Pepper platform docs (`softbank_*` keys above)

---

## 3) Model Inventory In Current Codebase

This is the concrete “what exists now” inventory (server + client).

## 3.1 Server-Side Models/Algorithms

1. Object detection:
- Backends: RF-DETR, RT-DETR, YOLO, OWLv2
- Code: `server/app/inference/detection/detectors.py`
- Default: RF-DETR (`server/config.yaml`)

2. ReID embedding model:
- Default: `nvidia/C-RADIOv4-SO400M`
- Code: `server/app/inference/tracking/embeddings.py`

3. Scene graph learned backend:
- RelTR
- Code: `server/app/inference/scene_graph/reltr_backend.py`, `server/app/inference/scene_graph/reltr_predictor.py`

4. Scene graph rule backend:
- Deterministic geometry rules
- Code: `server/app/inference/scene_graph/rule_backend.py`

5. Scene graph VLM backend:
- Provider-driven, currently Gemini in config
- Code: `server/app/inference/scene_graph/vlm_backend.py`, `server/app/providers/vlm/`

6. Captioning:
- Local BLIP path + provider fallback abstraction
- Code: `server/app/providers/caption/client.py`

7. Chat / QA generation:
- LLM provider abstraction for chat and QA pairs
- Code: `server/app/orchestration/services/chat.py`, `server/app/providers/llm/`

8. Segmentation/SoM masks:
- GrabCut default, SAM optional
- Code: `server/app/inference/scene_graph/som.py`

9. Translation/localization support:
- Google translation service + lexicons
- Code: `server/app/providers/translation/`

## 3.2 Client-Side “Models” And Sensors

Client does not run heavy neural inference; it runs robot-native perception signals and orchestration.

1. Camera acquisition:
- `ALVideoDevice` adapter
- Code: `client/app/scripts/pepper_client/perception/camera_adapter.py`

2. People geometry:
- `ALPeoplePerception`
- Code: `client/app/scripts/pepper_client/perception/people_adapter.py`

3. Face and social cues:
- `ALFaceDetection`, `ALFaceCharacteristics`, `ALGazeAnalysis`,
  `ALEngagementZones`, `ALSittingPeopleDetection`, `ALWavingDetection`
- Code: `client/app/scripts/pepper_client/perception/face_adapter.py`, `social_adapter.py`

4. Robot pose and sonar:
- `ALMotion`, `ALSonar`
- Code: `pose_adapter.py`, `sonar_adapter.py`

5. Dialog + speech runtime:
- QiChat topics + ALDialog concept updates + ALAnimatedSpeech/ALTextToSpeech
- Code: `dialog_adapter.py`, `speech_adapter.py`, `.top` files

6. Tablet memory UI:
- Local robot-hosted web app and JS bridge
- Code: `client/app/html/index.html`, `client/app/html/js/*`, `tablet_adapter.py`

---

## 4) Why These Choices Are Fit For This Project

High-level rationale:

1. **Distributed architecture** matches Pepper compute limits while preserving interaction quality.
2. **Hybrid scene understanding** (rules + learned + VLM) gives better robustness than any single backend alone.
3. **Persistent memory with IDs** is necessary for multi-turn grounding across scans.
4. **Deterministic dialog triggers + generative answers** balances safety/control with conversational flexibility.
5. **Provider-abstraction + hot reload** is practical for experimentation-heavy thesis workflows.

In short: the stack is explicitly optimized for **grounded, real-time-ish HRI under hardware constraints**, not for benchmark-only single-image caption quality.

---

## 5) Citation Key Map Used In Thesis Background Chapter

This is the full key set currently cited in `thesis-source/chapters/ch2_bckg_related_work.tex`.

HRI and Pepper platform:

- `kim_understanding_2024`
- `atuhurra_leveraging_2024`
- `softbank_peppers_tablet_2026`
- `softbank_alspeechrecognition_2026`
- `reyes_near_2018`
- `trinquet_future_2025`
- `brad_retrieval-augmented_2025`
- `latif_physicsassistant_2024`
- `zabala_exploring_2025`
- `wang_target_2024`
- `becerra_improving_2024`

Embodied planning with LLMs:

- `ahn_as_2022`
- `liang_code_2023`
- `catalini_llms_nodate`

Detection and visual perception:

- `zhao_detrs_2024`
- `vaswani2023attentionneed`
- `lin2015microsoftcococommonobjects`
- `liu_grounding_2024`
- `minderer_simple_2022`
- `pmlr-v139-radford21a`
- `dosovitskiy2021imageworth16x16words`
- `zhao2024detrsbeatyolosrealtime`
- `robinson2026rfdetrneuralarchitecturesearch`
- `minderer2024scalingopenvocabularyobjectdetection`
- `feng_vision-language_2025`
- `wojke_simple_2017`
- `zhang_bytetrack_2022`
- `ranzinger2026cradiov4techreport`

Scene graph and grounding:

- `johnson_image_2015`
- `cong_reltr_2023`
- `mascaro_scene_2025`
- `hughes_foundations_2024`
- `zhu_scene_2022`
- `wang_indvissgg_2025`
- `tam_let_2024`
- `yang_set-of-mark_2023`
- `kirillov_segment_2023`

LLM/VLM and alignment:

- `ouyang2022traininglanguagemodelsfollow`
- `liu_visual_2023`
- `dai_instructblip_2023`
- `driess_palm-e_2023`
- `shazeer2017outrageouslylargeneuralnetworks`
- `peng_kosmos-2_2023`

Dialogue and retrieval framing:

- `bocklisch_rasa_2017`
- `lewis_retrieval-augmented_2020`
- `noauthor_multi-vector_nodate`
- `fazlollahtabar_human-robot_2025`
- `Chan_2025`

---

## 6) Where To Modify What (Fast Edit Map)

If you need to change theory-driven behavior quickly, start here:

1. Detection backend/model behavior:
- `server/app/inference/detection/`
- `server/config.yaml` → `detection`

2. Tracking identity and memory association:
- `server/app/inference/tracking/`
- `server/app/inference/memory/state_store/`

3. Scene graph strategy:
- `server/app/inference/scene_graph/service.py`
- `server/app/inference/scene_graph/rule_backend.py`
- `server/app/inference/scene_graph/reltr_backend.py`
- `server/app/inference/scene_graph/vlm_backend.py`

4. Chat grounding + QA generation:
- `server/app/orchestration/services/chat.py`
- `server/app/api/v1/chat.py`
- `server/config.yaml` → `chat`, `qa_generation`, `pipeline_controls`

5. Robot-side triggers and UX:
- `client/app/pepper-grounded-client/*.top`
- `client/app/scripts/pepper_client/core/turn_manager.py`
- `client/app/scripts/peppergroundedclient.py`

6. Tablet memory rendering:
- `client/app/html/index.html`
- `client/app/html/js/*`
- `client/app/scripts/pepper_client/interaction/tablet_adapter.py`

---

## 7) Notes On Scope

- This page documents **implemented architecture and explicit thesis background links**.
- Some cited works are conceptual framing, not one-to-one code implementations.
- The exact active runtime choices are controlled by `server/config.yaml` and `client/app/scripts/client_config.json`.

