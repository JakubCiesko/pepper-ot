# Chapter 2 Bibliography Guide

This file maps the strongest literature to the current thesis structure and codebase. The focus is not on collecting every vaguely related paper. The goal is to identify the papers that actually strengthen the argument of the thesis, explain the implemented architecture, and provide defensible comparison points.

## How To Use This File

- Use the `Priority` field to decide what to read first.
- Use `Primary citation` papers as the main references in the thesis text.
- Use `Supporting citation` papers when you need breadth, contrast, or historical context.
- Use the `What to use it for` note to decide where the paper belongs in the chapter and what claim it should support.

## Chapter 2.1 Social Robotics and Human-Robot Interaction

### 1. `kim_understanding_2024`
- Title: *Understanding Large-Language Model (LLM)-powered Human-Robot Interaction*
- Venue/year: arXiv, 2024
- Priority: Highest
- Why it should be used:
  - It is one of the clearest HRI-specific overviews of what changes when LLMs are introduced into robot interaction.
  - It is directly useful for framing user expectations, embodiment, trust, and interaction design implications.
- What to use it for:
  - Introduction to LLM-enabled HRI.
  - Motivation for grounded interaction instead of generic chatbot behavior.
  - Evaluation discussion for user-facing interaction quality.

### 2. `atuhurra_leveraging_2024`
- Title: *Leveraging Large Language Models in Human-Robot Interaction: A Critical Analysis of Potential and Pitfalls*
- Venue/year: arXiv, 2024
- Priority: High
- Why it should be used:
  - It is broader and more cautionary than Kim et al.
  - It helps justify why grounding, transparency, and safety constraints matter in a social robot setting.
- What to use it for:
  - Benefits and risks of LLM/VLM integration.
  - Discussion of trust, bias, safety, and responsible deployment.

### 3. `grassi_grounding_2024`
- Title: *Grounding Conversational Robots on Vision Through Dense Captioning and Large Language Models*
- Venue/year: ICRA, 2024
- Priority: Highest
- Why it should be used:
  - This is the closest directly relevant prior system-level reference for Pepper-like visually grounded dialogue.
  - It provides an explicit comparison point for your own architecture.
- What to use it for:
  - Grounded interaction in HRI.
  - Comparison against caption-only grounding versus persistent scene-memory grounding.
  - Evaluation discussion for Pepper-based interaction.

## Chapter 2.1.2 Pepper Robot and the NAOqi Ecosystem

### 4. `softbank_python_sdk_2026`
- Title: *Python SDK*
- Source/year: Aldebaran / SoftBank Robotics documentation, accessed 2026
- Priority: High
- Why it should be used:
  - It documents the Python 2.7 SDK and the NAOqi programming stack actually used by the client implementation.
- What to use it for:
  - Explaining why the robot-side stack remains Python 2 based.
  - Describing the Pepper integration constraints.

### 5. `softbank_alspeechrecognition_2026`
- Title: *ALSpeechRecognition*
- Source/year: Aldebaran / SoftBank Robotics documentation, accessed 2026
- Priority: Medium
- Why it should be used:
  - It gives an authoritative description of the robot-native speech recognition capability and supported-language setup.
- What to use it for:
  - Background on robot-native speech services.
  - Motivation for translation-mediated or external dialogue processing.

### 6. `softbank_peppers_tablet_2026`
- Title: *Using Pepper's Tablet*
- Source/year: Aldebaran / SoftBank Robotics documentation, accessed 2026
- Priority: Medium
- Why it should be used:
  - It supports the client-side multimodal output path used in the implementation.
- What to use it for:
  - Explaining tablet-based visual feedback as part of the interaction stack.

### 7. `reyes_near_2018`
- Title: *Near Real-Time Object Recognition for Pepper based on Deep Neural Networks Running on a Backpack*
- Venue/year: IberSPEECH / arXiv mirror, 2018
- Priority: Medium
- Why it should be used:
  - It is a good older reference showing that Pepper's onboard compute is insufficient for modern vision workloads and that external compute is a recurring engineering pattern.
- What to use it for:
  - Motivation for server-side perception.
  - Historical background on Pepper visual augmentation.

## Chapter 2.2 Dialogue Systems for Robots

### 8. `chen_survey_2017`
- Title: *A Survey on Dialogue Systems: Recent Advances and New Frontiers*
- Venue/year: ACM SIGKDD Explorations, 2017
- Priority: High
- Why it should be used:
  - It gives a clean baseline taxonomy of dialogue systems before moving to robotics-specific and LLM-based systems.
- What to use it for:
  - Classical dialogue management background.
  - Distinguishing task-oriented and open-domain dialogue paradigms.

### 9. `lewis_retrieval-augmented_2020`
- Title: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Venue/year: NeurIPS, 2020
- Priority: High
- Why it should be used:
  - This is the standard reference for RAG.
  - Even though the implemented system is not a full vector-RAG architecture, the paper is useful for framing prompt-time grounding via external memory.
- What to use it for:
  - Conversation memory and context injection.
  - Positioning the thesis approach as structured scene-context grounding.

### 10. `schick_toolformer_2023`
- Title: *Toolformer: Language Models Can Teach Themselves to Use Tools*
- Venue/year: arXiv, 2023
- Priority: Medium
- Why it should be used:
  - It is useful when arguing that external tools and stateful context can complement a base language model.
- What to use it for:
  - Conceptual framing of external-state interaction.
  - Support for tool-augmented prompting.

### 11. `gou_critic_2023`
- Title: *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing*
- Venue/year: ICLR / arXiv, 2023-2024
- Priority: Medium
- Why it should be used:
  - It reinforces the claim that external feedback improves reliability.
  - It is especially useful for discussing hallucination mitigation and prompt-time validation.
- What to use it for:
  - Reliability discussion.
  - Motivation for grounding and structured outputs.

## Chapter 2.3 Visual Perception for Social Robots

### 12. `zhao_detrs_2024`
- Title: *DETRs Beat YOLOs on Real-time Object Detection*
- Venue/year: CVPR / arXiv, 2024
- Priority: High
- Why it should be used:
  - It is the core reference for RT-DETR, which is directly relevant to the implemented detector stack.
- What to use it for:
  - Transformer-based real-time detection.
  - Explaining the trade-off between accuracy and latency.

### 13. `liu_grounding_2024`
- Title: *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*
- Venue/year: ECCV / arXiv, 2023-2024
- Priority: High
- Why it should be used:
  - It is one of the strongest references for open-set and text-conditioned object detection.
- What to use it for:
  - Open-vocabulary detection.
  - Grounded language-conditioned perception.

### 14. `minderer_simple_2022`
- Title: *Simple Open-Vocabulary Object Detection with Vision Transformers*
- Venue/year: ECCV / arXiv, 2022
- Priority: High
- Why it should be used:
  - It is the canonical OWL-ViT reference.
  - It supports the discussion of open-vocabulary detection as a useful alternative to fixed-label detectors.
- What to use it for:
  - Comparison between closed-vocabulary and open-vocabulary detection backends.

### 15. `bewley_simple_2016`
- Title: *Simple Online and Realtime Tracking*
- Venue/year: ICIP, 2016
- Priority: High
- Why it should be used:
  - It is the standard minimal MOT baseline.
- What to use it for:
  - Tracking-by-detection background.
  - Hungarian matching and pragmatic real-time tracking discussion.

### 16. `wojke_simple_2017`
- Title: *Simple Online and Realtime Tracking with a Deep Association Metric*
- Venue/year: arXiv, 2017
- Priority: Highest
- Why it should be used:
  - It is the key appearance-based extension of SORT and is the closest conceptual reference for your implementation's appearance-plus-geometry association.
- What to use it for:
  - Re-identification and identity persistence.
  - Justifying appearance embeddings for conversational grounding.

### 17. `zhang_bytetrack_2022`
- Title: *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*
- Venue/year: ECCV, 2022
- Priority: High
- Why it should be used:
  - It is one of the strongest modern MOT references and a useful contrast to simpler trackers.
- What to use it for:
  - Showing what a stronger modern MOT baseline looks like.
  - Clarifying that your implementation is not a full ByteTrack reproduction.

## Chapter 2.4 Semantic Scene Understanding

### 18. `johnson_image_2015`
- Title: *Image Retrieval using Scene Graphs*
- Venue/year: CVPR, 2015
- Priority: Medium
- Why it should be used:
  - It is a foundational scene-graph reference that helps introduce the representation itself.
- What to use it for:
  - Defining scene graphs, object attributes, and relations.

### 19. `zhu_scene_2022`
- Title: *Scene Graph Generation: A Comprehensive Survey*
- Venue/year: arXiv survey, 2022
- Priority: Highest
- Why it should be used:
  - It is the best general survey for the scene-graph chapter.
  - It gives the taxonomy you need for rule-based, learned, and hybrid SGG positioning.
- What to use it for:
  - Scene graph generation overview.
  - Evaluation metrics and methodological taxonomy.

### 20. `agarwal_visual_2020`
- Title: *Visual Relationship Detection using Scene Graphs: A Survey*
- Venue/year: arXiv survey, 2020
- Priority: Medium
- Why it should be used:
  - It is useful as a supporting survey, especially for relationship-centric framing.
- What to use it for:
  - Supporting citations in the scene graph section.

### 21. `cong_reltr_2023`
- Title: *RelTR: Relation Transformer for Scene Graph Generation*
- Venue/year: journal / arXiv, 2022-2023
- Priority: High
- Why it should be used:
  - It is a clean reference for direct relation prediction from vision features.
- What to use it for:
  - The optional learned relation-prediction path in your server architecture.

### 22. `yang_set-of-mark_2023`
- Title: *Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V*
- Venue/year: arXiv, 2023
- Priority: Highest
- Why it should be used:
  - It is directly relevant to your SoM-based marked-image prompting path.
  - It is one of the rare papers that aligns very closely with an implemented subsystem.
- What to use it for:
  - Grounded prompting in VLM-based relation generation.
  - The marked-object reference mechanism in the thesis system.

### 23. `kirillov_segment_2023`
- Title: *Segment Anything*
- Venue/year: arXiv / ICCV, 2023
- Priority: Medium
- Why it should be used:
  - It supports the general idea of broad, promptable segmentation infrastructure behind marked-region prompting.
- What to use it for:
  - Background paragraph on mask-based visual grounding if you discuss segmentation-assisted SoM generation.

## Chapter 2.4.2 Neuro-Symbolic and Hybrid Reasoning

### 24. `rosinol_3d_2020`
- Title: *3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans*
- Venue/year: RSS, 2020
- Priority: High
- Why it should be used:
  - It is a strong robotics scene-graph reference with explicit actionability and dynamic entities.
- What to use it for:
  - Positioning scene graphs as robot-world models rather than image-only structures.

### 25. `hughes_hydra_2022`
- Title: *Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization*
- Venue/year: RSS, 2022
- Priority: Highest
- Why it should be used:
  - It is one of the best robotics references for real-time incremental scene graph construction.
- What to use it for:
  - Positioning your scene memory as a lightweight robot-world representation.
  - Hybrid fast perception plus slower semantic enrichment discussion.

### 26. `hughes_foundations_2024`
- Title: *Foundations of Spatial Perception for Robotics: Hierarchical Representations and Real-Time Systems*
- Venue/year: IJRR, 2024
- Priority: Highest
- Why it should be used:
  - It provides the strongest conceptual argument for hierarchical, persistent, actionable spatial representations in robotics.
- What to use it for:
  - Scene memory as a dynamic world model.
  - Motivation for structured intermediate representations between perception and dialogue.

### 27. `mascaro_scene_2025`
- Title: *Scene Representations for Robotic Spatial Perception*
- Venue/year: Annual Review of Control, Robotics, and Autonomous Systems, 2025
- Priority: Highest
- Why it should be used:
  - It is the best recent high-level survey for robotic spatial representations.
- What to use it for:
  - High-level framing of metric, semantic, and metric-semantic-topological representations.
  - Discussion of how your system fits within dynamic scene modeling.

## Chapter 2.5 Large Multimodal Models for Grounded Dialogue

### 28. `driess_palm-e_2023`
- Title: *PaLM-E: An Embodied Multimodal Language Model*
- Venue/year: arXiv, 2023
- Priority: Highest
- Why it should be used:
  - It is one of the most important embodied multimodal references.
  - It helps connect language, perception, and robotics in a single conceptual frame.
- What to use it for:
  - The broader motivation for multimodal grounded dialogue in robotics.

### 29. `peng_kosmos-2_2023`
- Title: *Kosmos-2: Grounding Multimodal Large Language Models to the World*
- Venue/year: arXiv, 2023
- Priority: High
- Why it should be used:
  - It is directly relevant to grounded object references and phrase grounding.
- What to use it for:
  - Grounded VLM prompting.
  - Comparison with structured scene-memory grounding.

### 30. `dai_instructblip_2023`
- Title: *InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning*
- Venue/year: NeurIPS / arXiv, 2023
- Priority: High
- Why it should be used:
  - It is a strong reference for instruction-tuned VLMs used in question answering and scene understanding.
- What to use it for:
  - VLM background.
  - Direct vision-chat mode and prompt-based visual question answering.

### 31. `liu_visual_2023`
- Title: *Visual Instruction Tuning*
- Venue/year: NeurIPS / arXiv, 2023
- Priority: High
- Why it should be used:
  - It is the canonical LLaVA reference.
- What to use it for:
  - Multimodal chat framing.
  - Open-model VLM background.

## Chapter 2.6 Evaluation of Grounded Multimodal Systems

### 32. Reuse these as primary evaluation anchors
- `kim_understanding_2024`
- `grassi_grounding_2024`
- `zhu_scene_2022`

### Why these should be reused
- `kim_understanding_2024` helps define user-facing and interaction-facing evaluation concerns in HRI.
- `grassi_grounding_2024` is the closest system-level comparison point for Pepper-based grounded dialogue.
- `zhu_scene_2022` helps justify scene-graph evaluation metrics and relation-level analysis.

## Minimum Reading Order

1. `grassi_grounding_2024`
2. `kim_understanding_2024`
3. `mascaro_scene_2025`
4. `hughes_foundations_2024`
5. `zhu_scene_2022`
6. `yang_set-of-mark_2023`
7. `zhao_detrs_2024`
8. `liu_grounding_2024`
9. `wojke_simple_2017`
10. `zhang_bytetrack_2022`
11. `driess_palm-e_2023`
12. `peng_kosmos-2_2023`
13. `dai_instructblip_2023`
14. `lewis_retrieval-augmented_2020`

## Recommended Thesis Use Pattern

- Chapter 1:
  - `kim_understanding_2024`
  - `grassi_grounding_2024`
  - `mascaro_scene_2025`
- Chapter 2:
  - Most of the entries in this file.
- Chapter 4:
  - `grassi_grounding_2024`
  - `hughes_foundations_2024`
  - `mascaro_scene_2025`
- Chapter 6:
  - `zhao_detrs_2024`
  - `liu_grounding_2024`
  - `minderer_simple_2022`
  - `wojke_simple_2017`
  - `zhang_bytetrack_2022`
  - `yang_set-of-mark_2023`
  - `cong_reltr_2023`
- Chapter 7:
  - `driess_palm-e_2023`
  - `peng_kosmos-2_2023`
  - `dai_instructblip_2023`
  - `liu_visual_2023`
  - `lewis_retrieval-augmented_2020`
- Chapter 8:
  - `zhu_scene_2022`
  - `kim_understanding_2024`
  - `grassi_grounding_2024`
