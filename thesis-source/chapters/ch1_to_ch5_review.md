# Review of Chapters 1-5

Files reviewed:

- `thesis-source/chapters/ch1_introduction.tex`
- `thesis-source/chapters/ch2_rewrite.tex`
- `thesis-source/chapters/ch3_rewrite.tex`
- `thesis-source/chapters/ch4_rewrite.tex`
- `thesis-source/chapters/ch5_rewrite.tex`

## Short Verdict

This is a good thesis draft with a clear technical spine. It does not read like five unrelated chapters. The line is understandable:

1. Human-robot interaction needs grounded, situated dialogue.
2. Pepper cannot do modern multimodal perception and LLM reasoning locally.
3. The thesis therefore defines scene-aware dialogue as a practical system problem.
4. Chapter 2 establishes the required computer-vision, language-model, retrieval, and Pepper background.
5. Chapter 3 turns those assumptions into a distributed architecture.
6. Chapter 4 explains the server-side perception, memory, scene graph, and dialogue machinery.
7. Chapter 5 explains how the robot client turns that server machinery into embodied interaction.

The strongest chapters are Chapters 2, 4, and 5. Chapter 3 works as a bridge, but should be structured more explicitly. Chapter 1 is currently the weakest part because it still contains draft markers, some overclaims, and a slightly different level of precision than the rewritten chapters.

The thesis can become strong if Chapter 1 is aligned with Chapters 2-5 and if the claimed contributions are made more exact. As it stands, the work is credible, technically coherent, and defensible, but not yet clean enough to send as a final thesis draft.

## Overall Assessment

### What Works

The topic is coherent. Scene-aware dialogue on Pepper is a concrete enough thesis problem: it is not merely "LLM on robot", but a system for connecting robot interaction, perception, scene memory, and grounded answering.

The chapters mostly obey the ML-writing rules you wanted. Chapter 2 defines tasks and prerequisites, compares methods rather than only listing them, and explains why standard assumptions fail in this system. Chapters 3-5 then instantiate those assumptions in the actual implementation.

The text has a clear "your style" direction. It is direct, formal, explanatory, and fond of explicit definitions. It often uses the structure "what the method assumes, why that assumption fails here, and what workaround the thesis uses." This is especially visible in Chapter 2 and Chapter 4.

The server chapter is convincing. It presents the system as engineering under constraints rather than as a magical AI pipeline. It explains state, APIs, perception, metadata fusion, scene graph generation, memory, dialogue, worker control, and dashboard inspection in a way that matches a real system.

The client chapter is also convincing. It makes clear that the robot is not the main compute node. It is the embodied interface, the turn manager, and the source of robot-native metadata.

### What Does Not Work Yet

Chapter 1 has not caught up with the rewritten thesis. It introduces the right topic, but it is less precise than Chapters 2-5 and still has unresolved editing artifacts.

The contribution claims are sometimes stronger than the demonstrated system. In particular, "structured subgraph RAG", "edge-cloud", and "open-source evaluation framework" need either exact implementation evidence or softer wording.

Chapter 3 is conceptually good, but it lacks sectioning. Since Chapters 4 and 5 are well-sectioned, Chapter 3 currently feels more essay-like than architectural. This is fixable.

The thesis still needs the evaluation chapter to carry part of the burden. Chapters 1-5 describe a plausible and well-designed system, but they do not prove that it works. That is acceptable only if the later evaluation chapter clearly tests the claims made in Chapter 1.

## Narrative Line

The overall line is clear and mostly successful. The best version of the thesis story is:

> Pepper robots are socially useful but computationally limited. Modern LLM dialogue is fluent but ungrounded unless connected to perception. This thesis builds a distributed scene-aware dialogue system for Pepper. The robot captures interaction and native perception signals; the server builds object-, person-, and relation-level scene memory; the dialogue layer uses that memory to answer broad and targeted questions. The contribution is the integration architecture and the operational system, not a new detector, tracker, VLM, or LLM.

That story is present in the chapters, but Chapter 1 should state it this cleanly. Right now, Chapter 1 sometimes sounds like the thesis contributes a broader evaluation framework or a general edge-cloud architecture. Chapters 3-5 show something more specific and better: a practical off-board architecture for grounded dialogue with Pepper.

## Chapter-by-Chapter Review

### Chapter 1: Introduction

Chapter 1 has the correct ingredients: HRI motivation, Pepper, LLM grounding problem, scene-aware dialogue, objectives, research questions, contributions, and thesis structure.

However, it is the least polished chapter.

Concrete issues:

- `ch1_introduction.tex:16` says "recent work by Latif et al." but cites `\cite{sievers_humanoid_2025}`. This looks wrong. Either the authors or the citation key must be corrected.
- `ch1_introduction.tex:28` contains `[PAGE]`. This must be removed before any serious review.
- `ch1_introduction.tex:50` claims "structured subgraph Retrieval-Augmented Generation (RAG)". If the implementation truly performs graph-local retrieval around objects, keep it. If it only filters or formats scene memory, soften this to "object-focused structured retrieval" or "structured scene-memory retrieval".
- `ch1_introduction.tex:55` uses "distributed edge-cloud system architecture". The actual architecture reads more like "distributed client-server architecture with off-board computation". Use "edge-cloud" only if there is a real cloud/edge distinction in deployment.
- `ch1_introduction.tex:88` contains `[MIGHTDROP]`. This must be removed. More importantly, the "open-source evaluation framework for embodied AI" contribution sounds too broad unless the thesis later presents and evaluates that framework explicitly.

The introduction also needs to align more tightly with Chapter 2's formal problem setting. Chapter 2 defines scene-aware dialogue using scene state, observations, dialogue history, and context construction. Chapter 1 should preview that framing instead of defining the problem only in prose.

Recommended direction:

- Make Chapter 1 more precise and less promotional.
- State that the thesis contributes a system architecture and implementation for scene-aware dialogue on Pepper.
- Avoid implying that the thesis contributes new object detection, tracking, ReID, SGG, LLM, or VLM methods.
- Replace broad phrases like "robust interaction" with testable language such as "more inspectable", "more grounded in detected scene state", or "able to answer questions using stored scene observations".

### Chapter 2: Background and Related Work

Chapter 2 is strong. It does the job of a background and related work chapter properly.

The best part is that it does not only describe methods. It repeatedly explains their assumptions and why those assumptions matter for this thesis. For example, the distinction between continuous video tracking and discrete robot observations is exactly the kind of reasoning that makes the chapter relevant rather than encyclopedic.

The problem setting section is useful and should influence the rest of the thesis. It gives the whole work a formal center.

The chapter also follows the "academic ancestors vs academic siblings" distinction reasonably well. It gives background for object detection, tracking, ReID, grounding, SGG, language models, RAG, and Pepper, while also explaining how related systems differ from this thesis.

Potential weaknesses:

- It is dense. That is not necessarily bad for a thesis, but the later chapters should not repeat too much of it.
- Some recent citations and model claims should be checked carefully against the bibliography and final source list.
- The claims around GraphRAG and subgraph retrieval should match what the implementation actually does.

Overall, this chapter gives the thesis credibility. It is probably the chapter that most makes the work look academically serious.

### Chapter 3: System Architecture and Design

Chapter 3 works as the bridge from theory to implementation. It explains why the system is distributed, what the main components are, how turns move through the system, and why memory and inspection matter.

The figures are useful. They are simple, but they communicate the split between robot, server, worker, memory, and dashboard. For a thesis, functional figures are better than decorative figures.

Main issue:

- `ch3_rewrite.tex` has no explicit `\section{...}` structure after the chapter title. Chapters 4 and 5 are sectioned, so Chapter 3 feels less structured by comparison.

Recommended sections:

- `\section{Design Requirements}`
- `\section{Distributed System Boundary}`
- `\section{Interaction Modes and Data Flow}`
- `\section{Scene Memory and Context Construction}`
- `\section{Inspection and Failure Handling}`

The content can stay mostly the same, but sectioning would make it read more like a thesis architecture chapter and less like a long design essay.

Chapter 3 should also explicitly prepare the reader for Chapters 4 and 5. It already does this implicitly, but a final paragraph saying that Chapter 4 explains the off-board server and Chapter 5 explains the robot-side client would improve continuity.

### Chapter 4: Server-Side Implementation

Chapter 4 is the strongest implementation chapter. It explains the server as the thesis's main technical object.

What works:

- The chapter clearly separates API, state, perception, robot metadata, scene graph generation, memory, dialogue, workers, and dashboard.
- It explains why the server owns coordination and why the robot does not run heavy models locally.
- It handles limitations honestly. The text does not pretend that detections, captions, metadata, and scene graphs are always correct.
- It presents inspection and dashboard functionality as part of research reproducibility, not merely as a UI feature.

The chapter also has a good thesis tone. It is implementation-specific without becoming a code dump.

Potential improvements:

- When naming concrete models such as RF-DETR, C-RADIO, Gemini, and BLIP, make sure the wording says "in the configuration used for this thesis" if those choices are configurable.
- Be careful with the term "scene graph generation". If the system builds a practical graph by combining detections, metadata, depth estimates, and VLM relation proposals, say that. Do not imply it is a standard end-to-end SGG benchmark model unless it is.
- If the evaluation later measures latency, correctness, or failure modes, Chapter 4 should use the same terms that Chapter 6 will use.

Overall, Chapter 4 is convincing and should remain the largest of the three system chapters.

### Chapter 5: Robot-Side Implementation

Chapter 5 reads well after Chapter 4. It explains the robot client as the embodied interaction layer rather than a duplicate of the server.

What works:

- The chapter makes the service boundary clear.
- It explains QiChat grammar and interaction entry points.
- It explains turn management, visual turns, metadata collection, server communication, local state, speech, tablet output, and limitations.
- The limitations section is honest and useful.

The strongest part is the framing: Pepper is not presented as a general-purpose AI computer. It is presented as a constrained social interface that delegates heavy reasoning to the server.

Potential improvements:

- Some points repeat the same idea: heavy models run off-board, Pepper remains responsive, and the robot client is thin. This is correct, but the final edit could compress repeated statements.
- The chapter should end with a stronger bridge to evaluation: what exactly should Chapter 6 verify after this implementation description?

Overall, Chapter 5 is coherent and fits the thesis line.

## Style Consistency

Chapters 2-5 are mostly stylistically consistent. They use formal definitions, explicit assumptions, and careful compare/contrast reasoning. This matches the style extracted from the old thesis and the ML-paper rules.

Chapter 1 is stylistically close, but not fully aligned. It has more generic academic motivation and more promotional contribution language. It should be rewritten in the more precise style of Chapter 2.

The comments of the form `%TL;DR ...` are useful as drafting structure. Since they are LaTeX comments, they will not print. They are acceptable if you want to keep them for later editing. If the final source will be shared with supervisors, they are still fine, but make sure none of them contain informal language.

## Does It Read as One Thesis?

Mostly yes.

The chapters follow each other logically:

- Chapter 1 asks why grounded Pepper dialogue matters.
- Chapter 2 defines the necessary concepts and limitations.
- Chapter 3 explains the architecture chosen under those limitations.
- Chapter 4 explains the server that implements the perception, memory, and dialogue side.
- Chapter 5 explains the robot client that implements embodied interaction.

The biggest continuity gap is between Chapter 1 and Chapter 2. Chapter 2 is much more exact than Chapter 1. After reading Chapter 2, the reader understands the thesis better than after reading Chapter 1. That should not happen. Chapter 1 should give a cleaner preview of the exact same problem that Chapter 2 formalizes.

## Main Risks Before Submission

### Risk 1: Unresolved Draft Markers

The `[PAGE]` marker at `ch1_introduction.tex:28` and `[MIGHTDROP]` marker at `ch1_introduction.tex:88` are serious because they immediately signal an unfinished draft.

### Risk 2: Contribution Overclaim

The thesis should not overclaim. Based on Chapters 3-5, the contribution is an integrated system for scene-aware dialogue on Pepper, with practical scene memory and multimodal grounding. That is already enough.

Avoid claiming:

- A general edge-cloud architecture unless cloud deployment is central.
- A general open-source evaluation framework unless this is actually developed and evaluated.
- A new GraphRAG method unless graph retrieval is formalized and measured.
- A new scene graph generation method unless the graph construction is presented as a methodological contribution.

### Risk 3: Evaluation Burden

The first five chapters set up claims that must be tested later. The evaluation chapter should check at least:

- Whether scene-aware context improves answer grounding over a baseline without scene memory.
- Whether targeted object or relation questions retrieve the correct context.
- Whether the system latency remains acceptable for interaction.
- Where failures come from: detector, ReID, metadata, VLM relation extraction, memory, or LLM answer generation.
- Whether the robot-side interaction remains usable despite off-board processing.

If the evaluation does not test these points, Chapter 1's objectives and contributions should be softened.

### Risk 4: Label and Integration Cleanup

The rewrite files use labels:

- `chap:system_architecture_rewrite`
- `chap:server_side_rewrite`
- `chap:robot_side_rewrite`

That is fine while drafting. Before final integration, decide whether these become the canonical labels. If the main thesis or other chapters reference older labels, normalize them.

## Priority Fix List

1. Fix Chapter 1 before editing the other chapters.
2. Remove all draft markers from Chapter 1.
3. Correct the suspicious Latif/Sievers citation mismatch in Chapter 1.
4. Rewrite Chapter 1's contribution list so it exactly matches what Chapters 3-5 implement and what Chapter 6 can evaluate.
5. Add explicit sections to Chapter 3.
6. Add a stronger bridge from Chapter 5 to the evaluation chapter.
7. Check that "CAG", "RAG", "GraphRAG", "subgraph retrieval", and "scene graph generation" are used consistently with the actual implementation.
8. Compile the full thesis and check for broken references, duplicate labels, missing citations, and bad figure placement.
9. Do a final citation audit for Chapter 2 because it carries most of the literature burden.

## Final Judgement

This can be a good thesis. The core is not weak. The system is specific, the implementation chapters are credible, and the background chapter gives the work academic context. The main problem is not lack of substance. The main problem is alignment: Chapter 1 must be made as precise as Chapters 2-5, and the contribution claims must be constrained to what the implementation and evaluation actually support.

If those fixes are made, the thesis should read as a coherent system thesis about scene-aware dialogue on Pepper, rather than as a collection of AI-generated component descriptions.
