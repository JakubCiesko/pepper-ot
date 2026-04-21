# Pepper Robot Client Documentation

This directory documents the current robot-side client under `client/`. It is the entry point for understanding how Pepper listens to QiChat grammar, captures camera frames and robot metadata, talks to the server, speaks answers, and renders local tablet memory views.

The client is a NAOqi Python 2 application packaged as the `pepper-grounded-client` robot app. Its central service is `PepperGroundedClient`, which is called by QiChat grammar and by the tablet JavaScript page. The service delegates long-running actions to `TurnManager`, sends HTTP requests through `PepperServerTransport`, stores short-lived dialogue and memory state in `SessionStore`, and uses robot adapters for camera, motion, people perception, speech, ALDialog, and ALTabletService.

## Start Here

Read these documents in this order if you are new to the robot client:

1. [`architecture-overview.md`](architecture-overview.md) explains the full client shape and the main runtime flows.
2. [`runtime-service-and-turns.md`](runtime-service-and-turns.md) explains `PepperGroundedClient`, `TurnManager`, asynchronous turn execution, busy handling, and bound Qi methods.
3. [`configuration.md`](configuration.md) explains `client_config.json`, defaults, normalization, runtime reload, and which settings affect each behavior.
4. [`server-communication.md`](server-communication.md) explains every server endpoint the client calls and how payloads are shaped.
5. [`perception-and-metadata.md`](perception-and-metadata.md) explains camera capture, scan planning, robot pose, people perception, social metadata, sonar, and `RobotMetadata` construction.
6. [`dialog-and-speech.md`](dialog-and-speech.md) explains QiChat topics, dynamic concepts, language resolution, TTS, acknowledgements, and fallback speech.
7. [`tablet-memory-ui.md`](tablet-memory-ui.md) explains the local tablet app, robot tablet adapter, fake tablet adapter, JavaScript bridge, and Q/A buttons.
8. [`session-state-and-dynamic-concepts.md`](session-state-and-dynamic-concepts.md) explains `SessionStore`, remembered labels, cached Q/A, concept refresh, and memory display payloads.
9. [`packaging-deployment-and-development.md`](packaging-deployment-and-development.md) explains manifest/PML packaging, Makefile deployment, local virtual robot mode, requirements, and generated files.
10. [`troubleshooting-and-extension-guide.md`](troubleshooting-and-extension-guide.md) maps common failures and feature changes to exact files.
11. [`file-inventory.md`](file-inventory.md) is a file-by-file locator for all client-side source, package, tablet, grammar, example, and support files.

## Current Core Flow

The normal spoken interaction path is:

1. QiChat recognizes a rule in `client/app/pepper-grounded-client/*.top`.
2. The rule calls a bound method on `PepperGroundedClient`, for example `look("en")`, `scan("cs")`, `askAboutObject("en", $1)`, or `showMemory("cs")`.
3. `PepperGroundedClient` forwards the action to `TurnManager` unless the call is a simple immediate helper.
4. `TurnManager` starts one daemon thread per accepted turn and rejects overlapping turns with a localized busy message.
5. The turn captures frames if needed, builds metadata from robot context, calls the server through `PepperServerTransport`, updates `SessionStore`, refreshes dynamic concepts when configured, and speaks or displays the result.
6. Server-side memory and Q/A data can be shown on the tablet through the local app under `/apps/pepper-grounded-client/`.

## Server Cross-References

The robot client depends on these server-side contracts:

- [`../server/api-reference.md`](../server/api-reference.md) documents `/api/v1/caption`, `/api/v1/detect`, `/api/v1/detect/panorama`, `/api/v1/chat`, `/api/v1/memory/summary`, `/api/v1/memory/reset`, and `/api/v1/chat/pregenerate_qa`.
- [`../server/data-models-and-contracts.md`](../server/data-models-and-contracts.md) documents `RobotMetadata`, `PersonMetadata`, social people payloads, chat request/response schemas, and memory summary structures.
- [`../server/detection-tracking-and-fusion.md`](../server/detection-tracking-and-fusion.md) explains how the server uses Pepper geometry and social metadata.
- [`../server/orchestration-and-conversations.md`](../server/orchestration-and-conversations.md) explains chat modes, object chat, conversation IDs, and Q/A pool behavior.
- [`../server/scene-memory-and-state.md`](../server/scene-memory-and-state.md) explains scene memory, memory summaries, graph SVG generation, and object crops.

## Important Current Design Facts

- The client is written for NAOqi Python 2. Several modules intentionally use `unicode`, Python 2 imports, and old-style classes.
- The same code is also run locally in a virtual robot/dev environment. That is why fake camera and fake tablet adapters exist.
- The tablet never opens remote dashboards. It serves and displays a local robot-hosted `html/index.html` page and pushes data into it through JavaScript.
- Dynamic concepts are populated from server memory summary data. They are not hard-coded object lists.
- Q/A button clicks on the tablet call back into the robot service through `qi.js` and reuse `answerCachedQuestion`.
- The server is the source of visual truth. The client captures images and robot metadata, but detection, captioning, scene graph generation, memory, chat, and Q/A generation happen server-side.
- `chat_id` is stored locally in `SessionStore` and passed back to the server so normal chat, object chat, and fallback cached-question chat can share a conversation.

## Documentation Maintenance Rule

When changing the client, update the dedicated document for that subsystem and [`file-inventory.md`](file-inventory.md). If the change affects server payloads or endpoint behavior, also update the corresponding server document under [`../server`](../server/index.md).
