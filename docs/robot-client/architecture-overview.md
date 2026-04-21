# Robot Client Architecture Overview

The robot client is a packaged Pepper application that bridges human speech, robot sensors, server-side perception, and tablet display. It is not a perception system by itself. Its job is to collect local context from Pepper, call server APIs, maintain enough state for smooth dialogue, and expose a simple Qi service interface to QiChat and tablet JavaScript.

## Top-Level Components

### `client/app/scripts/peppergroundedclient.py`

This is the main NAOqi service module. It defines `PepperGroundedClient`, registers bound methods with Qi, initializes all adapters, loads config, and owns the long-lived objects used by the client.

It constructs:

- `ServiceCache` for lazy access to NAOqi services.
- `SessionStore` for chat ID, memory labels, recent responses, and cached Q/A state.
- `PepperServerTransport` for all HTTP calls to the server.
- `CameraAdapter` or `FakeCameraAdapter` for image capture.
- `PoseAdapter` for head and body pose.
- `FaceAdapter`, `PeopleAdapter`, `SocialAdapter`, and `SonarAdapter` for robot metadata.
- `SpeechAdapter` for `ALTextToSpeech` and `ALAnimatedSpeech`.
- `TabletAdapter` or `FakeTabletAdapter` for the memory tablet page.
- `DialogAdapter` for ALDialog dynamic concepts.
- `RobotContextCollector` for combined pose/people/social/sonar snapshots.
- `MetadataBuilder` for server-compatible metadata JSON.
- `TurnManager` for asynchronous turns.

### `client/app/scripts/pepper_client/core/turn_manager.py`

`TurnManager` is the operational center of the client. It owns the actual workflows for quick look, scan, general ask, object ask, cached Q/A, memory display, visual refresh, and memory reset.

The Qi service methods should stay thin. If a feature is a long-running robot action or server call, it belongs in `TurnManager` rather than directly inside `PepperGroundedClient`.

### `client/app/scripts/pepper_client/core/transport.py`

`PepperServerTransport` is the server API client. It builds multipart requests for images, JSON requests for chat/memory, applies timeouts, handles TLS verification, parses JSON, and converts low-level `requests` errors into client-specific exceptions.

### `client/app/scripts/pepper_client/core/session_store.py`

`SessionStore` is local process state. It is not persistent storage. It keeps:

- Current server `chat_id`.
- Last caption/detection/chat response.
- Last visual refresh timestamp.
- Last memory summary.
- Remembered object labels.
- Remembered attributes and relations.
- Cached generated questions and exact question-to-answer mapping.

### `client/app/scripts/pepper_client/perception/*`

The perception package does not run ML. It reads Pepper sensors and robot memory:

- Camera bytes from `ALVideoDevice`.
- Head/body pose from `ALMotion`.
- People positions from `ALPeoplePerception`.
- Social metadata from ALMemory keys exposed by face characteristics, gaze, engagement zones, sitting, and waving services.
- Sonar values from `ALSonar` and ALMemory.

### `client/app/scripts/pepper_client/interaction/*`

The interaction package owns user-facing robot interaction:

- `SpeechAdapter` speaks localized text and applies TTS language.
- `speech_policy.py` chooses runtime language and localized acknowledgement/fallback messages.
- `DialogAdapter` updates ALDialog dynamic concepts.
- `TabletAdapter` displays and updates the local memory web page.

### `client/app/pepper-grounded-client/*.top`

The QiChat topics define spoken grammar. The English topic is `pepper-grounded-client_enu.top`; the Czech topic is `pepper-grounded-client_czc.top`. Both call the same Python Qi service methods with different language parameters.

### `client/app/html/*`

The tablet web app is a local robot-hosted page. It receives memory payloads from Python and renders:

- Detected object labels and counts.
- Attributes.
- Relationships.
- Server-provided scene graph SVG.
- Pregenerated Q/A buttons.

## Runtime Object Graph

The active runtime graph is:

```text
QiChat / Tablet JS
        |
        v
PepperGroundedClient Qi service
        |
        v
TurnManager
        |
        +--> CameraAdapter / FakeCameraAdapter
        +--> RobotContextCollector
        |       +--> PoseAdapter
        |       +--> PeopleAdapter
        |       +--> SocialAdapter + FaceAdapter
        |       +--> SonarAdapter
        +--> MetadataBuilder
        +--> PepperServerTransport
        +--> SessionStore
        +--> SpeechAdapter
        +--> DialogAdapter
        +--> TabletAdapter / FakeTabletAdapter
```

## Main Use Cases

### Quick Look

A quick look is triggered by grammar such as “what do you see” or “koukni”. The flow is:

1. QiChat calls `PepperGroundedClient.look(lang)`.
2. `TurnManager.start_look` accepts or rejects the turn.
3. `_run_look` captures one frame with `capture_mode="caption"`.
4. `_capture_with_metadata` gathers image bytes and robot context.
5. `PepperServerTransport.caption` posts `/api/v1/caption` with `run_detect` according to config.
6. `SessionStore.update_after_caption` saves the caption and optional detect request id.
7. `SpeechAdapter.say` speaks the caption.
8. Dynamic memory concepts refresh if `dialog.refresh_after_detect` is enabled.

### Full Scan

A scan is triggered by grammar such as “scan the room” or “oskenuj místnost”. The flow is:

1. QiChat calls `PepperGroundedClient.scan(lang)`.
2. `TurnManager._run_scan` snapshots the original head pose.
3. `scan_planner` chooses `panorama_detect` or `sequential_detect` from config.
4. `_prepare_scan_captures` moves the head through configured yaw angles, waits for settling, captures images, and builds per-frame metadata.
5. In panorama mode, `PepperServerTransport.panorama_detect` sends all images to `/api/v1/detect/panorama`.
6. In sequential mode, `PepperServerTransport.detect` sends each image to `/api/v1/detect`.
7. `SessionStore.update_after_detect` records the result and scan id.
8. The head pose is restored if configured.
9. Dynamic concepts refresh if configured.
10. A scan summary chat is optionally sent to `/api/v1/chat` and spoken.

### General Chat

A general chat is triggered by broad grammar fallbacks or directly by `ask(lang, query)`.

1. The user query is sanitized and truncated.
2. If visual context is stale and `behavior.auto_refresh_before_chat=true`, a detect refresh runs first.
3. `PepperServerTransport.chat_general` sends `mode="general"` to `/api/v1/chat`.
4. The server response updates the local `chat_id`.
5. The sentence returned by the server is spoken.

### Object Chat

Object chat is triggered by grammar using `_~memory_objects`.

1. Dynamic concept matching extracts the object label.
2. QiChat calls `PepperGroundedClient.askAboutObject(lang, object_label)`.
3. `TurnManager._run_object_ask` builds a default object query if no explicit query was provided.
4. `PepperServerTransport.chat_object` sends `mode="object"` and `object_label` to `/api/v1/chat`.
5. The response shares the same local conversation id as general chat.

### Memory Display

Memory display is triggered by grammar such as “show memory” or “ukaž paměť”.

1. QiChat calls `PepperGroundedClient.showMemory(lang)`.
2. `TurnManager._run_show_memory` calls `/api/v1/memory/summary` with `render_limit` and `lang`.
3. The summary updates `SessionStore` labels/attributes/relations.
4. The client calls `/api/v1/chat/pregenerate_qa` and caches returned Q/A pairs.
5. Dynamic concepts refresh from the new memory state.
6. A compact tablet payload is built.
7. `TabletAdapter.show_memory_page` opens the local app and injects the payload.
8. The page renders graph SVG, object labels, edge lists, and Q/A buttons.

## Threading Model

`PepperGroundedClient` methods are called by NAOqi and return quickly. Most work is delegated to `TurnManager._start_async`, which creates one daemon thread per accepted turn.

`TurnManager` uses `_busy` and `_active_turn` guarded by `RLock`. Only one turn is allowed at a time. This prevents a scan and a chat from simultaneously moving the head, capturing frames, or speaking over each other.

`SessionStore`, `SpeechAdapter`, and `FakeTabletAdapter` also use locks because they are accessed from multiple turn threads or HTTP server callbacks.

## Language Model

The client uses short runtime language codes internally:

- `en` for English.
- `cs` for Czech.

QiChat topic language codes are different:

- `enu` for English.
- `czc` for Czech.

Server output language values are different again:

- `english` for English.
- `czech` for Czech.

`speech_policy.py` is the single place that translates between these language names.

## Boundaries With The Server

The client does not make decisions about detection, scene graphs, memory identity, chat prompts, or Q/A generation. It only sends data and consumes responses. Server behavior is documented in [`../server/index.md`](../server/index.md).

The most important server docs for client changes are [`../server/api-reference.md`](../server/api-reference.md), [`../server/data-models-and-contracts.md`](../server/data-models-and-contracts.md), and [`../server/orchestration-and-conversations.md`](../server/orchestration-and-conversations.md).
