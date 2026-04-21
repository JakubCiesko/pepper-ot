# Robot Client File Inventory

This is a file-by-file locator for the current client tree. It separates runtime source from package resources, examples, generated artifacts, and editor support files.

## Runtime Entrypoint

### `client/app/scripts/peppergroundedclient.py`

Main NAOqi service module. Defines `PepperGroundedClient`, wires config, adapters, transport, session store, robot context, metadata builder, and turn manager. Exposes all Qi-bound methods used by QiChat and tablet JavaScript.

Change this file when:

- Adding a new public Qi service method.
- Changing lifecycle startup/shutdown behavior.
- Changing runtime adapter construction.
- Changing manual debug/reload controls.

Avoid placing long-running business logic here. Put that into `TurnManager`.

## Core Runtime Package

Directory: `client/app/scripts/pepper_client/core`

### `__init__.py`

Empty package marker.

### `session_store.py`

Thread-safe in-process state store for chat id, recent responses, visual refresh timestamps, remembered memory labels, remembered attributes/relations, and cached pregenerated Q/A pairs.

Change this file when:

- Adding client-side state required across turns.
- Adding a new dynamic concept source list.
- Changing how memory summary data is split into labels, attributes, and relations.

### `transport.py`

HTTP client for server APIs. Owns multipart image upload, JSON chat/memory requests, timeout handling, TLS verification, response validation, and conversion of request failures to client exceptions.

Change this file when:

- Adding a new server endpoint call.
- Changing request payloads.
- Updating response validation to match server contracts.
- Adding configurable paths/timeouts.

### `turn_manager.py`

Asynchronous workflow coordinator. Owns quick look, scan, general ask, object ask, cached answer, memory display, memory reset, visual refresh, dynamic concept refresh, and tablet payload creation.

Change this file when:

- Adding or changing robot behaviors.
- Changing scan flow.
- Changing how server responses are spoken or displayed.
- Changing when concepts refresh.
- Changing memory page payload shape.

## Interaction Package

Directory: `client/app/scripts/pepper_client/interaction`

### `__init__.py`

Empty package marker.

### `dialog_adapter.py`

Thin ALDialog wrapper for dynamic concepts. Resolves `enu`/`czc` topic language, cleans/caps values, and tries multiple `setConcept` signatures for binding compatibility.

Change this file when:

- Adding new dynamic concepts.
- Changing language resolution for concepts.
- Changing concept value cleaning/capping.

### `speech_adapter.py`

Speech wrapper for `ALAnimatedSpeech` and `ALTextToSpeech`. Cleans text, applies language, serializes speech calls with a lock, tries animated speech first, and falls back to plain TTS.

Change this file when:

- Changing speech service priority.
- Changing text encoding behavior.
- Changing TTS language application.

### `speech_policy.py`

Language and phrase policy. Contains acknowledgement phrases, generic phrases, language normalization, TTS-to-runtime language mapping, runtime-to-dialog/server language mapping, and random phrase selection.

Change this file when:

- Adding localized acknowledgements.
- Adding generic spoken messages.
- Changing language mapping behavior.

### `tablet_adapter.py`

Robot and fake tablet adapters. `TabletAdapter` uses `ALTabletService` to load local app, show/hide webview, wait for page readiness, and inject payload. `FakeTabletAdapter` serves the same HTML locally over HTTP and exposes `/payload.json`.

Change this file when:

- Changing local tablet load/show sequence.
- Changing JS bridge injection behavior.
- Changing fake tablet server behavior.
- Adding new tablet adapter methods used by turns.

## Perception Package

Directory: `client/app/scripts/pepper_client/perception`

### `__init__.py`

Empty package marker.

### `camera_adapter.py`

Real and fake camera adapters. `CameraAdapter` captures from `ALVideoDevice`, builds PIL RGB images, encodes JPEG bytes, reads camera FOV, and unsubscribes safely. `FakeCameraAdapter` randomly serves images from a local folder.

Change this file when:

- Changing capture parameters or image encoding.
- Adding camera metadata.
- Improving fake camera behavior.

### `face_adapter.py`

Face detection wrapper. Subscribes to `ALFaceDetection`, reads `FaceDetected`, extracts face yaw/pitch/label/confidence, and matches faces to people by angular proximity.

Change this file when:

- Changing face recognition extraction.
- Changing face-to-person matching.
- Adding face-related metadata.

### `people_adapter.py`

People perception wrapper. Subscribes to `ALPeoplePerception`, reads visible people ids, yaw/pitch angles, and distance from ALMemory.

Change this file when:

- Changing geometric people metadata.
- Adding new people perception fields.

### `pose_adapter.py`

Motion/pose wrapper. Reads head yaw/pitch and body yaw, moves head, and restores head pose after scans.

Change this file when:

- Changing scan movement behavior.
- Adding body pose metadata.
- Changing head restore timing.

### `robot_context.py`

Combines pose, people, social, and sonar snapshots into one context dict. Starts/stops people, social, and sonar collectors.

Change this file when:

- Adding a new robot context adapter.
- Changing top-level context snapshot structure.

### `scan_planner.py`

Pure config helper for scan yaw list, scan pitch, scan mode, panorama stitching, scan summary behavior, and memory render limit.

Change this file when:

- Adding scan modes.
- Changing how config maps to scan behavior.

### `social_adapter.py`

Social metadata collector. Subscribes to face characteristics, gaze analysis, engagement zones, sitting detection, and waving detection. Reads age, gender, smile, expression, looking-at-robot, head angles, engagement, sitting, waving, eye opening, and matched face labels.

Change this file when:

- Adding Pepper social metadata.
- Changing social attribute naming.
- Changing derived fields such as age bucket or gaze direction.

### `sonar_adapter.py`

Sonar collector. Subscribes to `ALSonar` and reads left/right ultrasonic sensor values from ALMemory.

Change this file when:

- Adding obstacle/proximity metadata.
- Changing sonar subscription behavior.

## Utility Package

Directory: `client/app/scripts/pepper_client/utils`

### `__init__.py`

Empty package marker.

### `config.py`

Default config, JSON loading, deep merge, normalization, saving, and script path helper.

Change this file when:

- Adding client config keys.
- Changing default values.
- Adding validation/normalization.
- Saving config differently.

### `error_policy.py`

Client exception classes and localized fallback messages.

Change this file when:

- Adding new error categories.
- Changing fallback speech.

### `ids.py`

UUID helper functions for turn ids, frame ids, and scan ids.

Change this file when:

- Changing id format or prefixes.

### `logging.py`

Small JSON-safe logging helper. `safe_json` tries `json.dumps(..., sort_keys=True)` and falls back to `str(data)`.

Change this file when:

- Changing compact payload logging behavior.

### `metadata_builder.py`

Builds server `RobotMetadata` JSON from capture and robot context.

Change this file when:

- Adding top-level metadata fields.
- Changing how people/social/sonar are included.

### `text.py`

Python 2 text cleanup utilities for normal strings, unicode strings, and query sanitization.

Change this file when:

- Fixing non-ASCII handling.
- Changing max-character truncation behavior.
- Changing whitespace normalization.

### `timing.py`

Wrapper around standard `time` module with `now_ts` and `sleep_seconds`. It exists to avoid Python 2 module shadowing problems with files named like builtins.

Change this file when:

- Adding timing helpers.

## STK Helper Package

Directory: `client/app/scripts/stk`

### `__init__.py`

Package marker.

### `events.py`

ALMemory event helper. Marked as unused/removal by comments. Not used by current runtime.

### `logging.py`

NAOqi logging wrapper. Converts unicode and containers to safe values before passing them to `qi.logging.Logger`.

### `runner.py`

Standalone robot runner helpers. Handles command-line `--qi-url`, off-robot prompt for robot host, `qi.Application` creation, service registration, and lifecycle calls.

### `services.py`

Lazy service cache. Fetches NAOqi services on demand from session and returns `None` if unavailable. Always refreshes `ALTabletService` rather than permanently caching it.

## Client Config And Startup Files

### `client/app/scripts/client_config.json`

Runtime configuration JSON. Merged with `DEFAULT_CONFIG` from `utils/config.py`. Current checked-in values are development-oriented: fake camera and fake tablet are enabled, server URL points at ngrok.

### `client/start_service.sh`

Local helper to run the service with Python 2 and connect to a virtual robot port.

### `client/setup.py`

Setuptools package definition for `pepper-grounded-client`, targeting Python `>=2.7,<3` and reading dependencies from `requirements.txt`.

### `client/requirements.txt`

Python dependencies for the client package:

- `Pillow==6.2.2`.
- `requests==2.27.1`.

### `client/testrun.py`

Tiny test script that connects to a robot host, fetches `PepperGroundedClient`, and prints `getStatus()`.

## QiChat And Dialog Package Files

Directory: `client/app/pepper-grounded-client`

### `pepper-grounded-client.dlg`

Multilanguage dialog mapping file. Links `enu` and `czc` to their topic files.

### `pepper-grounded-client_enu.top`

English QiChat grammar. Contains system explanation, quick look, panorama scan, memory display, reset flows, refresh concepts, object chat, list commands, count/presence/position/attribute/relation questions, suggested questions, direct cached Q/A, and explicit general ask fallback.

### `pepper-grounded-client_czc.top`

Czech QiChat grammar. Mirrors the English feature set in Czech.

## Tablet App Files

Directory: `client/app/html`

### `index.html`

Single tablet page. Declares sections for objects, attributes, relationships, scene graph SVG, and pregenerated Q/A. Loads `qi.js` and local JS modules.

### `css/style.css`

Tablet page styling. Defines dark layout, card grid, chips, graph SVG styling, Q/A buttons, answer previews, and status badge states.

### `js/state.js`

Creates shared `PepperMemoryState` global.

### `js/utils.js`

Creates `PepperMemoryUtils` with object parsing, HTML escaping, and query parameter parsing.

### `js/render.js`

Creates `PepperMemoryRender` with all DOM rendering logic and Q/A button binding.

### `js/service_bridge.js`

Creates `PepperMemoryService`, connects to robot service through `qi.js`, and handles Q/A button calls.

### `js/fake_tablet.js`

Creates `PepperMemoryFake`, detects fake mode, polls `/payload.json`, and renders fake tablet updates.

### `js/app.js`

Bootstraps fake/robot mode, exposes `window.PepperMemoryPage.renderFromBridge`, sets `window.PepperMemoryPageReady`, and starts initial render.

## Package Metadata

### `client/app/manifest.xml`

Robot package manifest. Defines package name, descriptions, supported languages, dialog content, NAOqi/robot requirements, and autorun service.

### `client/app/pepper-grounded-client.pml`

qipkg package file. Lists resources, topics, translations, ignored paths, and behavior description.

### `client/app/Makefile`

Build/install/deploy helper for qipkg packages.

### `client/app/icon.png`

Package icon.

### `client/app/testrun/behavior.xar`

Behavior file referenced by PML/manifest.

## Translations

Directory: `client/app/translations`

### `translation_cs_CZ.ts` and `translation_en_US.ts`

Minimal Qt translation source files.

### `translation_cs_CZ.qm` and `translation_en_US.qm`

Compiled translation files packaged as resources.

### `.translation_*.csp`

Temporary/generated translation files. Not source of truth.

## Examples And References

### `client/html example/dialog_presentation_nlp-master`

Reference app showing robot-hosted tablet content and JavaScript NAOqi connection patterns. Not part of runtime package.

Important files:

- `html/index.html`.
- `html/js/robotutils.js`.
- `html/js/main.js`.
- example `.top`, `.dlg`, `.pml`, and `manifest.xml`.

### `client/vs-code-highlight-qichat`

Local VS Code syntax highlighter for `.top` QiChat files. Not used by robot runtime.

Important files:

- `package.json`.
- `syntaxes/qichat.tmLanguage.json`.
- `qichat-0.0.1.vsix`.

## Editor And Generated Files

### `client/.idea/*`

JetBrains IDE project metadata. Not runtime.

### `client/.vscode/settings.json`

VS Code workspace settings. Not runtime.

### `*.pyc` and `__pycache__`

Generated Python bytecode/cache files. Not source. Many are excluded in the PML ignored paths.
