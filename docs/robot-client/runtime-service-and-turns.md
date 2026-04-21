# Runtime Service And Turns

This document explains the Python runtime: the `PepperGroundedClient` Qi service and the `TurnManager` workflows that actually execute robot actions.

## Main Service: `PepperGroundedClient`

File: `client/app/scripts/peppergroundedclient.py`

`PepperGroundedClient` is the NAOqi service exported to the robot. It is registered under the service name `PepperGroundedClient`, which is why QiChat rules can call `^pCall(PepperGroundedClient.look("en"))` and the tablet JS can call `service.answerCachedQuestion(...)` through `qi.js`.

### Construction

The constructor receives a `qi.Application` and reads `qiapp.session`. It then initializes the whole runtime stack.

Important construction steps:

- `stk.logging.get_logger` creates a UTF-8 safe NAOqi logger.
- `stk.services.ServiceCache` lazily resolves robot services such as `ALVideoDevice`, `ALMotion`, and `ALTextToSpeech`.
- `client_config.load_config` loads `client/app/scripts/client_config.json` and merges it with defaults.
- `SessionStore` starts with clean local state.
- `PepperServerTransport` is configured with the server base URL and endpoint paths.
- `_initialize_camera_adapter` chooses real or fake camera.
- `_initialize_tablet_adapter` chooses real or fake tablet.
- `TurnManager` receives references to all adapters and services.

The constructor intentionally wires dependencies explicitly. That makes feature placement clear: if a feature needs camera, server calls, speech, state, and dynamic concepts, it belongs in `TurnManager` because `TurnManager` already has all those dependencies.

### Lifecycle Hooks

`on_start` is marked with `@qi.nobind` because it is lifecycle code, not a public service method. It:

- Logs the current server and dialog language.
- Starts face detection.
- Starts robot context collectors.
- Refreshes memory concepts once from the server.

`on_stop` hides the tablet page, shuts down the turn manager speech path, stops context collectors, and stops face detection.

### Public Qi Methods

These methods are bound with `@qi.bind` and are callable from QiChat, tablet JS, or remote Qi clients.

| Method | Called by | Behavior |
|---|---|---|
| `look(lang_code)` | QiChat quick-look rules | Starts one caption/detect turn. |
| `scan(lang_code)` | QiChat full-scan rules | Starts panorama or sequential scan. |
| `ask(lang_code, query)` | QiChat general ask rules | Starts general chat without forced refresh. |
| `refreshAndAsk(lang_code, query)` | Potential grammar/API use | Starts general chat after forced visual refresh. |
| `resetConversation()` | QiChat reset conversation rules | Clears local chat state and asks server to reset existing conversation if present. |
| `askAboutObject(lang_code, object_label)` | Dynamic object rules | Starts object-focused chat. |
| `answerCachedQuestion(lang_code, question)` | Dynamic Q/A rules and tablet buttons | Speaks cached answer or falls back to general chat. |
| `listRelations(lang_code)` | QiChat list relation rules | Speaks sampled remembered relations. |
| `listAttributes(lang_code)` | QiChat list attribute rules | Speaks sampled remembered attributes. |
| `listObjects(lang_code)` | QiChat list object rules | Speaks sampled remembered object labels. |
| `listCachedQuestions(lang_code)` | QiChat suggested question rule | Speaks one cached suggested question. |
| `showMemory(lang_code)` | QiChat and possibly remote clients | Fetches memory summary and opens tablet memory view. |
| `hideMemory()` | Tablet/lifecycle helper | Hides memory webview. |
| `resetMemory(lang_code)` | QiChat reset memory rules | Clears server memory, local memory state, and local conversation. |
| `refreshMemoryConcepts(lang_code=None)` | QiChat nested refresh rules | Pulls memory summary and updates dynamic concepts. |
| `setDialogLanguage(mode)` | Debug/manual control | Persists `dialog.language`. Marked by TODO as likely not useful now. |
| `setServerBaseUrl(base_url)` | Debug/manual control | Updates server URL at runtime and persists config. |
| `reloadConfig()` | Debug/manual control | Reloads config from JSON and reinitializes affected adapters. |
| `say(text)` | Debug/manual control | Speaks raw text. |
| `getStatus()` | Debug/status tooling | Returns JSON status snapshot. |
| `stop()` | Debug/control tooling | Stops the Qi application. |

### Listing Dynamic Concepts

`listDynamicConcept` maps concept names to `SessionStore` getters:

- `objects` -> `get_memory_labels`.
- `attributes` -> `get_memory_attributes`.
- `relations` -> `get_memory_relations`.
- `cached_questions` -> `get_cached_questions`.
- `cached_answers` -> `get_cached_answers`.

For normal spoken list commands it samples up to a fixed number of entries, prefixes them with a localized phrase, replaces underscores with spaces, and speaks the result.

Current limitation: `listObjects`, `listAttributes`, and `listRelations` use a hard-coded sample size of 10. The file contains a TODO to make this tunable.

### Runtime Config Reload

`reloadConfig` calls `_apply_loaded_config`.

`_apply_loaded_config`:

- Stops robot context and face adapter.
- Replaces the existing config dict contents with the loaded config.
- Updates `SessionStore` server URL.
- Updates `PepperServerTransport` config.
- Updates `DialogAdapter` config.
- Restarts face adapter and robot context.
- Reinitializes camera and tablet adapters.
- Reassigns the new camera/tablet adapter references into the existing `TurnManager`.

Current limitation: most existing adapter objects keep their original config references except the adapters explicitly updated or recreated. If you add config that should hot-reload inside a long-lived adapter, either recreate that adapter or add an explicit `update_config` path.

## TurnManager

File: `client/app/scripts/pepper_client/core/turn_manager.py`

`TurnManager` owns all slow actions. Its public methods are intentionally named `start_*` because they return immediately after spawning a daemon thread.

### Busy Lock

`_start_async` creates a new turn id and checks `_busy` under `RLock`.

If another turn is active:

- It logs the rejected action.
- It speaks a localized busy fallback from `error_policy.fallback_message("busy", lang)`.
- It returns `False`.

If no turn is active:

- It sets `_busy=True` and stores `_active_turn` with id and kind.
- It optionally speaks an acknowledgement based on config.
- It starts a daemon thread named `pepper-turn-<turn_id>`.
- It returns `True`.

### Guarded Execution

Each thread runs `_run_guarded`. It catches:

- `CameraCaptureError` -> camera fallback speech.
- `ServerTimeoutError` -> timeout fallback speech.
- `ServerUnavailableError` -> server-unavailable fallback speech.
- `MalformedResponseError` -> invalid-server-response fallback speech.
- Any other exception -> generic unexpected fallback speech.

The `finally` block always clears `_busy` and `_active_turn`.

This pattern is important because QiChat should never block forever on a Python exception.

## Turn Workflows

### `_run_look`

Purpose: fast one-frame scene description.

Steps:

1. Resolve runtime and speech language.
2. Generate a frame id using `ids.new_frame_id`.
3. Capture one frame and metadata with capture mode `caption`.
4. Read `behavior.caption_run_detect` and `server.publish`.
5. Convert runtime language to server language.
6. Call `_caption_with_optional_retry`.
7. Store response through `SessionStore.update_after_caption`.
8. Speak `caption_response["caption"]`.
9. Refresh dynamic concepts when `dialog.refresh_after_detect=true`.

Server endpoint: `/api/v1/caption`.

### `_run_scan`

Purpose: multi-frame scene update.

Steps:

1. Resolve runtime and speech language.
2. Create a scan id using configured prefix.
3. Snapshot current head/body pose.
4. Choose scan mode through `scan_planner.scan_mode`.
5. Run panorama or sequential scan.
6. Restore head pose if `behavior.auto_restore_head_pose=true`.
7. Refresh dynamic concepts if `dialog.refresh_after_scan=true`.
8. If `scan_planner.summary_after_scan=true`, ask the server for a natural-language scan summary and speak it.
9. Otherwise speak a localized scan-complete message.

Server endpoints:

- `/api/v1/detect/panorama` for panorama mode.
- `/api/v1/detect` for sequential mode.
- `/api/v1/chat` for optional summary after scan.

### `_prepare_scan_captures`

Purpose: collect all images and metadata needed for a scan before sending them.

Steps for each configured yaw:

1. Move head to yaw and configured pitch.
2. Sleep for `capture.settle_seconds`.
3. Generate frame id.
4. Capture frame and metadata with capture mode `scan`.
5. Append `{index, image_bytes, metadata}` to the capture list.

This function is shared by panorama and sequential scan modes, so changes here affect both.

### `_run_scan_panorama`

Purpose: send multiple captures to the server in one panorama request.

Steps:

1. Prepare captures.
2. Read `panorama.stick_together`.
3. Call `transport.panorama_detect` with `resize_image=True`.
4. Store detect response in `SessionStore` with scan id.
5. Log object count and stitching mode.

### `_run_scan_sequential`

Purpose: send scan captures one at a time.

Steps:

1. Prepare captures.
2. For each capture, call `transport.detect`.
3. Store each successful response.
4. Track successes and last error.
5. If no frame succeeded, raise the last error or `ServerUnavailableError`.

This mode is useful if panorama stitching is not desired or if per-frame geometry should remain independent.

### `_run_ask`

Purpose: general chat.

Steps:

1. Sanitize and truncate the query.
2. Optionally refresh visual context when forced or stale.
3. Call `transport.chat_general`.
4. Update chat state.
5. Speak `chat_response["sentence"]`.

Current contract detail: the client expects the server chat response to contain `sentence` and `chat_id`.

### `_run_object_ask`

Purpose: object-focused chat.

Steps:

1. Clean object label.
2. Sanitize optional query.
3. If no query exists, build a default localized object query.
4. Call `transport.chat_object` with `mode="object"` and `object_label`.
5. Update chat state.
6. Speak the returned sentence.

### `_run_cached_answer`

Purpose: answer a pregenerated question quickly.

Steps:

1. Sanitize the question.
2. Look for exact match in `SessionStore.cached_answers`.
3. If exact match fails, do case-insensitive matching.
4. If found, speak the cached answer immediately.
5. If not found, fall back to `transport.chat_general`.

This is used by both QiChat dynamic `memory_cached_questions` rules and tablet Q/A buttons.

### `_run_show_memory`

Purpose: fetch memory summary, refresh concepts, and show tablet UI.

Steps:

1. Resolve language.
2. Compute render limit through `scan_planner.memory_render_limit`.
3. Call `transport.memory_summary(render_limit, language)`.
4. Store summary in `SessionStore`.
5. Call `transport.pregenerate_qa` with configured question count.
6. Store Q/A pairs in `SessionStore`.
7. Refresh dynamic concepts from summary.
8. Build tablet payload.
9. Call `tablet_adapter.show_memory_page(payload)`.
10. Speak fallback if the tablet display fails.

The tablet payload is not identical to the raw server summary. It is normalized for UI rendering:

- `object_labels` from summary labels.
- `label_counts` from summary counts.
- `attributes` from unary scene graph edges where `sub == obj`.
- `relationships` from binary edges.
- `graph_svg` from summary.
- `pregenerated_qa` from Q/A response or local cache.
- `ui_language` for button callback language.

### `_run_reset_memory`

Purpose: clear visual memory and conversation context.

Steps:

1. Save current chat id.
2. Call server memory reset.
3. Clear local memory state.
4. Clear local conversation state.
5. Ask server to reset old conversation if a chat id existed.
6. Refresh dynamic concepts if configured.
7. Speak reset acknowledgement.

## Helper Methods

### `_refresh_visual_context`

Captures one frame with capture mode `detect`, calls `/api/v1/detect`, updates local detect state, and refreshes dynamic concepts if configured.

This is used before chat when visual state is stale or when `refreshAndAsk` forces a refresh.

### `_capture_with_metadata`

Captures image bytes through the camera adapter, snapshots robot context, and builds a metadata dict through `MetadataBuilder`.

This is the only method that should combine image capture and robot context for server requests.

### `_caption_with_optional_retry`

Calls the caption endpoint. If it times out and `behavior.caption_retry_on_timeout=true`, it retries once.

### `_speech_lang` and `_speech_request_language`

These methods delegate to `speech_policy` and centralize language handling for turns.

## Where To Add New Turns

To add a new spoken action:

1. Add a bound method to `PepperGroundedClient` only if QiChat or tablet JS needs to call it directly.
2. Add `start_new_action` and `_run_new_action` to `TurnManager` if the action is long-running.
3. Add localized grammar rules to English and Czech `.top` files.
4. Update `SessionStore` only if the action creates reusable state.
5. Update `speech_policy` only if new acknowledgement/fallback text is needed.
6. Update this document and [`dialog-and-speech.md`](dialog-and-speech.md).
