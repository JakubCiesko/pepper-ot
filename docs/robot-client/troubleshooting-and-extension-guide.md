# Troubleshooting And Extension Guide

This guide maps common client failures and feature changes to the right files.

## Common Runtime Failures

### Robot says server is unavailable

Likely files:

- `client/app/scripts/pepper_client/core/transport.py`.
- `client/app/scripts/client_config.json`.

Check:

- `server.base_url` is reachable from the robot.
- `server.verify_tls` matches your certificate situation.
- ngrok/public URL is active if used.
- Endpoint paths match current server routes.
- Timeouts are not too short.

Server-side docs:

- [`../server/api-reference.md`](../server/api-reference.md).
- [`../server/startup-runtime-and-state.md`](../server/startup-runtime-and-state.md).

### Robot says server response is invalid

Likely cause: server response shape changed but client validation did not.

Check `PepperServerTransport` validation for:

- `caption`: requires `caption`.
- `detect`: requires `objects` list.
- `panorama_detect`: requires `objects` list.
- `chat`: requires `sentence` and `chat_id`.
- `pregenerate_qa`: requires `pregenerated_qa` list.
- `memory_summary`: requires JSON object.
- `reset_memory`: requires `ok`.

If server schemas changed, update both transport and docs.

### Robot fails on Czech or non-ASCII text

Likely files:

- `client/app/scripts/stk/logging.py`.
- `client/app/scripts/pepper_client/utils/text.py`.
- `client/app/scripts/pepper_client/interaction/speech_adapter.py`.

Check:

- Text is cleaned with `clean_text_unicode` before speech/listing when needed.
- Python 2 `unicode` strings are encoded to UTF-8 before NAOqi logging/speech where required.
- Avoid importing a local module named `time.py`; the project uses `utils/timing.py` to avoid shadowing the standard library.

### `module object is not callable` during concept refresh

Likely cause: importing the `text` module and calling it like a function, or similar namespace/function confusion.

Relevant files:

- `client/app/scripts/pepper_client/utils/text.py`.
- `client/app/scripts/pepper_client/core/session_store.py`.
- `client/app/scripts/pepper_client/core/turn_manager.py`.
- `client/app/scripts/pepper_client/interaction/dialog_adapter.py`.

Use explicit function names such as `text_utils.clean_text(...)`.

### `module object has no attribute sleep`

Likely cause: module shadowing around Python's `time` module. The current project uses `pepper_client.utils.timing` specifically because Python 2 import behavior can be fragile when a local file has a builtin-like name.

Use:

```python
from pepper_client.utils import timing as time_utils

time_utils.sleep_seconds(seconds)
time_utils.now_ts()
```

Do not create `utils/time.py` or import it as if it were the standard library.

### Camera capture fails

Likely files:

- `client/app/scripts/pepper_client/perception/camera_adapter.py`.
- `client/app/scripts/client_config.json`.

Check:

- `ALVideoDevice` is available.
- `capture.camera_id`, `resolution`, `color_space`, and `fps` are valid for Pepper.
- The camera is not held by another subscriber.
- In fake mode, `fake_camera_path` exists and contains images.

### Scan moves head but does not process images

Likely files:

- `client/app/scripts/pepper_client/core/turn_manager.py`.
- `client/app/scripts/pepper_client/perception/scan_planner.py`.
- `client/app/scripts/pepper_client/core/transport.py`.

Check:

- `panorama.enabled` and `panorama.mode`.
- `capture.scan_yaws_deg`.
- `capture.settle_seconds`.
- Server panorama endpoint path.
- Server timeout long enough for multi-image processing.

### Tablet page does not show memory

Likely files:

- `client/app/scripts/pepper_client/interaction/tablet_adapter.py`.
- `client/app/html/index.html`.
- `client/app/html/js/app.js`.
- `client/app/html/js/render.js`.
- `client/app/pepper-grounded-client.pml`.

Check robot mode:

- `tablet.fake_tablet=false`.
- `ALTabletService` exists.
- `local_app_name` matches app package name.
- `html/index.html` and JS files are listed in PML resources.
- `window.PepperMemoryPageReady` becomes true.

Check fake mode:

- `tablet.fake_tablet=true`.
- Fake server logs URL.
- Browser can load `/`, `/js/*`, `/css/*`, and `/payload.json`.

### Q/A tablet buttons are disabled

Likely files:

- `client/app/html/js/service_bridge.js`.
- `client/app/html/js/render.js`.
- `client/app/scripts/peppergroundedclient.py`.

Check:

- Page is not in fake mode. Fake mode intentionally cannot call robot service.
- `/libs/qi/2/qi.js` loaded.
- JS resolves service name `PepperGroundedClient`.
- Python service is registered.
- `showMemory` payload includes `pregenerated_qa`.

### Dynamic concepts do not update

Likely files:

- `client/app/scripts/pepper_client/interaction/dialog_adapter.py`.
- `client/app/scripts/pepper_client/core/turn_manager.py`.
- `client/app/scripts/pepper_client/core/session_store.py`.
- QiChat `.top` files.

Check:

- `dialog.enable_dynamic_memory_concepts=true`.
- Server `/api/v1/memory/summary` returns labels and scene graph.
- `DialogAdapter.set_dynamic_concept` logs successful setConcept attempt.
- Current dialog language matches the topic language.
- Dynamic concept names match exactly: `memory_objects`, `memory_attributes`, `memory_relations`, `memory_cached_questions`.

## Where To Add Features

### Add a new spoken intent

Files:

- `client/app/pepper-grounded-client/pepper-grounded-client_enu.top`.
- `client/app/pepper-grounded-client/pepper-grounded-client_czc.top`.
- `client/app/scripts/peppergroundedclient.py` if a new bound method is needed.
- `client/app/scripts/pepper_client/core/turn_manager.py` if the intent is long-running.

Pattern:

1. Define concepts and rules in QiChat.
2. Keep grammar response short.
3. Call a Python method with `^pCall`.
4. Let Python handle server calls and speech.

### Add a new server API call

Files:

- `client/app/scripts/pepper_client/core/transport.py`.
- `client/app/scripts/pepper_client/utils/error_policy.py` if new errors need distinct handling.
- `client/app/scripts/client_config.json` and `utils/config.py` if path/timeout should be configurable.

Pattern:

1. Add a method to `PepperServerTransport`.
2. Validate response shape immediately.
3. Raise `MalformedResponseError` for schema mismatch.
4. Call from `TurnManager`, not from QiChat directly.

### Add a new metadata field

Files:

- Relevant adapter under `client/app/scripts/pepper_client/perception`.
- `client/app/scripts/pepper_client/utils/metadata_builder.py`.
- Server `server/app/schemas/robot.py`.
- Server memory/fusion code.

Pattern:

1. Read robot value in adapter.
2. Add to context snapshot.
3. Add to metadata builder if top-level.
4. Add server schema field.
5. Use it server-side.

### Add a new tablet section

Files:

- `client/app/html/index.html`.
- `client/app/html/css/style.css`.
- `client/app/html/js/render.js`.
- `client/app/scripts/pepper_client/core/turn_manager.py` for payload shape.
- `client/app/pepper-grounded-client.pml` if adding assets.

Pattern:

1. Add HTML target node.
2. Add render function.
3. Include field in Python payload.
4. Ensure fake and robot modes render the same payload.

### Add a new dynamic concept

Files:

- Both `.top` files for `dynamic:` declaration and usage.
- `SessionStore` for local values.
- `DialogAdapter.refresh_memory_concepts` for setting it.
- `TurnManager._refresh_dynamic_concepts_from_summary` for extraction.

Pattern:

1. Store source values in `SessionStore`.
2. Cap and clean values in `DialogAdapter`.
3. Update concepts after memory summary.
4. Use `_~concept_name` in QiChat.

### Add a new scan mode

Files:

- `client/app/scripts/pepper_client/perception/scan_planner.py`.
- `client/app/scripts/pepper_client/core/turn_manager.py`.
- `client/app/scripts/pepper_client/utils/config.py`.
- `client/app/scripts/client_config.json`.

Pattern:

1. Normalize the mode in config.
2. Return it from `scan_planner.scan_mode`.
3. Add a `_run_scan_<mode>` helper.
4. Keep `_prepare_scan_captures` shared if possible.

## Known Code Smells And Current Limitations

These are not necessarily bugs, but they are useful to know before changing code.

### Python 2 Compatibility Is Real

The client source uses Python 2 constructs such as `unicode`, `raw_input`, old imports, and `ur"..."` regex strings. Do not casually modernize a single file without checking robot runtime.

### Direct Debug Mode Is Hard-Coded

`peppergroundedclient.py` currently has `run_local=True` and `czech=True` in the `__main__` block. That is convenient for local testing but not a clean command-line mode selector.

### Some Bound Methods Are Mostly Debug Controls

`setDialogLanguage`, `setServerBaseUrl`, `reloadConfig`, `say`, `getStatus`, and `stop` are useful for manual control but are not core user-facing grammar paths.

### Generated Files Are Present In Source Tree

There are many `.pyc` and `__pycache__` files under `client/app/scripts`. Treat them as generated artifacts, not source.

### Transport Reset Conversation Path Is Hard-Coded

`PepperServerTransport.reset_conversation` does not use config for the conversation reset route.

### Chat Response Field Is Client-Specific

The client expects `sentence` in chat response. If the server response field changes or aliases are removed, chat fails as malformed.

### Dynamic Concept Attributes Store Relation Names Only

Attributes and relations dynamic concepts contain only `rel` names, not full triples. This is sufficient for current grammar but not for questions that need direct object-edge binding.

### Fake Camera Geometry Is Weak

Fake camera uses fixed FOV values and random image selection. It is good for flow testing but poor for debugging geometry fusion.

## Minimal Validation Commands

From repository root, useful non-robot checks are:

```sh
python2 -m py_compile client/app/scripts/peppergroundedclient.py
```

```sh
python2 -m py_compile client/app/scripts/pepper_client/core/*.py client/app/scripts/pepper_client/interaction/*.py client/app/scripts/pepper_client/perception/*.py client/app/scripts/pepper_client/utils/*.py
```

For tablet JS syntax, a modern Node parser can catch obvious syntax errors, but it does not guarantee old tablet WebKit compatibility:

```sh
node --check client/app/html/js/app.js
node --check client/app/html/js/render.js
node --check client/app/html/js/service_bridge.js
```

For package completeness, inspect `pepper-grounded-client.pml` whenever new files are added.
