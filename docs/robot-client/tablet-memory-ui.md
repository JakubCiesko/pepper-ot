# Tablet Memory UI

The tablet UI is a local robot-hosted web app. It does not open the server dashboard or any remote URL. Its only job is to display the current server memory summary and let the user tap pregenerated Q/A buttons.

## File Layout

| File | Purpose |
|---|---|
| `client/app/html/index.html` | Single page entrypoint. |
| `client/app/html/css/style.css` | Tablet page styling. |
| `client/app/html/js/state.js` | Shared global page state. |
| `client/app/html/js/utils.js` | ES5 helper functions. |
| `client/app/html/js/render.js` | DOM rendering for objects, edges, graph SVG, and Q/A buttons. |
| `client/app/html/js/service_bridge.js` | `qi.js` session and `PepperGroundedClient` service bridge. |
| `client/app/html/js/fake_tablet.js` | Fake desktop tablet polling mode. |
| `client/app/html/js/app.js` | Page bootstrap and Python bridge API exposure. |

The package file `client/app/pepper-grounded-client.pml` includes all of these files as resources.

## Local Robot URL

`TabletAdapter.local_app_url()` returns:

```text
http://198.18.0.1/apps/<local_app_name>/
```

The default app name is `pepper-grounded-client`, so the normal URL is:

```text
http://198.18.0.1/apps/pepper-grounded-client/
```

This matches Pepper tablet packaging rules: the app content is under the app's packaged `html/` folder and loaded locally by ALTabletService.

## TabletAdapter

File: `client/app/scripts/pepper_client/interaction/tablet_adapter.py`

`TabletAdapter` wraps `ALTabletService`.

### `show_memory_page(payload=None)`

Steps:

1. Gets `ALTabletService` from `ServiceCache`.
2. Loads the local app through `_ensure_local_app_loaded`.
3. Shows the webview through `_ensure_webview_visible`.
4. Injects the memory payload through `push_memory_payload`.

If tablet service is unavailable or loading fails, it returns `False`.

### Local App Loading

`_ensure_local_app_loaded` first tries:

```python
tablet.loadApplication(app_name)
```

If that fails, it falls back to:

```python
tablet.loadUrl(local_app_url)
```

This sequence is intentional because ALTabletService docs recommend loading an app or URL first and then showing the webview.

### Showing Webview

`_ensure_webview_visible` first tries:

```python
tablet.showWebview()
```

If that fails, it falls back to:

```python
tablet.showWebview(url)
```

The fallback exists for binding differences.

### Payload Injection

`push_memory_payload(payload)`:

1. Reads retry attempts and interval from `tablet` config.
2. Calls `_wait_page_ready` until `window.PepperMemoryPageReady` is truthy.
3. Serializes payload to JSON.
4. Executes JavaScript that calls:

```javascript
window.PepperMemoryPage.renderFromBridge(payload)
```

5. Retries until JS returns true.

The page-ready handshake prevents injecting data before `index.html` has loaded its scripts.

### Hiding

`hide_memory_page()` calls `ALTabletService.hideWebview()` and returns true even if hiding fails because the webview may already be hidden.

## FakeTabletAdapter

File: `client/app/scripts/pepper_client/interaction/tablet_adapter.py`

`FakeTabletAdapter` is a local development mirror. It is selected when `tablet.fake_tablet=true`.

It implements the same runtime methods used by `TurnManager`:

- `show_memory_page(payload=None)`.
- `hide_memory_page()`.
- `push_memory_payload(payload)`.

### Fake HTTP Server

On first show, it starts a daemon `HTTPServer` bound to `tablet.fake_host:tablet.fake_port`.

Routes:

| Route | Response |
|---|---|
| `/` | Serves `client/app/html/index.html`. |
| `/index.html` | Serves `client/app/html/index.html`. |
| `/payload.json` | Serves latest stored payload JSON. |
| `/health` | Returns `ok`. |
| `/js/*`, `/css/*`, file paths with extension | Serves static files from `client/app/html`. |

Static file serving uses `_safe_static_file` to prevent path traversal outside the HTML root.

### Fake URL

The logged fake URL has this shape:

```text
http://127.0.0.1:8766/?fake_tablet=1&poll_ms=1000
```

Open this in a desktop browser while the client service runs locally.

The browser page polls `/payload.json` and re-renders whenever payload changes.

## HTML Entrypoint

File: `client/app/html/index.html`

The page defines five visual sections:

- Detected Objects.
- Detected Attributes.
- Detected Relationships.
- Scene Graph.
- Pregenerated Q/A.

It includes scripts in this order:

1. `/libs/qi/2/qi.js`.
2. `js/state.js`.
3. `js/utils.js`.
4. `js/render.js`.
5. `js/service_bridge.js`.
6. `js/fake_tablet.js`.
7. `js/app.js`.

This order is required because the scripts use globals rather than ES modules.

## JavaScript Architecture

The tablet JS is ES5-style global namespace code for compatibility with older tablet WebKit.

### `state.js`

Creates `window.PepperMemoryState` with `appState`:

- `payload`.
- `language`.
- `serviceReady`.
- `service`.
- `session`.
- `connecting`.
- `fakeMode`.
- `fakePollIntervalMs`.
- `buttonCooldownMs`.
- `cooldownByQuestion`.

### `utils.js`

Creates `window.PepperMemoryUtils` with:

- `asObject`.
- `escapeHtml`.
- `parseQueryParams`.

`escapeHtml` is used for text values. The graph SVG is intentionally injected raw because it is internal server output.

### `render.js`

Creates `window.PepperMemoryRender`.

Important functions:

| Function | Behavior |
|---|---|
| `setServiceStatus(text, mode)` | Updates status badge. |
| `showQaError(text)` | Shows/hides Q/A error text. |
| `renderObjects(payload)` | Renders labels with counts as chips. |
| `renderEdgeList(...)` | Renders attributes or relationships. |
| `renderGraph(payload)` | Injects `graph_svg` or empty state. |
| `renderQA(payload)` | Renders question buttons and answer previews. |
| `refreshButtonAvailability()` | Disables Q/A buttons unless robot service is connected. |
| `setQuestionClickHandler(handlerFn)` | Injects click behavior from service bridge. |
| `render(payload)` | Main render entrypoint. |

Expected payload fields:

```json
{
  "ui_language": "en",
  "object_labels": ["person", "cat"],
  "label_counts": {"person": 1, "cat": 1},
  "attributes": [{"sub": "cat_2", "rel": "is_gray", "obj": "cat_2"}],
  "relationships": [{"sub": "person_1", "rel": "holding", "obj": "cat_2"}],
  "graph_svg": "<svg ...></svg>",
  "pregenerated_qa": [{"question": "...", "answer": "..."}],
  "qa_metadata": {}
}
```

### `service_bridge.js`

Creates `window.PepperMemoryService`.

It connects to the Python service through `qi.js`:

1. `ensureServiceConnected()` checks fake mode and avoids duplicate connection attempts.
2. It calls `QiSession(onConnect, onDisconnect)`.
3. It tries service names `PepperGroundedClient` and `peppergroundedclient`.
4. When resolved, Q/A buttons become enabled.

Button click flow:

1. User taps a Q/A button.
2. JS reads `data-question`.
3. JS checks service readiness and cooldown.
4. JS calls:

```javascript
service.answerCachedQuestion(currentLanguage(), question)
```

5. Python receives the call and starts cached answer turn.

The button cooldown prevents accidental multi-tap duplicate speech.

### `fake_tablet.js`

Creates `window.PepperMemoryFake`.

It detects fake mode through query string:

```text
?fake_tablet=1&poll_ms=1000
```

In fake mode it:

- Sets status to “Fake tablet mode”.
- Polls `/payload.json` with XHR.
- Calls `render.render(payload)`.

It uses `XMLHttpRequest` instead of `fetch` for older browser compatibility.

### `app.js`

Bootstrap script.

It:

1. Initializes fake mode from query.
2. Exposes Python bridge API:

```javascript
window.PepperMemoryPage = {
  renderFromBridge: function (payload) { ... }
};
```

3. Sets:

```javascript
window.PepperMemoryPageReady = true;
```

4. Starts fake polling or real Qi service connection.
5. Renders initial empty state.

## Q/A Buttons

Q/A buttons depend on two layers:

- The Python show-memory flow must fetch Q/A pairs and include them in payload.
- The tablet JS must connect to the Python Qi service.

The button call does not hit the server directly. It calls `PepperGroundedClient.answerCachedQuestion`, which lets Python use local cache and speech policy.

If cache misses, Python falls back to normal chat, so the button remains useful even when exact cache state changed.

## Styling

File: `client/app/html/css/style.css`

The UI uses a dark two-column card grid with:

- Responsive-ish fixed tablet viewport target.
- Object chips.
- Edge lists.
- White graph SVG background.
- Q/A buttons with answer previews.
- Status badge modes: `ok`, `warn`, `err`.

The `index.html` viewport is fixed to width 1280 for Pepper tablet density compatibility:

```html
<meta name="viewport" content="width=1280, user-scalable=no" />
```

## Where To Change Tablet UI

- Change content sections: `index.html` and `render.js`.
- Change visual style: `style.css`.
- Change Q/A button behavior: `service_bridge.js`.
- Change fake tablet polling: `fake_tablet.js`.
- Change Python payload shape: `TurnManager._build_memory_page_payload`.
- Change local robot loading: `TabletAdapter`.
- Change package inclusion: `pepper-grounded-client.pml` resources.
