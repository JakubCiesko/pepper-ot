# Dashboard and Operator UI

## Files Covered

- `app/dashboard.py`
- `app/static/templates/dashboard.html`
- `app/static/templates/dashboard/pages/*.html`
- `app/static/js/dashboard/app.js`
- `app/static/js/dashboard/core/*.js`
- `app/static/js/dashboard/features/config/index.js`
- `app/static/js/dashboard/features/live/index.js`
- `app/static/js/dashboard/features/conversation/index.js`
- `app/static/js/dashboard/features/memory/*.js`
- `app/static/js/dashboard/features/scene_graph/index.js`
- `app/static/js/dashboard/features/ui_shell/*.js`

## Purpose

The dashboard is the operator-facing surface for observing and controlling the running system.

It is not just a demo page. It is a real runtime control plane for:
- config edits
- worker control
- live frame review
- chat
- memory editing
- scene graph inspection

## Backend glue: `app/dashboard.py`

Exposes:
- dashboard HTML route
- dashboard websocket route
- model listing endpoint
- dashboard chat message endpoint

## Frontend app bootstrap

Implemented in `static/js/dashboard/app.js`.

Main jobs:
- initialize feature modules
- create dashboard websocket
- dispatch websocket messages to feature handlers

## Core frontend utilities

### `core/http.js`
- safe JSON parsing
- JSON request helper

### `core/notifications.js`
- transient status messages to user

### `core/ws.js`
- dashboard websocket creation and reconnect behavior

## Feature: Config editor

Implemented in `features/config/index.js`.

This is the largest frontend module because it mirrors a large portion of `AppConfig`.

### Responsibilities

- load active/saved config from backend
- populate form controls
- build JSON patch payloads
- validate JSON-like kwargs textareas
- apply/save/reload/upload/download config
- show hot-reload vs hard-reload warnings
- surface pipeline presets and derived summaries
- expose worker settings, scene graph settings, provider settings, tracking settings, visualization settings, and storage settings

### Important coupling

Field IDs in the DOM must match config serialization assumptions in this file. If you rename config fields or schema structure, update this module.

## Feature: Live panel

Implemented in `features/live/index.js`.

Responsibilities:
- render latest processed frame
- maintain a recent frame carousel
- render detection list
- render metrics
- render compact scene graph summaries
- load last persisted state on page init

## Feature: Conversation panel

Implemented in `features/conversation/index.js`.

Responsibilities:
- send general chat requests
- send vision chat requests
- maintain active chat id
- append/replace conversation history in UI
- handle websocket chat updates

## Feature: Memory panel

Files:
- `features/memory/index.js`
- `features/memory/actions.js`
- `features/memory/api.js`
- `features/memory/dom_refs.js`
- `features/memory/parsers.js`
- `features/memory/render.js`

Responsibilities:
- fetch current memory
- render object and relation lists
- fill editors for manual CRUD operations
- parse bbox/attribute forms
- submit memory mutation requests

## Feature: Scene graph panel

Implemented in `features/scene_graph/index.js`.

Responsibilities:
- convert graph and memory into visualization elements
- initialize Cytoscape-like graph panel if present
- color-code labels/types

## UI shell utilities

Files:
- `features/ui_shell/index.js`
- `navigation.js`
- `tabs.js`
- `sidebar.js`
- `theme.js`

Responsibilities:
- page navigation
- grouped tabs
- sidebar behavior
- theme toggle

## Websocket message consumers

The dashboard reacts to runtime messages such as:
- detection payloads
- chat message payloads
- memory updates

If the dashboard looks stale but API calls still work, inspect the websocket dispatch chain from backend broadcast to `app.js` to feature-specific handler.

## Best Tweak Points

- improving operator ergonomics in config editor
- adding new runtime metrics to live panel
- adding visualization filters to memory/scene graph panels

## Risky Tweak Points

- changing payload shapes without updating feature handlers
- changing DOM IDs without updating `config/index.js`
- adding fields to config without exposing sensible defaults in UI
