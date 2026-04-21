# Assets, Tests, and Support Files

This document explains non-core-code server files: config, prompts, ontology, translation lexicons, static assets, runtime state, and tests/support files.

## Root Server Files

### `server/config.yaml`

Default runtime config loaded by `AppConfig.load`. It is also the file saved by `/api/v1/config/save` and reloaded by `/api/v1/config/reload`.

When adding config fields, update:

- `server/app/schemas/config.py`
- `server/config.yaml`
- config reload rules
- dashboard template/JS if operator-facing
- docs in this directory

### `server/app/main.py`

FastAPI entrypoint. Mounts static files, API, and dashboard.

## Prompts

Prompt sources can be inline in YAML or stored under allowed prompt roots. The config manager validates uploaded prompt paths for safety.

Prompt fields currently exist for:

- scene graph VLM system/user prompt
- chat general system/user prompt
- object chat system/user prompt
- caption system/user prompt

Prompt rendering uses `server/app/core/prompting/renderer.py`. It is simple placeholder replacement, not Jinja.

Common placeholders:

- `{context}`
- `{caption}`
- `{captions_recent}`
- `{caption_recent}`
- `{predicates}`
- `{query}`
- `{history}`
- object-chat placeholders such as `{object_context}` and `{matched_ids}`

## Ontology

Detection and VLM scene graph can use ontology terms from config or files.

Detection ontology:

- inline `detection.ontology`
- or `detection.ontology_path`

VLM scene graph ontology:

- `scene_graph.vlm.ontology.predicates`
- `scene_graph.vlm.ontology.objects`
- optional ontology path

The vocabulary translator warms Czech display translations from these terms.

## Translation Lexicons

Files:

- `server/app/providers/translation/lexicons/labels_cs.json`
- `server/app/providers/translation/lexicons/attributes_cs.json`
- `server/app/providers/translation/lexicons/relations_cs.json`
- `server/app/providers/translation/lexicons_user/labels_cs.user.json`
- `server/app/providers/translation/lexicons_user/attributes_cs.user.json`
- `server/app/providers/translation/lexicons_user/relations_cs.user.json`

Static lexicons are shipped defaults. User lexicons are editable through dashboard and override static entries.

`VocabularyTranslationService` creates missing user lexicon files if needed and atomically writes updates.

## Static Dashboard Assets

Files:

- `server/app/static/templates/dashboard.html`
- `server/app/static/templates/dashboard/pages/*.html`
- `server/app/static/js/dashboard/**/*.js`
- `server/app/static/css/style.css`
- `server/app/static/pepper_icon.png`

The dashboard is served from `/dashboard` and static assets from `/static`.

JavaScript is modular ES modules under `static/js/dashboard`. The root module is `dashboard/app.js`.

## Runtime State Files

Runtime-generated files may live under `server/state` depending on config and backend behavior.

Examples:

- persisted last-state JSON
- persisted last image when `storage.store_image=true`
- temporary RelTR input images during RelTR prediction

Do not treat `state` files as source-of-truth database storage. Scene memory and QA pool are in memory unless explicitly persisted by last-state support.

## Model Files

Detector and RelTR checkpoints may be referenced by config:

- `detection.weights_path`
- `scene_graph.reltr.checkpoint_path`

The exact location is deployment-specific. Model registry code may download or load backend-specific weights.

## Tests

Server tests live under `server/tests` when present. Use tests to verify:

- endpoint contracts
- config validation and reload behavior
- detection pipeline branches
- memory CRUD
- chat language behavior
- scene graph parsing/filtering
- QA pool behavior

Run typical server tests with:

```bash
pytest server/tests -q
```

## Local Development Checks

Useful checks:

```bash
python3 -m py_compile $(find server/app -name '*.py' | sort)
```

```bash
node --check server/app/static/js/dashboard/features/config/index.js
```

For all dashboard JS:

```bash
for f in $(find server/app/static/js/dashboard -name '*.js' | sort); do node --check "$f" || exit 1; done
```

## Safe Path Validation

`config_manager._validate_paths` restricts uploaded YAML prompt and ontology paths. Prompt paths must stay under prompt roots; ontology paths must stay under ontology roots. Absolute paths and `..` traversal are rejected.

## Documentation Updates

When changing support files:

- If config shape changes, update `configuration-and-reload.md`.
- If dashboard assets change, update `dashboard-and-operator-ui.md`.
- If prompts/ontology behavior changes, update this file and provider/pipeline docs.
- If tests are added for a subsystem, mention them in the subsystem doc if they define expected behavior.
