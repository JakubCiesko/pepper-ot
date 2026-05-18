# Packaging Deployment And Development

The client is a robot application package plus Python source. It targets Pepper/NAOqi Python 2 but also has local development paths for a virtual robot, fake camera, and fake tablet.

## Package Root

Package root: `client/app`

Important package files:

| File | Purpose |
|---|---|
| `manifest.xml` | Robot package metadata, service declaration, supported languages, NAOqi requirements. |
| `pepper-grounded-client.pml` | qipkg package manifest listing scripts, topics, resources, translations, and ignored paths. |
| `Makefile` | qipkg package, install, and remote deploy helpers. |
| `icon.png` | App icon. |
| `testrun/behavior.xar` | Behavior content referenced by the package. |

## Manifest

File: `client/app/manifest.xml`

Key facts:

- Package uuid is `pepper-grounded-client`.
- Package version is currently `0.1.39`.
- Supported languages are `en_US` and `cs_CZ`.
- Dialog content includes both English and Czech topic files.
- NAOqi requirement is min/max `2.5`.
- Robot model requirement is `JULIETTE_Y20`.
- Service declaration starts:

```xml
<service name="PepperGroundedClient" execStart="/usr/bin/python2 scripts/peppergroundedclient.py" autorun="true"/>
```

This service name must stay aligned with QiChat and tablet JS service lookup.

## PML Package File

File: `client/app/pepper-grounded-client.pml`

The PML lists:

- Manifest source.
- Behavior description.
- Dialog mapping.
- All Python source files.
- Config JSON.
- Translation files.
- Tablet HTML/CSS/JS files.
- Icon.
- Topic files.
- Ignored generated files.

If you add a new source file or tablet asset, add it to `<Resources>`. Otherwise it may work locally but not after packaging to the robot.

Current important resource groups:

- `scripts/peppergroundedclient.py`.
- `scripts/stk/*.py`.
- `scripts/pepper_client/core/*.py`.
- `scripts/pepper_client/interaction/*.py`.
- `scripts/pepper_client/perception/*.py`.
- `scripts/pepper_client/utils/*.py`.
- `html/index.html`.
- `html/css/style.css`.
- `html/js/*.js`.
- `translations/*.qm` and `.ts`.

## Makefile

File: `client/app/Makefile`

Targets:

| Target | Behavior |
|---|---|
| `pkg` | Bumps package version and builds `.pkg` using `qipkg`. |
| `install` | Installs latest package from `$(MYAPPS_DIR)` using `install_pkg.py`. |
| `deploy` | Builds package, uploads it to a remote machine, and runs remote install command. |

Environment variables used:

- `PEPPER_ROOT`.
- `FI_MUNI_USER`.
- `FI_THESIS_HOST`.
- `FI_THESIS_REMOTE_INSTALL_CMD`.

The target uses `rm -fv` to remove old package artifacts in the app directory during packaging. Be careful when invoking it in a dirty worktree.

## Python Package Metadata

File: `client/setup.py`

This is a Python package definition for `pepper-grounded-client`:

- Source package dir is `app/scripts`.
- Packages are discovered under `app/scripts`.
- Requirements are loaded from `client/requirements.txt`.
- Python version is `>=2.7,<3`.

Requirements:

```text
Pillow==6.2.2
requests==2.27.1
```

NAOqi/qi modules are expected from the robot/SDK environment, not from this requirements file.

## Local Start Script

File: `client/start_service.sh`

Current content:

```sh
$PEPPER_ROOT/pyenv/bin/python2 app/scripts/peppergroundedclient.py --qi-url tcp://127.0.0.1:$VIRTUAL_ROBOT_PORT
```

This is for local virtual robot testing. It assumes:

- `PEPPER_ROOT` points to an SDK/root with Python 2 environment.
- `VIRTUAL_ROBOT_PORT` is set.
- The service script can connect to virtual NAOqi.

## Main Script Debug Mode

At the bottom of `peppergroundedclient.py`, the `__main__` block currently has:

```python
run_local = True
czech = True
```

When executed directly, it:

1. Starts a `qi.Application`.
2. Registers `PepperGroundedClient` manually.
3. Calls `on_start`.
4. Optionally sets ALDialog/TTS language to Czech.
5. Runs the app.
6. Calls `on_stop` after app exits.

If `run_local` were false, it would use `stk.runner.run_service`.

This direct debug path is useful but also means behavior differs from packaged robot autorun. Keep that in mind when debugging startup.

## STK Helpers

Directory: `client/app/scripts/stk`

These are small local helper modules inspired by Aldebaran sample tooling.

### `stk/services.py`

`ServiceCache` lazily resolves services from `qi.Session` through `__getattr__`.

Special behavior: `ALTabletService` is refreshed each time because tablet service availability can change or stale proxies are common.

### `stk/logging.py`

`SafeLogger` wraps `qi.logging.Logger` and converts unicode/list/dict arguments to UTF-8-safe values before logging.

This is important because Python 2 NAOqi logging can fail on non-ASCII Czech strings.

### `stk/runner.py`

Provides:

- Robot/local detection.
- Prompt for robot host when off robot.
- `qi.Application` initialization.
- `run_activity` and `run_service` helpers.

### `stk/events.py`

Contains `EventHelper` for ALMemory event subscriptions, but it is marked as unused/removal in comments. Current production client code does not use it.

## Translation Files

Directory: `client/app/translations`

Files:

- `translation_cs_CZ.ts`.
- `translation_en_US.ts`.
- Compiled `.qm` files.

They are minimal placeholders currently. The actual spoken strings are mostly in QiChat topics and Python `speech_policy.py`, not Qt translation catalogs.

## Generated And Editor Files

The repository currently contains many generated files under `client/app/scripts`:

- `.pyc` files.
- `__pycache__` directories.
- `.csp` temporary translation files.

The PML has an extensive `<IgnoredPaths>` section to exclude many of them from packaging.

These generated files are not source of truth. Do not edit them. If packaging includes a new generated file by mistake, update ignored paths or clean the app directory.

Editor/support files:

- `client/.idea/*`.
- `client/.vscode/settings.json`.
- `client/vs-code-highlight-qichat/*`.

These are development aids, not robot runtime files.

## Example Tablet Application

Directory: `client/html example/dialog_presentation_nlp-master`

This is an external example app used as a reference for robot-hosted tablet content and `qimessaging`/Qi JS patterns. It is not part of the `pepper-grounded-client` package.

Useful reference file:

- `html/js/robotutils.js`.

The current client does not use `robotutils.js`; it directly uses `/libs/qi/2/qi.js` from the robot-hosted page.

## VS Code QiChat Highlighter

Directory: `client/vs-code-highlight-qichat`

This is a local VS Code extension package for `.top` syntax highlighting. It is unrelated to robot runtime.

Files:

- `package.json` language and grammar contribution.
- `syntaxes/qichat.tmLanguage.json` TextMate grammar.
- `qichat-0.0.1.vsix` packaged extension.

## Development Modes

### Real Robot Mode

Use when deployed to Pepper:

- `capture.fake_camera=false`.
- `tablet.fake_tablet=false`.
- Server URL reachable from the robot network.
- App installed as robot package.
- `ALTabletService`, `ALVideoDevice`, `ALMotion`, `ALDialog`, and speech services available.

### Virtual Robot / Desktop Mode

Use for local iteration:

- `capture.fake_camera=true` and `capture.fake_camera_path` points to images.
- `tablet.fake_tablet=true`.
- Start service through `client/start_service.sh` or direct Python 2 command.
- Open logged fake tablet URL in a browser.

### Mixed Mode

You can use fake camera with real server or fake tablet with real robot, but be explicit about what is fake. Fake camera uses artificial FOV and random image selection, so geometry metadata is not representative.

## Packaging Checklist

Before deploying a client change:

1. Add new Python files to PML resources.
2. Add new tablet JS/CSS/image files to PML resources.
3. Ensure QiChat topics are referenced in both `.dlg` and PML topics.
4. Ensure `manifest.xml` service name matches `PepperGroundedClient` calls.
5. Remove or ignore generated `.pyc`/cache files.
6. Confirm `client_config.json` values are deploy-safe.
7. Build package with `make pkg` from `client/app`.
8. Install through `make install` or the remote deploy path.
