import logging

from app.core.config import config_apply
from app.core.config import config_manager
from app.core.runtime.state import app_state
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/config")
async def get_config():
    """
    Returns the active and saved system configuration.

    Includes:
    - active: current runtime configuration
    - saved: configuration stored on disk
    - active_resolved: runtime config after defaults and inheritance
    - contracts: runtime behavior contracts/capabilities
    """
    saved = config_manager.load_config()
    active = app_state.config or saved
    hard_reload_fields = config_apply.hard_fields()
    return {
        "active": config_manager.dump_config(active),
        "saved": config_manager.dump_config(saved),
        "active_resolved": config_manager.resolve_config(active),
        "contracts": {
            **config_manager.behavior_contracts(),
            "hard_reload_fields": hard_reload_fields,
            "allowed_providers": [
                "openai",
                "gemini",
                "openai_compatible",
                "local_hf",
                "local_4bit",
            ],
        },
    }


@router.get("/state")
async def get_state():
    """
    Returns the latest pipeline output state.

    This includes:
    - last processed image
    - detected objects
    - scene graph
    - memory snapshot
    """
    return app_state.last_state or {
        "image": None,
        "objects": [],
        "scene_graph": [],
        "memory": {"objects": [], "relationships": [], "timestamp": None},
    }


@router.patch("/config")
async def patch_config(request: Request):
    """
    Applies a partial update to the runtime configuration.

    This endpoint:
    1. Validates and merges patch data.
    2. Computes differences between old and new configuration.
    3. Applies in-place, so-called hot changes immediately when possible.
    4. Rebuilds the pipeline if required (so-called hard changes)
    """
    if app_state.config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")
    data = await request.json()
    try:
        logger.info("Applying patch to config, with data: %s", data)
        updated = config_manager.apply_patch(app_state.config, data)
        diff = config_apply.diff_config(app_state.config, updated)
        # needs rebuild
        if diff.hard:
            logger.info(
                "Hard changes detected: %s. Rebuilding with new config.", diff.hard
            )
            await app_state.apply_config(updated)
            return {
                "ok": True,
                "reloaded": True,
                "applied": diff.hot,
                "requires_reload": diff.hard,
            }
        # can change settings, no need to rebuild
        await config_apply.apply_hot_config(app_state, updated)
        return {
            "ok": True,
            "reloaded": False,
            "applied": diff.hot,
            "requires_reload": [],
        }
    except ValueError as exc:
        logger.error(
            "Error applying patch to config: %s, error: %s",
            None or app_state.config,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/config/save")
async def save_config():
    """
    Saves the current runtime configuration to disk atomically.

    Uses a temporary file and replace strategy to avoid corruption.
    """
    if app_state.config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")
    path = config_manager.config_path()
    yaml_text = config_manager.dump_config_yaml(app_state.config)
    tmp = path.with_suffix(".tmp")
    logger.debug("Saving config to temp path: %s", tmp)
    tmp.write_text(yaml_text, encoding="utf-8")
    logger.debug("Replacing temp %s for %s", tmp, path)
    tmp.replace(path)
    logger.info("Configuration saved")
    return {"ok": True, "path": str(path)}


@router.post("/config/reload")
async def reload_config():
    """
    Reloads the configuration from disk and rebuilds the pipeline.
    """
    logger.info("Reloading config")
    cfg = config_manager.load_config()
    await app_state.apply_config(cfg)
    return {"ok": True}


@router.post("/config/upload")
async def upload_config(file: UploadFile = File(...)):
    """
    Uploads and applies a YAML configuration file.

    This replaces the current runtime configuration.
    """
    content = await file.read()
    try:
        logger.info("Parsing and applying uploaded yaml config: %s", content)
        cfg = config_manager.parse_uploaded_yaml(content)
        await app_state.apply_config(cfg)
        return {"ok": True}
    except ValueError as exc:
        logger.error("Invalid uploaded configuration")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/config/download")
async def download_config(source: str | None = None):
    """
    Downloads the active or saved configuration as a YAML file.

    Query params:
    - source=saved: download saved config
    - default: active config
    """
    if source == "saved":
        cfg = config_manager.load_config()
    else:
        cfg = app_state.config or config_manager.load_config()

    yaml_text = config_manager.dump_config_yaml(cfg)
    filename = "config_saved.yaml" if source == "saved" else "config.yaml"
    logger.info("User downloading config from source=%s", source)
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(
        content=yaml_text,
        media_type="application/x-yaml",
        headers=headers,
    )
