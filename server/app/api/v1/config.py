import logging

from app.core import config_manager
from app.core.state import ml_state
from app.inference.detection.detectors import DetectionModelType
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter()

# TODO: make sort of updatable just that one changed thing


@router.get("/config")
async def get_config():
    saved = config_manager.load_config()
    active = ml_state.config or saved
    return {
        "active": config_manager.dump_config(active),
        "saved": config_manager.dump_config(saved),
        "active_resolved": config_manager.resolve_config(active),
    }


@router.patch("/config")
async def patch_config(request: Request):
    if ml_state.config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")
    data = await request.json()
    try:
        logger.info(f"Applying patch to config, with data: {data}")
        updated = config_manager.apply_patch(ml_state.config, data)
        await ml_state.apply_config(updated)
        return {"ok": True}
    except ValueError as exc:
        logger.error(
            f"Error applying patch to config: {None or ml_state.config}, error: {exc}"
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/config/save")
async def save_config():
    if ml_state.config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")
    path = config_manager.config_path()
    yaml_text = config_manager.dump_config_yaml(ml_state.config)
    tmp = path.with_suffix(".tmp")
    logger.info(f"Saving config to temp path: {tmp}")
    tmp.write_text(yaml_text, encoding="utf-8")
    logger.info(f"Replacing temp {tmp} for {path}")
    tmp.replace(path)
    return {"ok": True, "path": str(path)}


@router.post("/config/reload")
async def reload_config():
    logger.info("Reloading config")
    cfg = config_manager.load_config()
    await ml_state.apply_config(cfg)
    return {"ok": True}


@router.post("/config/upload")
async def upload_config(file: UploadFile = File(...)):
    content = await file.read()
    try:
        logger.info(f"Parsing uploaded yaml config: {content}")
        cfg = config_manager.parse_uploaded_yaml(content)
        await ml_state.apply_config(cfg)
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/config/download")
async def download_config(source: str | None = None):
    if source == "saved":
        cfg = config_manager.load_config()
    else:
        cfg = ml_state.config or config_manager.load_config()
    # TODO: HIDE API KEYS IF PRESENT!
    yaml_text = config_manager.dump_config_yaml(cfg)
    filename = "config_saved.yaml" if source == "saved" else "config.yaml"
    logger.info(f"Downloading config from source={source} as file={filename}")
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(
        content=yaml_text,
        media_type="application/x-yaml",
        headers=headers,
    )


@router.post("/threshold")
async def set_threshold(request: Request):
    """Dynamically updates the confidence threshold."""
    data = await request.json()
    new_threshold = float(data.get("threshold", 0.5))

    # Update global config safely
    ml_state.config.detection.confidence_threshold = new_threshold
    ml_state.pipeline.set_detection_threshold(new_threshold)

    logger.info(f"Threshold updated to {new_threshold}")
    return {"ok": True, "threshold": new_threshold}


@router.post("/model")
async def set_model(request: Request):
    """Reloads the AI engine with a different weights file."""
    data = await request.json()
    model_name = data.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="No model specified")

    logger.info(f"Reloading AI Engine with model: {model_name}")

    # Update config
    if model_name in {m.value for m in DetectionModelType}:
        ml_state.config.detection.backend = model_name
    else:
        ml_state.config.detection.weights_path = model_name

    # Trigger engine reload (assuming ml_state has a method for this)
    await ml_state.reload_pipeline()

    return {"ok": True, "selected_model": model_name}


@router.post("/language")
async def set_language(request: Request):
    """Changes the output translation language."""
    data = await request.json()
    lang = data.get("language", "en").strip().lower()

    logger.info(f"Changing language to: {lang}")
    ml_state.config.system.language = lang
    # Trigger translation reload if applicable

    return {"ok": True, "language": lang}
