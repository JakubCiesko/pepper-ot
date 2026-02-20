import logging

from app.core.state import ml_state
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/threshold")
async def set_threshold(request: Request):
    """Dynamically updates the confidence threshold."""
    data = await request.json()
    new_threshold = float(data.get("threshold", 0.5))

    # Update global config safely
    ml_state.config.detection.confidence_threshold = new_threshold

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
