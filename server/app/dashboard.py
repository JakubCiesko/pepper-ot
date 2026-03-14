import logging

from fastapi import APIRouter
from fastapi import Request
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.infra.ws_manager import ws_manager
from app.inference.detection.detectors import DetectionModelType

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/dashboard")
async def dashboard(request: Request) -> HTMLResponse:
    """Serves html dashboard page."""
    logger.info("Received request to fetch dashboard html")
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.websocket("/dashboard/events")
async def dashboard_ws(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"Dashboard Events WebSocket Error: {e}")
        ws_manager.disconnect(websocket)


@router.get("/dashboard/config/get_models")
async def list_models():
    """Return a list of available detection model backends."""
    try:
        return {"models": [m.value for m in DetectionModelType]}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.post("/dashboard/sentence")
async def dashboard_sentence(payload: dict):
    """
    Receive sentences spoken by Pepper and broadcast them to WebSocket clients.
    Example payload: { "sentence": "Hello, I am Pepper!" }
    """
    sentence = payload.get("sentence", "")
    if not sentence:
        return {"status": "error", "msg": "No sentence"}

    await ws_manager.broadcast({"type": "sentence", "text": sentence})

    return {"status": "ok"}
