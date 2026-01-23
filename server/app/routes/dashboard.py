import logging

from app.routes.detect import DETECTION_SERVICE
from app.services.ws_manager import ws_manager
from fastapi import APIRouter
from fastapi import Request
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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
        logger.error(f"Dashboard Events WebSocker Error: {e}")
        ws_manager.disconnect(websocket)


@router.get("/dashboard/config/get_models")
async def list_models():
    """Return a list of available detection models stored in detection_models folder."""
    try:
        models = DETECTION_SERVICE.available_models
        return {"models": models}
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

    # Broadcast to all connected WebSocket clients
    await ws_manager.broadcast({"type": "sentence", "text": sentence})

    return {"status": "ok"}
