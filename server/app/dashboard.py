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


@router.post("/dashboard/chat_message")
async def dashboard_chat_message(payload: dict):
    """
    Broadcast a canonical chat message event to dashboard WebSocket clients.
    Example payload:
    {
      "text": "Hello",
      "role": "assistant",
      "chat_id": "optional-conversation-id"
    }
    """
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"status": "error", "msg": "No text"}

    role = str(payload.get("role", "assistant")).strip().lower()
    if role not in {"user", "assistant"}:
        return {"status": "error", "msg": "Invalid role"}

    chat_id = payload.get("chat_id")
    await ws_manager.broadcast(
        {
            "type": "chat_message",
            "chat_id": chat_id,
            "message": {
                "id": None,
                "role": role,
                "text": text,
                "timestamp": None,
            },
        }
    )
    return {"status": "ok"}
