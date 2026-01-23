import asyncio
import base64
import io
import logging
import random

from app.models.detection_settings import DETECTION_CONFIG
from app.services.detection import DetectionService
from app.services.ws_manager import ws_manager
from fastapi import APIRouter
from fastapi import Request
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

logger = logging.getLogger(__name__)
router = APIRouter()
DETECTION_SERVICE = DetectionService()


def get_color_encoding(objects: list[dict]) -> dict[str, str]:
    logger.info("Running color encoding, computing unique colors for labels...")
    unique_labels = list({obj["label"] for obj in objects})
    colors = {
        label: tuple(random.choices(range(50, 256), k=3)) for label in unique_labels
    }
    logger.info(f"{len(colors)} colors assigned")
    return colors


def annotate_image(
    img_bytes: bytes, objects: list[dict], colors: dict[str, str]
) -> str:
    logger.info(f"Annotating image with {len(objects)} objects")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    num_objects = max(1, len(objects))
    font_size = max(10, int(h * 0.05 / (num_objects**0.5)))  # bulharske konstanty
    try:
        # font supportingunicode is beter
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()
    rect_width = max(2, int(h * 0.005))
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        conf = obj["confidence"]
        color = colors[label]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=rect_width)
        text = f"{label} {conf:.2f}"
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        text_bg = [x1, y1 - text_h - 2, x1 + text_w + 2, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 1, y1 - text_h - 1), text, fill="black", font=font)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@router.post("/detect")
async def detect(image: UploadFile) -> JSONResponse:
    """API endpoint which runs object recognition inference on a single image instance."""
    img_bytes = await image.read()
    try:
        logger.info("Running detection endpoint...")
        response = await run_in_threadpool(DETECTION_SERVICE.detect, img_bytes)
        response_dict = response.model_dump()
        confidence_threshold = DETECTION_CONFIG.get("confidence_threshold")
        if confidence_threshold is not None:
            response_dict["objects"] = [
                obj
                for obj in response_dict["objects"]
                if obj["confidence"] > confidence_threshold
            ]
        colors = get_color_encoding(response_dict["objects"])
        annotated_image_b64 = await run_in_threadpool(
            annotate_image, img_bytes, response_dict["objects"], colors
        )
        broadcast_message = {
            "objects": response_dict["objects"],
            "image": annotated_image_b64,
            "colors": colors,
        }
        asyncio.create_task(ws_manager.broadcast(broadcast_message))
        return JSONResponse(status_code=200, content=response_dict)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return JSONResponse(content={"error": "Inference failed", "detail": str(e)})


@router.post("/config/threshold")
async def set_threshold(request: Request):
    """Api config endpoint which sets detection threshold.
    Object with confidence below threshold will be ignored."""
    data = await request.json()
    logger.info(
        f"Received request to change confidence threshold to {data['threshold']}"
    )
    global DETECTION_CONFIG
    DETECTION_CONFIG["confidence_threshold"] = float(data["threshold"])
    logger.info(
        f"Threshold set to {DETECTION_CONFIG['confidence_threshold']}. Detection config: {DETECTION_CONFIG}"
    )
    return {"ok": True, "threshold": DETECTION_CONFIG["confidence_threshold"]}


@router.post("/config/model")
async def set_model(request: Request):
    """Api config endpoint which sets detection model from the possible already downloaded detection models.

    NOTE: run model download manually."""
    data = await request.json()

    global DETECTION_SERVICE, DETECTION_CONFIG

    logger.info(f"Received request to change detection model config with: {data}")
    model_name = data.get("model")
    if model_name not in DETECTION_SERVICE.available_models:
        return {"ok": False, "error": "Model not available"}
    logger.info("Reloading detection service with model: " + model_name)
    DETECTION_CONFIG["model"] = model_name
    await DETECTION_SERVICE.reload_with_model(model_name)
    logger.info(f"Detection config: {DETECTION_CONFIG}")
    return {"ok": True, "selected_model": model_name}


@router.post("/config/language")
async def set_language(request: Request):
    """
    API config endpoint which sets the language for label translations.

    Accepted values: any language supported by googletrans (e.g., 'en', 'cs', 'fr'...).
    After changing the language, translations are reloaded in DetectionService.
    """
    data = await request.json()
    lang = data.get("language", "en").strip().lower()

    global DETECTION_SERVICE, DETECTION_CONFIG

    logger.info(f"Received request to change detection language to: {lang}")
    DETECTION_CONFIG["language"] = lang
    DETECTION_SERVICE.settings.language = lang
    DETECTION_SERVICE.translate_to = lang
    DETECTION_SERVICE.translation_path = DETECTION_SERVICE.model_path.with_suffix(
        f".{DETECTION_SERVICE.translate_to}.labels.json"
    )
    await DETECTION_SERVICE.initialize_translations()

    logger.info(f"Language set to {lang}")
    return {"ok": True, "language": lang}
