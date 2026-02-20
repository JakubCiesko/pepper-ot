import logging
from pathlib import Path
import urllib.request

from rfdetr import RFDETRMedium
import torch
from ultralytics import RTDETR
from ultralytics import YOLO

from app.inference.detection.detectors import BaseDetector
from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.detectors import RoboflowDetector
from app.inference.detection.detectors import UltralyticsDetector

logger = logging.getLogger(__name__)


class DetectionModelRegistry:
    """Handles integrity, paths, and downloads for all object detection models."""

    MODELS_DIR = Path.cwd().parent.parent / "detection_models"

    # Registry of models used in the project; can be expanded
    REGISTRY = {
        DetectionModelType.RT_DETR: "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
        DetectionModelType.YOLO: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        DetectionModelType.RF_DETR: None,  # no weights file
    }

    @classmethod
    def get_model_path(cls, model_name: DetectionModelType) -> Path:
        return cls.MODELS_DIR / cls.REGISTRY[model_name.value].split("/")[-1]

    @classmethod
    def ensure_model(cls, model_name: DetectionModelType) -> Path | None:
        logger.info(f"Ensuring model {model_name}")
        url = cls.REGISTRY[model_name]
        if url is None:
            return None

        path = cls.MODELS_DIR / Path(url).name
        if not path.exists():
            cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        return path

    @classmethod
    def load_detector(
        cls,
        model_name: DetectionModelType,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> BaseDetector:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = cls.ensure_model(model_name)
        match model_name:
            case DetectionModelType.YOLO:
                logger.info("Loading detection model with Ultralytics YOLO Backend")
                return UltralyticsDetector(YOLO(path), device, threshold)
            case DetectionModelType.RT_DETR:
                logger.info("Loading detection model with Ultralytics RT-DETR Backend")
                return UltralyticsDetector(RTDETR(path), device, threshold)
            case DetectionModelType.RF_DETR:
                logger.info("Loading detection model with Roboflow RF-DETR Backend")
                return RoboflowDetector(RFDETRMedium(), device, threshold)
