from pathlib import Path
import urllib.request

from rfdetr import RFDETRMedium
import torch
from ultralytics import RTDETR
from ultralytics import YOLO

from .detector import BaseDetector
from .detector import DetectionModel
from .detector import RoboflowDetector
from .detector import UltralyticsDetector


class ModelManager:
    """Handles integrity, paths, and downloads for all object detection models."""

    MODELS_DIR = Path.cwd().parent.parent / "detection_models"

    # Registry of models used in the project; can be expanded
    REGISTRY = {
        DetectionModel.RT_DETR: "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
        DetectionModel.YOLO: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        DetectionModel.RF_DETR: None,  # no weights file
    }

    @classmethod
    def get_model_path(cls, model_name: DetectionModel) -> Path:
        return cls.MODELS_DIR / cls.REGISTRY[model_name.value].split("/")[-1]

    @classmethod
    def ensure_model(cls, model_name: DetectionModel) -> Path | None:
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
        model_name: DetectionModel,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> BaseDetector:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = cls.ensure_model(model_name)
        match model_name:
            case DetectionModel.YOLO:
                return UltralyticsDetector(YOLO(path), device, threshold)
            case DetectionModel.RT_DETR:
                return UltralyticsDetector(RTDETR(path), device, threshold)
            case DetectionModel.RF_DETR:
                return RoboflowDetector(RFDETRMedium(), device, threshold)
