import logging
from pathlib import Path

from PIL import Image
import torch

from ..types import DetectionObject
from .detector import DetectionModel
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class DetectionService:
    """Stateless service for object detection using YOLO/RT-DETR."""

    # here it might depend on the backend whether ultralytics or roboflow... but that is easy to guess...
    def __init__(
        self,
        model_name: DetectionModel = DetectionModel.RF_DETR,
        model_path: Path | None = None,
        device: str | None = None,
        threshold: float = 0.5,
    ):
        self.model_path = (
            ModelManager.ensure_model(model_name) if model_path is None else model_path
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Detection Model: {self.model_path} on {self.device}")
        self.model = ModelManager.load_detector(model_name, self.device, threshold)

    def detect(self, image: Image.Image) -> list[DetectionObject]:
        return self.model.predict(image)
