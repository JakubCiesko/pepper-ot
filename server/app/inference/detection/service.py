import logging
from pathlib import Path

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.model_registry import DetectionModelRegistry
from app.inference.types import DetectionObject
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class DetectionService:
    """Stateless service for object detection using YOLO/RT-DETR."""

    # here it might depend on the backend whether ultralytics or roboflow... but that is easy to guess...
    def __init__(
        self,
        model_name: DetectionModelType = DetectionModelType.RF_DETR,
        model_path: Path | None = None,
        device: str | None = None,
        threshold: float = 0.5,
    ):
        self.model_path = (
            DetectionModelRegistry.ensure_model(model_name)
            if model_path is None
            else model_path
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Detection Model: {self.model_path} on {self.device}")
        self.model = DetectionModelRegistry.load_detector(
            model_name, self.device, threshold
        )

    def detect(self, image: Image.Image) -> list[DetectionObject]:
        return self.model.predict(image)
