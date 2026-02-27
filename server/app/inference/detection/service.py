import logging
from pathlib import Path

from PIL import Image
import torch

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.model_registry import DetectionModelRegistry
from app.inference.types import DetectionObject

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
        logger.info(
            f"Loading DetectionService with model_name={model_name}, model_path={model_path}, device={device}, threshold={threshold}"
        )
        self.model_path = (
            DetectionModelRegistry.ensure_model(model_name)
            if model_path is None
            else model_path
        )
        self._threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Detection Model: {self.model_path} on {self.device}")
        self.model = DetectionModelRegistry.load_detector(
            model_name, self.device, threshold
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        logger.info(
            f"Updating DetectionService threshold to {value}. Propagating down to the model"
        )
        self._threshold = value

        # propagate to actual model
        if hasattr(self.model, "threshold"):
            self.model.threshold = value
            logger.info(f"Threshold set to {self.model.threshold}")

    def detect(self, image: Image.Image) -> list[DetectionObject]:
        return self.model.predict(image)
