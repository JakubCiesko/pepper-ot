import logging

from app.models.detection import DetectionObject
from app.services.model_manager import ModelManager
from PIL import Image
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionService:
    """Stateless service for object detection using YOLO/RT-DETR."""

    def __init__(self, model_name: str = "rtdetr-x.pt", device: str = None):
        self.model_path = ModelManager.ensure_model(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Detection Model: {self.model_path} on {self.device}")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def detect(self, image: Image.Image) -> list[DetectionObject]:
        results = self.model.predict(image, device=self.device, verbose=False)
        detections = []

        for r in results:
            for box, cls, conf in zip(
                r.boxes.xyxy, r.boxes.cls, r.boxes.conf, strict=True
            ):
                label = self.model.names[int(cls)]
                detections.append(
                    DetectionObject(
                        label=label,
                        confidence=float(conf),
                        bbox=[float(x) for x in box],
                    )
                )
        return detections
