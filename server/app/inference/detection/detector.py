from abc import ABC
from abc import abstractmethod
from enum import StrEnum

from PIL import Image
from rfdetr.detr import RFDETR
from rfdetr.util.coco_classes import COCO_CLASSES
from ultralytics.engine.model import Model

from ..types import DetectionObject


# TODO: maybe set to float16, no need for float32
class BaseDetector(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> list[DetectionObject]:
        pass


class DetectionModel(StrEnum):
    RT_DETR = "rt_detr"
    YOLO = "yolo"
    RF_DETR = "rf_detr"


class UltralyticsDetector(BaseDetector):
    def __init__(self, model: Model, device: str, threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.fuse()
        self.model.eval()

    def predict(self, image: Image.Image) -> list[DetectionObject]:
        results = self.model.predict(image, device=self.device, verbose=False)

        detections = [
            DetectionObject(
                class_id=int(cls),
                object_id=i,
                label=self.model.names[int(cls)],
                confidence=float(conf),
                bbox=[float(x) for x in box],
            )
            for r in results
            for i, (box, cls, conf) in enumerate(
                zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf, strict=True)
            )
        ]
        return list(filter(lambda det: det.confidence >= self.threshold, detections))


class RoboflowDetector(BaseDetector):
    def __init__(self, model: RFDETR, device: str, threshold: float = 0.5):
        self.model = model
        self.dummy_device = device
        self.device = (
            model.model.device
        )  # right now, i just let roboflow get the device
        # model.model.model.to(device)
        self.threshold = threshold
        self.model.optimize_for_inference()

    def predict(self, image: Image.Image) -> list[DetectionObject]:
        results = self.model.predict(image, threshold=self.threshold)
        detections = [
            DetectionObject(
                class_id=class_id,
                object_id=i,
                label=COCO_CLASSES.get(class_id),
                confidence=float(score),
                bbox=list(map(float, box)),
            )
            for i, (box, score, class_id) in enumerate(
                zip(results.xyxy, results.confidence, results.class_id, strict=True)
            )
        ]
        return detections
