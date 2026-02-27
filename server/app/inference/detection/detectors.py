from abc import ABC
from abc import abstractmethod
from enum import StrEnum

from PIL import Image
from rfdetr.detr import RFDETR
from rfdetr.util.coco_classes import COCO_CLASSES
import torch
from transformers import Owlv2ForObjectDetection
from transformers import Owlv2Processor
from ultralytics.engine.model import Model

from app.inference.types import DetectionObject


# TODO: maybe set to float16, no need for float32
class BaseDetector(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> list[DetectionObject]:
        pass


class DetectionModelType(StrEnum):
    RT_DETR = "rt_detr"
    YOLO = "yolo"
    RF_DETR = "rf_detr"
    OWL_V2 = "owl_v2"


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


class Owlv2Detector(BaseDetector):
    def __init__(
        self,
        model_path: str | None = None,
        processor_path: str | None = None,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        default_path = "google/owlv2-base-patch16-ensemble"
        self.processor_path = processor_path or default_path
        self.model_path = model_path or default_path
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_path).to(device)
        self.processor = Owlv2Processor.from_pretrained(self.processor_path)
        self.device = device
        self.threshold = threshold

    def predict(
        self, image: Image.Image, ontology: list[str] | None = None
    ) -> list[DetectionObject]:
        ontology: list[str] = ontology or list(COCO_CLASSES.values())
        text_labels: list[list[str]] = [ontology]
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
            text_labels=text_labels,
        )
        # there is just one picture...
        detections = [
            DetectionObject(
                class_id=text_labels[0].index(text_label),
                object_id=i,
                label=text_label,
                confidence=float(score),
                bbox=list(map(float, box)),
            )
            for result in results
            for i, (box, score, text_label) in enumerate(
                zip(
                    result["boxes"],
                    result["scores"],
                    result["text_labels"],
                    strict=True,
                )
            )
        ]
        return detections
