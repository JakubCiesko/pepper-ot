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

from app.inference.types import InferenceDetectionObject


class DetectionModelType(StrEnum):
    RT_DETR = "rt_detr"
    YOLO = "yolo"
    RF_DETR = "rf_detr"
    OWL_V2 = "owl_v2"


# TODO: maybe set to float16, no need for float32
class BaseDetector(ABC):
    """
    Abstract base class for all object detectors.

    Defines the interface every detector must implement:
      - `device` property for getting/setting computation device.
      - `predict` method for performing inference on a PIL.Image.
    """

    @property
    @abstractmethod
    def device(self):
        pass

    # it is needed to have this reimplemented as it is different for different providers.
    @device.setter
    @abstractmethod
    def device(self, value: str):
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> list[InferenceDetectionObject]:
        """
        Run inference on a single image.

        Args:
            image (PIL.Image.Image): The image to run detection on.

        Returns:
            list[InferenceDetectionObject]: A list of DetectionObject instances.
        """
        pass


class UltralyticsDetector(BaseDetector):
    """
    Detector wrapper for Ultralytics YOLO models.

    Args:
        model (ultralytics.engine.model.Model): A preloaded YOLO model.
        device (str): 'cpu' or 'cuda' device string.
        threshold (float): Confidence threshold for filtering detections.
    """

    def __init__(self, model: Model, device: str, threshold: float = 0.5):
        self.model = model.to(device)
        self._device = device
        self.threshold = threshold
        self.model.fuse()
        self.model.eval()

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value
        self.model.to(self._device)
        self.model.fuse()
        self.model.eval()

    def predict(self, image: Image.Image) -> list[InferenceDetectionObject]:
        results = self.model.predict(image, device=self.device, verbose=False)

        detections = [
            InferenceDetectionObject(
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
    """
    Detector wrapper for RF-DETR / Roboflow models.

    Args:
        model (rfdetr.detr.RFDETR): Preloaded RF-DETR model.
        device (str): 'cpu' or 'cuda' device string.
        threshold (float): Confidence threshold for filtering predictions.
    """

    def __init__(self, model: RFDETR, device: str, threshold: float = 0.5):
        self.model = model
        self._device = device
        self.threshold = threshold
        self.model.optimize_for_inference()

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value
        self.model.__init__(device=value)
        self.model.optimize_for_inference()

    def predict(self, image: Image.Image) -> list[InferenceDetectionObject]:
        results = self.model.predict(image, threshold=self.threshold)
        detections = [
            InferenceDetectionObject(
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
    """
    Detector wrapper for Open-vocabulary OWL-ViT models.

    Supports caching the ontology encoding for faster inference.

    Args:
        model_path (str | None): Path or HuggingFace identifier for the model.
        processor_path (str | None): Path or HF identifier for the processor.
        ontology (list[str] | None): List of class labels to detect. Defaults to COCO classes.
        device (str): Device to run inference on ('cpu' or 'cuda').
        threshold (float): Confidence threshold for filtering predictions.
    """

    def __init__(
        self,
        model_path: str | None = None,
        processor_path: str | None = None,
        ontology: list[str] | None = None,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        default_path = "google/owlv2-base-patch16-ensemble"
        self.processor_path = processor_path or default_path
        self.model_path = model_path or default_path
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_path).to(device)
        self.processor = Owlv2Processor.from_pretrained(
            self.processor_path, backend="torchvision"  # use_fast=True
        )
        self._device = device
        self.threshold = threshold
        self._ontology = None
        self._text_inputs = None
        self.set_ontology(ontology)

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value
        self.model.to(self._device)
        if self._text_inputs is not None:
            self._text_inputs = {
                k: v.to(self._device) for k, v in self._text_inputs.items()
            }

    def set_ontology(self, ontology: list[str] | None = None):
        """
        Encode and cache the ontology for efficient inference.

        Args:
            ontology (list[str] | None): List of class labels. If None, defaults to COCO classes.
        """
        self._ontology = ontology or list(COCO_CLASSES.values())
        self._text_inputs = self.processor(
            text=[self._ontology], return_tensors="pt", padding=True
        ).to(self.device)

    def predict(
        self, image: Image.Image, ontology: list[str] | None = None
    ) -> list[InferenceDetectionObject]:
        """
        Run OWL-ViT inference on an image using the cached or updated ontology.

        Args:
            image (PIL.Image.Image): Input image.
            ontology (list[str] | None): Optional ontology override for this prediction.

        Returns:
            list[InferenceDetectionObject]: List of detected objects with labels, confidence, and bounding boxes.
        """
        # custom ontology
        if ontology is not None:
            self.set_ontology(ontology)

        if self._text_inputs is None:
            # if ontology is None, it will just use cococlass...
            self.set_ontology(ontology)

        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**image_inputs, **self._text_inputs)

        target_sizes = [(image.height, image.width)]
        text_labels = [self._ontology]

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
            text_labels=text_labels,
        )
        detections = [
            InferenceDetectionObject(
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
