import logging
from pathlib import Path

from PIL import Image
import torch

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.model_registry import DetectionModelRegistry
from app.inference.types import InferenceDetectionObject

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Stateless service for running object detection on images using multiple backends.

    Supports YOLO, RT-DETR, RF-DETR, and OWL-ViT via DetectionModelRegistry.

    Attributes:
        backend (DetectionModelType): Backend model type.
        model_path (Path | None): Path to model weights, if applicable, or HF id.
        _ontology (list[str] | None): Optional list of class labels for open-vocabulary models.
        _threshold (float): Confidence threshold for filtering detections.
        _device (str | torch.device): Device for inference ('cpu' or 'cuda').
        model (BaseDetector): Loaded detector object for inference.
    """

    def __init__(
        self,
        model_name: DetectionModelType = DetectionModelType.RF_DETR,
        model_path: Path | None = None,
        ontology: list[str] | None = None,
        device: str | None = None,
        threshold: float = 0.5,
    ):
        logger.info(
            "Loading DetectionService with model_name=%s, model_path=%s, device=%s, threshold=%.2f",
            model_name,
            model_path,
            device,
            threshold,
        )
        self.backend = model_name
        self.model_path = (
            DetectionModelRegistry.ensure_model(model_name)
            if model_path is None
            else model_path
        )
        self._ontology = ontology
        self._threshold = threshold
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Detection Model: %s on %s", self.model_path, self.device)
        self.model = DetectionModelRegistry.load_detector(
            model_name, self.device, threshold, ontology
        )

    # All these properties need to propagate all the way to the model
    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        logger.debug(
            "Updating DetectionService threshold to %f. Propagating down to the model",
            value,
        )
        self._threshold = value

        if hasattr(self.model, "threshold"):
            self.model.threshold = value
            logger.info("Detection Threshold set to %.2f", self.model.threshold)

    @property
    def device(self) -> str | None | torch.device:
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
        logger.debug("Updating DetectionService device to %s", device)
        self._device = device
        if hasattr(self.model, "device"):
            self.model.device = device
            logger.info("Detection Model device set to %s", device)

    @property
    def ontology(self) -> list[str] | None:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: list[str] | None):
        self._ontology = ontology
        if hasattr(self.model, "set_ontology"):
            logger.info(
                "Updating DetectionService ontology to %d objects: %s",
                len(ontology) if ontology else 0,
                ontology[:5] if ontology else [],
            )
            self.model.set_ontology(ontology)

    def detect(self, image: Image.Image) -> list[InferenceDetectionObject]:
        """
        Run object detection on a single image.

        Args:
            image (PIL.Image.Image): Image to run detection on.

        Returns:
            list[InferenceDetectionObject]: List of detected objects with bounding boxes, labels, and confidence scores.
        """
        logger.debug("Running DetectionService with backend: %s", self.backend)
        return self.model.predict(image)

    def detect_batch(
        self, images: list[Image.Image]
    ) -> list[list[InferenceDetectionObject]]:
        return self.model.predict_batch(images)

    def __call__(
        self, image: Image.Image
    ) -> list[InferenceDetectionObject] | list[list[InferenceDetectionObject]]:
        if isinstance(image, list):
            return self.detect_batch(image)
        return self.detect(image)
