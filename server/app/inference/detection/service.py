import logging
from pathlib import Path

from PIL import Image
import torch

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.model_registry import DetectionModelRegistry
from app.inference.types import DetectionObject

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
            f"Loading DetectionService with model_name={model_name}, model_path={model_path}, device={device}, threshold={threshold}"
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
        logger.info(f"Loading Detection Model: {self.model_path} on {self.device}")
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
            f"Updating DetectionService threshold to {value}. Propagating down to the model"
        )
        self._threshold = value

        if hasattr(self.model, "threshold"):
            self.model.threshold = value
            logger.info(f"Detection Threshold set to {self.model.threshold}")

    @property
    def device(self) -> str | None | torch.device:
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
        logger.debug(f"Updating DetectionService device to {device}")
        self._device = device
        if hasattr(self.model, "device"):
            self.model.device = device
            logger.info(f"Detection Model device set to {device}")

    @property
    def ontology(self) -> list[str] | None:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: list[str] | None):
        self._ontology = ontology
        if hasattr(self.model, "set_ontology"):
            logger.info(
                f"Updating DetectionService ontology to {len(ontology) if ontology else 0} objects: [{','.join(ontology[:5] if ontology else [])}...]"
            )
            self.model.set_ontology(ontology)

    def detect(self, image: Image.Image) -> list[DetectionObject]:
        """
        Run object detection on a single image.

        Args:
            image (PIL.Image.Image): Image to run detection on.

        Returns:
            list[DetectionObject]: List of detected objects with bounding boxes, labels, and confidence scores.
        """
        logger.debug(f"Running DetectionService with backend: {self.backend}")
        return self.model.predict(image)

    def __call__(self, image: Image.Image) -> list[DetectionObject]:
        return self.detect(image)
