import logging
from pathlib import Path
from typing import Literal

from PIL import Image
import torch
from torchvision.ops import nms
from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.model_registry import DetectionModelRegistry
from app.inference.types import InferenceDetectionObject
from collections import defaultdict

logger = logging.getLogger(__name__)

def apply_nms(
    detections: list[InferenceDetectionObject], iou_threshold: float = 0.5
) -> list[InferenceDetectionObject]:
    if not detections:
        return detections
    boxes = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
    scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)

    keep = nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep.tolist()]

def apply_nms_per_class(
    detections: list[InferenceDetectionObject], iou_threshold: float = 0.5
) -> list[InferenceDetectionObject]:
    if not detections:
        return detections
    grouped = defaultdict(list)

    for d in detections:
        grouped[d.class_id].append(d)

    final = []

    for cls, dets in grouped.items():
        boxes = torch.tensor([d.bbox for d in dets], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in dets], dtype=torch.float32)

        keep = nms(boxes, scores, iou_threshold)
        final.extend([dets[i] for i in keep.tolist()])

    return final

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
        run_nms_post_filter: bool = False,
        nms_iou_threshold: float = 0.5,
        nms_type: Literal["general", "per_class"] = "per_class",
    ):
        logger.info(
            "Loading DetectionService with model_name=%s, model_path=%s, device=%s, threshold=%.2f, run_nms_post_filter=%s, nms_type=%s, nms_iou_threshold=%.2f",
            model_name,
            model_path,
            device,
            threshold,
            run_nms_post_filter,
            nms_type,
            nms_iou_threshold,
        )
        self.backend = model_name
        self.model_path = (
            DetectionModelRegistry.ensure_model(model_name)
            if model_path is None
            else model_path
        )
        self._ontology = ontology
        self._threshold = threshold
        self._run_nms_post_filter = run_nms_post_filter
        self._nms_iou_threshold = 0.5
        self._nms_type: Literal["general", "per_class"] = "per_class"
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_type = nms_type
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
    def run_nms_post_filter(self) -> bool:
        return self._run_nms_post_filter

    @run_nms_post_filter.setter
    def run_nms_post_filter(self, value: bool):
        self._run_nms_post_filter = bool(value)
        logger.info("Detection NMS post-filter enabled=%s", self._run_nms_post_filter)

    @property
    def nms_iou_threshold(self) -> float:
        return self._nms_iou_threshold

    @nms_iou_threshold.setter
    def nms_iou_threshold(self, value: float):
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError("nms_iou_threshold must be between 0.0 and 1.0")
        self._nms_iou_threshold = numeric
        logger.info("Detection NMS IoU threshold set to %.2f", self._nms_iou_threshold)

    @property
    def nms_type(self) -> Literal["general", "per_class"]:
        return self._nms_type

    @nms_type.setter
    def nms_type(self, value: Literal["general", "per_class"]):
        if value not in ("general", "per_class"):
            raise ValueError("nms_type must be either 'general' or 'per_class'")
        self._nms_type = value
        logger.info("Detection NMS type set to %s", self._nms_type)

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

    def _maybe_apply_nms(
        self, detections: list[InferenceDetectionObject], source: str
    ) -> list[InferenceDetectionObject]:
        if not self._run_nms_post_filter or not detections:
            return detections

        before_count = len(detections)
        if self._nms_type == "general":
            filtered = apply_nms(detections, self._nms_iou_threshold)
        else:
            filtered = apply_nms_per_class(detections, self._nms_iou_threshold)
        after_count = len(filtered)
        logger.debug(
            "%s: applied NMS type=%s iou=%.2f before=%d after=%d",
            source,
            self._nms_type,
            self._nms_iou_threshold,
            before_count,
            after_count,
        )
        return filtered

    def detect(self, image: Image.Image) -> list[InferenceDetectionObject]:
        """
        Run object detection on a single image.

        Args:
            image (PIL.Image.Image): Image to run detection on.

        Returns:
            list[InferenceDetectionObject]: List of detected objects with bounding boxes, labels, and confidence scores.
        """
        logger.debug(
            "Running DetectionService detect backend=%s threshold=%.2f device=%s run_nms_post_filter=%s nms_type=%s nms_iou_threshold=%.2f",
            self.backend,
            self._threshold,
            self._device,
            self._run_nms_post_filter,
            self._nms_type,
            self._nms_iou_threshold,
        )
        detections = self.model.predict(image)
        return self._maybe_apply_nms(detections, source="detect")

    def detect_batch(
        self, images: list[Image.Image]
    ) -> list[list[InferenceDetectionObject]]:
        logger.debug(
            "Running DetectionService detect_batch backend=%s threshold=%.2f device=%s run_nms_post_filter=%s nms_type=%s nms_iou_threshold=%.2f batch_size=%d",
            self.backend,
            self._threshold,
            self._device,
            self._run_nms_post_filter,
            self._nms_type,
            self._nms_iou_threshold,
            len(images),
        )
        detections_batch = self.model.predict_batch(images)
        if not self._run_nms_post_filter:
            return detections_batch

        filtered_batch = []
        before_total = 0
        after_total = 0
        for idx, detections in enumerate(detections_batch):
            before_total += len(detections)
            filtered = self._maybe_apply_nms(
                detections, source=f"detect_batch[{idx}]"
            )
            after_total += len(filtered)
            filtered_batch.append(filtered)
        logger.debug(
            "detect_batch: NMS summary before_total=%d after_total=%d",
            before_total,
            after_total,
        )
        return filtered_batch

    def __call__(
        self, image: Image.Image
    ) -> list[InferenceDetectionObject] | list[list[InferenceDetectionObject]]:
        if isinstance(image, list):
            return self.detect_batch(image)
        return self.detect(image)
