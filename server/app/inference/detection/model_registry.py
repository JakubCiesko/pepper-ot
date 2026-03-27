import logging
from pathlib import Path
import urllib.request

from rfdetr import RFDETRLarge
import torch
from ultralytics import RTDETR
from ultralytics import YOLO

from app.inference.detection.detectors import BaseDetector
from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.detectors import Owlv2Detector
from app.inference.detection.detectors import RoboflowDetector
from app.inference.detection.detectors import UltralyticsDetector

logger = logging.getLogger(__name__)


class DetectionModelRegistry:
    """
    Registry and manager for all object detection models.

    Handles:
      - Ensuring local model files exist.
      - Downloading weights if missing.
      - Loading detector objects with correct backend and device.

    Attributes:
        MODELS_DIR (Path): Directory where model weights are stored.
        REGISTRY (dict[DetectionModelType, str|None]): Maps model types to download URLs.
    """

    MODELS_DIR = Path(__file__).resolve().parents[3] / "detection_models"

    # Registry of models used in the project; can be expanded
    REGISTRY = {
        DetectionModelType.RT_DETR: "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
        DetectionModelType.YOLO: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        DetectionModelType.RF_DETR: None,  # no weights file
        DetectionModelType.OWL_V2: None,
    }

    @classmethod
    def get_model_path(cls, model_name: DetectionModelType) -> Path:
        """
        Returns the local path for a given model.

        Args:
            model_name (DetectionModelType): The type of detection model.

        Returns:
            Path: Path to the local model file.
        """
        return cls.MODELS_DIR / cls.REGISTRY[model_name].split("/")[-1]

    @classmethod
    def ensure_model(cls, model_name: DetectionModelType) -> Path | None:
        """
        Ensure that the model file exists locally, downloading it if necessary.

        Args:
            model_name (DetectionModelType): The type of detection model.

        Returns:
            Path | None: Path to the model file, or None if no file is required (e.g., RF-DETR or OWL-V2).
        """
        logger.info("Ensuring model %s", model_name)
        url = cls.REGISTRY[model_name]
        if url is None:
            return None

        path = cls.MODELS_DIR / Path(url).name
        if not path.exists():
            cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        return path

    @classmethod
    def load_detector(
        cls,
        model_name: DetectionModelType,
        device: str | None = None,
        threshold: float = 0.5,
        ontology: list[str] | None = None,
    ) -> BaseDetector:
        """
        Load a detector object for a given model type.

        Args:
            model_name (DetectionModelType): Which detector backend to use.
            device (str | None): Device to run inference on ('cpu' or 'cuda').
                                 Defaults to CUDA if available.
            threshold (float): Confidence threshold for filtering detections.
            ontology (list[str] | None): Optional ontology for open-vocabulary models.

        Returns:
            BaseDetector: A detector object ready for inference.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = cls.ensure_model(model_name)
        match model_name:
            case DetectionModelType.YOLO:
                logger.info("Loading detection model with Ultralytics YOLO Backend")
                return UltralyticsDetector(YOLO(path), device, threshold)
            case DetectionModelType.RT_DETR:
                logger.info("Loading detection model with Ultralytics RT-DETR Backend")
                return UltralyticsDetector(RTDETR(str(path)), device, threshold)
            case DetectionModelType.RF_DETR:
                logger.info("Loading detection model with Roboflow RF-DETR Backend")
                return RoboflowDetector(RFDETRLarge(device=device), device, threshold)
            case DetectionModelType.OWL_V2:
                logger.info(
                    "Loading detection model with Google OpenVocab OwlV2 Backend"
                )
                # for now processor located at the same place
                # ontologies too probably...
                return Owlv2Detector(path, path, ontology, device, threshold)
