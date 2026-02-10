import logging
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles integrity, paths, and downloads for all object detection models."""

    MODELS_DIR = Path(__file__).parent.parent.parent / "detection_models"

    # Registry of models used in the project; can be expanded
    REGISTRY: dict[str, str] = {
        "rtdetr-x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
        "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    }

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        return cls.MODELS_DIR / model_name

    @classmethod
    def ensure_model(cls, model_name: str):
        path = cls.get_model_path(model_name)
        if not path.exists():
            if model_name not in cls.REGISTRY:
                raise FileNotFoundError(
                    f"Model {model_name} not found and no URL registered."
                )

            logger.info(f"Downloading model {model_name}...")
            cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(cls.REGISTRY[model_name], path)
            logger.info(f"Downloaded {model_name} to {path}")
        return path
