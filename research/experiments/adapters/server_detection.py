from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import ensure_server_app_importable


class ServerDetectionAdapter:
    def __init__(self, backend: str):
        ensure_server_app_importable()
        from app.inference.detection.detectors import DetectionModelType
        from app.inference.detection.service import DetectionService

        backend_enum = DetectionModelType(backend)
        self._service = DetectionService(backend_enum)
        self._backend_enum = DetectionModelType

    @property
    def service(self) -> Any:
        return self._service

    def detect_image(self, image: Image.Image) -> list[dict[str, Any]]:
        detections = self._service.detect(image)
        return [det.model_dump() for det in detections]

    def detect_images(
        self, image_paths: list[Path], batch_size: int = 4
    ) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        batch: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path in image_paths:
            with Image.open(path) as img:
                batch.append(img.convert("RGB").copy())
                batch_paths.append(path)
            if len(batch) < batch_size:
                continue
            results = self._service.detect_batch(batch)
            for p, dets in zip(batch_paths, results, strict=True):
                output[str(p)] = [det.model_dump() for det in dets]
            batch, batch_paths = [], []

        if batch:
            results = (
                self._service.detect_batch(batch)
                if len(batch) > 1
                else [self._service.detect(batch[0])]
            )
            for p, dets in zip(batch_paths, results, strict=True):
                output[str(p)] = [det.model_dump() for det in dets]
        return output
