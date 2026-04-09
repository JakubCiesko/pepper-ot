from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

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

    def model_optimization(self, current_batch_size: int, batch_size: int = 4) -> bool:
        needs_reoptimization = current_batch_size != batch_size
        reoptimized = False
        if needs_reoptimization and self._service.backend == self._backend_enum.RF_DETR:
            self._service.model.model.optimize_for_inference(
                batch_size=current_batch_size
            )
            reoptimized = True
        return reoptimized

    def detect_images(
        self, image_paths: list[Path], batch_size: int = 4
    ) -> dict[str, list[dict[str, Any]]]:
        self._service.model.model.optimize_for_inference(batch_size=batch_size)
        output: dict[str, list[dict[str, Any]]] = {}
        batch: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path in tqdm(image_paths, desc="Running inference on images", leave=False):
            with Image.open(path) as img:
                batch.append(img.convert("RGB").copy())
                batch_paths.append(path)
            # what will happen if data size is N*batch_size + 1?
            if len(batch) < batch_size:
                continue
            reoptimized = self.model_optimization(len(batch), batch_size)
            # weird thing from roboflow fix:
            if reoptimized and len(batch) == 1:
                results = [self._service.detect(batch[0])]
            else:
                results = self._service.detect_batch(batch)
            for p, dets in zip(batch_paths, results, strict=True):
                output[str(p)] = [det.model_dump() for det in dets]
            batch, batch_paths = [], []

        if batch:
            reoptimized = self.model_optimization(len(batch), batch_size)

            if reoptimized and len(batch) == 1:
                results = [self._service.detect(batch[0])]
            else:
                results = (
                    self._service.detect_batch(batch)
                    if len(batch) > 1
                    else [self._service.detect(batch[0])]
                )

            for p, dets in zip(batch_paths, results, strict=True):
                output[str(p)] = [det.model_dump() for det in dets]
        return output
