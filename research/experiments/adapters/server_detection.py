from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .bootstrap import ensure_server_app_importable
from .utils import resize_pil_with_scale
from .utils import scale_xyxy_bbox


class ServerDetectionAdapter:
    def __init__(self, backend: str, confidence: float = 0.5):
        ensure_server_app_importable()
        from app.inference.detection.detectors import DetectionModelType
        from app.inference.detection.service import DetectionService

        backend_enum = DetectionModelType(backend)
        self._service = DetectionService(backend_enum, threshold=confidence)
        self._backend_enum = DetectionModelType

    @property
    def service(self) -> Any:
        return self._service

    def detect_image(
        self, image: Image.Image, max_image_size: int | None = None
    ) -> list[dict[str, Any]]:
        scale_x = scale_y = 1.0
        if max_image_size:
            image, (scale_x, scale_y) = resize_pil_with_scale(image, max_image_size)
        detections = self._service.detect(image)
        out: list[dict[str, Any]] = []
        for det in detections:
            payload = det.model_dump()
            if max_image_size:
                payload["bbox"] = scale_xyxy_bbox(payload["bbox"], scale_x, scale_y)
            out.append(payload)
        return out

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
        self,
        image_paths: list[Path],
        batch_size: int = 4,
        max_image_size: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        self._service.model.model.optimize_for_inference(batch_size=batch_size)
        output: dict[str, list[dict[str, Any]]] = {}
        batch: list[Image.Image] = []
        batch_paths: list[Path] = []
        batch_scales: list[tuple[float, float]] = []
        for path in tqdm(image_paths, desc="Running inference on images", leave=False):
            with Image.open(path) as img:
                img_pil = img.convert("RGB").copy()
                scale_x = scale_y = 1.0
                if max_image_size:
                    img_pil, (scale_x, scale_y) = resize_pil_with_scale(
                        img_pil, max_image_size
                    )
                batch.append(img_pil)
                batch_paths.append(path.resolve())
                batch_scales.append((scale_x, scale_y))
            # what will happen if data size is N*batch_size + 1?
            if len(batch) < batch_size:
                continue
            reoptimized = self.model_optimization(len(batch), batch_size)
            # weird thing from roboflow fix:
            if reoptimized and len(batch) == 1:
                results = [self._service.detect(batch[0])]
            else:
                results = self._service.detect_batch(batch)
            for p, dets, (scale_x, scale_y) in zip(
                batch_paths, results, batch_scales, strict=True
            ):
                rescaled: list[dict[str, Any]] = []
                for det in dets:
                    payload = det.model_dump()
                    if max_image_size:
                        payload["bbox"] = scale_xyxy_bbox(
                            payload["bbox"], scale_x, scale_y
                        )
                    rescaled.append(payload)
                output[str(p)] = rescaled
            batch, batch_paths, batch_scales = [], [], []

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

            for p, dets, (scale_x, scale_y) in zip(
                batch_paths, results, batch_scales, strict=True
            ):
                rescaled: list[dict[str, Any]] = []
                for det in dets:
                    payload = det.model_dump()
                    if max_image_size:
                        payload["bbox"] = scale_xyxy_bbox(
                            payload["bbox"], scale_x, scale_y
                        )
                    rescaled.append(payload)
                output[str(p)] = rescaled
        return output
