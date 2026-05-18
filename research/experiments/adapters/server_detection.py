from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .bootstrap import ensure_server_app_importable
from .utils import resize_pil_with_scale
from .utils import scale_xyxy_bbox


class ServerDetectionAdapter:
    """In-process adapter around the server detection service.

    The description phase uses this adapter to produce detections with the same
    detector implementations as the server pipeline, while returning serialized
    dictionaries suitable for research artifacts.
    """

    def __init__(self, backend: str, confidence: float = 0.5):
        """Create a detection adapter for a server detector backend.

        Args:
            backend: DetectionModelType value such as rf_detr, rt_detr, yolo, or
                owl_v2.
            confidence: Detection confidence threshold.
        """
        ensure_server_app_importable()
        from app.inference.detection.detectors import DetectionModelType
        from app.inference.detection.service import DetectionService

        backend_enum = DetectionModelType(backend)
        self._service = DetectionService(backend_enum, threshold=confidence)
        self._backend_enum = DetectionModelType

    @property
    def service(self) -> Any:
        """Return the underlying server DetectionService instance."""
        return self._service

    def detect_image(
        self, image: Image.Image, max_image_size: int | None = None
    ) -> list[dict[str, Any]]:
        """Detect objects in one PIL image.

        Args:
            image: Image to run through the server detector.
            max_image_size: Optional longest-side resize limit before detection.

        Returns:
            Serialized detection rows. If resizing was applied, bbox
            coordinates are scaled back to the original image coordinate space.
        """
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
        """Retune RF-DETR inference optimization when batch size changes.

        Args:
            current_batch_size: Number of images in the next batch.
            batch_size: Previously optimized batch size.

        Returns:
            True when the model was reoptimized, otherwise False.
        """
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
        """Detect objects for a list of image paths in batches.

        Args:
            image_paths: Image files to load and process.
            batch_size: Preferred detector batch size.
            max_image_size: Optional longest-side resize limit before
                detection.

        Returns:
            Mapping from resolved image path to serialized detection rows.

        Side Effects:
            Optimizes the detector for the requested batch size and may
            reoptimize for a final short batch when required by the backend.
        """
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
