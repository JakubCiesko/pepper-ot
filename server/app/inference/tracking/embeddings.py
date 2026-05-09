from io import BytesIO
import logging

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from transformers import AutoModel
from transformers import CLIPImageProcessor

from app.inference.types import InferenceDetectionObject

logger = logging.getLogger(__name__)


def _normalize_labels(labels: list[str] | None) -> set[str]:
    return {
        str(label).strip().lower() for label in (labels or []) if str(label).strip()
    }


class _SingleModelExtractor:
    def __init__(
        self,
        reid_model: str,
        target_size: tuple[int, int] | None,
        resampling_method: Image.Resampling | str,
        device: str,
        target_aspect_ratio: float | None = None,
    ):
        self.device = device
        self.reid_model = reid_model
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading model (%s) on device=%s...", reid_model, self.device)
        self.resampling_method = self._resolve_resampling_method(resampling_method)
        self.model = (
            AutoModel.from_pretrained(reid_model, trust_remote_code=True, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            reid_model, trust_remote_code=True
        )
        self.target_size = target_size or self._model_target_size() or (384, 384)
        self.target_aspect_ratio = (
            target_aspect_ratio
            if target_aspect_ratio is not None
            else self._model_target_aspect_ratio()
        )
        self.embedding_dim = self._model_embedding_dim()
        logger.info(
            "Loaded extractor model=%s target_size=%s aspect_ratio=%s embedding_dim=%s",
            reid_model,
            self.target_size,
            self.target_aspect_ratio,
            self.embedding_dim,
        )

    @staticmethod
    def _resolve_resampling_method(
        resampling_method: Image.Resampling | str,
    ) -> Image.Resampling:
        if isinstance(resampling_method, str):
            try:
                return getattr(Image.Resampling, resampling_method)
            except Exception as exc:
                logger.exception(
                    "Resampling method %s not supported: %s",
                    resampling_method,
                    exc,
                )
                logger.info("Fallback to BICUBIC resampling.")
                return Image.Resampling.BICUBIC
        return resampling_method

    def _model_target_size(self) -> tuple[int, int] | None:
        raw = getattr(self.model.config, "reid_target_size", None) or getattr(
            self.model.config, "target_size", None
        )
        if raw is None or len(raw) != 2:
            return None
        return int(raw[0]), int(raw[1])

    def _model_target_aspect_ratio(self) -> float | None:
        raw = getattr(self.model.config, "reid_aspect_ratio", None)
        if raw is None:
            return None
        ratio = float(raw)
        return ratio if ratio > 0 else None

    def _model_embedding_dim(self) -> int:
        for attr in ("embedding_dim", "projection_dim", "hidden_size"):
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 0

    def set_device(self, device: str):
        logger.info(
            "Setting extractor device to %s for model=%s", device, self.reid_model
        )
        self.device = device
        self.model = self.model.to(self.device)

    def set_target_size(self, target_size: tuple[int, int] | None):
        if target_size:
            self.target_size = tuple(target_size)

    def set_target_aspect_ratio(self, target_aspect_ratio: float | None):
        self.target_aspect_ratio = (
            target_aspect_ratio
            if target_aspect_ratio and target_aspect_ratio > 0
            else None
        )

    def set_resampling_method(self, resampling_method: Image.Resampling | str):
        self.resampling_method = self._resolve_resampling_method(resampling_method)

    def _expand_bbox_to_aspect_ratio(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image: Image.Image,
    ) -> tuple[int, int, int, int]:
        if self.target_aspect_ratio is None:
            return x1, y1, x2, y2

        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        current_ratio = width / height
        if current_ratio < self.target_aspect_ratio:
            width = height * self.target_aspect_ratio
        else:
            height = width / self.target_aspect_ratio

        new_x1 = int(round(center_x - width / 2.0))
        new_x2 = int(round(center_x + width / 2.0))
        new_y1 = int(round(center_y - height / 2.0))
        new_y2 = int(round(center_y + height / 2.0))

        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        if new_x2 > image.width:
            overflow = new_x2 - image.width
            new_x1 = max(0, new_x1 - overflow)
            new_x2 = image.width
        if new_y2 > image.height:
            overflow = new_y2 - image.height
            new_y1 = max(0, new_y1 - overflow)
            new_y2 = image.height

        return new_x1, new_y1, max(new_x1 + 1, new_x2), max(new_y1 + 1, new_y2)

    def prepare_crops(
        self,
        image: Image.Image,
        detections: list[InferenceDetectionObject],
        debug_show: bool = False,
    ) -> list[Image.Image]:
        crops = []
        logger.info(
            "Preparing crops of detected images with size: %s for model=%s",
            self.target_size,
            self.reid_model,
        )
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det.bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                crop = Image.new("RGB", self.target_size)
            else:
                x1, y1, x2, y2 = self._expand_bbox_to_aspect_ratio(
                    x1, y1, x2, y2, image
                )
                crop = image.crop((x1, y1, x2, y2))
                crop = crop.resize(self.target_size, self.resampling_method)

            if debug_show:
                crop.save(f"/tmp/pepper_server/crop_{i}.png")
            crops.append(crop)

        logger.info(
            "Prepared %d crops with size=%s resampling=%s model=%s",
            len(crops),
            self.target_size,
            self.resampling_method,
            self.reid_model,
        )
        return crops

    @staticmethod
    def _encode_crop_bytes(crops: list[Image.Image]) -> list[bytes]:
        encoded: list[bytes] = []
        for crop in crops:
            buffer = BytesIO()
            rgb_crop = crop.convert("RGB")
            rgb_crop.save(buffer, format="JPEG", quality=90, optimize=True)
            encoded.append(buffer.getvalue())
        return encoded

    def extract_with_crops(
        self, image: Image.Image, detections: list[InferenceDetectionObject]
    ) -> tuple[NDArray, list[bytes]]:
        if not detections:
            logger.info("No detections, returning empty embedding/crop arrays.")
            return np.array([]), []

        crops = self.prepare_crops(image, detections, debug_show=False)
        with torch.no_grad():
            inputs = self.processor(
                images=crops, return_tensors="pt", do_resize=False, do_center_crop=False
            )
            pixel_values = inputs.pixel_values.to(self.model.dtype).to(self.device)
            summary, _ = self.model(pixel_values)
            summary = summary / summary.norm(p=2, dim=-1, keepdim=True)
            self.embedding_dim = int(summary.shape[-1])

        logger.info(
            "Extracted %d normalized embeddings with model=%s",
            len(summary),
            self.reid_model,
        )
        return summary.cpu().float().numpy(), self._encode_crop_bytes(crops)


class FeatureExtractor:
    """Extracts visual embeddings with optional human-specific model routing."""

    REPO = "nvidia/C-RADIOv4-SO400M"
    TARGET_SIZE = (384, 384)

    def __init__(
        self,
        reid_model: str | None = None,
        target_size: tuple[int, int] | None = None,
        resampling_method: Image.Resampling | str = Image.Resampling.BICUBIC,
        device: str | None = None,
        human_reid_enabled: bool = False,
        human_reid_model: str | None = None,
        human_reid_target_size: tuple[int, int] | None = None,
        human_labels: list[str] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.human_reid_enabled = human_reid_enabled
        self.default_extractor = self._build_model_extractor(
            reid_model or self.REPO,
            target_size or self.TARGET_SIZE,
            resampling_method,
            self.device,
        )
        self.human_extractor: _SingleModelExtractor | None = None
        if human_reid_enabled and human_reid_model:
            self.human_extractor = self._build_model_extractor(
                human_reid_model,
                human_reid_target_size,
                resampling_method,
                self.device,
                self._size_to_aspect_ratio(human_reid_target_size),
            )
        elif human_reid_enabled:
            logger.warning(
                "human_reid_enabled=true but no human_reid_model configured; "
                "falling back to generic ReID model."
            )
        self.human_labels = _normalize_labels(human_labels)
        self.output_dim = max(
            self.default_extractor.embedding_dim,
            self.human_extractor.embedding_dim if self.human_extractor else 0,
        )

    @staticmethod
    def _size_to_aspect_ratio(target_size: tuple[int, int] | None) -> float | None:
        if target_size is None or len(target_size) != 2 or target_size[1] <= 0:
            return None
        return float(target_size[0]) / float(target_size[1])

    def _build_model_extractor(
        self,
        reid_model: str,
        target_size: tuple[int, int] | None,
        resampling_method: Image.Resampling | str,
        device: str,
        target_aspect_ratio: float | None = None,
    ) -> _SingleModelExtractor:
        return _SingleModelExtractor(
            reid_model=reid_model,
            target_size=target_size,
            resampling_method=resampling_method,
            device=device,
            target_aspect_ratio=target_aspect_ratio,
        )

    @property
    def target_size(self) -> tuple[int, int]:
        return self.default_extractor.target_size

    @target_size.setter
    def target_size(self, value: tuple[int, int]):
        self.default_extractor.set_target_size(value)

    @property
    def resampling_method(self) -> Image.Resampling:
        return self.default_extractor.resampling_method

    @resampling_method.setter
    def resampling_method(self, value: Image.Resampling | str):
        self.default_extractor.set_resampling_method(value)
        if self.human_extractor is not None:
            self.human_extractor.set_resampling_method(value)

    def set_device(self, device: str):
        self.device = device
        self.default_extractor.set_device(device)
        if self.human_extractor is not None:
            self.human_extractor.set_device(device)

    def update_config(
        self,
        human_target_size: tuple[int, int] | None,
        human_labels: list[str] | None,
    ):
        if human_target_size and self.human_extractor is not None:
            self.human_extractor.set_target_size(human_target_size)
            self.human_extractor.set_target_aspect_ratio(
                self._size_to_aspect_ratio(human_target_size)
            )
        self.human_labels = _normalize_labels(human_labels)

    def _is_human_detection(self, detection: InferenceDetectionObject) -> bool:
        if not self.human_reid_enabled or self.human_extractor is None:
            return False
        return detection.label.strip().lower() in self.human_labels

    @staticmethod
    def _pad_embeddings(embeddings: NDArray, width: int) -> NDArray:
        if embeddings.size == 0 or embeddings.shape[-1] == width:
            return embeddings
        padded = np.zeros((embeddings.shape[0], width), dtype=np.float32)
        padded[:, : embeddings.shape[-1]] = embeddings
        return padded

    def _extract_group(
        self,
        extractor: _SingleModelExtractor,
        image: Image.Image,
        detections: list[InferenceDetectionObject],
    ) -> tuple[NDArray, list[bytes]]:
        embeddings, crop_bytes = extractor.extract_with_crops(image, detections)
        if embeddings.size == 0:
            return embeddings, crop_bytes
        self.output_dim = max(self.output_dim, embeddings.shape[-1])
        return self._pad_embeddings(embeddings, self.output_dim), crop_bytes

    def extract_with_crops(
        self, image: Image.Image, detections: list[InferenceDetectionObject]
    ) -> tuple[NDArray, list[bytes]]:
        if not detections:
            logger.info("No detections, returning empty embedding/crop arrays.")
            return np.array([]), []

        embeddings = np.zeros((len(detections), self.output_dim), dtype=np.float32)
        crop_bytes: list[bytes] = [b""] * len(detections)
        human_indices = [
            idx for idx, det in enumerate(detections) if self._is_human_detection(det)
        ]
        human_index_set = set(human_indices)
        generic_indices = [
            idx for idx in range(len(detections)) if idx not in human_index_set
        ]

        for indices, extractor in (
            (generic_indices, self.default_extractor),
            (human_indices, self.human_extractor),
        ):
            if not indices or extractor is None:
                continue
            subset = [detections[idx] for idx in indices]
            subset_embeddings, subset_crop_bytes = self._extract_group(
                extractor, image, subset
            )
            if subset_embeddings.size == 0:
                continue
            if subset_embeddings.shape[-1] > embeddings.shape[-1]:
                expanded = np.zeros(
                    (len(detections), subset_embeddings.shape[-1]), dtype=np.float32
                )
                expanded[:, : embeddings.shape[-1]] = embeddings
                embeddings = expanded
            for position, det_index in enumerate(indices):
                embeddings[det_index, : subset_embeddings.shape[-1]] = (
                    subset_embeddings[position]
                )
                crop_bytes[det_index] = subset_crop_bytes[position]

        logger.info(
            "Extracted %d embeddings using generic_indices=%s human_indices=%s",
            len(detections),
            generic_indices,
            human_indices,
        )
        return embeddings, crop_bytes

    def extract(
        self, image: Image.Image, detections: list[InferenceDetectionObject]
    ) -> NDArray:
        if not detections:
            logger.info("No detections, returning empty embedding array.")
            return np.array([])
        embeddings, _ = self.extract_with_crops(image, detections)
        return embeddings
