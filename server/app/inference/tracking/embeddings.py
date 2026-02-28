import logging

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from transformers import AutoModel
from transformers import CLIPImageProcessor

from app.inference.types import DetectionObject

logger = logging.getLogger(__name__)


# TODO: potentially extend for other CLIPImageProcessor embedding models.
class FeatureExtractor:
    """Extracts visual embeddings (fingerprints) for ReID using Nvidia RADIO."""

    REPO = "nvidia/C-RADIOv4-SO400M"
    TARGET_SIZE = (384, 384)

    def __init__(
        self,
        reid_model: str | None = None,
        target_size: tuple[int] | None = None,
        resampling_method: Image.Resampling | str = Image.Resampling.BICUBIC,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        reid_model = reid_model or self.REPO
        target_size = target_size or self.TARGET_SIZE
        self.target_size = target_size
        logger.info(f"Loading model ({reid_model}) on {self.device}...")
        if isinstance(resampling_method, str):
            try:
                resampling_method = getattr(Image.Resampling, resampling_method)
            except Exception as e:
                logger.exception(
                    f"Resampling method {resampling_method} not supported: {e}"
                )
                logger.info("Fallback to BICUBIC resampling.")
                resampling_method = Image.Resampling.BICUBIC
        self.resampling_method = resampling_method
        self.model = (
            AutoModel.from_pretrained(reid_model, trust_remote_code=True, dtype=dtype)
            .to(self.device)
            .eval()
        )

        self.processor = CLIPImageProcessor.from_pretrained(
            reid_model, trust_remote_code=True
        )
        logger.info("Image FeatureExtractor Model loaded.")

    def set_device(self, device: str):
        logger.info(f"Setting extractor device to {device}")
        self.device = device
        self.model = self.model.to(self.device)

    def prepare_crops(self, image: Image.Image, detections: list[DetectionObject]):
        crops = []
        logger.info(f"Preparing crops of detected images with size: {self.target_size}")
        for det in detections:
            # bbox is [x1, y1, x2, y2]
            # Convert float bbox to int
            x1, y1, x2, y2 = map(int, det.bbox)

            # Clamp to image bounds to avoid errors
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            # Handle degenerate boxes (width or height 0)
            if x2 <= x1 or y2 <= y1:
                # Create a black dummy crop if detection is invalid
                crop = Image.new("RGB", self.target_size)
            else:
                crop = image.crop((x1, y1, x2, y2))
                crop = crop.resize(self.target_size, self.resampling_method)

            crops.append(crop)
        logger.info(
            f"Prepared {len(crops)} crops of detected objects in image with size: {self.target_size}, resampling method: {self.resampling_method}"
        )
        return crops

    def extract(self, image: Image.Image, detections: list[DetectionObject]) -> NDArray:
        """
        Crops the image based on detections and returns a batch of embeddings.
        Returns: (N, D) array where N=len(detections)
        """
        if not detections:
            logger.info("No detections, returning empty embedding array.")
            return np.array([])

        # 1. Prepare Crops with FIXED Resolution
        # We must use a fixed size (e.g. 384x384) so we can batch them into one tensor.
        # 384 is a standard resolution that works well with C-RADIO.
        crops = self.prepare_crops(image, detections)

        # 2. Batch Inference
        with torch.no_grad():
            # We explicitly disable resizing in the processor since we did it manually
            inputs = self.processor(
                images=crops, return_tensors="pt", do_resize=False, do_center_crop=False
            )

            pixel_values = inputs.pixel_values.to(self.model.dtype).to(self.device)

            summary, _ = self.model(pixel_values)

            # 3. Normalize for Cosine Similarity
            summary = summary / summary.norm(p=2, dim=-1, keepdim=True)
        logger.info(f"Extracted {len(summary)} normalized embeddings.")
        return summary.cpu().float().numpy()
