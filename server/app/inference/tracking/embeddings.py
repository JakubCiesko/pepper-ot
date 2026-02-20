import logging

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from transformers import AutoModel
from transformers import CLIPImageProcessor

from ..types import DetectionObject

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts visual embeddings (fingerprints) for ReID using Nvidia RADIO."""

    REPO = "nvidia/C-RADIOv4-SO400M"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"Loading Nvidia RADIO ({self.REPO}) on {self.device}...")
        self.model = (
            AutoModel.from_pretrained(self.REPO, trust_remote_code=True, dtype=dtype)
            .to(self.device)
            .eval()
        )

        self.processor = CLIPImageProcessor.from_pretrained(
            self.REPO, trust_remote_code=True
        )

    def extract(self, image: Image.Image, detections: list[DetectionObject]) -> NDArray:
        """
        Crops the image based on detections and returns a batch of embeddings.
        Returns: (N, D) array where N=len(detections)
        """
        if not detections:
            return np.array([])

        # 1. Prepare Crops with FIXED Resolution
        # We must use a fixed size (e.g. 384x384) so we can batch them into one tensor.
        # 384 is a standard resolution that works well with C-RADIO.
        TARGET_SIZE = (384, 384)

        crops = []
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
                crop = Image.new("RGB", TARGET_SIZE)
            else:
                crop = image.crop((x1, y1, x2, y2))
                crop = crop.resize(TARGET_SIZE, Image.Resampling.BICUBIC)

            crops.append(crop)

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

        return summary.cpu().float().numpy()
