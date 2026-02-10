import logging

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from transformers import AutoModel
from transformers import CLIPImageProcessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts visual embeddings (fingerprints) for ReID using Nvidia RADIO."""

    REPO = "nvidia/C-RADIOv4-SO400M"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"Loading Nvidia RADIO ({self.REPO}) on {self.device}...")
        self.model = (
            AutoModel.from_pretrained(
                self.REPO, trust_remote_code=True, torch_dtype=dtype
            )
            .to(self.device)
            .eval()
        )

        self.processor = CLIPImageProcessor.from_pretrained(
            self.REPO, trust_remote_code=True
        )

    def extract_batch(self, image_crops: list[Image.Image]) -> NDArray:
        if not image_crops:
            return np.array([])

        # Radio requires specific resolutions
        resized = []
        for img in image_crops:
            h, w = img.size[1], img.size[0]
            h2, w2 = self.model.get_nearest_supported_resolution(h, w)
            resized.append(img.resize((w2, h2), Image.Resampling.BICUBIC))

        with torch.no_grad():
            pixel_values = (
                self.processor(images=resized, return_tensors="pt", do_resize=False)
                .pixel_values.to(self.model.dtype)
                .to(self.device)
            )

            summary, _ = self.model(pixel_values)
            # Normalize for cosine similarity
            summary = summary / summary.norm(p=2, dim=-1, keepdim=True)

        return summary.cpu().numpy()
