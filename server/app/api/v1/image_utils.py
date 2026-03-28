import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_image_bytes(image_bytes: bytes, max_dim: int = 1024) -> bytes:
    """
    Resizes an image byte stream using OpenCV for maximum throughput.
    Uses INTER_AREA interpolation, which is mathematically optimal for decimation.
    """

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    h, w = img.shape[:2]
    logger.info("Ensuring image resolution (%d, %d) matching max_dim=%d", h, w, max_dim)
    # in bounds check
    if max(h, w) <= max_dim:
        logger.info("Image already optimal, dim: (%d, %d)", h, w)
        return image_bytes
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    logger.info("Resizing image bytes to (%d, %d)", new_w, new_h)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode image")

    return encoded.tobytes()
