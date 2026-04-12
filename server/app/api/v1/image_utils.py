import io
import logging

import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps

logger = logging.getLogger(__name__)


def save_debug(image_bytes: bytes, name: str):
    with open(pth := f"/tmp/{name}.jpg", "wb") as f:
        f.write(image_bytes)
        logger.info("Saved debug image: %s to %s", name, pth)


def resize_image_bytes(
    image_bytes: bytes, max_dim: int = 1024, debug_show: bool = False
) -> tuple[bytes, tuple[int, int]]:
    """
    Resizes an image byte stream using OpenCV for maximum throughput.
    Corrects EXIF orientation using Pillow before resizing.

    Uses INTER_AREA interpolation, which is optimal for downscaling.
    """

    # Step 1: Correct EXIF orientation using Pillow
    if debug_show:
        save_debug(image_bytes, "BEFORE")
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")  # Ensure compatibility

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()
    except Exception as exc:
        logger.info("EXIF orientation correction skipped: %s", exc)

    # Step 2: Decode with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    h, w = img.shape[:2]
    logger.info(
        "Ensuring image resolution (%d, %d) matching max_dim=%d",
        h,
        w,
        max_dim,
    )

    # Step 3: Skip resizing if already within bounds
    if max(h, w) <= max_dim:
        logger.info("Image already optimal, dim: (%d, %d)", h, w)
        return image_bytes, (int(w), int(h))

    # Step 4: Resize while preserving aspect ratio
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    logger.info("Resizing image bytes to (%d, %d)", new_w, new_h)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    # Step 5: Encode back to JPEG
    success, encoded = cv2.imencode(
        ".jpg",
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not success:
        raise RuntimeError("Failed to encode image")
    result_bytes = encoded.tobytes()
    if debug_show:
        save_debug(result_bytes, "AFTER")
    return result_bytes, (new_w, new_h)


def create_panorama(imgs: list[bytes]) -> bytes:
    """
    Create a horizontal panorama from a list of images.

    Args:
        imgs: List of images as byte streams.

    Returns:
        A byte stream of the stitched panorama in JPEG format.

    Raises:
        ValueError: If the input list is empty or an image cannot be decoded.
    """
    if not imgs:
        raise ValueError("No images provided for panorama creation.")

    # Decode images into PIL format
    pil_images = []
    for img_bytes in imgs:
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                pil_images.append(pil_img)
        except Exception as exc:
            raise ValueError(f"Failed to decode image bytes: {exc}") from exc

    # Compute panorama dimensions
    widths, heights = zip(*(img.size for img in pil_images), strict=True)
    total_width = sum(widths)
    max_height = max(heights)

    # Create blank canvas
    panorama = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    x_offset = 0
    for img in pil_images:
        panorama.paste(img, (x_offset, 0))
        x_offset += img.width

    # Encode panorama to bytes
    buffer = io.BytesIO()
    panorama.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
