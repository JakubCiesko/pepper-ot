from PIL import Image


def resize_pil(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resize a PIL image so its longest side is at most max_dim.

    Args:
        img: Source image.
        max_dim: Maximum allowed width or height.

    Returns:
        Original image when already small enough, otherwise a resized image that
        preserves aspect ratio.
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def resize_pil_with_scale(
    img: Image.Image, max_dim: int = 1024
) -> tuple[Image.Image, tuple[float, float]]:
    """Resize a PIL image and return factors for restoring coordinates.

    Args:
        img: Source image.
        max_dim: Maximum allowed width or height.

    Returns:
        Tuple of resized image and (scale_x, scale_y), where each scale maps
        resized coordinates back into the original image coordinate space.
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, (1.0, 1.0)
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS), (
        w / new_size[0],
        h / new_size[1],
    )


def scale_xyxy_bbox(box: list[float], scale_x: float, scale_y: float) -> list[float]:
    """Scale an [x1, y1, x2, y2] box by separate x and y factors."""
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
