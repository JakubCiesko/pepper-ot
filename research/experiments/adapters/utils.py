from PIL import Image


def resize_pil(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def resize_pil_with_scale(
    img: Image.Image, max_dim: int = 1024
) -> tuple[Image.Image, tuple[float, float]]:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, (1.0, 1.0)
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS), (w / new_size[0], h / new_size[1])


def scale_xyxy_bbox(box: list[float], scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
