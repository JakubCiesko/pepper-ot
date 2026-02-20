import asyncio
import base64
import json
from pathlib import Path


def load_last_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        image = data.get("image")
        image_path = data.get("image_path")
        if image is None and image_path:
            img_path = Path(image_path)
            if not img_path.is_absolute():
                img_path = path.parent / img_path
            if img_path.exists():
                img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                data["image"] = img_b64
        return data
    except Exception:
        return None


def save_last_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def save_last_image(path: Path, image_b64: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(image_b64.encode("utf-8"))
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


async def save_last_state_async(path: Path, payload: dict) -> None:
    await asyncio.to_thread(save_last_state, path, payload)


async def save_last_image_async(path: Path, image_b64: str) -> None:
    await asyncio.to_thread(save_last_image, path, image_b64)
