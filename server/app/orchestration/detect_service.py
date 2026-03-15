import base64
import io
import json
import logging
from pathlib import Path
import time
from typing import Any

from fastapi import HTTPException
from PIL import Image

from app.core.infra.storage import save_last_image_async
from app.core.infra.storage import save_last_state_async
from app.core.infra.ws_manager import ws_manager
from app.core.runtime.state import AppState
from app.orchestration.runtime_adapter import memory_payload
from app.orchestration.runtime_adapter import resolve_runtime_adapter
from app.schemas.robot import PersonMetadata
from app.schemas.robot import RobotMetadata
from app.schemas.scene import DetectionResponse

logger = logging.getLogger(__name__)


class DetectService:
    def __init__(self, state: AppState):
        self.state = state

    @staticmethod
    def parse_metadata(metadata_json: str | None) -> RobotMetadata:
        if not metadata_json:
            return RobotMetadata(head_yaw=0.0, head_pitch=0.0)
        try:
            raw = json.loads(metadata_json)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Invalid metadata JSON"
            ) from exc

        people_raw = raw.get("people")
        people = []
        if isinstance(people_raw, list):
            people.extend([PersonMetadata(**item) for item in people_raw])

        return RobotMetadata(
            head_yaw=float(raw.get("head_yaw", 0.0)),
            head_pitch=float(raw.get("head_pitch", 0.0)),
            body_yaw=raw.get("body_yaw"),
            camera_hfov=raw.get("camera_hfov"),
            camera_vfov=raw.get("camera_vfov"),
            image_width=raw.get("image_width"),
            image_height=raw.get("image_height"),
            timestamp=raw.get("timestamp"),
            frame_id=raw.get("frame_id"),
            scan_id=raw.get("scan_id"),
            people=people,
        )

    async def process(
        self, image_bytes: bytes, metadata: RobotMetadata, publish: bool
    ) -> DetectionResponse:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

        adapter = resolve_runtime_adapter(self.state)
        try:
            result = await adapter.detect(image_bytes, metadata)
        except Exception as exc:
            logger.exception("Detection failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        payload = {
            "type": "detection",
            "image": result.get("image_b64"),
            "objects": result.get("objects", []),
            "scene_graph": result.get("scene_graph", []),
            "memory": result.get("memory", memory_payload()),
            "metrics": result.get("metrics", {}),
            "executed_stages": result.get("executed_stages", []),
        }

        if publish:
            await ws_manager.broadcast(payload)
            await self._update_and_persist(payload)

        return DetectionResponse(
            objects=payload["objects"],
            timestamp=time.time(),
            image_width=int(result.get("image_width", image.width)),
            image_height=int(result.get("image_height", image.height)),
        )

    async def _update_and_persist(self, payload: dict[str, Any]):
        self.state.last_state = payload
        cfg = self.state.config
        if cfg is None or not cfg.storage.persist_last_state:
            return

        base_dir = (
            cfg._config_path.parent if cfg._config_path is not None else Path.cwd()
        )
        state_path = base_dir / cfg.storage.last_state_path
        persist_payload = dict(payload)

        if cfg.storage.store_image and payload.get("image"):
            image_path = state_path.with_suffix(".jpg")
            persist_payload["image"] = None
            persist_payload["image_path"] = str(image_path.relative_to(base_dir))
            await save_last_image_async(image_path, payload["image"])
        else:
            persist_payload["image"] = None
            persist_payload.pop("image_path", None)

        await save_last_state_async(state_path, persist_payload)


def decode_base64_image(image_b64: str) -> bytes:
    return base64.b64decode(image_b64.encode("utf-8"))
