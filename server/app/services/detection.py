import asyncio
import io
import json
import logging
import os
from pathlib import Path
import urllib.request

from app.models.detection_result import DetectionObject
from app.models.detection_result import DetectionResponse
from app.models.detection_settings import YOLOSettings
from googletrans import Translator
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def download_model(model_url: str, model_path: Path):
    logger.info(f"Downloading model from {model_url} to {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path)


class DetectionService:
    def __init__(self, settings: YOLOSettings = None):
        self.settings = settings or YOLOSettings()
        self.model_path: Path = self.settings.model_path
        self.device = self.settings.device_actual
        self.imgsz = self.settings.imgsz
        self.model = self.load_model()

        self.translate_to = self.settings.language or "en"
        self.translation_path = self.model_path.with_suffix(
            f".{self.translate_to}.labels.json"
        )
        self.translations = {}

    @property
    def available_models(self) -> list[str]:
        models_dir = self.model_path.parent  # pretty dirty
        return [f for f in os.listdir(models_dir) if f.endswith(".pt")]

    def load_model(self):
        if not self.model_path.exists():
            download_model(self.settings.model_url, self.model_path)
        model = YOLO(str(self.model_path))
        logger.info(f"Loaded model from {self.model_path}")
        if self.settings.fuse_model:
            logger.info("Fusing model")
            model.fuse()
        model.to(self.device)
        logger.info(f"Model fully loaded to device: {self.device}")
        return model

    async def initialize_translations(self):
        if self.translate_to.strip().lower() != "en":
            logger.info(f"Initializing translation to language: {self.translate_to}")
            self.translations = await self.load_or_create_translations()
        else:
            logger.info(
                f"No translation needed the language is set to: {self.translate_to}"
            )

    async def load_or_create_translations(self) -> dict:
        if self.translation_path.exists():
            logger.info(f"Loading existing translations from {self.translation_path}")
            with open(self.translation_path, encoding="utf-8") as f:
                return json.load(f)

        logger.info("No translation file found. Generating translations...")
        all_labels = list(self.model.names.values())

        # CHANGED: Called directly with await, no asyncio.run()
        translations = await self.translate_all_labels(all_labels)

        with open(self.translation_path, "w", encoding="utf-8") as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved translations to {self.translation_path}")
        return translations

    async def translate_label(self, label: str) -> tuple:
        translator = Translator()
        result = await translator.translate(label, dest=self.translate_to)
        return label, result.text

    async def translate_all_labels(self, labels: list[str]) -> dict:
        tasks = [asyncio.create_task(self.translate_label(lbl)) for lbl in labels]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def detect(self, image_bytes: bytes) -> DetectionResponse:
        logger.info("Running detection on image")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self.model.predict(
            img, device=self.device, imgsz=self.imgsz, verbose=False
        )
        objects = []
        for r in results:
            for box, cls, conf in zip(
                r.boxes.xyxy, r.boxes.cls, r.boxes.conf, strict=True
            ):
                en_label = self.model.names[int(cls)]
                label = (
                    en_label
                    if self.translate_to.strip().lower() == "en"
                    else self.translations.get(en_label)
                )
                objects.append(
                    DetectionObject(
                        label=label,
                        confidence=float(conf),
                        bbox=[float(x) for x in box],
                    )
                )
        logger.info(f"Found {len(objects)} objects in provided image.")
        return DetectionResponse(objects=objects)

    async def reload_with_model(self, model_name: str):
        logger.info(f"Reloading DetectionService with model {model_name}")
        settings = YOLOSettings(model_name=model_name)
        self.settings = settings
        self.model_path = settings.model_path
        self.model = self.load_model()
        self.translation_path = self.model_path.with_suffix(
            f".{self.translate_to}.labels.json"
        )
        self.translations = {}
        await self.initialize_translations()
