import logging
from pathlib import Path

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
import cv2
from src.models.data_generation import DataGenConfig
import supervision as sv
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)


# TODO: decide whether to resize images BEFORE detection or AFTER detection
class KnowledgeDistiller:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        logger.info(f"Loading config for Teacher Model from {self.config_path}")
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        self.config = DataGenConfig(**raw_config)
        if self.config.teacher is None:
            raise ValueError(
                f"Teacher Model not defined. DataGenConfig: {self.config.model_dump()}"
            )
        logger.info(
            f"Initializing Grounding DINO. Device: {self.config.teacher.device}"
        )
        self.ontology = CaptionOntology(self.config.teacher.ontology.objects)
        self.model = GroundingDINO(ontology=self.ontology)
        logger.info("Preparing data directories")
        self.input_dir = self.config.paths.input_dir
        self.output_dir = self.config.paths.output_dir
        self.images_dir = self.output_dir / "images/train"
        self.labels_dir = self.output_dir / "labels/train"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def distill(self):
        """Main loop: read image -> predict -> save ground truth in YOLO format"""
        # TODO: maybe move this to config
        supported_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in self.input_dir.iterdir() if f.suffix.lower() in supported_ext
        ]
        logger.info(f"Starting distillation on {len(image_files)} images")
        total_detections = 0
        images_with_detections = 0
        for img_path in tqdm(image_files):
            detections = self.model.predict(str(img_path))
            mask = detections.confidence > self.config.teacher.box_threshold
            detections = detections[mask]
            if len(detections) == 0:
                continue
            images_with_detections += 1
            total_detections += len(detections)
            self._save_yolo_pair(img_path, detections)
        self._create_dataset_yaml()
        logger.info(
            f"Finished distillation on {len(image_files)} images. Images with detections: {images_with_detections} "
            f"Total detected objects: {total_detections}"
        )
        logger.info(f"Results saved to {self.output_dir}")

    def _save_yolo_pair(self, img_path: Path, detections: sv.Detections):
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Could not read {img_path}")
            return
        h, w, _ = image.shape
        target_resolution = self.config.target_resolution
        if target_resolution is not None:
            target_w, target_h = target_resolution
            if (w, h) != (target_w, target_h):
                image = cv2.resize(image, (target_w, target_h))

        save_img_path = self.images_dir / img_path.name
        cv2.imwrite(str(save_img_path), image)
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for xyxy, _, _, class_id, _, _ in detections:

                x_center = ((xyxy[0] + xyxy[2]) / 2) / w
                y_center = ((xyxy[1] + xyxy[3]) / 2) / h
                width = (xyxy[2] - xyxy[0]) / w
                height = (xyxy[3] - xyxy[1]) / h

                # Sanity Check (Clamp to 0-1 to avoid YOLO errors)
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

    def _create_dataset_yaml(self):
        """
        Creates the data.yaml file required by YOLO/RT-DETR training.
        """
        # Get class names in the correct order (0, 1, 2...)
        class_names = self.ontology.classes()

        yaml_content = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/train",  # TODO: no validation yet
            "names": dict(enumerate(class_names)),
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
