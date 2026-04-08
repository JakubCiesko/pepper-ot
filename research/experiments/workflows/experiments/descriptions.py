from pathlib import Path

from research.experiments.adapters import ServerCaptionAdapter
from research.experiments.adapters import ServerDetectionAdapter
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import iter_image_paths
from research.experiments.io import save_json


def _objects_from_detection_row(row: list[dict]) -> list[str]:
    return [str(item.get("label", "")).strip() for item in row if item.get("label")]


async def run_descriptions(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting description phase")
    image_paths = list(iter_image_paths(config.paths.images_dir))
    run.logger.info("Found images=%d", len(image_paths))

    detections: dict[str, list[dict]] = {}
    if config.detection.enabled:
        detector = ServerDetectionAdapter(config.detection.backend)
        detections = detector.detect_images(
            image_paths=image_paths,
            batch_size=config.detection.batch_size,
        )
        save_json(run.run_dir / config.paths.detections_file, detections)
        run.logger.info("Saved detections for %d images", len(detections))

    captioner = ServerCaptionAdapter(
        model_provider=config.description_model.provider,
        model_id=config.description_model.model_id,
        system_prompt=config.descriptions.system_prompt,
    )

    def prompt_builder(path: Path) -> str:
        if not config.prompting.include_detection_labels_in_descriptions:
            return config.descriptions.user_prompt_template.replace("{objects}", "[]")
        labels = _objects_from_detection_row(detections.get(str(path), []))
        return config.descriptions.user_prompt_template.replace(
            "{objects}", str(labels)
        )

    descriptions = await captioner.caption_images(
        image_paths=image_paths,
        prompt_builder=prompt_builder,
        max_concurrent=config.descriptions.max_concurrent_batches,
    )
    save_json(run.run_dir / config.paths.descriptions_file, descriptions)
    run.logger.info("Saved descriptions for %d images", len(descriptions))
    return descriptions
