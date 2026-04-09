import asyncio
from pathlib import Path
from time import perf_counter

from PIL import Image
from tqdm.auto import tqdm

from research.experiments.adapters import ServerCaptionAdapter
from research.experiments.adapters import ServerDetectionAdapter
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import iter_image_paths
from research.experiments.io import load_json
from research.experiments.io import save_json


def _objects_from_detection_row(row: list[dict]) -> list[str]:
    return [str(item.get("label", "")).strip() for item in row if item.get("label")]


#  TODO: cache save?
async def run_descriptions(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting description phase")
    stage_metrics = StageMetrics(stage="descriptions")
    image_paths = list(iter_image_paths(config.paths.images_dir))
    run.logger.info("Found images=%d", len(image_paths))

    detections: dict[str, list[dict]] = {}
    detection_path: Path = run.run_dir / config.paths.detections_file
    if config.detection.enabled:
        detector = ServerDetectionAdapter(
            config.detection.backend, confidence=config.detection.confidence
        )
        run.logger.info(
            "Initialized detector with config: %s", config.detection.model_dump()
        )
        detections = detector.detect_images(
            image_paths=image_paths,
            batch_size=config.detection.batch_size,
            max_image_size=config.detection.max_image_size,
        )
        save_json(detection_path, detections)
        run.logger.info("Saved detections for %d images", len(detections))
    else:
        run.logger.info(
            "Detection stage bypassed trying to load detections from file %s",
            detection_path,
        )
        detections = load_json(detection_path, {})
        run.logger.info(
            "Loaded detections for %d images from file %s",
            len(detections),
            detection_path,
        )

    captioner = ServerCaptionAdapter(
        model_provider=config.description_model.provider,
        model_id=config.description_model.model_id,
        system_prompt=config.descriptions.system_prompt,
    )
    run.logger.info(
        "Initialized captioner with config:  %s, %s, %s",
        config.description_model.provider,
        config.description_model.model_id,
        config.descriptions.model_dump(),
    )

    def prompt_builder(path: Path) -> str:
        if not config.prompting.include_detection_labels_in_descriptions:
            return config.descriptions.user_prompt_template.replace("{objects}", "[]")
        labels = _objects_from_detection_row(detections.get(str(path.resolve()), []))
        return config.descriptions.user_prompt_template.replace(
            "{objects}", str(labels)
        )

    semaphore = asyncio.Semaphore(config.descriptions.max_concurrent_batches)
    max_image_size = config.descriptions.max_image_size
    descriptions: dict[str, dict] = {}
    progress = tqdm(
        total=len(image_paths), desc="Generating Captions For Images", unit="img"
    )

    async def process_one(path: Path):
        t0 = perf_counter()
        async with semaphore:
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                prompt = prompt_builder(path)
                run.logger.debug("Prompt built successfully: %s", prompt)
                payload = await captioner.caption_image(
                    image, prompt_override=prompt, max_image_size=max_image_size
                )
                descriptions[str(path.resolve())] = payload
                stage_metrics.record_ok(perf_counter() - t0)
            except Exception:
                stage_metrics.record_failed(
                    "caption_generation_error", perf_counter() - t0
                )
            finally:
                progress.update(1)

    await asyncio.gather(*(process_one(path) for path in image_paths))
    progress.close()

    stage_metrics.finish()
    save_json(run.run_dir / config.paths.descriptions_file, descriptions)
    save_json(run.run_dir / "metrics_descriptions.json", stage_metrics.to_dict())
    run.logger.info("Saved descriptions for %d images", len(descriptions))
    run.logger.info(
        "Descriptions summary ok=%d failed=%d skipped=%d duration=%.3fs",
        stage_metrics.ok,
        stage_metrics.failed,
        stage_metrics.skipped,
        stage_metrics.duration_s,
    )
    return descriptions
