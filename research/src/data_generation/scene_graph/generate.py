import asyncio
import json
import logging
from pathlib import Path

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
import cv2
from tqdm import tqdm
import yaml

from research.src.data_generation.scene_graph.llm_client import SceneGraphGenerator
from research.src.data_generation.scene_graph.painter import SoMPainter
from research.src.models.data_generation import DataGenConfig

logger = logging.getLogger(__name__)


def generate_som_images(config: DataGenConfig) -> list[tuple[Path, Path]]:
    """
    Phase 1: GPU Heavy.
    Runs Grounding DINO to detect objects and 'paints' the SoM overlay on images.
    Returns a list of tuples: [(original_img_path, som_img_path), ...]
    """
    logger.info("Phase 1: Initializing Grounding DINO & Painter...")

    ontology_dict = config.teacher.ontology_config.objects
    detector = GroundingDINO(ontology=CaptionOntology(ontology_dict))
    painter = SoMPainter()

    input_dir = config.paths.input_dir
    output_dir = config.paths.output_dir
    som_images_dir = output_dir / "som_images"

    output_dir.mkdir(parents=True, exist_ok=True)
    som_images_dir.mkdir(exist_ok=True)

    supported_ext = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_ext]

    valid_pairs = []

    logger.info(f"Processing {len(image_files)} images...")

    for img_path in tqdm(image_files, desc="Painting SoM"):
        detections = detector.predict(str(img_path))
        detections = detections[detections.confidence > config.teacher.box_threshold]

        if len(detections) < 2:
            continue
        original_image = cv2.imread(str(img_path))
        if original_image is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        som_image = painter.paint(original_image, detections)

        som_filename = f"som_{img_path.name}"
        som_save_path = som_images_dir / som_filename
        cv2.imwrite(str(som_save_path), som_image)
        valid_pairs.append((img_path, som_save_path))

    logger.info(f"Phase 1 Complete. {len(valid_pairs)} SoM images ready.")
    return valid_pairs


def generate_scene_graphs(config: DataGenConfig, valid_pairs: list[tuple[Path, Path]]):
    """
    Phase 2: I/O Heavy (Async).
    Sends the generated SoM images to GPT-4o to extract scene graphs.
    Saves the final JSONL dataset.
    """
    if not valid_pairs:
        logger.warning("No valid images provided for Scene Graph generation.")
        return

    logger.info("Phase 2: Initializing LLM Client...")

    # 1. Setup LLM
    llm = SceneGraphGenerator(
        config.llm_labeler, ontology_config=config.teacher.ontology_config
    )

    # Extract just the SoM paths for the batch generator
    som_paths_only = [p[1] for p in valid_pairs]
    jsonl_path = config.paths.output_dir / "train.jsonl"

    logger.info(f"Sending {len(som_paths_only)} requests to OpenAI...")

    results = asyncio.run(llm.batch_generate(som_paths_only, batch_size=20))

    logger.info("Saving training data...")
    saved_count = 0

    results_map = dict(results)

    with open(jsonl_path, "w") as f:
        for original_path, som_path in valid_pairs:
            scene_graph = results_map.get(som_path)

            # Skip failures (empty lists)
            if not scene_graph:
                continue

            entry = {
                "image": str(som_path.absolute()),  # VLM trains on SoM version
                "original_image": str(original_path.absolute()),
                "scene_graph": scene_graph,
            }

            json.dump(entry, f)
            f.write("\n")
            saved_count += 1

    logger.info(f"Phase 2 Complete! Saved {saved_count} samples to {jsonl_path}")


# TODO: Use DINO or my RF-DETR?
def run_generation(config_path: Path):
    """
    Orchestrator function called by the CLI.
    """
    logger.info(f"Loading config from: {config_path}")

    with config_path.open("r") as f:
        raw_config = yaml.safe_load(f)
    config = DataGenConfig(**raw_config)

    logger.info(f"Config loaded. Output dir: {config.paths.output_dir}")

    # Step 1: Vision (Synchronous/GPU)
    valid_pairs = generate_som_images(config)

    # Step 2: Language (Asynchronous/Network)
    generate_scene_graphs(config, valid_pairs)
