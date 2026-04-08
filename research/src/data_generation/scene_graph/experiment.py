import asyncio
import json
from pathlib import Path

from app.inference.caption.service import CaptionInferenceService
from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.service import DetectionService
from app.providers.llm.client import LLMClient
from app.schemas.config import CaptionConfig
from app.schemas.config import LLMConfig
from app.schemas.config import PromptSource
from app.schemas.config import StructuredOutputConfig
from PIL import Image
from pydantic import BaseModel
from pydantic import Field


class ImagePredicatesAttributes(BaseModel):
    predicates: list[str] = Field(default_factory=list, description="Predicate names")
    attributes: list[str] = Field(default_factory=list, description="Attribute names")


class GeneralPredicates(BaseModel):
    predicates: list[str] = Field(
        default_factory=list, description="General, broadly applicable predicate names"
    )


class GeneralAttributes(BaseModel):
    attributes: list[str] = Field(
        default_factory=list, description="General, broadly applicable attribute names"
    )


# First run detection, then pass image + list of detected objects to the caption service.


def resize_pil(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def load_images(
    image_dir: Path,
    extensions: tuple[str] = (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"),
    batch_size: int = 32,
):
    batch = []
    paths_batch = []
    for p in image_dir.glob("**/*"):
        if p.suffix not in extensions:
            continue
        with Image.open(p) as img:
            img = resize_pil(img)
            batch.append(img.copy())
            paths_batch.append(p)
        if len(batch) == batch_size:
            yield paths_batch, batch
            batch = []
            paths_batch = []
    if paths_batch and batch:
        yield paths_batch, batch


def run_detection(
    detection_service: DetectionService,
    input_path: Path,
    batch_size: int = 32,
    output_path: Path = None,
):
    output = {}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if detection_service.backend is DetectionModelType.RF_DETR:
        detection_service.model.model.optimize_for_inference(batch_size=batch_size)
    for i, (image_paths, image_batch) in enumerate(
        load_images(input_path, batch_size=batch_size)
    ):
        if (
            len(image_batch) != batch_size
            and detection_service.backend is DetectionModelType.RF_DETR
        ):
            detection_service.model.model.remove_optimized_model()
        if (
            len(image_batch) == 1
            and detection_service.backend is DetectionModelType.RF_DETR
        ):
            results = [detector.detect(image_batch[0])]
        else:
            results = detector.detect_batch(image_batch)
        for image_path, detections in zip(image_paths, results, strict=True):
            output[str(image_path)] = [det.model_dump() for det in detections]
        if output_path is not None:
            with output_path.open(mode="w") as f:
                json.dump(output, f)
        if i % 10 == 0:
            print(f"Running batch {i} of size {len(image_paths)}")
    return output


async def run_caption_batch(image_paths, image_batch, caption_service, data, semaphore):
    results = []
    async with semaphore:
        batch_tasks = []
        for p, img in zip(image_paths, image_batch, strict=True):
            detected_objects = data.get(str(p), [])
            user_prompt = f"Mention these detected objects: {[d['label'] for d in detected_objects]}"
            batch_tasks.append(
                caption_service.caption_image(img, prompt_override=user_prompt)
            )
        batch_results = await asyncio.gather(*batch_tasks)
        for p, caption in zip(image_paths, batch_results, strict=True):
            results.append((str(p), caption.__dict__))
    return results


async def run_caption(
    image_dir: Path,
    caption_service,
    image_detection_output_path: Path,
    batch_size: int = 4,
    max_concurrent_batches: int = 1,
    output_path: Path = None,
):
    with image_detection_output_path.open("r") as f:
        data = json.load(f)

    semaphore = asyncio.Semaphore(max_concurrent_batches)
    all_results = []

    for i, (image_paths, image_batch) in enumerate(
        load_images(image_dir, batch_size=batch_size)
    ):
        batch_results = await run_caption_batch(
            image_paths, image_batch, caption_service, data, semaphore
        )
        all_results.extend(batch_results)
        if i % 10 == 0:
            print(f"Running caption batch {i} of size {len(image_paths)}")

    all_results = dict(all_results)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode="w") as f:
            json.dump(all_results, f)
    return all_results


async def extract_structured_from_captions(
    captions_path: Path,
    detection_output_path: Path,
    llm_client: LLMClient,
    output_path: Path,
    batch_size: int = 4,
    max_concurrent_batches: int = 2,
    n_predicates: int = 50,
    n_attributes: int = 25,
):
    """
    Extract structured predicates and attributes from captions using LLMClient.

    captions_path: JSON file {image_path: caption}
    detection_output_path: JSON file {image_path: [detected_objects]}
    llm_client: configured LLMClient instance
    output_path: path to save structured output JSON
    """

    # Load files
    with captions_path.open("r") as f:
        captions = json.load(f)
    with detection_output_path.open("r") as f:
        detected_objects = json.load(f)

    semaphore = asyncio.Semaphore(max_concurrent_batches)
    all_results: dict[str, ImagePredicatesAttributes] = {}
    system_prompt = (
        f"Extract {n_predicates} predicates and {n_attributes} attributes "
        "from the provided image caption. "
        "Focus mainly on objects mentioned in the provided list of objects. "
        "Try to pick the most representative predicates and attributes which can be employed over multiple images. "
        "Keep in mind these predicates and attributes will be used by robots in conversations with humans, "
        "so they should be simple, practical and conversation relevant. "
        "Output just the predicates and attributes, do not repeat the object names."
    )

    # Split captions into batches
    image_items = list(captions.items())
    for i in range(0, len(image_items), batch_size):
        batch = image_items[i : i + batch_size]

        async def process_batch(batch_items):
            batch_results = {}
            async with semaphore:
                tasks = []
                for image_path, caption in batch_items:
                    objects = detected_objects.get(image_path, [])
                    # Take top N objects or all
                    top_objects = [d["label"] for d in objects]  # configurable
                    user_prompt = (
                        f"Caption: {caption}\n"
                        f"List of the Most important objects in the image: {top_objects}\n"
                        "Return a JSON object with list of 'predicates', and list of 'attributes'."
                    )
                    tasks.append(
                        llm_client.generate(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            output_schema=ImagePredicatesAttributes,
                        )
                    )
                batch_responses = await asyncio.gather(*tasks)
                for (image_path, _), resp in zip(
                    batch_items, batch_responses, strict=True
                ):
                    batch_results[image_path] = resp.parsed
            return batch_results

        batch_results = await process_batch(batch)
        all_results.update(batch_results)

    # Save structured output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {k: v.model_dump() if v else None for k, v in all_results.items()},
            f,
            indent=2,
        )

    return all_results


async def consolidate_vocabulary(
    final_vocab_path: Path,
    vocabulary_path: Path,
    llm_client: LLMClient,
    n_predicates: int = 50,
    n_attributes: int = 25,
):

    with vocabulary_path.open("r") as f:
        vocabulary = json.load(f)
    predicates, attributes = [], []
    for image_vocab in vocabulary.values():
        predicates.extend(image_vocab["predicates"])
        attributes.extend(image_vocab["attributes"])

    system_prompt = (
        "Extract or create the most general forms of predicates from the provided list of predicates. "
        f"Extract or create {n_predicates} predicates. "
        "The predicates are relations between **two** real, concrete objects "
        "(ex. Man plays with dog produces predicate: plays). "
        "Focus on semantic, positional (ex. is_right_of), functional (ex. speaking_to, holding), "
        "and comparative predicates (ex. is_bigger_than). "
        "Write them with underscore (_) instead of spaces. "
        "Keep in mind these predicates will be used by robots in conversations with humans, "
        "so they should be simple, practical and conversation relevant."
    )
    user_prompt = f"Extract or create general, applicable form of predicates from this List of predicates: {predicates}"
    pred_resp = await llm_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=GeneralPredicates,
    )
    system_prompt = (
        "Extract or create the most general forms of attributes from the provided list of attributes. "
        f"Extract or create {n_attributes} attributes. "
        "The attributes are attributes, qualities, colors, materials, or states of real objects. "
        "Write them with underscore (_) instead of spaces. "
        "If applicable, use is_ as prefix (ex. is_green, is_wooden). "
        "Keep in mind these attributes will be used by robots in conversations with humans, "
        "so they should be simple, practical and conversation relevant."
    )
    user_prompt = f"Extract or create general, applicable form of attributes from this List of attributes: {attributes}"

    attr_resp = await llm_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=GeneralAttributes,
    )
    output = {
        "predicates": pred_resp.parsed.model_dump(),
        "attributes": attr_resp.parsed.model_dump(),
    }
    final_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with final_vocab_path.open("w") as f:
        json.dump(output, f, indent=2)
    return output


if __name__ == "__main__":

    # RF-Detr Model, confidence = 0.5
    RUN_DETECTION = False
    RUN_CAPTION = False
    RUN_VOCABULARY_BUILDER = False

    input_path = Path("/home/jakub-ciesko/Work/pepper_ot/data/images")
    image_detection_output_path = Path(
        "/home/jakub-ciesko/Work/pepper_ot/data/images/output/detections.json"
    )
    caption_output_path = Path(
        "/home/jakub-ciesko/Work/pepper_ot/data/images/output/captions.json"
    )
    vocabulary_output_path = Path(
        "/home/jakub-ciesko/Work/pepper_ot/data/images/output/vocabulary.json"
    )
    final_vocab = Path(
        "/home/jakub-ciesko/Work/pepper_ot/data/images/output/final_vocabulary.json"
    )
    caption_config = CaptionConfig(
        mode="prompted",
        max_words=None,
        system_prompt=PromptSource(
            text="Describe the provided image very thoroughly. Mention all provided objects and their "
            "relationships to each other or to other objects in the image. Mention their attributes too."
        ),
        user_prompt=None,
        provider="openai",
        model_id="gpt-5-nano-2025-08-07",
    )

    GPU_BATCH_SIZE = 4
    LLM_BATCH_SIZE = 16
    LLM_CONCURRENT_BATCHES = 2

    if RUN_DETECTION:
        detector = DetectionService(DetectionModelType.RT_DETR)
        run_detection(
            detector,
            input_path,
            GPU_BATCH_SIZE,
            output_path=image_detection_output_path,
        )
    if RUN_CAPTION:
        caption_service = CaptionInferenceService(
            caption_config,
            system_prompt=str(caption_config.system_prompt),
            user_prompt=caption_config.user_prompt,
        )
        captions = asyncio.run(
            run_caption(
                input_path,
                caption_service,
                image_detection_output_path,
                output_path=caption_output_path,
                batch_size=LLM_BATCH_SIZE,
                max_concurrent_batches=LLM_CONCURRENT_BATCHES,
            )
        )
    llm_config = LLMConfig(
        provider="openai",
        model_id="gpt-5-nano-2025-08-07",
        structured_output=StructuredOutputConfig(mode="provider_native"),
    )
    llm_client = LLMClient(llm_config)
    n_predicates, n_attributes = 100, 50

    if RUN_VOCABULARY_BUILDER:

        structured_output = asyncio.run(
            extract_structured_from_captions(
                captions_path=caption_output_path,
                detection_output_path=image_detection_output_path,
                llm_client=llm_client,
                output_path=vocabulary_output_path,
                batch_size=LLM_BATCH_SIZE,
                max_concurrent_batches=LLM_CONCURRENT_BATCHES,
                n_predicates=n_predicates,
                n_attributes=n_attributes,
            )
        )
    output = asyncio.run(
        consolidate_vocabulary(
            final_vocab,
            vocabulary_output_path,
            llm_client,
            n_predicates=n_predicates,
            n_attributes=n_attributes,
        )
    )
    print(output)
