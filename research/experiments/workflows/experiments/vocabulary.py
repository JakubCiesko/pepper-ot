import asyncio
from time import perf_counter

from tqdm.auto import tqdm

from research.experiments.adapters import ServerLLMAdapter
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import GeneralAttributes
from research.experiments.schemas import GeneralPredicates
from research.experiments.schemas import ImagePredicatesAttributes


async def run_vocabulary_mining(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting vocabulary mining phase")
    stage_metrics = StageMetrics(stage="vocabulary")

    descriptions = load_json(run.run_dir / config.paths.descriptions_file, default={})
    run.logger.info(
        "Loaded descriptions for %d images from file %s",
        len(descriptions),
        run.run_dir / config.paths.descriptions_file,
    )
    detections = load_json(run.run_dir / config.paths.detections_file, default={})
    run.logger.info(
        "Loaded detections for %d images from file %s",
        len(detections),
        run.run_dir / config.paths.detections_file,
    )
    if not descriptions:
        raise RuntimeError(
            f"Descriptions file {run.run_dir / config.paths.descriptions_file} is empty. "
            f"Run description phase first."
        )

    llm = ServerLLMAdapter(
        provider=config.vocabulary_model.provider,
        model_id=config.vocabulary_model.model_id,
        structured_mode=config.vocabulary_model.structured_mode,
    )

    semaphore = asyncio.Semaphore(config.vocabulary.max_concurrent_batches)
    per_image: dict[str, dict] = {}
    progress = tqdm(total=len(descriptions), desc="vocab_extract", unit="img")

    async def process_one(image_path: str, payload: dict):
        t0 = perf_counter()
        async with semaphore:
            try:
                caption = str(payload.get("text", "")).strip()
                if not caption:
                    per_image[image_path] = {"predicates": [], "attributes": []}
                    stage_metrics.record_skipped("missing_caption")
                    return
                labels = [
                    d.get("label")
                    for d in detections.get(image_path, [])
                    if d.get("label")
                ]
                user_prompt = (
                    f"Caption: {caption}\n"
                    f"Detected objects: {labels}\n"
                    "Return JSON with predicates and attributes, focus mainly on the objects "
                    "mentioned in the list of detected objects."
                )
                resp = await llm.generate_structured(
                    system_prompt=config.vocabulary.extract_system_prompt,
                    user_prompt=user_prompt,
                    output_schema=ImagePredicatesAttributes,
                )
                parsed = getattr(resp, "parsed", None)
                per_image[image_path] = (
                    parsed.model_dump()
                    if parsed
                    else {"predicates": [], "attributes": []}
                )
                if parsed is None:
                    stage_metrics.record_failed(
                        "structured_parse_missing", perf_counter() - t0
                    )
                else:
                    stage_metrics.record_ok(perf_counter() - t0)
            except Exception:
                per_image[image_path] = {"predicates": [], "attributes": []}
                stage_metrics.record_failed("vocab_extract_error", perf_counter() - t0)
            finally:
                progress.update(1)

    await asyncio.gather(
        *(process_one(path, payload) for path, payload in descriptions.items())
    )
    progress.close()
    save_json(run.run_dir / config.paths.vocabulary_candidates_file, per_image)
    run.logger.info("Saved per-image vocabulary candidates")

    predicates: list[str] = []
    attributes: list[str] = []
    for row in per_image.values():
        predicates.extend(row.get("predicates", []))
        attributes.extend(row.get("attributes", []))

    consolidation_progress = tqdm(total=2, desc="vocab_consolidate", unit="call")
    pred_resp = await llm.generate_structured(
        system_prompt=config.vocabulary.consolidate_predicates_prompt,
        user_prompt=f"Input predicates: {predicates}",
        output_schema=GeneralPredicates,
    )
    consolidation_progress.update(1)
    attr_resp = await llm.generate_structured(
        system_prompt=config.vocabulary.consolidate_attributes_prompt,
        user_prompt=f"Input attributes: {attributes}",
        output_schema=GeneralAttributes,
    )
    consolidation_progress.update(1)
    consolidation_progress.close()

    pred_parsed = getattr(pred_resp, "parsed", None)
    attr_parsed = getattr(attr_resp, "parsed", None)
    final_vocab = {
        "predicates": (pred_parsed.predicates if pred_parsed else [])[
            : config.vocabulary.predicates_target
        ],
        "attributes": (attr_parsed.attributes if attr_parsed else [])[
            : config.vocabulary.attributes_target
        ],
    }
    save_json(run.run_dir / config.paths.vocabulary_final_file, final_vocab)
    stage_metrics.finish()
    save_json(run.run_dir / "metrics_vocabulary.json", stage_metrics.to_dict())
    run.logger.info(
        "Saved final vocabulary predicates=%d attributes=%d",
        len(final_vocab["predicates"]),
        len(final_vocab["attributes"]),
    )
    run.logger.info(
        "Vocabulary summary ok=%d failed=%d skipped=%d duration=%.3fs",
        stage_metrics.ok,
        stage_metrics.failed,
        stage_metrics.skipped,
        stage_metrics.duration_s,
    )
    return final_vocab
