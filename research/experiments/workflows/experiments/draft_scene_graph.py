import asyncio
from pathlib import Path
from time import perf_counter

from tqdm.auto import tqdm

from research.experiments.adapters import ServerVLMAdapter
from research.experiments.adapters.bootstrap import ensure_server_app_importable
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import StageMetrics
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft

from .scene_graph_common import build_prompt_image
from .scene_graph_common import objects_for_prompt
from .scene_graph_common import pil_to_jpeg_bytes
from .scene_graph_common import render_template
from .scene_graph_common import vocabulary_for_prompt


async def run_draft_scene_graph(config: ExperimentConfig, run: RunContext) -> dict:
    """Generate draft scene graphs from descriptions, detections, and vocabulary.

    Args:
        config: Experiment configuration containing draft SGG prompt settings,
            SoM rendering settings, VLM model settings, and artifact paths.
        run: Run context containing previous phase artifacts and receiving draft
            outputs.

    Returns:
        Mapping from image path to a draft scene graph payload. Each payload
        includes image metadata, prompt objects, prompt vocabulary, parsed
        relationships, and optionally the raw model response.

    Raises:
        RuntimeError: If descriptions or final vocabulary are missing. Adapter
            setup errors may also propagate before per-image processing starts.

    Side Effects:
        Reads descriptions.json, detections.json, and vocabulary_final.json;
        optionally writes SoM prompt images; writes draft_scene_graph.json and
        metrics_draft_scene_graph.json under run.run_dir.
    """
    run.logger.info("Starting draft scene graph phase")
    stage_metrics = StageMetrics(stage="draft_scene_graph")
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
    vocabulary = load_json(run.run_dir / config.paths.vocabulary_final_file, default={})
    run.logger.info(
        "Loaded vocabulary with %d predicates, %d attributes from file %s",
        len(vocabulary.get("predicates", [])),
        len(vocabulary.get("attributes", [])),
        run.run_dir / config.paths.vocabulary_final_file,
    )

    if not descriptions or not vocabulary:
        raise RuntimeError(
            "Descriptions or vocabulary missing. Run previous phases first."
        )

    ensure_server_app_importable()
    from app.inference.scene_graph.som import SoMPainter

    vlm = ServerVLMAdapter(
        provider=config.draft_sgg_model.provider,
        model_id=config.draft_sgg_model.model_id,
        structured_mode=config.draft_sgg_model.structured_mode,
        device=config.draft_scene_graph.som_device,
        base_url=config.draft_sgg_model.base_url,
    )

    painter = SoMPainter(
        line_thickness=config.draft_scene_graph.som_line_thickness,
        color_lookup=config.draft_scene_graph.som_color_lookup,
        mask_opacity=config.draft_scene_graph.som_mask_opacity,
        mask_backend=config.draft_scene_graph.som_mask_backend,
        device=config.draft_scene_graph.som_device,
    )

    som_dir = run.run_dir / config.draft_scene_graph.som_output_dir
    if config.draft_scene_graph.save_som_images:
        som_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.draft_scene_graph.max_concurrent_batches)
    drafts: dict[str, dict] = {}
    progress = tqdm(total=len(descriptions), desc="draft_sgg", unit="img")

    async def process_one(image_path: str, payload: dict):
        t0 = perf_counter()
        async with semaphore:
            try:
                path = Path(image_path)
                if not path.exists():
                    stage_metrics.record_skipped("missing_image_path")
                    drafts[image_path] = {"relationships": []}
                    return

                caption = str(payload.get("text", "")).strip()
                detected_rows = detections.get(image_path, [])
                objects = objects_for_prompt(detected_rows)

                render_values = {
                    "caption": (
                        caption
                        if config.prompting.include_caption_in_sgg_prompt
                        else ""
                    ),
                    "vocabulary": vocabulary_for_prompt(
                        vocabulary, config.draft_scene_graph.vocab_mode
                    ),
                    "objects": objects,
                }
                system_prompt = render_template(
                    config.draft_scene_graph.system_prompt, render_values
                )
                user_prompt = render_template(
                    config.draft_scene_graph.user_prompt_template, render_values
                )
                if config.draft_scene_graph.vocab_mode == "open":
                    system_prompt = system_prompt.replace(
                        "Use ONLY the values in the provided PREDICATES and ATTRIBUTES lists. Do not invent new ones.",
                        "Use concise relation and attribute names grounded in the image. Do not invent objects.",
                    )
                    user_prompt = user_prompt.replace(
                        "Allowed predicates and attributes: \n", ""
                    )

                prompt_image, som_image = build_prompt_image(
                    image_path=path,
                    detected_rows=detected_rows,
                    config=config,
                    painter=painter,
                )

                som_path: str | None = None
                if config.draft_scene_graph.save_som_images and som_image is not None:
                    som_file = som_dir / f"som_{path.name}"
                    som_image.save(som_file)
                    som_path = str(som_file)
                raw_text, parsed = await vlm.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_bytes=pil_to_jpeg_bytes(prompt_image),
                    output_schema=SceneGraphDraft,
                )
                parsed_payload = (
                    parsed.model_dump() if parsed else {"relationships": []}
                )

                drafts[image_path] = {
                    "image_path": image_path,
                    "som_image_path": som_path,
                    "visual_mode": (
                        "raw"
                        if not config.draft_scene_graph.use_som_image
                        else config.draft_scene_graph.visual_mode
                    ),
                    "vocab_mode": config.draft_scene_graph.vocab_mode,
                    "caption": caption,
                    "objects": objects,
                    "vocabulary": render_values["vocabulary"],
                    **parsed_payload,
                }
                if config.draft_scene_graph.include_raw_response:
                    drafts[image_path]["raw_response"] = raw_text
                if parsed is None:
                    stage_metrics.record_failed(
                        "structured_parse_missing", perf_counter() - t0
                    )
                else:
                    stage_metrics.record_ok(perf_counter() - t0)
            except Exception:
                drafts[image_path] = {"relationships": []}
                stage_metrics.record_failed(
                    "draft_generation_error", perf_counter() - t0
                )
            finally:
                progress.update(1)

    await asyncio.gather(
        *(process_one(path, payload) for path, payload in descriptions.items())
    )
    progress.close()
    stage_metrics.finish()
    save_json(run.run_dir / config.paths.draft_scene_graph_file, drafts)
    save_json(run.run_dir / "metrics_draft_scene_graph.json", stage_metrics.to_dict())
    run.logger.info("Saved draft scene graphs for %d images", len(drafts))
    run.logger.info(
        "Draft SGG summary ok=%d failed=%d skipped=%d duration=%.3fs",
        stage_metrics.ok,
        stage_metrics.failed,
        stage_metrics.skipped,
        stage_metrics.duration_s,
    )
    return drafts
