import asyncio

from research.experiments.adapters import ServerLLMAdapter
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft


async def run_draft_scene_graph(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting draft scene graph phase")
    descriptions = load_json(run.run_dir / config.paths.descriptions_file, default={})
    detections = load_json(run.run_dir / config.paths.detections_file, default={})
    vocabulary = load_json(run.run_dir / config.paths.vocabulary_final_file, default={})
    if not descriptions or not vocabulary:
        raise RuntimeError(
            "Descriptions or vocabulary missing. Run previous phases first."
        )

    llm = ServerLLMAdapter(
        provider=config.draft_sgg_model.provider,
        model_id=config.draft_sgg_model.model_id,
        structured_mode=config.draft_sgg_model.structured_mode,
    )

    semaphore = asyncio.Semaphore(config.draft_scene_graph.max_concurrent_batches)
    drafts: dict[str, dict] = {}

    async def process_one(image_path: str, payload: dict):
        async with semaphore:
            caption = str(payload.get("text", "")).strip()
            objects = [
                {
                    "id": det.get("object_id"),
                    "label": det.get("label"),
                    "bbox": det.get("bbox"),
                }
                for det in detections.get(image_path, [])
            ]
            user_prompt = config.draft_scene_graph.user_prompt_template
            user_prompt = user_prompt.replace("{objects}", str(objects))
            user_prompt = user_prompt.replace("{vocabulary}", str(vocabulary))
            if config.prompting.include_caption_in_sgg_prompt:
                user_prompt = user_prompt.replace("{caption}", caption)
            else:
                user_prompt = user_prompt.replace("{caption}", "")

            resp = await llm.generate_structured(
                system_prompt=config.draft_scene_graph.system_prompt,
                user_prompt=user_prompt,
                output_schema=SceneGraphDraft,
            )
            parsed = getattr(resp, "parsed", None)
            drafts[image_path] = (
                parsed.model_dump() if parsed else {"relationships": []}
            )

    await asyncio.gather(
        *(process_one(path, payload) for path, payload in descriptions.items())
    )
    save_json(run.run_dir / config.paths.draft_scene_graph_file, drafts)
    run.logger.info("Saved draft scene graphs for %d images", len(drafts))
    return drafts
