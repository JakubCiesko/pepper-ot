import asyncio
import random

from research.experiments.adapters import ServerLLMAdapter
from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext
from research.experiments.io import load_json
from research.experiments.io import save_json
from research.experiments.schemas import SceneGraphDraft


def _build_vocab_slices(vocab: dict, min_size: int, step: int) -> list[dict]:
    predicates = list(vocab.get("predicates", []))
    attributes = list(vocab.get("attributes", []))
    max_len = len(predicates) + len(attributes)
    sizes = list(range(min_size, max_len + 1, step))
    if not sizes or sizes[-1] != max_len:
        sizes.append(max_len)
    out: list[dict] = []
    for size in sizes:
        keep_pred = min(len(predicates), max(1, size // 2))
        keep_attr = min(len(attributes), size - keep_pred)
        out.append(
            {
                "predicates": predicates[:keep_pred],
                "attributes": attributes[:keep_attr],
            }
        )
    return out


async def run_context_rot(config: ExperimentConfig, run: RunContext) -> dict:
    run.logger.info("Starting context-rot phase")
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

    random.seed(config.seed)
    vocab_slices = _build_vocab_slices(
        vocabulary,
        min_size=config.context_rot.min_vocab_size,
        step=config.context_rot.step,
    )

    results: dict[str, dict] = {}
    sample_items = list(descriptions.items())

    async def evaluate_one(image_path: str, payload: dict, sliced_vocab: dict):
        caption = str(payload.get("text", "")).strip()
        objects = [
            {"id": det.get("object_id"), "label": det.get("label")}
            for det in detections.get(image_path, [])
        ]
        prompt = config.draft_scene_graph.user_prompt_template
        prompt = prompt.replace("{objects}", str(objects))
        prompt = prompt.replace("{vocabulary}", str(sliced_vocab))
        prompt = prompt.replace(
            "{caption}",
            caption if config.prompting.include_caption_in_sgg_prompt else "",
        )
        resp = await llm.generate_structured(
            system_prompt=config.draft_scene_graph.system_prompt,
            user_prompt=prompt,
            output_schema=SceneGraphDraft,
        )
        parsed = getattr(resp, "parsed", None)
        rel_count = len(parsed.relationships) if parsed else 0
        return {"relationship_count": rel_count}

    for sliced in vocab_slices:
        key = f"vocab_{len(sliced.get('predicates', [])) + len(sliced.get('attributes', []))}"
        stats = {"images": 0, "relationship_count_sum": 0}
        for _ in range(config.context_rot.rounds_per_size):
            tasks = [
                evaluate_one(path, payload, sliced) for path, payload in sample_items
            ]
            batch = await asyncio.gather(*tasks)
            for item in batch:
                stats["images"] += 1
                stats["relationship_count_sum"] += item["relationship_count"]
        stats["relationship_count_avg"] = (
            stats["relationship_count_sum"] / stats["images"]
            if stats["images"]
            else 0.0
        )
        results[key] = stats

    save_json(run.run_dir / config.paths.context_rot_file, results)
    run.logger.info("Saved context-rot results")
    return results
