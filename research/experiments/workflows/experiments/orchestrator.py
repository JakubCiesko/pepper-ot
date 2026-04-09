from research.experiments.config.models import ExperimentConfig
from research.experiments.io import RunContext

from .context_rot import run_context_rot
from .descriptions import run_descriptions
from .draft_scene_graph import run_draft_scene_graph
from .evaluate_scene_graph import run_scene_graph_evaluation
from .vocabulary import run_vocabulary_mining


async def run_all_phases(config: ExperimentConfig, run: RunContext) -> dict:
    outputs: dict[str, dict] = {}
    if config.descriptions.enabled:
        outputs["descriptions"] = await run_descriptions(config, run)
    if config.vocabulary.enabled:
        outputs["vocabulary"] = await run_vocabulary_mining(config, run)
    if config.draft_scene_graph.enabled:
        outputs["draft_scene_graph"] = await run_draft_scene_graph(config, run)
    if config.context_rot.enabled:
        outputs["context_rot"] = await run_context_rot(config, run)
    if config.evaluation.enabled:
        outputs["evaluation"] = await run_scene_graph_evaluation(config, run)
    run.logger.info("All enabled phases completed")
    return outputs
