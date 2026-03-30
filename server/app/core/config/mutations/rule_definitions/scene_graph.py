from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_pipeline_group

rules = [
    ReloadRule(
        "scene_graph.vlm.provider", attrgetter("scene_graph.vlm.provider"), "hard"
    ),
    ReloadRule(
        "scene_graph.vlm.model_id", attrgetter("scene_graph.vlm.model_id"), "hard"
    ),
    ReloadRule(
        "scene_graph.vlm.base_url", attrgetter("scene_graph.vlm.base_url"), "hard"
    ),
    ReloadRule(
        "scene_graph.vlm.timeout_seconds",
        attrgetter("scene_graph.vlm.timeout_seconds"),
        "hard",
    ),
    ReloadRule(
        "scene_graph.vlm.api_key_env",
        attrgetter("scene_graph.vlm.api_key_env"),
        "hard",
    ),
    ReloadRule(
        "scene_graph.vlm.client_init_kwargs",
        attrgetter("scene_graph.vlm.client_init_kwargs"),
        "hard",
    ),
    ReloadRule("scene_graph.vlm.device", attrgetter("scene_graph.vlm.device"), "hard"),
    ReloadRule(
        "scene_graph.vlm.call_kwargs",
        attrgetter("scene_graph.vlm.call_kwargs"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.structured_output",
        attrgetter("scene_graph.vlm.structured_output"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.structured_schema",
        attrgetter("scene_graph.vlm.structured_schema"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.local_vlm_hints",
        attrgetter("scene_graph.vlm.local_vlm_hints"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.system_prompt",
        attrgetter("scene_graph.vlm.system_prompt"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.user_prompt",
        attrgetter("scene_graph.vlm.user_prompt"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.vlm.ontology",
        attrgetter("scene_graph.vlm.ontology"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "scene_graph.mode", attrgetter("scene_graph.mode"), "hot", _apply_pipeline_group
    ),
    ReloadRule(
        "scene_graph.rules",
        attrgetter("scene_graph.rules"),
        "hot",
        _apply_pipeline_group,
    ),
]
