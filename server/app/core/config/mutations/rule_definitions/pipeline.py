from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_pipeline_group

rules = [
    ReloadRule(
        "pipeline_controls",
        attrgetter("pipeline_controls"),
        "hot",
        _apply_pipeline_group,
    ),
]
