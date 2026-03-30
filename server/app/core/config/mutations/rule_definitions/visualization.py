from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_pipeline_group

rules = [
    ReloadRule(
        "visualization.mask_backend",
        attrgetter("visualization.mask_backend"),
        "hard",
    ),
    ReloadRule(
        "visualization.device",
        attrgetter("visualization.device"),
        "hard",
    ),
    ReloadRule(
        "visualization", attrgetter("visualization"), "hot", _apply_pipeline_group
    ),
]
