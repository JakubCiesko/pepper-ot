from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_qa_group

rules = [
    ReloadRule(
        "qa_generation",
        attrgetter("qa_generation"),
        "hot",
        _apply_qa_group,
    ),
]
