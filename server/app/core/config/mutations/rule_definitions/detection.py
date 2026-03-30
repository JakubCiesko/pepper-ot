from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_pipeline_group

rules = [
    ReloadRule("detection.backend", attrgetter("detection.backend"), "hard"),
    ReloadRule("detection.weights_path", attrgetter("detection.weights_path"), "hard"),
    ReloadRule(
        "detection.confidence_threshold",
        attrgetter("detection.confidence_threshold"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "detection.device",
        attrgetter("detection.device"),
        "hard",
        _apply_pipeline_group,
    ),
    # objects
    ReloadRule(
        "detection.ontology",
        attrgetter("detection.ontology"),
        "hot",
        _apply_pipeline_group,
    ),
    ReloadRule(
        "detection.ontology_path",
        attrgetter("detection.ontology_path"),
        "hot",
        _apply_pipeline_group,
    ),
]
