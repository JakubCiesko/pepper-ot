from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_caption_group

rules = [
    ReloadRule("caption.provider", attrgetter("caption.provider"), "hard"),
    ReloadRule("caption.model_id", attrgetter("caption.model_id"), "hard"),
    ReloadRule("caption.base_url", attrgetter("caption.base_url"), "hard"),
    ReloadRule(
        "caption.timeout_seconds", attrgetter("caption.timeout_seconds"), "hard"
    ),
    ReloadRule("caption.api_key_env", attrgetter("caption.api_key_env"), "hard"),
    ReloadRule(
        "caption.client_init_kwargs", attrgetter("caption.client_init_kwargs"), "hard"
    ),
    ReloadRule("caption.device", attrgetter("caption.device"), "hard"),
    ReloadRule(
        "caption.call_kwargs",
        attrgetter("caption.call_kwargs"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule(
        "caption.structured_output",
        attrgetter("caption.structured_output"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule("caption.mode", attrgetter("caption.mode"), "hot", _apply_caption_group),
    ReloadRule(
        "caption.max_words",
        attrgetter("caption.max_words"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule(
        "caption.system_prompt",
        attrgetter("caption.system_prompt"),
        "hot",
        _apply_caption_group,
    ),
    ReloadRule(
        "caption.user_prompt",
        attrgetter("caption.user_prompt"),
        "hot",
        _apply_caption_group,
    ),
]
