from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule
from app.core.config.mutations.reload_rule import _apply_chat_group

rules = [
    ReloadRule("chat.provider", attrgetter("chat.provider"), "hard"),
    ReloadRule("chat.model_id", attrgetter("chat.model_id"), "hard"),
    ReloadRule("chat.base_url", attrgetter("chat.base_url"), "hard"),
    ReloadRule("chat.timeout_seconds", attrgetter("chat.timeout_seconds"), "hard"),
    ReloadRule("chat.api_key_env", attrgetter("chat.api_key_env"), "hard"),
    ReloadRule(
        "chat.client_init_kwargs", attrgetter("chat.client_init_kwargs"), "hard"
    ),
    ReloadRule("chat.device", attrgetter("chat.device"), "hot", _apply_chat_group),
    ReloadRule(
        "chat.call_kwargs", attrgetter("chat.call_kwargs"), "hot", _apply_chat_group
    ),
    ReloadRule(
        "chat.structured_output",
        attrgetter("chat.structured_output"),
        "hot",
        _apply_chat_group,
    ),
    ReloadRule(
        "chat.system_prompt", attrgetter("chat.system_prompt"), "hot", _apply_chat_group
    ),
]
