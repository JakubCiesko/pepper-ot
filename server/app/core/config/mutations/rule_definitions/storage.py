from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule

rules = [
    ReloadRule(
        "storage.persist_last_state", attrgetter("storage.persist_last_state"), "hot"
    ),
    ReloadRule("storage.last_state_path", attrgetter("storage.last_state_path"), "hot"),
    ReloadRule("storage.store_image", attrgetter("storage.store_image"), "hot"),
]
