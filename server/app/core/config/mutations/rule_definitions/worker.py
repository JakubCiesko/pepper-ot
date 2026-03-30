from operator import attrgetter

from app.core.config.mutations.reload_rule import ReloadRule

rules = [
    ReloadRule("worker.enabled", attrgetter("worker.enabled"), "hard"),
    ReloadRule("worker.host", attrgetter("worker.host"), "hard"),
    ReloadRule("worker.port", attrgetter("worker.port"), "hard"),
    ReloadRule(
        "worker.startup_timeout_seconds",
        attrgetter("worker.startup_timeout_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.shutdown_grace_seconds",
        attrgetter("worker.shutdown_grace_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.max_startup_queue", attrgetter("worker.max_startup_queue"), "hard"
    ),
    ReloadRule(
        "worker.healthcheck_interval_seconds",
        attrgetter("worker.healthcheck_interval_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.restart_max_attempts", attrgetter("worker.restart_max_attempts"), "hard"
    ),
    ReloadRule(
        "worker.restart_window_seconds",
        attrgetter("worker.restart_window_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.restart_backoff_seconds",
        attrgetter("worker.restart_backoff_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.circuit_breaker_cooldown_seconds",
        attrgetter("worker.circuit_breaker_cooldown_seconds"),
        "hard",
    ),
    ReloadRule(
        "worker.idle_timeout_seconds", attrgetter("worker.idle_timeout_seconds"), "hot"
    ),
    ReloadRule(
        "worker.idle_check_interval_seconds",
        attrgetter("worker.idle_check_interval_seconds"),
        "hot",
    ),
    ReloadRule(
        "worker.request_timeout_seconds",
        attrgetter("worker.request_timeout_seconds"),
        "hot",
    ),
]
