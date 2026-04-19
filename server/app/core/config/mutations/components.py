from app.core.config.mutations.reload_rule import ConfigDiff
from app.core.config.mutations.reload_rule import ReloadRule
import app.core.config.mutations.rule_definitions as RuleDefinitions
from app.core.runtime.state import AppState
from app.schemas.config import AppConfig

RELOAD_RULES: list[ReloadRule] = (
    RuleDefinitions.DetectionRules
    + RuleDefinitions.CaptionRules
    + RuleDefinitions.SceneGraphRules
    + RuleDefinitions.VisualizationRules
    + RuleDefinitions.StorageRules
    + RuleDefinitions.WorkerRules
    + RuleDefinitions.TrackingRules
    + RuleDefinitions.ChatRules
    + RuleDefinitions.QAGenerationRules
    + RuleDefinitions.PipelineRules
)


def _assert_unique_paths():
    seen: set[str] = set()
    dupes: set[str] = set()
    for rule in RELOAD_RULES:
        if rule.path in seen:
            dupes.add(rule.path)
        seen.add(rule.path)
    if dupes:
        raise ValueError(f"Duplicate reload rule paths: {sorted(dupes)}")


_assert_unique_paths()


def rule_index() -> dict[str, ReloadRule]:
    return {rule.path: rule for rule in RELOAD_RULES}


def diff_config(old: AppConfig, new: AppConfig) -> ConfigDiff:
    hot: list[str] = []
    hard: list[str] = []
    for rule in RELOAD_RULES:
        if rule.getter(old) != rule.getter(new):
            (hot if rule.mode == "hot" else hard).append(rule.path)
    return ConfigDiff(hot=hot, hard=hard)


def hard_reload_fields() -> list[str]:
    return [rule.path for rule in RELOAD_RULES if rule.mode == "hard"]


async def apply_hot_changes(
    ml_state: "AppState", old: AppConfig, new: AppConfig
) -> ConfigDiff:
    diff = diff_config(old, new)

    ml_state.config = new
    ml_state.config_version += 1

    idx = rule_index()
    applied_handlers: set[int] = set()
    for path in diff.hot:
        rule = idx.get(path)
        if rule is None or rule.apply_hot is None:
            continue
        marker = id(rule.apply_hot)
        if marker in applied_handlers:
            continue
        rule.apply_hot(ml_state, new)
        applied_handlers.add(marker)

    if ml_state.worker_manager:
        await ml_state.worker_manager.apply_hot_config(new, ml_state.config_version)
    return diff
