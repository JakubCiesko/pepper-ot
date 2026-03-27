from app.core.config.reload_contract import ConfigDiff
from app.core.config.reload_contract import apply_hot_changes
from app.core.config.reload_contract import diff_config as contract_diff_config
from app.core.config.reload_contract import hard_reload_fields
from app.core.runtime.state import AppState
from app.schemas.config import AppConfig


def diff_config(old, new) -> ConfigDiff:
    return contract_diff_config(old, new)


async def apply_hot_config(ml_state: AppState, new: AppConfig):
    old = ml_state.config or new
    await apply_hot_changes(ml_state, old, new)


def hard_fields() -> list[str]:
    return hard_reload_fields()
