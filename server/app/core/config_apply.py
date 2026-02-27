from dataclasses import dataclass

from app.core.state import MLState
from app.schemas.config import AppConfig


@dataclass
class ConfigDiff:
    hot: list[str]
    hard: list[str]


def _changed(old, new) -> bool:
    return old != new


def diff_config(old: AppConfig, new: AppConfig) -> ConfigDiff:
    hot: list[str] = []
    hard: list[str] = []

    # Detection: backend/weights are hard, threshold is hot
    if _changed(old.detection.backend, new.detection.backend):
        hard.append("detection.backend")
    if _changed(old.detection.weights_path, new.detection.weights_path):
        hard.append("detection.weights_path")
    if _changed(old.detection.confidence_threshold, new.detection.confidence_threshold):
        hot.append("detection.confidence_threshold")

    # Understanding (VLM)
    if _changed(old.understanding.backend, new.understanding.backend):
        hard.append("understanding.backend")
    if _changed(old.understanding.model_id, new.understanding.model_id):
        hard.append("understanding.model_id")
    if _changed(
        old.understanding.inference.get("backend_kwargs"),
        new.understanding.inference.get("backend_kwargs"),
    ):
        hard.append("understanding.inference.backend_kwargs")
    if _changed(
        old.understanding.inference.get("temperature"),
        new.understanding.inference.get("temperature"),
    ):
        hot.append("understanding.inference.temperature")
    if _changed(
        old.understanding.inference.get("max_tokens"),
        new.understanding.inference.get("max_tokens"),
    ):
        hot.append("understanding.inference.max_tokens")
    if _changed(old.understanding.system_prompt, new.understanding.system_prompt):
        hot.append("understanding.system_prompt")
    if _changed(old.understanding.user_prompt, new.understanding.user_prompt):
        hot.append("understanding.user_prompt")
    if _changed(old.understanding.ontology, new.understanding.ontology):
        hot.append("understanding.ontology")

    # Chat
    if _changed(old.chat.backend, new.chat.backend):
        hard.append("chat.backend")
    if _changed(old.chat.model_id, new.chat.model_id):
        hard.append("chat.model_id")
    if _changed(
        old.chat.inference.get("backend_kwargs"),
        new.chat.inference.get("backend_kwargs"),
    ):
        hard.append("chat.inference.backend_kwargs")
    if _changed(
        old.chat.inference.get("temperature"), new.chat.inference.get("temperature")
    ):
        hot.append("chat.inference.temperature")
    if _changed(
        old.chat.inference.get("max_tokens"), new.chat.inference.get("max_tokens")
    ):
        hot.append("chat.inference.max_tokens")
    if _changed(old.chat.system_prompt, new.chat.system_prompt):
        hot.append("chat.system_prompt")
    if _changed(old.chat.context_template, new.chat.context_template):
        hot.append("chat.context_template")

    # Visualization
    if _changed(old.visualization, new.visualization):
        hot.append("visualization")

    # SGG rules/mode
    if _changed(old.sgg.mode, new.sgg.mode):
        hot.append("sgg.mode")
    if _changed(old.sgg.rules, new.sgg.rules):
        hot.append("sgg.rules")

    # Memory pruning
    if _changed(
        old.tracking.memory_max_age_seconds, new.tracking.memory_max_age_seconds
    ):
        hot.append("tracking.memory_max_age_seconds")
    if _changed(old.tracking.memory_max_objects, new.tracking.memory_max_objects):
        hot.append("tracking.memory_max_objects")
    if _changed(old.tracking.memory_max_relations, new.tracking.memory_max_relations):
        hot.append("tracking.memory_max_relations")

    # Storage (hot)
    if _changed(old.storage.persist_last_state, new.storage.persist_last_state):
        hot.append("storage.persist_last_state")
    if _changed(old.storage.last_state_path, new.storage.last_state_path):
        hot.append("storage.last_state_path")
    if _changed(old.storage.store_image, new.storage.store_image):
        hot.append("storage.store_image")

    # Tracking backend changes are hard
    if _changed(old.tracking.reid_model, new.tracking.reid_model):
        hard.append("tracking.reid_model")

    # System language (hot)
    if _changed(old.system.get("language"), new.system.get("language")):
        hot.append("system.language")

    return ConfigDiff(hot=hot, hard=hard)


async def apply_hot_config(ml_state: MLState, new: AppConfig) -> None:
    ml_state.config = new

    if ml_state.pipeline:
        ml_state.pipeline.detector.threshold = new.detection.confidence_threshold
        ml_state.pipeline.vis_config = new.visualization
        ml_state.pipeline.sgg_mode = new.sgg.mode
        ml_state.pipeline.fusion_config = new.fusion

        # Update memory pruning settings
        if hasattr(ml_state.pipeline, "memory") and ml_state.pipeline.memory:
            ml_state.pipeline.memory.set_limits(
                new.tracking.memory_max_age_seconds,
                new.tracking.memory_max_objects,
                new.tracking.memory_max_relations,
            )

        # Update VLM prompts/ontology + inference params
        base_dir = new._config_path.parent if new._config_path is not None else None
        if ml_state.pipeline.sgg:
            sgg = ml_state.pipeline.sgg
            if base_dir is not None:
                sgg.system_prompt = new.understanding.system_prompt.resolve(base_dir)
                sgg.user_prompt = (
                    new.understanding.user_prompt.resolve(base_dir)
                    if new.understanding.user_prompt is not None
                    else None
                )
                predicates, objects = new.understanding.ontology.resolve(base_dir)
                sgg.predicates = predicates
                sgg.objects = objects

            sgg.config.temperature = new.understanding.inference.get(
                "temperature", sgg.config.temperature
            )
            sgg.config.max_tokens = new.understanding.inference.get(
                "max_tokens", sgg.config.max_tokens
            )

        if ml_state.pipeline.rules_sgg:
            ml_state.pipeline.rules_sgg.rules_config = new.sgg.rules

    if ml_state.chat_service:
        base_dir = new._config_path.parent if new._config_path is not None else None
        if base_dir is not None:
            ml_state.chat_service.system_prompt = new.chat.system_prompt.resolve(
                base_dir
            )
            ml_state.chat_service.context_template = (
                new.chat.context_template.resolve(base_dir)
                if new.chat.context_template is not None
                else None
            )
        ml_state.chat_service.llm.config.inference = new.chat.inference
