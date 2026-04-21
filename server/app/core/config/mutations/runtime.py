from pathlib import Path
from typing import Any

from app.schemas.config import AppConfig


def resolve_base_dir(cfg: AppConfig) -> Path | None:
    return cfg._config_path.parent if cfg._config_path is not None else None


def resolve_prompt_text(
    source, base_dir: Path | None, fallback: str | None = None
) -> str | None:
    if source is None:
        return fallback
    if base_dir is not None:
        return source.resolve(base_dir)
    if getattr(source, "text", None) is not None:
        return str(source.text).strip()
    return fallback


def apply_pipeline_runtime_updates(
    pipeline: Any, cfg: AppConfig, base_dir: Path | None
):
    pipeline.detector.threshold = cfg.detection.confidence_threshold
    pipeline.detector.run_nms_post_filter = cfg.detection.run_nms_post_filter
    pipeline.detector.nms_iou_threshold = cfg.detection.nms_iou_threshold
    pipeline.detector.nms_type = cfg.detection.nms_type
    pipeline.detector.device = cfg.detection.device
    pipeline.detector.ontology = (
        cfg.detection.resolve_ontology(base_dir)
        if base_dir is not None
        else cfg.detection.ontology
    )
    pipeline.pipeline_controls = cfg.pipeline_controls
    pipeline.fusion_config = cfg.fusion
    pipeline.vis_config = cfg.visualization
    if getattr(pipeline, "qa_service", None) is not None:
        pipeline.qa_service.update_runtime(
            cfg.chat,
            pairs_per_update=cfg.qa_generation.pairs_per_update,
        )
    if hasattr(pipeline, "memory") and pipeline.memory:
        pipeline.memory.set_limits(
            cfg.tracking.memory_max_age_seconds,
            cfg.tracking.memory_max_objects,
            cfg.tracking.memory_max_relations,
            max_captions=cfg.tracking.memory_max_captions,
            caption_max_age_seconds=cfg.tracking.caption_max_age_seconds,
        )
        pipeline.memory.set_max_dormant_frames(cfg.tracking.max_dormant_frames)
        pipeline.memory.set_association_config(cfg.tracking.association)
        pipeline.memory.set_feature_extraction_config(cfg.tracking.feature_extraction)


def apply_scene_graph_runtime_updates(
    scene_graph_service: Any, cfg: AppConfig, base_dir: Path | None
):
    scene_graph_service.parallel_execution = cfg.scene_graph.parallel_execution
    vlm_backend = scene_graph_service.vlm_backend
    system_prompt = resolve_prompt_text(
        cfg.scene_graph.vlm.system_prompt,
        base_dir,
        fallback=vlm_backend.system_prompt,
    )
    user_prompt = resolve_prompt_text(
        cfg.scene_graph.vlm.user_prompt,
        base_dir,
        fallback=None,
    )
    predicates, objects = (
        cfg.scene_graph.vlm.ontology.resolve(base_dir)
        if base_dir is not None
        else (vlm_backend.predicates, vlm_backend.objects)
    )
    vlm_backend.update_runtime(
        config=cfg.scene_graph.vlm,
        predicates=predicates,
        objects=objects,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        rebuild_client=False,
    )
    scene_graph_service.rule_backend.rules_config = cfg.scene_graph.rules
    scene_graph_service.reltr_backend.update_runtime(cfg.scene_graph.reltr)
