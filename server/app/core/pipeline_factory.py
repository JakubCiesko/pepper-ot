from pathlib import Path

from app.inference.caption.service import CaptionInferenceService
from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.service import DetectionService
from app.inference.memory.scene_memory import SceneMemory
from app.inference.pipeline import PerceptionPipeline
from app.inference.scene_graph.reltr_backend import RelTRSceneGraphGenerator
from app.inference.scene_graph.rules_backend import RuleSceneGraphGenerator
from app.inference.scene_graph.service import SceneGraphService
from app.inference.scene_graph.som import SoMPainter
from app.inference.scene_graph.vlm_backend import VLMSceneGraphGenerator
from app.schemas.config import AppConfig


def build_perception_pipeline(config: AppConfig) -> PerceptionPipeline:
    base_dir = (
        config._config_path.parent if config._config_path is not None else Path.cwd()
    )

    detection_backend = DetectionModelType(config.detection.backend)
    model_path = (
        Path(config.detection.weights_path) if config.detection.weights_path else None
    )
    detector = DetectionService(
        model_name=detection_backend,
        model_path=model_path,
        device=config.detection.device,
        threshold=config.detection.confidence_threshold,
        ontology=config.detection.resolve_ontology(base_dir),
    )

    memory = SceneMemory(
        memory_max_age_seconds=config.tracking.memory_max_age_seconds,
        memory_max_objects=config.tracking.memory_max_objects,
        memory_max_relations=config.tracking.memory_max_relations,
        memory_max_captions=config.tracking.memory_max_captions,
        caption_max_age_seconds=config.tracking.caption_max_age_seconds,
        max_dormant_frames=config.tracking.max_dormant_frames,
        association_config=config.tracking.association,
        feature_extraction_config=config.tracking.feature_extraction,
    )
    painter = SoMPainter(
        mask_backend=config.visualization.mask_backend,
        line_thickness=config.visualization.line_thickness,
        color_lookup=config.visualization.color_lookup,
        mask_opacity=config.visualization.mask_opacity,
        device=config.visualization.device,
    )
    vlm_system_prompt = config.scene_graph.vlm.system_prompt.resolve(base_dir)
    vlm_user_prompt = (
        config.scene_graph.vlm.user_prompt.resolve(base_dir)
        if config.scene_graph.vlm.user_prompt is not None
        else None
    )
    predicates, objects = config.scene_graph.vlm.ontology.resolve(base_dir)
    vlm_backend = VLMSceneGraphGenerator(
        config.scene_graph.vlm,
        predicates=predicates,
        objects=objects,
        system_prompt=vlm_system_prompt,
        user_prompt=vlm_user_prompt,
    )
    caption_system_prompt = config.caption.system_prompt.resolve(base_dir)
    caption_user_prompt = (
        config.caption.user_prompt.resolve(base_dir)
        if config.caption.user_prompt is not None
        else None
    )
    caption_service = CaptionInferenceService(
        config.caption,
        system_prompt=caption_system_prompt,
        user_prompt=caption_user_prompt,
    )
    rule_backend = RuleSceneGraphGenerator(config.scene_graph.rules)
    reltr_backend = RelTRSceneGraphGenerator(config.scene_graph.reltr)
    scene_graph_service = SceneGraphService(
        vlm_backend=vlm_backend,
        rule_backend=rule_backend,
        reltr_backend=reltr_backend,
    )
    return PerceptionPipeline(
        detector=detector,
        memory=memory,
        painter=painter,
        scene_graph_service=scene_graph_service,
        caption_service=caption_service,
        fusion_config=config.fusion,
        vis_config=config.visualization,
        pipeline_controls=config.pipeline_controls,
    )
