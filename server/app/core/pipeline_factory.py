from pathlib import Path

from app.inference.detection.detectors import DetectionModelType
from app.inference.detection.service import DetectionService
from app.inference.memory.scene_memory import SceneMemory
from app.inference.pipeline import VisualPipeline
from app.inference.scene_graph.rules_backend import RuleBasedSceneGraphBackend
from app.inference.scene_graph.service import SceneGraphService
from app.inference.scene_graph.som import SoMPainter
from app.inference.scene_graph.vlm_backend import VLMSceneGraphBackend
from app.schemas.config import AppConfig


def build_visual_pipeline(config: AppConfig) -> VisualPipeline:
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
        max_dormant_frames=config.tracking.max_dormant_frames,
        association_config=config.tracking.association,
        feature_extraction_config=config.tracking.feature_extraction,
    )
    painter = SoMPainter(
        line_thickness=config.visualization.line_thickness,
        color_lookup=config.visualization.color_lookup,
        mask_opacity=config.visualization.mask_opacity,
    )
    vlm_system_prompt = config.scene_graph.vlm.system_prompt.resolve(base_dir)
    vlm_user_prompt = (
        config.scene_graph.vlm.user_prompt.resolve(base_dir)
        if config.scene_graph.vlm.user_prompt is not None
        else None
    )
    predicates, objects = config.scene_graph.vlm.ontology.resolve(base_dir)
    vlm_backend = VLMSceneGraphBackend(
        config.scene_graph.vlm,
        predicates=predicates,
        objects=objects,
        system_prompt=vlm_system_prompt,
        user_prompt=vlm_user_prompt,
    )
    rule_backend = RuleBasedSceneGraphBackend(config.scene_graph.rules)
    scene_graph_service = SceneGraphService(
        mode=config.scene_graph.mode,
        vlm_backend=vlm_backend,
        rule_backend=rule_backend,
    )
    return VisualPipeline(
        detector=detector,
        memory=memory,
        painter=painter,
        scene_graph_service=scene_graph_service,
        fusion_config=config.fusion,
        vis_config=config.visualization,
        pipeline_controls=config.pipeline_controls,
    )
