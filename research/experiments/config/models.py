"""Pydantic contracts for research experiment configs and matrix variants."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class PathsConfig(BaseModel):
    """
    Artifact naming contract for a run. These fields decide where each phase reads
    previous outputs from and where it writes its own JSON artifacts.
    """

    images_dir: Path
    manifest_file: Path | None = None
    output_root: Path = Path("research/artifacts")
    detections_file: str = "detections.json"
    descriptions_file: str = "descriptions.json"
    vocabulary_candidates_file: str = "vocabulary_candidates.json"
    vocabulary_final_file: str = "vocabulary_final.json"
    draft_scene_graph_file: str = "draft_scene_graph.json"
    context_rot_file: str = "context_rot.json"
    ground_truth_scene_graph_file: str = "ground_truth_scene_graph.json"
    scene_graph_metrics_per_image_file: str = "metrics_scene_graph_per_image.json"
    scene_graph_metrics_summary_file: str = "metrics_scene_graph_summary.json"
    image_potency_metrics_file: str = "metrics_image_potency.json"
    sensitivity_metrics_file: str = "metrics_sensitivity_curves.json"


class LLMModelConfig(BaseModel):
    """Model adapter settings used by caption, vocabulary, and draft SGG phases.

    The provider selects the adapter implementation, model_id selects the model
    within that provider, structured_mode controls how typed responses are
    obtained, and base_url/api_key support OpenAI-compatible or hosted backends.
    """

    provider: Literal["openai", "gemini", "openai_compatible", "local_hf", "local_4bit"]
    model_id: str
    structured_mode: Literal["provider_native", "parse_output", "instructor"] = (
        "provider_native"
    )
    base_url: str | None = None
    api_key: str | None = None


class DetectionStageConfig(BaseModel):
    """Object detection settings for the description phase.

    When enabled, detections are generated before captioning and saved as the
    run detection artifact. When disabled, the phase expects detections to
    already exist at paths.detections_file if prompts need object labels.
    """

    enabled: bool = True
    backend: Literal["yolo", "rt_detr", "rf_detr", "owl_v2"] = "rf_detr"
    batch_size: int = 4
    max_image_size: int | None = None
    confidence: float = 0.5


class DescriptionStageConfig(BaseModel):
    """Caption generation settings for image descriptions.

    The phase reads images from paths.images_dir or paths.manifest_file, builds
    prompts from the configured templates, and writes descriptions plus stage
    metrics into the run directory.
    """

    enabled: bool = True
    # unused
    batch_size: int = 2
    max_concurrent_batches: int = 2
    system_prompt: str = (
        "Describe the image in detail, mention visible objects, attributes, and relations."
    )
    user_prompt_template: str = (
        "Detected objects: {objects}\nDescribe the scene thoroughly."
    )
    max_image_size: int | None = None


class VocabularyStageConfig(BaseModel):
    """Vocabulary mining settings for predicates and attributes.

    current_run mode extracts candidates from generated descriptions and
    consolidates them through the configured vocabulary model. frozen_file mode
    copies an existing vocabulary artifact and skips candidate extraction.
    """

    enabled: bool = True
    # unused
    batch_size: int = 2
    max_concurrent_batches: int = 2
    predicates_target: int = 50
    attributes_target: int = 25
    source_mode: Literal["current_run", "frozen_file"] = "current_run"
    frozen_file: Path | None = None
    extract_system_prompt: str = (
        "Extract practical predicates and attributes from the caption for scene graph use."
    )
    consolidate_predicates_prompt: str = (
        "Consolidate these predicates into general practical predicates using underscore naming."
    )
    consolidate_attributes_prompt: str = (
        "Consolidate these attributes into general practical attributes using underscore naming."
    )


class DraftSceneGraphStageConfig(BaseModel):
    """Draft scene graph generation and prompt-image settings.

    The phase consumes descriptions, detections, and final vocabulary; formats
    prompts according to vocab_mode; optionally paints SoM images; and writes
    parsed relationships plus prompt/debug metadata for each image.
    """

    enabled: bool = True
    # unused
    batch_size: int = 8
    max_concurrent_batches: int = 2
    save_som_images: bool = True
    use_som_image: bool = True
    visual_mode: Literal["raw", "som"] = "som"
    vocab_mode: Literal["open", "closed", "soft", "list"] = "list"
    som_output_dir: str = "som_images_draft"
    include_raw_response: bool = True
    som_show_bbox: bool = True
    som_show_mask: bool = False
    som_show_polygon: bool = False
    som_show_labels: bool = True
    som_line_thickness: int = 2
    som_mask_opacity: float = 0.5
    som_color_lookup: Literal["index", "class", "track"] = "index"
    som_mask_backend: Literal["grabcut", "sam"] = "grabcut"
    som_device: str = "cuda"
    max_image_size: int | None = None
    system_prompt: str = (
        "Generate scene graph JSON relations using only object IDs and provided vocabulary."
    )
    user_prompt_template: str = (
        "Objects: {objects}\nAllowed predicates/attributes: {vocabulary}\nCaption: {caption}"
    )
    som_grab_cut_scale: float = 0.35
    som_grab_cut_iter_count: int = 8
    som_use_roi_grab_cut: bool = True
    som_max_mask_workers: int | None = 4


class ContextRotStageConfig(BaseModel):
    """Vocabulary-reduction sensitivity settings.

    Context-rot creates automatic or manual vocabulary levels, reruns draft SGG
    under each reduced vocabulary, remaps predictions into that level, and can
    evaluate each level against ground truth.
    """

    enabled: bool = True
    min_vocab_size: int = 10
    step: int = 5
    strategy: Literal["semantic", "frequency", "random", "llm_drop", "random_drop"] = (
        "semantic"
    )
    rounds_per_size: int = 1
    evaluate_against_ground_truth: bool = True
    levels_file: Path | None = None


class EvaluationStageConfig(BaseModel):
    """Scene graph evaluation settings.

    These flags control key alignment, missing GT/prediction behavior,
    normalization, optional graph edit and per-predicate metrics, potency
    diagnostics, and bootstrap confidence intervals.
    """

    enabled: bool = False
    keyspace: Literal["all", "gt_only"] = "all"
    strict_id_match: bool = True
    normalize_ids: bool = True
    normalize_relations: bool = True
    compute_ged: bool = False
    compute_per_predicate: bool = True
    compute_pair_metrics: bool = True
    compute_attribute_f1: bool = True
    compute_potency: bool = True
    missing_policy: Literal["skip", "empty"] = "empty"
    complexity_bins: list[int] = Field(default_factory=lambda: [3, 5, 8, 12, 20])
    bootstrap_rounds: int = 1000


class PromptingConfig(BaseModel):
    """Cross-phase prompt switches shared by description and SGG prompts."""

    include_detection_labels_in_descriptions: bool = True
    include_caption_in_sgg_prompt: bool = True


class ExperimentConfig(BaseModel):
    """
    Top-level experiment contract. Every CLI phase receives this normalized config
    and uses its paths, model settings, stage toggles, and prompt controls.
    """

    name: str = "scene_graph_research"
    seed: int = 42
    paths: PathsConfig
    description_model: LLMModelConfig
    vocabulary_model: LLMModelConfig
    draft_sgg_model: LLMModelConfig
    experiment_id: str | None = None
    detection: DetectionStageConfig = Field(default_factory=DetectionStageConfig)
    descriptions: DescriptionStageConfig = Field(default_factory=DescriptionStageConfig)
    vocabulary: VocabularyStageConfig = Field(default_factory=VocabularyStageConfig)
    draft_scene_graph: DraftSceneGraphStageConfig = Field(
        default_factory=DraftSceneGraphStageConfig
    )
    context_rot: ContextRotStageConfig = Field(default_factory=ContextRotStageConfig)
    evaluation: EvaluationStageConfig = Field(default_factory=EvaluationStageConfig)
    prompting: PromptingConfig = Field(default_factory=PromptingConfig)
