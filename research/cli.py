from pathlib import Path

import click

from research.experiments.workflows.data_generation.distillation.distill import (
    run_distillation,
)
from research.experiments.workflows.data_generation.scene_graph.generate import (
    run_generation,
)
from research.experiments.workflows.training.detector import run_detector_training
from research.experiments.workflows.training.vlm import run_vlm_training


@click.group()
def cli():
    """
    Pepper Research CLI.

    Tools for knowledge distillation, synthetic scene graph data generation,
    and model training. These commands are legacy entrypoints around the
    research workflows and expect explicit YAML config files for each task.
    """
    pass


@cli.group()
def data():
    """Tools for generating synthetic datasets.

    Commands in this group create training or evaluation data artifacts from raw
    images, teacher models, and scene graph generation configs.
    """
    pass


@data.command(name="distill")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="configs/distillation_config.yaml",
    help="Path to the GroundingDINO -> YOLO config.",
    show_default=True,
)
def cmd_distill(config):
    """Run Grounding DINO to auto-label raw images.

    Args:
        config: Distillation YAML describing input images, ontology labels,
            teacher model settings, and output dataset paths.

    Side Effects:
        Runs the distillation workflow and writes YOLO-style labels and dataset
        metadata according to the config.
    """
    run_distillation(config)


@data.command(name="scene-graph")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="configs/scene_graph_config.yaml",
    help="Path to the SoM -> GPT-4o config.",
    show_default=True,
)
def cmd_scene_graph(config):
    """Generate synthetic scene graph data from configured images.

    Args:
        config: Scene graph generation YAML describing image sources, SoM
            rendering, teacher model settings, prompts, and output paths.

    Side Effects:
        Runs the configured scene graph generation workflow and writes generated
        annotations according to the config.
    """
    run_generation(config)


@cli.group()
def train():
    """Tools for fine-tuning research models.

    Commands in this group start detector or VLM training jobs from local YAML
    configs. They may require optional training dependencies and GPU support.
    """
    pass


@train.command(name="detector")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="configs/detector_train.yaml",
    help="Config for RT-DETR training.",
    show_default=True,
)
def cmd_train_detector(config):
    """Train or fine-tune an RT-DETR detector.

    Args:
        config: Detector training YAML with model, dataset, and training output
            settings.

    Side Effects:
        Starts the detector training workflow and writes checkpoints/logs as
        configured by the training script.
    """
    run_detector_training(config)


@train.command(name="vlm")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="configs/vlm_train.yaml",
    help="Config for Qwen2.5-VL Unsloth training.",
    show_default=True,
)
def cmd_train_vlm(config):
    """Train or fine-tune a VLM scene graph model.

    Args:
        config: VLM training YAML with dataset paths, system prompt, LoRA, and
            trainer hyperparameters.

    Side Effects:
        Starts the VLM training workflow and writes model outputs according to
        the config.
    """
    run_vlm_training(config)


if __name__ == "__main__":
    cli()
