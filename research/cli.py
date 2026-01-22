from pathlib import Path

import click

from research.src.data_generation.distillation.distill import run_distillation


@click.group()
def cli():
    """
    Pepper Research CLI.

    Tools for Knowledge Distillation, Scene Graph Generation, and Model Training (VLM and OT).
    """
    pass


@cli.group()
def data():
    """Tools for generating synthetic datasets."""
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
    """Run Grounding DINO to auto-label raw images."""
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
    pass


@cli.group()
def train():
    """Tools for Fine-Tuning models."""
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
    pass


@train.command(name="vlm")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="configs/vlm_train.yaml",
    help="Config for Qwen2.5-VL Unsloth training.",
    show_default=True,
)
def cmd_train_vlm(config):
    pass
