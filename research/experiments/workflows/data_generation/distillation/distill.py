from pathlib import Path
import sys

import click

from .teacher import KnowledgeDistiller


def run_distillation(config_path: Path):
    """
    The main entrypoint that the CLI will call.

    Run Knowledge Distillation Dataset Generation (Grounding DINO -> YOLO).

    This script loads a teacher configuration, runs the Grounding DINO model
    on raw images, and saves the detections in YOLO format for student training.

    """
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    click.secho("Starting Distillation Data Generation Pipeline", fg="green", bold=True)
    click.echo(f"\tConfig path: {config_path}")
    try:
        click.echo("Loading Knowledge Distiller")
        distiller = KnowledgeDistiller(config_path)
        click.echo(
            f"Running Knowledge Distillation with config: {distiller.config.model_dump()}"
        )
        distiller.distill()
    except Exception as e:
        click.secho("\nCRITICAL FAILURE during distillation:", fg="red", bold=True)
        click.echo(e)
        sys.exit(1)
