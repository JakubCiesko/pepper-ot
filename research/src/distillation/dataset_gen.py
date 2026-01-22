from pathlib import Path
import sys

import click
from src.distillation.teacher import KnowledgeDistiller


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="../../configs/distillation_config.yaml",
    help="Path to the distillation configuration YAML file.",
    show_default=True,
)
def main(config: Path):
    """
    Run Knowledge Distillation Dataset Generation (Grounding DINO -> YOLO).

    This script loads a teacher configuration, runs the Grounding DINO model
    on raw images, and saves the detections in YOLO format for student training.
    """
    config_path = config.resolve()
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
        click.secho("\n❌ CRITICAL FAILURE during distillation:", fg="red", bold=True)
        click.echo(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
