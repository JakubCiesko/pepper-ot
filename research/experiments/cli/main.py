import asyncio
from pathlib import Path

import click

from ..config import load_experiment_config
from ..io import start_run
from ..workflows.experiments import run_all_phases
from ..workflows.experiments import run_context_rot
from ..workflows.experiments import run_descriptions
from ..workflows.experiments import run_draft_scene_graph
from ..workflows.experiments import run_vocabulary_mining

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "experiments" / "default.yaml"
)


@click.group()
def main() -> None:
    """Research CLI for scene-graph experiments."""


def _prepare(config_path: Path):
    config, raw = load_experiment_config(config_path)
    run = start_run(config.paths.output_root, config.name, config.experiment_id, raw)
    run.logger.info("Using config=%s", config_path)
    return config, run


@main.command("run-all")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def run_all_command(config_path: Path) -> None:
    config, run = _prepare(config_path)
    asyncio.run(run_all_phases(config, run))


@main.command("describe")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def describe_command(config_path: Path) -> None:
    config, run = _prepare(config_path)
    asyncio.run(run_descriptions(config, run))


@main.command("mine-vocab")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def mine_vocab_command(config_path: Path) -> None:
    config, run = _prepare(config_path)
    asyncio.run(run_vocabulary_mining(config, run))


@main.command("draft-sgg")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def draft_sgg_command(config_path: Path) -> None:
    config, run = _prepare(config_path)
    asyncio.run(run_draft_scene_graph(config, run))


@main.command("context-rot")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def context_rot_command(config_path: Path) -> None:
    config, run = _prepare(config_path)
    asyncio.run(run_context_rot(config, run))


if __name__ == "__main__":
    main()
