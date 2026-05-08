import asyncio
import logging
import shutil
from functools import partial
from http.server import ThreadingHTTPServer
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import click

from ..config import load_experiment_config
from ..annotation_app import build_annotation_bundle
from ..annotation_app import import_annotation_export
from ..harness.manifest import write_gqa_manifest
from ..harness.manifest import write_local_manifest
from ..harness.matrix import run_matrix
from ..harness.pipeline_batch import run_pipeline_batch
from ..harness.reports import aggregate_runs
from ..harness.templates import write_ground_truth_template
from ..io import RunContext
from ..io import start_run
from ..workflows.experiments import run_all_phases
from ..workflows.experiments import run_context_rot
from ..workflows.experiments import run_descriptions
from ..workflows.experiments import run_draft_scene_graph
from ..workflows.experiments import run_scene_graph_evaluation
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


@main.command("evaluate-sgg")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def evaluate_sgg_command(config_path: Path) -> None:
    config, run = _prepare(config_path)
    asyncio.run(run_scene_graph_evaluation(config, run))


@main.command("make-manifest")
@click.option(
    "--source",
    type=click.Choice(["local", "gqa"]),
    required=True,
    help="Dataset source to materialize into manifest.jsonl.",
)
@click.option("--images-dir", type=click.Path(path_type=Path), default=Path("data/subset"))
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--max-samples", type=int, default=10)
@click.option("--seed", type=int, default=42)
@click.option("--dataset-name", default="local")
def make_manifest_command(
    source: str,
    images_dir: Path,
    out: Path,
    max_samples: int,
    seed: int,
    dataset_name: str,
) -> None:
    if source == "local":
        rows = write_local_manifest(
            images_dir=images_dir,
            out=out,
            dataset=dataset_name,
            max_samples=max_samples,
            seed=seed,
        )
    else:
        rows = write_gqa_manifest(
            out=out,
            image_root=images_dir,
            max_samples=max_samples,
            seed=seed,
        )
    click.echo(f"Wrote {len(rows)} rows to {out}")


@main.command("run-matrix")
@click.option("--config", "matrix_path", type=click.Path(path_type=Path), required=True)
def run_matrix_command(matrix_path: Path) -> None:
    results = asyncio.run(run_matrix(matrix_path))
    click.echo(f"Completed {len(results)} variants")


@main.command("make-gt-template")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def make_gt_template_command(run_dir: Path, out: Path | None) -> None:
    path = write_ground_truth_template(run_dir, out)
    click.echo(f"Wrote ground-truth template to {path}")


@main.command("evaluate-run")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--gt", "gt_path", type=click.Path(path_type=Path), required=True)
def evaluate_run_command(run_dir: Path, gt_path: Path) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise click.ClickException(f"Missing run metadata: {metadata_path}")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_config = metadata["config"]
    config = load_experiment_config_from_raw(raw_config)
    shutil.copyfile(gt_path, run_dir / config.paths.ground_truth_scene_graph_file)
    logger = logging.getLogger(f"research.run.evaluate.{metadata['run_id']}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    run = RunContext(
        run_id=metadata["run_id"],
        run_dir=run_dir,
        log_path=run_dir / "run.log",
        logger=logger,
    )
    asyncio.run(run_scene_graph_evaluation(config, run))
    click.echo(f"Evaluated {run_dir}")


@main.command("plot-runs")
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=Path("research/artifacts/runs"),
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=Path("research/artifacts/reports/latest"),
)
def plot_runs_command(runs_root: Path, out_dir: Path) -> None:
    rows = aggregate_runs(runs_root, out_dir)
    click.echo(f"Aggregated {len(rows)} runs into {out_dir}")


@main.command("pipeline-batch")
@click.option(
    "--server-config",
    type=click.Path(path_type=Path),
    default=Path("server/config.yaml"),
)
@click.option("--manifest", type=click.Path(path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
@click.option("--preset", default="full")
@click.option("--limit", type=int, default=None)
def pipeline_batch_command(
    server_config: Path, manifest: Path, out_dir: Path, preset: str, limit: int | None
) -> None:
    result = asyncio.run(
        run_pipeline_batch(
            server_config=server_config,
            manifest=manifest,
            out_dir=out_dir,
            preset=preset,
            limit=limit,
        )
    )
    click.echo(f"Pipeline batch wrote {result['summary']['images']} rows to {out_dir}")


@main.command("export-annotation-bundle")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
def export_annotation_bundle_command(run_dir: Path, out_dir: Path) -> None:
    path = build_annotation_bundle(run_dir, out_dir)
    click.echo(f"Exported annotation bundle to {path}")


@main.command("import-annotation-export")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--annotations", type=click.Path(path_type=Path), required=True)
def import_annotation_export_command(run_dir: Path, annotations: Path) -> None:
    out = import_annotation_export(run_dir, annotations)
    click.echo(f"Wrote imported annotations to {out}")


@main.command("serve-annotation-bundle")
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
@click.option("--port", type=int, default=8000)
def serve_annotation_bundle_command(bundle_dir: Path, port: int) -> None:
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.exists():
        raise click.ClickException(f"Bundle directory does not exist: {bundle_dir}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(bundle_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    click.echo(f"Serving {bundle_dir} at http://127.0.0.1:{port}/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping server")
    finally:
        server.server_close()


def load_experiment_config_from_raw(raw_config: dict):
    from ..config.models import ExperimentConfig

    return ExperimentConfig(**raw_config)


if __name__ == "__main__":
    main()
