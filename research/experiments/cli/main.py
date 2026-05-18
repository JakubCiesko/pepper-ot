import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
import logging
from pathlib import Path
import shutil

import click

from ..annotation_app import build_annotation_bundle
from ..annotation_app import import_annotation_export
from ..config import load_experiment_config
from ..harness.manifest import write_gqa_manifest
from ..harness.manifest import write_local_manifest
from ..harness.matrix import run_matrix
from ..harness.pipeline_batch import run_pipeline_batch
from ..harness.reports import aggregate_runs
from ..harness.templates import write_ground_truth_template
from ..io import RunContext
from ..io import resume_run
from ..io import start_run
from ..workflows.experiments import run_all_phases
from ..workflows.experiments import run_context_rot
from ..workflows.experiments import run_descriptions
from ..workflows.experiments import run_draft_scene_graph
from ..workflows.experiments import run_scene_graph_evaluation
from ..workflows.experiments import run_vocabulary_mining

# Default experiment config used by CLI commands when --config is not provided.
# Individual commands may start a fresh run or resume from run_metadata.json.
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "experiments" / "default.yaml"
)


@click.group()
def main() -> None:
    """Research CLI for scene-graph experiments.

    Commands in this group create or resume experiment runs, materialize
    manifests, evaluate completed runs, export annotation bundles, and aggregate
    reports. The commands write artifacts under the configured research output
    root unless an explicit output path is provided.
    """


def _config_from_metadata(metadata: dict):
    """Rebuild an ExperimentConfig from a saved run_metadata.json payload.

    Args:
        metadata: Parsed run metadata dictionary produced by start_run.

    Returns:
        A tuple of the validated ExperimentConfig and the raw config dictionary.

    Raises:
        click.ClickException: If the metadata does not contain a config object.
    """
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        raise click.ClickException(
            "Run metadata does not contain a valid config object."
        )
    return load_experiment_config_from_raw(raw_config), raw_config


def _prepare(config_path: Path, *, command: str):
    """Create a new run context from a config file.

    Args:
        config_path: YAML experiment config to load.
        command: CLI command name recorded in run_metadata.json.

    Returns:
        The validated ExperimentConfig and a RunContext pointing at the new run
        directory.

    Side Effects:
        Creates the run directory, writes run_metadata.json, opens run.log, and
        records config, git, and model metadata.
    """
    config, raw = load_experiment_config(config_path)
    run = start_run(
        config.paths.output_root,
        config.name,
        config.experiment_id,
        raw,
        command=command,
    )
    run.logger.info("Using config=%s", config_path)
    return config, run


def _prepare_or_resume(config_path: Path, run_dir: Path | None, *, command: str):
    """Create a new run or resume an existing run directory.

    Args:
        config_path: YAML config used only when run_dir is not provided.
        run_dir: Existing run directory containing run_metadata.json, or None
            for a fresh run.
        command: CLI command name recorded for fresh runs.

    Returns:
        The ExperimentConfig and RunContext for the selected run.

    Side Effects:
        Fresh runs write new metadata. Resumed runs reuse the config embedded in
        run_metadata.json so later phases are reproducible even if the source
        YAML changed.
    """
    if run_dir is None:
        return _prepare(config_path, command=command)
    run, metadata = resume_run(run_dir)
    config, _ = _config_from_metadata(metadata)
    run.logger.info("Using resumed config from %s", run_dir / "run_metadata.json")
    return config, run


@main.command("run-all")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
def run_all_command(config_path: Path) -> None:
    """Run all enabled experiment phases from a config.

    Args:
        config_path: Experiment YAML to validate and execute.

    Side Effects:
        Creates a fresh run and writes every artifact produced by enabled
        phases, including descriptions, vocabulary, draft scene graphs,
        context-rot outputs, and evaluation metrics.
    """
    config, run = _prepare(config_path, command="run-all")
    asyncio.run(run_all_phases(config, run))


@main.command("describe")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
@click.option("--run", "run_dir", type=click.Path(path_type=Path), default=None)
def describe_command(config_path: Path, run_dir: Path | None) -> None:
    """Generate image descriptions for a new or resumed run.

    Args:
        config_path: Experiment YAML used for fresh runs.
        run_dir: Optional existing run directory to resume.

    Side Effects:
        Writes detections.json when detection is enabled, descriptions.json,
        and metrics_descriptions.json under the run directory.
    """
    config, run = _prepare_or_resume(config_path, run_dir, command="describe")
    asyncio.run(run_descriptions(config, run))


@main.command("mine-vocab")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
@click.option("--run", "run_dir", type=click.Path(path_type=Path), default=None)
def mine_vocab_command(config_path: Path, run_dir: Path | None) -> None:
    """Mine and consolidate scene graph vocabulary from descriptions.

    Args:
        config_path: Experiment YAML used for fresh runs.
        run_dir: Optional existing run directory to resume.

    Raises:
        RuntimeError: If descriptions are missing or frozen vocabulary config is
            invalid.

    Side Effects:
        Writes vocabulary_candidates.json, vocabulary_final.json, and
        metrics_vocabulary.json.
    """
    config, run = _prepare_or_resume(config_path, run_dir, command="mine-vocab")
    asyncio.run(run_vocabulary_mining(config, run))


@main.command("draft-sgg")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
@click.option("--run", "run_dir", type=click.Path(path_type=Path), default=None)
def draft_sgg_command(config_path: Path, run_dir: Path | None) -> None:
    """Generate draft scene graphs for run images.

    Args:
        config_path: Experiment YAML used for fresh runs.
        run_dir: Optional existing run directory to resume.

    Side Effects:
        Reads descriptions, detections, and final vocabulary; writes
        draft_scene_graph.json, optional SoM images, and
        metrics_draft_scene_graph.json.
    """
    config, run = _prepare_or_resume(config_path, run_dir, command="draft-sgg")
    asyncio.run(run_draft_scene_graph(config, run))


@main.command("context-rot")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
@click.option("--run", "run_dir", type=click.Path(path_type=Path), default=None)
def context_rot_command(config_path: Path, run_dir: Path | None) -> None:
    """Run context-rot sensitivity evaluation over vocabulary levels.

    Args:
        config_path: Experiment YAML used for fresh runs.
        run_dir: Optional existing run directory to resume.

    Side Effects:
        Writes context_rot_vocab_slices.json, per-level context_rot_levels
        artifacts, context_rot.json, and metrics_context_rot.json.
    """
    config, run = _prepare_or_resume(config_path, run_dir, command="context-rot")
    asyncio.run(run_context_rot(config, run))


@main.command("evaluate-sgg")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG
)
@click.option("--run", "run_dir", type=click.Path(path_type=Path), default=None)
def evaluate_sgg_command(config_path: Path, run_dir: Path | None) -> None:
    """Evaluate generated scene graphs for a new or resumed run.

    Args:
        config_path: Experiment YAML used for fresh runs.
        run_dir: Optional existing run directory to resume.

    Side Effects:
        Reads ground truth and predictions; writes per-image metrics, summary
        metrics, potency metrics, sensitivity metrics, and stage metrics.
    """
    config, run = _prepare_or_resume(config_path, run_dir, command="evaluate-sgg")
    asyncio.run(run_scene_graph_evaluation(config, run))


@main.command("make-manifest")
@click.option(
    "--source",
    type=click.Choice(["local", "gqa"]),
    required=True,
    help="Dataset source to materialize into manifest.jsonl.",
)
@click.option(
    "--images-dir", type=click.Path(path_type=Path), default=Path("data/subset")
)
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
    """Create a manifest.jsonl from local images or GQA samples.

    Args:
        source: Either local files or the GQA scene graph dataset.
        images_dir: Source image directory or destination cache for GQA images.
        out: Manifest JSONL path to write.
        max_samples: Maximum images to include.
        seed: Sampling seed for deterministic local subsets.
        dataset_name: Dataset label used for local manifest rows.

    Side Effects:
        Writes a JSONL manifest and, for GQA, may materialize sampled images
        under images_dir.
    """
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
    """Run every variant defined in an experiment matrix file.

    Args:
        matrix_path: YAML matrix with a base config, variants, overrides, and
            optional artifact reuse.

    Side Effects:
        Creates one run per variant and writes matrix_result.json into each run
        directory.
    """
    results = asyncio.run(run_matrix(matrix_path))
    click.echo(f"Completed {len(results)} variants")


@main.command("make-gt-template")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option(
    "--prefill-draft/--blank",
    default=False,
    help="Prefill relationships from draft SGG or create blank annotation rows.",
)
def make_gt_template_command(
    run_dir: Path, out: Path | None, prefill_draft: bool
) -> None:
    """Write a human-editable ground-truth scene graph template.

    Args:
        run_dir: Existing run directory containing detections and optionally
            draft scene graphs.
        out: Optional output file; defaults to ground_truth_scene_graph_template.json.
        prefill_draft: When true, copy draft relationships into the template.

    Side Effects:
        Writes a JSON file that can be edited by a human and passed to
        evaluate-run.
    """
    path = write_ground_truth_template(run_dir, out, prefill_draft=prefill_draft)
    click.echo(f"Wrote ground-truth template to {path}")


@main.command("evaluate-run")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--gt", "gt_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--gt-only",
    is_flag=True,
    help="Evaluate only images present in the ground-truth file.",
)
def evaluate_run_command(run_dir: Path, gt_path: Path, gt_only: bool) -> None:
    """Evaluate an existing run against a ground-truth scene graph file.

    Args:
        run_dir: Existing run directory with run_metadata.json and
            draft_scene_graph.json.
        gt_path: Ground-truth scene graph JSON to copy into the run.
        gt_only: Evaluate only keys present in the ground-truth file.

    Side Effects:
        Copies gt_path to the configured ground-truth artifact name and rewrites
        evaluation metric artifacts for the run.
    """
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise click.ClickException(f"Missing run metadata: {metadata_path}")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config, _ = _config_from_metadata(metadata)
    if gt_only:
        config.evaluation.keyspace = "gt_only"
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
    """Aggregate run metrics into report tables and plots.

    Args:
        runs_root: Directory containing completed run subdirectories.
        out_dir: Directory where report artifacts are written.

    Side Effects:
        Writes metrics_summary.csv, metrics_summary.json, report.md, and
        available plots.
    """
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
    """Run the server perception pipeline over a manifest batch.

    Args:
        server_config: Server AppConfig YAML used to build the pipeline.
        manifest: Manifest JSONL with image_id and image_path rows.
        out_dir: Directory for batch metrics and per-image output.
        preset: Pipeline controls preset to apply before running.
        limit: Optional maximum number of manifest rows to process.

    Side Effects:
        Writes pipeline_batch_per_image.json and pipeline_batch_summary.json.
    """
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
    """Export a static annotation bundle for a completed run.

    Args:
        run_dir: Run directory containing images, detections, descriptions,
            vocabulary, and draft scene graphs.
        out_dir: Directory where the static annotation app is written.

    Side Effects:
        Copies raw and SoM image assets and writes bundle.json, app.js,
        style.css, and index.html.
    """
    path = build_annotation_bundle(run_dir, out_dir)
    click.echo(f"Exported annotation bundle to {path}")


@main.command("import-annotation-export")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--annotations", type=click.Path(path_type=Path), required=True)
def import_annotation_export_command(run_dir: Path, annotations: Path) -> None:
    """Import annotation UI exports as run ground truth.

    Args:
        run_dir: Run directory where ground_truth_scene_graph.json should be
            written.
        annotations: JSON export from the static annotation UI.

    Side Effects:
        Normalizes relationships and writes ground_truth_scene_graph.json in the
        run directory.
    """
    out = import_annotation_export(run_dir, annotations)
    click.echo(f"Wrote imported annotations to {out}")


@main.command("serve-annotation-bundle")
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
@click.option("--port", type=int, default=8000)
def serve_annotation_bundle_command(bundle_dir: Path, port: int) -> None:
    """Serve an exported annotation bundle over local HTTP.

    Args:
        bundle_dir: Directory produced by export-annotation-bundle.
        port: Local TCP port to bind; defaults to 8000.

    Side Effects:
        Starts a blocking ThreadingHTTPServer until interrupted.
    """
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.exists():
        raise click.ClickException(f"Bundle directory does not exist: {bundle_dir}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(bundle_dir))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    click.echo(f"Serving {bundle_dir} at http://127.0.0.1:{port}/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping server")
    finally:
        server.server_close()


def load_experiment_config_from_raw(raw_config: dict):
    """Validate a raw config dictionary as an ExperimentConfig.

    Args:
        raw_config: Parsed config dictionary from metadata or YAML.

    Returns:
        A validated ExperimentConfig instance.
    """
    from ..config.models import ExperimentConfig

    return ExperimentConfig(**raw_config)


if __name__ == "__main__":
    main()
