# Research Experiments

The research experiment package provides repeatable scene graph experiments around the same detection, captioning, VLM, and evaluation concepts used by the server. It is designed for offline batches, prompt and vocabulary studies, human annotation, and metric aggregation.

## CLI Entrypoint

Run the CLI as a module from the repository root:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main --help
```

The default config is `research/configs/experiments/default.yaml`. Most commands accept `--config`; phase commands also accept `--run` to resume from an existing run directory and reuse the config stored in `run_metadata.json`.

Core phase commands:

| Command | Purpose |
|---|---|
| `run-all` | Runs all enabled phases in config order. |
| `describe` | Produces image descriptions from configured image inputs. |
| `mine-vocab` | Extracts and consolidates predicates and attributes. |
| `draft-sgg` | Generates draft scene graphs, optionally using SoM images. |
| `context-rot` | Evaluates sensitivity to reduced vocabulary context. |
| `evaluate-sgg` | Computes scene graph metrics when evaluation is enabled. |

Support commands cover dataset manifests, matrices, batch server-pipeline runs, reports, and human annotation bundles:

| Command | Purpose |
|---|---|
| `make-manifest` | Writes `manifest.jsonl` from local image folders or GQA samples. |
| `run-matrix` | Runs every variant from a matrix YAML file. |
| `pipeline-batch` | Runs server-style perception over a manifest. |
| `make-gt-template` | Creates a JSON ground-truth template for manual editing. |
| `evaluate-run` | Evaluates an existing run against a supplied ground-truth file. |
| `plot-runs` | Aggregates run metrics into report outputs. |
| `export-annotation-bundle` | Builds a static browser annotation bundle. |
| `import-annotation-export` | Imports annotation exports as run ground truth. |
| `serve-annotation-bundle` | Serves a bundle locally for annotation. |

## Config And Matrix Layout

Experiment configs are YAML files validated by `research/experiments/config/models.py`. The top-level contract includes:

- `name`, `experiment_id`, and `seed` for run identity and reproducibility.
- `paths` for inputs, output root, and artifact file names.
- `description_model`, `vocabulary_model`, and `draft_sgg_model` for provider and model selection.
- Stage blocks for `detection`, `descriptions`, `vocabulary`, `draft_scene_graph`, `context_rot`, and `evaluation`.
- `prompting` switches that control whether detection labels and captions are included in downstream prompts.

Matrix files, such as `research/configs/experiments/matrix_smoke.yaml`, declare a `base_config`, optional `common_overrides`, optional reusable artifacts, and a list of `variants`. Each variant deep-merges its overrides onto the base config and gets its own run directory.

## Data Expectations

Image inputs come from either `paths.images_dir` or `paths.manifest_file`.

- Directory mode walks `images_dir` recursively and accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
- Manifest mode reads JSON Lines where each row contains an `image_path`.
- Relative image paths are interpreted by the command or workflow that consumes them; keep manifests stable relative to the repository root or use absolute paths for external datasets.

Training workflows live under `research/experiments/workflows/training`. They are separate from the main scene graph evaluation CLI and should write their model artifacts outside run directories unless a specific experiment needs them as inputs. Evaluation workflows expect detections, descriptions, vocabulary, draft scene graphs, and optional ground truth to use the file names configured in `paths`.

## Artifact Layout

Runs are written below:

```text
research/artifacts/runs/<run_id>/
```

Each run starts with:

- `run_metadata.json`, including command, config, config hash, optional manifest hash, model metadata, platform details, and git state.
- `run.log`, the per-run log file.

Common phase artifacts use the names from `paths`:

| Artifact | Contents |
|---|---|
| `detections.json` | Detected objects keyed by image. |
| `descriptions.json` | Generated image descriptions keyed by image. |
| `vocabulary_candidates.json` | Raw vocabulary extraction candidates. |
| `vocabulary_final.json` | Consolidated predicates and attributes. |
| `draft_scene_graph.json` | Generated relationships and attributes keyed by image. |
| `context_rot.json` | Context reduction sensitivity results. |
| `ground_truth_scene_graph.json` | Human or imported reference annotations. |
| `metrics_scene_graph_per_image.json` | Per-image scene graph metrics. |
| `metrics_scene_graph_summary.json` | Aggregate scene graph metrics. |
| `metrics_image_potency.json` | Image-level potency metrics. |
| `metrics_sensitivity_curves.json` | Vocabulary sensitivity metrics. |

Draft scene graph runs may also write SoM images under the configured `som_output_dir`, usually `som_images_draft`.

## Human Evaluation

Human evaluation can use either a JSON template or the browser annotation bundle.

For direct JSON editing, generate a template from a run:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main make-gt-template --run research/artifacts/runs/<run_id>
```

The template lists detected object IDs, labels, and boxes for each image. Annotators fill the `relationships` list with rows shaped like:

```json
{"sub": "1", "rel": "on", "obj": "2"}
```

For browser annotation, export and serve a static bundle:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main export-annotation-bundle --run research/artifacts/runs/<run_id> --out research/artifacts/annotation/<run_id>
PYTHONPATH=. python3 -m research.experiments.cli.main serve-annotation-bundle --bundle research/artifacts/annotation/<run_id>
```

Import the exported annotations back into the run with `import-annotation-export`. Evaluation then reads `ground_truth_scene_graph.json`, or a supplied `--gt` file when using `evaluate-run`.
