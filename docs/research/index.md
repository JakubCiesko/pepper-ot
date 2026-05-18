# Research Experiments

The research package is the offline experiment harness for the thesis system. It reuses the same server-side detection, captioning, VLM, Set-of-Mark, and scene graph concepts, but it runs them over image batches and writes every intermediate artifact to disk so prompt, vocabulary, model, and visualization choices can be compared reproducibly.

The package is not the robot runtime. The robot and dashboard call the server. The research package is where you run controlled batches, create human ground truth, evaluate generated scene graphs, compare matrices, and produce reports.

## Mental Model

An experiment run is a directory containing a frozen config snapshot, logs, and phase artifacts. Each phase reads the previous artifacts and writes the next one:

```text
images/manifest
  -> detections + descriptions
  -> vocabulary candidates + final vocabulary
  -> draft scene graphs
  -> optional context-rot reduced-vocabulary runs
  -> optional evaluation metrics against human ground truth
```

The important rule is that resumed phase commands use the config stored in `run_metadata.json`, not whatever the source YAML says today. That is intentional: a run must stay reproducible after the original config file changes.

## CLI Entrypoint

Run commands from the repository root:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main --help
```

The default config is `research/configs/experiments/default.yaml`. Most commands accept `--config`. Phase commands also accept `--run research/artifacts/runs/<run_id>` to continue an existing run.

Core phase commands:

| Command | Reads | Writes | Purpose |
|---|---|---|---|
| `run-all` | Configured images or manifest | All enabled phase artifacts | Runs enabled phases in dependency order. |
| `describe` | Images or manifest, optional existing detections | `detections.json`, `descriptions.json`, `metrics_descriptions.json` | Runs detection when enabled, then generates caption-style scene descriptions. |
| `mine-vocab` | `descriptions.json`, `detections.json`, or a frozen vocab file | `vocabulary_candidates.json`, `vocabulary_final.json`, `metrics_vocabulary.json` | Extracts candidate predicates/attributes per image, then consolidates a final vocabulary. |
| `draft-sgg` | `descriptions.json`, `detections.json`, `vocabulary_final.json` | `draft_scene_graph.json`, optional `som_images_draft/`, `metrics_draft_scene_graph.json` | Prompts a VLM to output relationships over detected object IDs. |
| `context-rot` | Descriptions, detections, vocabulary, optional ground truth | `context_rot_vocab_slices.json`, `context_rot_levels/`, `context_rot.json`, `metrics_context_rot.json` | Re-runs draft SGG under reduced vocabulary levels to measure sensitivity to context size/composition. |
| `evaluate-sgg` | Ground truth, predictions, detections, vocabulary, optional context-rot output | Scene graph, potency, sensitivity, and stage metrics | Computes graph metrics and diagnostics for a run. |

Support commands:

| Command | Purpose |
|---|---|
| `make-manifest` | Writes `manifest.jsonl` from local image folders or streamed GQA samples. |
| `run-matrix` | Runs variants from a matrix YAML, with deep-merged overrides and optional artifact reuse. |
| `pipeline-batch` | Runs the live server perception pipeline over a manifest for latency or server-like replay. |
| `make-gt-template` | Creates a JSON ground-truth template containing detected object IDs and empty or draft-prefilled relationships. |
| `evaluate-run` | Copies a supplied ground-truth file into an existing run and evaluates it. |
| `plot-runs` | Aggregates completed run directories into CSV, JSON, Markdown, and plots. |
| `export-annotation-bundle` | Builds a static browser annotation bundle from one run. |
| `import-annotation-export` | Converts annotation UI export JSON into `ground_truth_scene_graph.json`. |
| `serve-annotation-bundle` | Serves an exported annotation bundle locally, on port 8000 by default. |

## Config Contract

Experiment configs are YAML files validated by `research/experiments/config/models.py`.

Top-level fields:

- `name`, `experiment_id`, `seed`: run identity and deterministic sampling/order choices.
- `paths`: image input, optional manifest, output root, and artifact file names.
- `description_model`, `vocabulary_model`, `draft_sgg_model`: provider, model ID, structured output mode, and optional base URL.
- `detection`: server detector backend, confidence, image resize, and batch settings.
- `descriptions`: caption prompts and concurrency.
- `vocabulary`: candidate extraction, consolidation prompts, target vocabulary sizes, and `current_run` vs `frozen_file` mode.
- `draft_scene_graph`: SoM rendering, prompt-image mode, vocabulary prompt mode, VLM concurrency, raw-response storage, and scene graph prompts.
- `context_rot`: automatic or manual vocabulary levels, strategy, rounds, and optional ground-truth evaluation.
- `evaluation`: key alignment, missing-pair policy, ID/relation normalization, per-predicate metrics, potency, and bootstrap settings.
- `prompting`: cross-phase switches such as whether detections enter description prompts and whether captions enter SGG prompts.

`paths.output_root` is the root for run directories. Normal runs go under:

```text
<output_root>/runs/<run_id>/
```

Matrix configs use this shape:

```yaml
name: MyMatrix
base_config: eval_sgg_server_like_base.yaml
reuse_artifacts:
  from_run: research/artifacts/runs/<existing_run>
  files:
    - detections.json
    - descriptions.json
    - vocabulary_final.json
common_overrides:
  descriptions:
    enabled: false
variants:
  - name: gpt_som
    overrides:
      experiment_id: gpt_som
      draft_scene_graph:
        use_som_image: true
```

`common_overrides` and each variant's `overrides` are recursively deep-merged onto the base config. `reuse_artifacts` copies already-computed files into each new variant run before phases execute, which is how you avoid re-running expensive detection/description/vocabulary stages when testing only SGG prompt or SoM changes.

## Data Inputs

Image inputs come from either `paths.images_dir` or `paths.manifest_file`.

- Directory mode walks `images_dir` recursively and accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
- Manifest mode reads JSON Lines with at least `image_path`; `image_id`, `dataset`, `split`, tags, and provenance are also supported by `ManifestRow`.
- Use absolute image paths for external datasets. If paths are relative, keep command execution rooted at the repository root.

The scene graph tasks assume relationships are over detector object IDs. A relationship row is:

```json
{"sub": "1", "rel": "on", "obj": "2"}
```

Unary attributes use the same object ID for subject and object:

```json
{"sub": "1", "rel": "is_red", "obj": "1"}
```

## Phase Details

### Description Phase

`run_descriptions` discovers images, optionally runs the server detector through `ServerDetectionAdapter`, and captions each image through `ServerCaptionAdapter`.

When detection is enabled, `detections.json` is keyed by resolved image path and contains serialized server detection rows: labels, bounding boxes, confidence, class ID, and object ID when available. The caption prompt can include the detected labels if `prompting.include_detection_labels_in_descriptions=true`.

Output:

- `detections.json`
- `descriptions.json`
- `metrics_descriptions.json`

### Vocabulary Phase

`run_vocabulary_mining` has two modes:

- `source_mode: current_run`: read descriptions and detections, ask the LLM for per-image predicates/attributes, then consolidate them into final predicate and attribute lists.
- `source_mode: frozen_file`: copy an existing vocabulary artifact into the run and record provenance.

The final vocabulary is not just a list. It also stores provenance, including source images, raw candidate counts, model settings, and prompts when mined from the current run.

Output:

- `vocabulary_candidates.json`
- `vocabulary_final.json`
- `metrics_vocabulary.json`

### Draft Scene Graph Phase

`run_draft_scene_graph` loads descriptions, detections, and final vocabulary. It may render a Set-of-Mark prompt image with object IDs, boxes, labels, masks, or polygons. The VLM is then prompted to output structured relationships over the visible object IDs.

Vocabulary prompt modes:

- `closed`: pass the structured predicates/attributes dictionary and instruct the model to use only those terms.
- `list`: pass a shuffled flat list of allowed terms.
- `soft`: pass vocabulary as guidance, not a hard constraint.
- `open`: remove the closed-vocabulary wording and let the model choose concise labels.

Output:

- `draft_scene_graph.json`
- optional SoM images under `draft_scene_graph.som_output_dir`, usually `som_images_draft/`
- `metrics_draft_scene_graph.json`

### Context-Rot Phase

Context-rot measures how sensitive SGG is to vocabulary context. The phase builds reduced vocabulary levels, reruns draft SGG for each level, remaps predictions into the reduced vocabulary, and optionally remaps/evaluates ground truth the same way.

Automatic levels use `min_vocab_size`, `step`, `strategy`, `seed`, and `rounds_per_size`. Manual levels come from `context_rot.levels_file` and can include explicit predicate/attribute maps plus `drop_unmapped`.

Output:

- `context_rot_vocab_slices.json`: vocabulary levels actually used.
- `context_rot_levels/<level>/vocabulary.json`: reduced vocabulary for that level.
- `context_rot_levels/<level>/raw_predictions.json`: raw VLM graph output.
- `context_rot_levels/<level>/remapped_predictions.json`: graph after reduced-vocabulary remapping.
- optional `remapped_ground_truth.json` and per-level metrics when ground truth is enabled.
- `context_rot.json`: aggregate per-level summary.
- `metrics_context_rot.json`: stage metrics.

### Evaluation Phase

Evaluation aligns keys through `manifest.jsonl` aliases when available, then compares ground truth and predictions. It accepts graph payloads with `relationships`, `edges`, `no_label_edges`, or a single `{sub, rel, obj}` row.

Metric groups:

- `strict_triplet`: exact subject, relation, object match.
- `binary_triplet`: exact match for non-unary edges only.
- `attribute`: exact match for unary attribute edges where `sub == obj`.
- `pair_ordered`: subject/object pair match ignoring relation label.
- `pair_unordered`: object pair match ignoring relation label and direction.
- `predicate_only`: multiset match of relation labels for binary edges.
- `diagnostics`: invalid object IDs, hallucinated objects, duplicates, out-of-vocabulary labels, and reversed-direction errors.
- `potency`: object count, possible ordered pairs, relation density, and attribute density.
- `sensitivity`: vocabulary/context-rot curve and prompt/model comparison table across runs.

Output:

- `metrics_scene_graph_per_image.json`
- `metrics_scene_graph_summary.json`
- `metrics_image_potency.json`
- `metrics_sensitivity_curves.json`
- `metrics_scene_graph_evaluation_stage.json`

## Human Evaluation

Human evaluation can use direct JSON editing or the static browser bundle.

Template workflow:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main make-gt-template \
  --run research/artifacts/runs/<run_id>
```

This writes `ground_truth_scene_graph.template.json` unless `--out` is supplied. Annotators fill each image's `relationships` list with `{sub, rel, obj}` rows over the listed object IDs.

Bundle workflow:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main export-annotation-bundle \
  --run research/artifacts/runs/<run_id> \
  --out research/artifacts/annotation/<run_id>

PYTHONPATH=. python3 -m research.experiments.cli.main serve-annotation-bundle \
  --bundle research/artifacts/annotation/<run_id>
```

`serve-annotation-bundle` uses local HTTP and defaults to `http://127.0.0.1:8000/index.html`. After annotation, import the exported JSON:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main import-annotation-export \
  --run research/artifacts/runs/<run_id> \
  --annotations path/to/export.json
```

Then evaluate:

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main evaluate-run \
  --run research/artifacts/runs/<run_id> \
  --gt research/artifacts/runs/<run_id>/ground_truth_scene_graph.json \
  --gt-only
```

Use `--gt-only` when the ground-truth file covers only a subset of run images. Without it, missing predictions or missing ground truth are handled according to `evaluation.missing_policy`.

## Common Workflows

### Smoke A Local Dataset

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main make-manifest \
  --source local \
  --images-dir data/subset \
  --out data/eval/local_smoke.jsonl \
  --max-samples 10 \
  --dataset-name local_smoke

PYTHONPATH=. python3 -m research.experiments.cli.main run-all \
  --config research/configs/experiments/default.yaml
```

For a real smoke config, set `paths.manifest_file` to the manifest and keep max samples small until every phase writes the expected artifacts.

### Resume One Phase

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main draft-sgg \
  --run research/artifacts/runs/<run_id>
```

This reads the config embedded in `run_metadata.json`. Use this when previous artifacts already exist and you only need to rerun a downstream phase.

### Run A Matrix

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main run-matrix \
  --config research/configs/experiments/matrix_eval_sgg_server_like.yaml
```

Use matrices when the question is comparative: raw image vs SoM image, labels vs no labels, captions vs no captions, provider A vs provider B, or frozen vocabulary vs mined vocabulary.

### Replay The Server Pipeline Offline

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main pipeline-batch \
  --server-config server/config.yaml \
  --manifest data/eval/local_smoke.jsonl \
  --out research/artifacts/latency/full_local_smoke \
  --preset full \
  --limit 10
```

This builds the server perception pipeline in-process and writes per-image outputs plus latency summaries. It is useful for measuring server presets without using the robot client or dashboard upload path.

### Aggregate Reports

```sh
PYTHONPATH=. python3 -m research.experiments.cli.main plot-runs \
  --runs-root research/artifacts/runs \
  --out research/artifacts/reports/latest
```

Report outputs include `metrics_summary.csv`, `metrics_summary.json`, `report.md`, and plots when `matplotlib` is installed.

## How To Read A Run Directory

Start with these files:

1. `run_metadata.json`: exact config, command, model settings, git state, and manifest/config hashes.
2. `run.log`: phase progress and failures.
3. `metrics_*stage*.json` or `metrics_<phase>.json`: item counts, failures, latency, and throughput.
4. `draft_scene_graph.json`: model output and prompt context for each image.
5. `metrics_scene_graph_summary.json`: run-level evaluation numbers.
6. `metrics_scene_graph_per_image.json`: per-image failures and outliers.

If the summary looks wrong, inspect one image across `detections.json`, `descriptions.json`, `vocabulary_final.json`, `draft_scene_graph.json`, and `ground_truth_scene_graph.json`. Most evaluation surprises are key mismatches, object ID mismatches, or relations that use labels outside the reduced/frozen vocabulary.

## Maintenance Notes

- Keep docs and config examples consistent with `research/experiments/config/models.py`.
- Keep command examples using `PYTHONPATH=.` from the repository root.
- If a phase adds a new artifact, update this page, `paths` config defaults, and report/evaluation docs if the artifact affects metrics.
- Do not treat training workflows under `research/experiments/workflows/training` as part of the main scene graph evaluation pipeline. They are separate model-development paths.
