# Experiment Runbook

## Day 1: Smoke Run

Create a 10-image local manifest:

```bash
PYTHONPATH=research python -m research.experiments.cli.main make-manifest \
  --source local \
  --images-dir data/subset \
  --out data/eval/local_smoke.jsonl \
  --max-samples 10 \
  --dataset-name local_smoke
```

Run the SoM/raw VLM smoke matrix:

```bash
PYTHONPATH=research python -m research.experiments.cli.main run-matrix \
  --config research/configs/experiments/matrix_smoke.yaml
```

Create an editable ground-truth template for one run:

```bash
PYTHONPATH=research python -m research.experiments.cli.main make-gt-template \
  --run research/artifacts/runs/<run_id>
```

After editing the template, evaluate the run:

```bash
PYTHONPATH=research python -m research.experiments.cli.main evaluate-run \
  --run research/artifacts/runs/<run_id> \
  --gt research/artifacts/runs/<run_id>/ground_truth_scene_graph.template.json
```

Export a bundle for browser annotation:

```bash
PYTHONPATH=research python -m research.experiments.cli.main export-annotation-bundle \
  --run research/artifacts/runs/<run_id> \
  --out research/artifacts/annotation_bundles/<bundle_id>
```

Serve the bundle locally:

```bash
PYTHONPATH=research python -m research.experiments.cli.main serve-annotation-bundle \
  --bundle research/artifacts/annotation_bundles/<bundle_id> \
  --port 8000
```

## Day 2: Vocabulary and Context-Rot

Run semantic and random vocabulary shrinking with the actual VLM image path:

```bash
PYTHONPATH=research python -m research.experiments.cli.main run-matrix \
  --config research/configs/experiments/matrix_vocab_context.yaml
```

Aggregate plots and tables:

```bash
PYTHONPATH=research python -m research.experiments.cli.main plot-runs \
  --runs-root research/artifacts/runs \
  --out research/artifacts/reports/latest
```

Use the context-rot scree plot/table to freeze the chosen vocabulary size.

## Day 3: SGG Settings

Duplicate `matrix_smoke.yaml` into a scaled matrix and vary:

- `draft_scene_graph.use_som_image`: true/false
- `draft_scene_graph.som_show_labels`: true/false
- `prompting.include_caption_in_sgg_prompt`: true/false
- `draft_sgg_model.provider`: gemini/openai/local_hf
- server SGG settings through `pipeline-batch` for rules, RelTR, and hybrids

## Day 4: Latency

Run replay latency against a manifest:

```bash
PYTHONPATH=research python -m research.experiments.cli.main pipeline-batch \
  --server-config server/config.yaml \
  --manifest data/eval/local_smoke.jsonl \
  --out research/artifacts/latency/full_local_smoke \
  --preset full \
  --limit 10
```

Repeat with presets such as `caption_only`, `detect_only`, `rules_only`, and `minimal`.

## Dataset Notes

Use local images first. For GQA, install the research package dependencies and run:

```bash
PYTHONPATH=research python -m research.experiments.cli.main make-manifest \
  --source gqa \
  --images-dir data/gqa_smoke/images \
  --out data/eval/gqa_smoke.jsonl \
  --max-samples 10
```

Then scale `--max-samples` after the 10-image smoke run works.
