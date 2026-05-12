from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from research.experiments.io import load_json


DEFAULT_RUNS = [
    "train_vocab2_gemini",
    "train_vocab2_gpt",
    "train_vocab3_gemini",
    "train_vocab3_gpt",
    "train_vocab3_qwen36_35b_a3b",
]


PREDICATE_COLOR = "#F2D45C"
ATTRIBUTE_COLOR = "#0000dc"


def _counts_from_vocab(vocab: dict[str, Any], kind: str) -> Counter[str]:
    provenance = vocab.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}

    count_key = "%s_counts" % kind
    counts = provenance.get(count_key, {})
    if isinstance(counts, dict) and counts:
        return Counter(
            {
                str(term): int(count)
                for term, count in counts.items()
                if str(term).strip() and int(count) > 0
            }
        )

    # Older or partially written artifacts may only contain the final selected
    # vocabulary. Count each selected term once so the run can still contribute
    # to overlap statistics.
    terms = vocab.get(kind, [])
    if isinstance(terms, list):
        return Counter(str(term) for term in terms if str(term).strip())
    return Counter()


def _image_support_from_candidates(
    candidates: dict[str, Any],
    kind: str,
) -> Counter[str]:
    support: Counter[str] = Counter()
    for row in candidates.values():
        if not isinstance(row, dict):
            continue
        for term in set(row.get(kind, []) or []):
            term = str(term).strip()
            if term:
                support[term] += 1
    return support


def _load_run_counts(run_dir: Path) -> dict[str, Any] | None:
    vocab_path = run_dir / "vocabulary_final.json"
    if not vocab_path.exists():
        print("Skipping %s: missing vocabulary_final.json" % run_dir)
        return None

    vocab = load_json(vocab_path, default={})
    if not isinstance(vocab, dict):
        print("Skipping %s: vocabulary_final.json is not an object" % run_dir)
        return None

    candidates = load_json(run_dir / "vocabulary_candidates.json", default={})
    if not isinstance(candidates, dict):
        candidates = {}

    return {
        "predicates": _counts_from_vocab(vocab, "predicate"),
        "attributes": _counts_from_vocab(vocab, "attribute"),
        "predicate_image_support": _image_support_from_candidates(
            candidates, "predicates"
        ),
        "attribute_image_support": _image_support_from_candidates(
            candidates, "attributes"
        ),
        "image_count": len(candidates),
    }


def _aggregate(
    run_counts: dict[str, dict[str, Any]],
    key: str,
) -> tuple[Counter[str], Counter[str], dict[str, Counter[str]]]:
    total_counts: Counter[str] = Counter()
    run_presence: Counter[str] = Counter()
    per_run: dict[str, Counter[str]] = {}

    for run_name, counts_by_kind in run_counts.items():
        counts = counts_by_kind[key]
        per_run[run_name] = counts
        total_counts.update(counts)
        for term in counts:
            run_presence[term] += 1

    return total_counts, run_presence, per_run


def _write_frequency_csv(
    path: Path,
    total_mentions: Counter[str],
    total_image_support: Counter[str],
    run_presence: Counter[str],
    per_run_mentions: dict[str, Counter[str]],
    per_run_image_support: dict[str, Counter[str]],
    run_image_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_names = list(per_run_mentions.keys())
    total_image_observations = sum(run_image_counts.values())
    fieldnames = [
        "term",
        "total_mentions",
        "total_image_support",
        "normalized_image_support",
        "run_count",
    ]
    for run_name in run_names:
        fieldnames.extend(
            [
                "%s_mentions" % run_name,
                "%s_image_support" % run_name,
                "%s_image_support_rate" % run_name,
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        terms = set(total_mentions) | set(total_image_support)
        for term in sorted(
            terms,
            key=lambda item: (
                -total_image_support.get(item, 0),
                -total_mentions.get(item, 0),
                item,
            ),
        ):
            image_support = total_image_support.get(term, 0)
            row = {
                "term": term,
                "total_mentions": total_mentions.get(term, 0),
                "total_image_support": image_support,
                "normalized_image_support": (
                    image_support / float(total_image_observations)
                    if total_image_observations
                    else 0.0
                ),
                "run_count": run_presence.get(term, 0),
            }
            for run_name in run_names:
                image_count = run_image_counts.get(run_name, 0)
                support = per_run_image_support[run_name].get(term, 0)
                row["%s_mentions" % run_name] = per_run_mentions[run_name].get(term, 0)
                row["%s_image_support" % run_name] = support
                row["%s_image_support_rate" % run_name] = (
                    support / float(image_count) if image_count else 0.0
                )
            writer.writerow(row)


def _write_rank_plot(out_dir: Path, kind: str, counts: Counter[str]) -> None:
    if not counts:
        return
    import matplotlib.pyplot as plt

    values = [count for _, count in counts.most_common()]
    ranks = list(range(1, len(values) + 1))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(ranks, values, marker=".", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Term rank")
    ax.set_ylabel("Frequency across runs")
    ax.set_title("%s rank-frequency distribution" % kind.capitalize())
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / ("%s_rank_frequency.png" % kind), dpi=180)
    fig.savefig(out_dir / ("%s_rank_frequency.pdf" % kind))
    plt.close(fig)


def _write_normalized_rank_plot(
    out_dir: Path,
    kind: str,
    image_support: Counter[str],
    total_image_observations: int,
) -> None:
    if not image_support or not total_image_observations:
        return
    import matplotlib.pyplot as plt

    values = [
        count / float(total_image_observations)
        for _, count in image_support.most_common()
    ]
    ranks = list(range(1, len(values) + 1))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(ranks, values, marker=".", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Term rank")
    ax.set_ylabel("Normalized image support")
    ax.set_title("%s image-support distribution" % kind.capitalize())
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / ("%s_image_support_rank_frequency.png" % kind), dpi=180)
    fig.savefig(out_dir / ("%s_image_support_rank_frequency.pdf" % kind))
    plt.close(fig)


def _write_combined_normalized_rank_plot(
    out_dir: Path,
    predicate_image_support: Counter[str],
    attribute_image_support: Counter[str],
    total_image_observations: int,
) -> None:
    if (
        not predicate_image_support
        or not attribute_image_support
        or not total_image_observations
    ):
        return
    import matplotlib.pyplot as plt

    predicate_values = [
        count / float(total_image_observations)
        for _, count in predicate_image_support.most_common()
    ]
    attribute_values = [
        count / float(total_image_observations)
        for _, count in attribute_image_support.most_common()
    ]
    predicate_ranks = list(range(1, len(predicate_values) + 1))
    attribute_ranks = list(range(1, len(attribute_values) + 1))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        predicate_ranks,
        predicate_values,
        marker=".",
        linewidth=1.4,
        color=PREDICATE_COLOR,
        label="Predicates",
    )
    ax.plot(
        attribute_ranks,
        attribute_values,
        marker=".",
        linewidth=1.4,
        color=ATTRIBUTE_COLOR,
        label="Attributes",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Term rank")
    ax.set_ylabel("Normalized image support")
    ax.set_title("Predicate and attribute image-support distributions")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "vocabulary_image_support_rank_frequency.png", dpi=180)
    fig.savefig(out_dir / "vocabulary_image_support_rank_frequency.pdf")
    plt.close(fig)


def _write_combined_raw_rank_plot(
    out_dir: Path,
    predicate_counts: Counter[str],
    attribute_counts: Counter[str],
) -> None:
    if not predicate_counts or not attribute_counts:
        return
    import matplotlib.pyplot as plt

    predicate_values = [count for _, count in predicate_counts.most_common()]
    attribute_values = [count for _, count in attribute_counts.most_common()]
    predicate_ranks = list(range(1, len(predicate_values) + 1))
    attribute_ranks = list(range(1, len(attribute_values) + 1))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        predicate_ranks,
        predicate_values,
        marker=".",
        linewidth=1.4,
        color=PREDICATE_COLOR,
        label="Predicates",
    )
    ax.plot(
        attribute_ranks,
        attribute_values,
        marker=".",
        linewidth=1.4,
        color=ATTRIBUTE_COLOR,
        label="Attributes",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Term rank")
    ax.set_ylabel("Raw extraction frequency")
    ax.set_title("Predicate and attribute rank-frequency distributions")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "vocabulary_raw_rank_frequency.png", dpi=180)
    fig.savefig(out_dir / "vocabulary_raw_rank_frequency.pdf")
    plt.close(fig)


def _plot_color(kind: str) -> str | None:
    if kind == "predicate":
        return PREDICATE_COLOR
    if kind == "attribute":
        return ATTRIBUTE_COLOR
    return None


def _write_top_terms_plot(
    out_dir: Path,
    kind: str,
    counts: Counter[str],
    top_k: int,
    denominator: int | None = None,
    xlabel: str = "Frequency across runs",
    filename_prefix: str = "top",
) -> None:
    if not counts:
        return
    import matplotlib.pyplot as plt

    top_items = counts.most_common(top_k)
    labels = [term for term, _ in reversed(top_items)]
    if denominator:
        values = [count / float(denominator) for _, count in reversed(top_items)]
    else:
        values = [count for _, count in reversed(top_items)]

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, len(labels) * 0.22)))
    ax.barh(labels, values, color=_plot_color(kind))
    ax.set_xlabel(xlabel)
    ax.set_title("Top %s %s" % (len(labels), kind))
    fig.tight_layout()
    fig.savefig(out_dir / ("%s_%s.png" % (filename_prefix, kind)), dpi=180)
    fig.savefig(out_dir / ("%s_%s.pdf" % (filename_prefix, kind)))
    if filename_prefix == "top_image_support":
        fig.savefig(out_dir / ("top_%s.png" % kind), dpi=180)
        fig.savefig(out_dir / ("top_%s.pdf" % kind))
    plt.close(fig)


def _write_overlap_plot(
    out_dir: Path,
    kind: str,
    run_presence: Counter[str],
    run_total: int,
) -> None:
    if not run_presence:
        return
    import matplotlib.pyplot as plt

    overlap_hist = Counter(run_presence.values())
    xs = list(range(1, run_total + 1))
    ys = [overlap_hist.get(x, 0) for x in xs]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xlabel("Number of runs containing term")
    ax.set_ylabel("Unique terms")
    ax.set_title("%s cross-run overlap" % kind.capitalize())
    fig.tight_layout()
    fig.savefig(out_dir / ("%s_run_overlap.png" % kind), dpi=180)
    fig.savefig(out_dir / ("%s_run_overlap.pdf" % kind))
    plt.close(fig)


def _write_plots(
    out_dir: Path,
    kind: str,
    total_mentions: Counter[str],
    total_image_support: Counter[str],
    run_presence: Counter[str],
    run_total: int,
    total_image_observations: int,
    top_k: int,
) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed; wrote CSV files only")
        return

    _write_rank_plot(out_dir, kind, total_mentions)
    _write_normalized_rank_plot(
        out_dir,
        kind,
        total_image_support,
        total_image_observations,
    )
    _write_top_terms_plot(
        out_dir,
        kind,
        total_mentions,
        top_k,
        xlabel="Raw extracted mentions across runs",
        filename_prefix="top_mentions",
    )
    _write_top_terms_plot(
        out_dir,
        kind,
        total_image_support,
        top_k,
        denominator=total_image_observations,
        xlabel="Normalized image support",
        filename_prefix="top_image_support",
    )
    _write_overlap_plot(out_dir, kind, run_presence, run_total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot predicate and attribute frequency distributions across vocabulary runs."
    )
    parser.add_argument(
        "--runs-root",
        default="research/artifacts/runs",
        help="Directory containing experiment run directories.",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="Run directory name to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="research/artifacts/reports/vocabulary_distribution",
        help="Directory where CSV files and plots will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top predicates/attributes to show in bar charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    run_names = args.runs or DEFAULT_RUNS

    run_counts: dict[str, dict[str, Any]] = {}
    for run_name in run_names:
        loaded = _load_run_counts(runs_root / run_name)
        if loaded is not None:
            run_counts[run_name] = loaded

    if not run_counts:
        raise SystemExit("No complete vocabulary runs found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_counts_by_kind: dict[str, Counter[str]] = {}
    image_support_by_kind: dict[str, Counter[str]] = {}
    total_image_observations = sum(
        int(payload.get("image_count", 0)) for payload in run_counts.values()
    )
    for kind in ("predicates", "attributes"):
        support_key = "%s_image_support" % kind[:-1]
        total_mentions, run_presence, per_run_mentions = _aggregate(run_counts, kind)
        total_image_support, _, per_run_image_support = _aggregate(
            run_counts, support_key
        )
        raw_counts_by_kind[kind] = total_mentions
        image_support_by_kind[kind] = total_image_support
        run_image_counts = {
            run_name: int(payload.get("image_count", 0))
            for run_name, payload in run_counts.items()
        }
        _write_frequency_csv(
            out_dir / ("%s_frequencies.csv" % kind[:-1]),
            total_mentions,
            total_image_support,
            run_presence,
            per_run_mentions,
            per_run_image_support,
            run_image_counts,
        )
        _write_plots(
            out_dir,
            kind[:-1],
            total_mentions,
            total_image_support,
            run_presence,
            len(run_counts),
            total_image_observations,
            max(1, args.top_k),
        )

    _write_combined_raw_rank_plot(
        out_dir,
        raw_counts_by_kind.get("predicates", Counter()),
        raw_counts_by_kind.get("attributes", Counter()),
    )
    _write_combined_normalized_rank_plot(
        out_dir,
        image_support_by_kind.get("predicates", Counter()),
        image_support_by_kind.get("attributes", Counter()),
        total_image_observations,
    )

    print("Processed %s runs" % len(run_counts))
    print("Wrote vocabulary distribution report to %s" % out_dir)


if __name__ == "__main__":
    main()
