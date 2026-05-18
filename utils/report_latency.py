#!/usr/bin/env python3
from collections import Counter
from collections import defaultdict
import math
from pathlib import Path
import re
import statistics

LATENCY_LOG = Path("latency.logs")
MANUAL_NOTES = Path("latencies.txt")

LATENCY_RE = re.compile(
    r"LATENCY "
    r"turn_id=(?P<turn_id>\S+) "
    r"kind=(?P<kind>\S+) "
    r"event=(?P<event>\S+) "
    r"phase=(?P<phase>\S+) "
    r"elapsed_s=(?P<elapsed>[-0-9.]+) "
    r"wall_ts=(?P<wall_ts>[-0-9.]+)"
)

METRIC_NAMES = {
    "caption_memory_update_time",
    "caption_time",
    "detection_time",
    "memory_update_time",
    "qa_generation_time",
    "scene_graph_generation_time",
    "scene_graph_memory_update_time",
    "som_image_paint_time",
    "total_processing",
    "wall_processing_time",
}

SECTION_HINTS = [
    "setup",
    "to text",
    "bez reltr",
    "bez pregenerate",
    "no vlm sgg",
    "eng",
    "chat local llm",
    "gemini chat",
    "waht do you see",
    "what do you see",
]


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    return ("%." + str(digits) + "f s") % value


def fmt_mean_sd(values):
    values = clean_values(values)
    if not values:
        return "n/a"
    if len(values) == 1:
        return fmt(values[0])
    return "%.2f +/- %.2f s" % (statistics.mean(values), statistics.stdev(values))


def clean_values(values):
    return [v for v in values if v is not None and not math.isnan(v)]


def median(values):
    values = clean_values(values)
    if not values:
        return None
    return statistics.median(values)


def mean(values):
    values = clean_values(values)
    if not values:
        return None
    return statistics.mean(values)


def parse_latency_logs(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            match = LATENCY_RE.search(line)
            if not match:
                continue
            row = match.groupdict()
            row["line_no"] = line_no
            row["phase"] = None if row["phase"] == "None" else row["phase"]
            row["elapsed"] = parse_float(row["elapsed"])
            row["wall_ts"] = parse_float(row["wall_ts"])
            rows.append(row)
    return rows


def first_event(rows, event, phase=None):
    for row in rows:
        if row["event"] != event:
            continue
        if phase is not None and row["phase"] != phase:
            continue
        return row
    return None


def summarize_turns(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["turn_id"]].append(row)

    completed = []
    errors = []
    for turn_id, turn_rows in grouped.items():
        turn_rows = sorted(turn_rows, key=lambda row: row["line_no"])
        kind = turn_rows[0]["kind"]
        ack = first_event(turn_rows, "speech_start", "ack")
        answer = first_event(turn_rows, "speech_start", "answer")
        error = first_event(turn_rows, "speech_start", "error")
        server_start = first_event(turn_rows, "server_request_start")
        server_end = first_event(turn_rows, "server_response_received")

        if answer is not None:
            answer_latency = answer["elapsed"]
            ack_latency = ack["elapsed"] if ack is not None else None
            server_start_elapsed = (
                server_start["elapsed"] if server_start is not None else None
            )
            server_end_elapsed = (
                server_end["elapsed"] if server_end is not None else None
            )
            server_duration = None
            post_server = None
            if server_start_elapsed is not None and server_end_elapsed is not None:
                server_duration = server_end_elapsed - server_start_elapsed
            if answer_latency is not None and server_end_elapsed is not None:
                post_server = answer_latency - server_end_elapsed

            silent_after_ack = None
            if ack_latency is not None and answer_latency is not None:
                silent_after_ack = answer_latency - ack_latency

            completed.append(
                {
                    "turn_id": turn_id,
                    "kind": kind,
                    "answer_latency": answer_latency,
                    "ack_latency": ack_latency,
                    "server_start": server_start_elapsed,
                    "server_duration": server_duration,
                    "post_server": post_server,
                    "silent_after_ack": silent_after_ack,
                }
            )
        elif error is not None:
            errors.append(
                {
                    "turn_id": turn_id,
                    "kind": kind,
                    "error_latency": error["elapsed"],
                }
            )

    return completed, errors


def print_latency_summary(rows, completed, errors):
    print("# Robot Latency Summary\n")
    print("Source: `%s`\n" % LATENCY_LOG)
    print("- Parsed latency rows: `%d`" % len(rows))
    print("- Turns with final answers: `%d`" % len(completed))
    print("- Error turns: `%d`" % len(errors))
    print()

    print("## Main Table\n")
    print(
        "| Operation | n | Final answer median | Final answer mean +/- sd | Server time median | Pre-server median | Silent after ack median |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")

    labels = [
        ("look", "Look / describe scene"),
        ("ask", "Ask question"),
        ("cached_answer", "Cached answer"),
    ]
    for kind, label in labels:
        items = [item for item in completed if item["kind"] == kind]
        print(
            "| %s | %d | %s | %s | %s | %s | %s |"
            % (
                label,
                len(items),
                fmt(median([item["answer_latency"] for item in items])),
                fmt_mean_sd([item["answer_latency"] for item in items]),
                fmt(median([item["server_duration"] for item in items])),
                fmt(median([item["server_start"] for item in items])),
                fmt(median([item["silent_after_ack"] for item in items])),
            )
        )
    print()

    print("## Error Turns\n")
    if not errors:
        print("No error turns found.\n")
    else:
        print("| Kind | n | Error latency median | Error latency mean +/- sd |")
        print("|---|---:|---:|---:|")
        for kind in sorted(set(item["kind"] for item in errors)):
            items = [item for item in errors if item["kind"] == kind]
            values = [item["error_latency"] for item in items]
            print(
                "| %s | %d | %s | %s |"
                % (kind, len(items), fmt(median(values)), fmt_mean_sd(values))
            )
        print()

    print("## Useful Interpretation\n")
    look_items = [item for item in completed if item["kind"] == "look"]
    ask_items = [item for item in completed if item["kind"] == "ask"]
    cached_items = [item for item in completed if item["kind"] == "cached_answer"]
    look_med = median([item["answer_latency"] for item in look_items])
    look_server = median([item["server_duration"] for item in look_items])
    look_pre = median([item["server_start"] for item in look_items])
    ask_med = median([item["answer_latency"] for item in ask_items])
    ask_server = median([item["server_duration"] for item in ask_items])
    cached_med = median([item["answer_latency"] for item in cached_items])

    print(
        "For scene description turns, the median time to final answer speech "
        "was %s. The median server request itself took %s, while the median "
        "delay before the server request was %s. This means a substantial part "
        "of the observed response time happened before the server call, for "
        "example during robot-side turn handling, acknowledgement speech, and "
        "image capture/preparation. For direct question answering, the median "
        "final-answer latency was %s, with a median server time of %s. Cached "
        "or pregenerated answers were effectively instant in this run (%s), "
        "which supports using pregenerated or administrator-provided Q&A items "
        "when immediate answers are important."
        % (
            fmt(look_med),
            fmt(look_server),
            fmt(look_pre),
            fmt(ask_med),
            fmt(ask_server),
            fmt(cached_med, digits=3),
        )
    )
    print()


def looks_like_section(line):
    low = line.strip().lower()
    if not low:
        return False
    return any(hint in low for hint in SECTION_HINTS)


def parse_manual_notes(path):
    if not path.exists():
        return [], []

    sections = []
    current = {
        "name": "initial",
        "phone_ranges": [],
        "phone_singles": [],
        "metrics": defaultdict(list),
    }

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if looks_like_section(line):
            if (
                current["phone_ranges"]
                or current["phone_singles"]
                or current["metrics"]
            ):
                sections.append(current)
            current = {
                "name": line,
                "phone_ranges": [],
                "phone_singles": [],
                "metrics": defaultdict(list),
            }
            i += 1
            continue

        metric = line
        if metric in METRIC_NAMES and i + 1 < len(lines):
            value_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*s", lines[i + 1])
            if value_match:
                current["metrics"][metric].append(float(value_match.group(1)))
                i += 2
                continue

        range_match = re.fullmatch(
            r"([0-9]+(?:\.[0-9]+)?)\s*[-,]\s*([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if range_match:
            ack = float(range_match.group(1))
            final = float(range_match.group(2))
            current["phone_ranges"].append((ack, final, max(0.0, final - ack)))
            i += 1
            continue

        single_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*s?", line)
        if single_match:
            current["phone_singles"].append(float(single_match.group(1)))
            i += 1
            continue

        i += 1

    if current["phone_ranges"] or current["phone_singles"] or current["metrics"]:
        sections.append(current)

    return sections, lines


def print_manual_summary(sections):
    print("## Manual Phone Notes And Backend Metrics\n")
    print("Source: `%s`\n" % MANUAL_NOTES)
    if not sections:
        print("No manual notes parsed.\n")
        return

    print(
        "| Section | ranges n | feedback median | final median | silent median | singles n | single median | wall median | sgg median | qa median |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for section in sections:
        ranges = section["phone_ranges"]
        singles = section["phone_singles"]
        metrics = section["metrics"]
        print(
            "| %s | %d | %s | %s | %s | %d | %s | %s | %s | %s |"
            % (
                section["name"].replace("|", "/"),
                len(ranges),
                fmt(median([item[0] for item in ranges])),
                fmt(median([item[1] for item in ranges])),
                fmt(median([item[2] for item in ranges])),
                len(singles),
                fmt(median(singles)),
                fmt(median(metrics.get("wall_processing_time", []))),
                fmt(median(metrics.get("scene_graph_generation_time", []))),
                fmt(median(metrics.get("qa_generation_time", []))),
            )
        )
    print()

    fastest = []
    for section in sections:
        wall = median(section["metrics"].get("wall_processing_time", []))
        if wall is not None:
            fastest.append((wall, section))
    if fastest:
        fastest.sort(key=lambda item: item[0])
        wall, section = fastest[0]
        print(
            "Fastest backend setup in the manual notes: `%s`, with median "
            "`wall_processing_time` %s." % (section["name"], fmt(wall))
        )
        print()


def main():
    rows = parse_latency_logs(LATENCY_LOG)
    completed, errors = summarize_turns(rows)
    print_latency_summary(rows, completed, errors)

    sections, _ = parse_manual_notes(MANUAL_NOTES)
    print_manual_summary(sections)

    counts = Counter(row["kind"] for row in rows)
    if counts:
        print("## Parsed Event Counts\n")
        print("| Kind | latency rows |")
        print("|---|---:|")
        for kind, count in sorted(counts.items()):
            print("| %s | %d |" % (kind, count))
        print()


if __name__ == "__main__":
    main()
