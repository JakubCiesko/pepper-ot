#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

INVERSE_RULES = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "under": "above",
    "in_front_of": "behind",
    "behind": "in_front_of",
}

SYMMETRIC_RULES = {
    "next_to",
    "touching",
    "connected_to",
    "parallel_to",
    "perpendicular_to",
    "interacting_with",
    "shaking_hands",
    "talking_to",
}

SUPPORT_RULES = {
    "on": "supporting",
    "sitting_on": "supporting",
    "standing_on": "supporting",
    "lying_on": "supporting",
    "resting_on": "supporting",
    "leaning_on": "supporting",
    "mounted_on": "supporting",
    "hanging_from": "supporting",
    "stacked_on": "below",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Complete scene graph annotations with inverse and symmetric "
            "relations. Standalone utility; no project imports."
        )
    )
    parser.add_argument("input", type=Path, help="Input SGG JSON file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Required unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file. Otherwise the script is non-destructive.",
    )
    return parser.parse_args()


def normalize_edge(edge):
    if not isinstance(edge, dict):
        return None
    sub = edge.get("sub")
    rel = edge.get("rel")
    obj = edge.get("obj")
    if sub is None or rel is None or obj is None:
        return None
    sub = str(sub).strip()
    rel = str(rel).strip()
    obj = str(obj).strip()
    if not sub or not rel or not obj:
        return None
    return {"sub": sub, "rel": rel, "obj": obj}


def derived_relation(rel):
    if rel in INVERSE_RULES:
        return INVERSE_RULES[rel]
    if rel in SYMMETRIC_RULES:
        return rel
    if rel in SUPPORT_RULES:
        return SUPPORT_RULES[rel]
    return None


def complete_item(item, added_by_rule):
    if not isinstance(item, dict):
        return {"relationships": []}, 0, 0, 0

    raw_relationships = item.get("relationships", [])
    if not isinstance(raw_relationships, list):
        raw_relationships = []

    output = []
    seen = set()
    malformed = 0
    duplicates = 0

    for raw_edge in raw_relationships:
        edge = normalize_edge(raw_edge)
        if edge is None:
            malformed += 1
            continue
        key = (edge["sub"], edge["rel"], edge["obj"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append(edge)

    original_valid = len(output)
    for edge in list(output):
        new_rel = derived_relation(edge["rel"])
        if not new_rel:
            continue
        derived = {
            "sub": edge["obj"],
            "rel": new_rel,
            "obj": edge["sub"],
        }
        key = (derived["sub"], derived["rel"], derived["obj"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append(derived)
        added_by_rule[(edge["rel"], new_rel)] += 1

    new_item = dict(item)
    new_item["relationships"] = output
    added = len(output) - original_valid
    return new_item, malformed, duplicates, added


def complete_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object keyed by image path/id.")

    completed = {}
    added_by_rule = Counter()
    totals = Counter()

    for key, item in payload.items():
        original_count = 0
        if isinstance(item, dict) and isinstance(item.get("relationships"), list):
            original_count = len(item["relationships"])
        new_item, malformed, duplicates, added = complete_item(item, added_by_rule)
        completed[str(key)] = new_item
        totals["images"] += 1
        totals["original_edges"] += original_count
        totals["malformed_skipped"] += malformed
        totals["duplicate_skipped"] += duplicates
        totals["added_edges"] += added
        totals["final_edges"] += len(new_item.get("relationships", []))

    return completed, totals, added_by_rule


def output_path(args):
    if args.in_place:
        if args.out is not None:
            raise ValueError("Use either --in-place or --out, not both.")
        return args.input
    if args.out is None:
        raise ValueError("Provide --out unless --in-place is used.")
    return args.out


def print_report(totals, added_by_rule, out_path):
    print(f"Images processed: {totals['images']}")
    print(f"Original edges: {totals['original_edges']}")
    print(f"Malformed skipped: {totals['malformed_skipped']}")
    print(f"Duplicate existing/derived skipped: {totals['duplicate_skipped']}")
    print(f"Added edges: {totals['added_edges']}")
    print(f"Final edges: {totals['final_edges']}")
    print(f"Output: {out_path}")

    if not added_by_rule:
        print("Added by rule: none")
        return

    print()
    print("Added by rule:")

    for (source, derived), count in added_by_rule.most_common():
        print(f"  {source} -> {derived}: {count}")


def main():
    args = parse_args()
    try:
        out_path = output_path(args)
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        completed, totals, added_by_rule = complete_payload(payload)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(completed, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print_report(totals, added_by_rule, out_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
