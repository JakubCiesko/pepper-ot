from collections.abc import Iterable
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CanonicalEdge:
    sub: str
    rel: str
    obj: str


def normalize_node(value: object, *, normalize_ids: bool) -> str:
    text = str(value).strip()
    if not normalize_ids:
        return text
    match = re.search(r"(\d+)$", text)
    return match.group(1) if match else text


def normalize_relation(value: object, *, normalize_relations: bool) -> str:
    text = str(value).strip()
    if not normalize_relations:
        return text
    return text.lower().replace(" ", "_")


def _relationship_rows(payload: object) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("relationships", "edges", "no_label_edges"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    if all(key in payload for key in ("sub", "rel", "obj")):
        return [payload]
    return []


def canonicalize_edges(
    payload: object,
    *,
    normalize_ids: bool = True,
    normalize_relations: bool = True,
) -> set[CanonicalEdge]:
    rows = _relationship_rows(payload)
    out: set[CanonicalEdge] = set()
    for row in rows:
        if not all(key in row for key in ("sub", "rel", "obj")):
            continue
        sub = normalize_node(row.get("sub"), normalize_ids=normalize_ids)
        obj = normalize_node(row.get("obj"), normalize_ids=normalize_ids)
        rel = normalize_relation(
            row.get("rel"), normalize_relations=normalize_relations
        )
        if not sub or not rel or not obj:
            continue
        out.add(CanonicalEdge(sub=sub, rel=rel, obj=obj))
    return out


def split_unary_binary(
    edges: Iterable[CanonicalEdge],
) -> tuple[set[CanonicalEdge], set[CanonicalEdge]]:
    unary: set[CanonicalEdge] = set()
    binary: set[CanonicalEdge] = set()
    for edge in edges:
        if edge.sub == edge.obj:
            unary.add(edge)
        else:
            binary.add(edge)
    return unary, binary
