from collections.abc import Iterable
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CanonicalEdge:
    """Normalized scene graph edge used as the comparison unit.

    Attributes:
        sub: Normalized subject object identifier.
        rel: Normalized relation or attribute label.
        obj: Normalized object identifier. Unary attributes use the same value
            for sub and obj.
    """

    sub: str
    rel: str
    obj: str


def normalize_node(value: object, *, normalize_ids: bool) -> str:
    """Normalize an object identifier for graph comparison.

    Args:
        value: Raw object identifier from a detection or graph payload.
        normalize_ids: When true, use only the trailing numeric component, so
            values like object_12 and 12 compare as the same node.

    Returns:
        Stripped identifier text, optionally reduced to the trailing number.
    """
    text = str(value).strip()
    if not normalize_ids:
        return text
    match = re.search(r"(\d+)$", text)
    return match.group(1) if match else text


def normalize_relation(value: object, *, normalize_relations: bool) -> str:
    """Normalize a relation or attribute label for graph comparison.

    Args:
        value: Raw relation or attribute label.
        normalize_relations: When true, lowercase the label and replace spaces
            with underscores.

    Returns:
        Normalized label text.
    """
    text = str(value).strip()
    if not normalize_relations:
        return text
    return text.lower().replace(" ", "_")


def _relationship_rows(payload: object) -> list[dict]:
    """Extract relationship rows from supported graph payload shapes.

    Args:
        payload: None, a list of rows, a graph dictionary with relationships,
            edges, or no_label_edges, or a single {sub, rel, obj} row.

    Returns:
        List of relationship dictionaries. Unsupported payloads and non-dict
        rows are ignored.
    """
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
    """Convert a graph payload into a deduplicated set of canonical edges.

    Args:
        payload: Graph payload containing relationships in any supported shape.
        normalize_ids: Whether to normalize object identifiers with
            normalize_node.
        normalize_relations: Whether to normalize labels with normalize_relation.

    Returns:
        Set of CanonicalEdge objects. Rows missing sub, rel, or obj and rows
        that normalize to empty values are skipped.
    """
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
    """Split canonical edges into attribute and relationship sets.

    Args:
        edges: Canonical edges from ground truth or predictions.

    Returns:
        A tuple of unary attribute edges and binary relationship edges. Unary
        attributes are represented by edges where sub equals obj.
    """
    unary: set[CanonicalEdge] = set()
    binary: set[CanonicalEdge] = set()
    for edge in edges:
        if edge.sub == edge.obj:
            unary.add(edge)
        else:
            binary.add(edge)
    return unary, binary
