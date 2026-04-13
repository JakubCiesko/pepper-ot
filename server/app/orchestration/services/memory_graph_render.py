import html
import math

from app.schemas.scene import MemorySummary
from app.schemas.scene import SceneGraphRelation
from app.schemas.scene import SceneState


class MemoryGraphRenderService:
    """Builds a merged memory graph view and a lightweight SVG rendering."""

    CARD_WIDTH = 220
    CARD_BASE_HEIGHT = 56
    CARD_LINE_HEIGHT = 18
    COLUMNS = 3
    PADDING_X = 40
    PADDING_Y = 40
    GAP_X = 70
    GAP_Y = 80

    def build_summary(self, state: SceneState) -> MemorySummary:
        labels, label_counts = self._labels_and_counts(state)
        relations = self._build_relations(state)
        graph_svg = self.render_svg(state)
        return MemorySummary(
            labels=labels,
            label_counts=label_counts,
            scene_graph=relations,
            graph_svg=graph_svg,
            timestamp=state.timestamp,
        )

    def _labels_and_counts(self, state: SceneState) -> tuple[list[str], dict[str, int]]:
        counts: dict[str, int] = {}
        for obj in state.objects:
            label = str(obj.label).strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        labels = sorted(counts)
        return labels, counts

    def _build_relations(self, state: SceneState) -> list[SceneGraphRelation]:
        object_map = {obj.id: obj for obj in state.objects}
        relations: list[SceneGraphRelation] = []
        seen: set[tuple[str, str, str]] = set()

        for obj in sorted(state.objects, key=lambda item: item.id):
            node_name = f"{obj.label}_{obj.id}"
            for attribute in sorted(set(obj.attributes or [])):
                key = (node_name, attribute, node_name)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    SceneGraphRelation(sub=node_name, rel=attribute, obj=node_name)
                )

        for rel in sorted(
            state.relationships,
            key=lambda item: (item.subject_id, item.predicate, item.object_id),
        ):
            subject = object_map.get(rel.subject_id)
            obj = object_map.get(rel.object_id)
            if subject is None or obj is None:
                continue
            sub_name = f"{subject.label}_{subject.id}"
            obj_name = f"{obj.label}_{obj.id}"
            key = (sub_name, rel.predicate, obj_name)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                SceneGraphRelation(sub=sub_name, rel=rel.predicate, obj=obj_name)
            )
        return relations

    def render_svg(self, state: SceneState) -> str:
        objects = sorted(state.objects, key=lambda item: item.id)
        if not objects:
            return self._empty_svg()

        rows = max(1, math.ceil(len(objects) / self.COLUMNS))
        widths = (
            self.PADDING_X * 2
            + self.COLUMNS * self.CARD_WIDTH
            + (self.COLUMNS - 1) * self.GAP_X
        )
        heights = self.PADDING_Y * 2 + rows * 180 + (rows - 1) * self.GAP_Y

        object_positions: dict[int, tuple[int, int, int]] = {}
        node_fragments: list[str] = []
        for index, obj in enumerate(objects):
            col = index % self.COLUMNS
            row = index // self.COLUMNS
            x = self.PADDING_X + col * (self.CARD_WIDTH + self.GAP_X)
            y = self.PADDING_Y + row * (180 + self.GAP_Y)
            attributes = sorted(set(obj.attributes or []))
            card_height = (
                self.CARD_BASE_HEIGHT + min(len(attributes), 8) * self.CARD_LINE_HEIGHT
            )
            object_positions[obj.id] = (x, y, card_height)
            node_fragments.append(
                self._render_node(obj.label, obj.id, attributes, x, y, card_height)
            )

        object_map = {obj.id: obj for obj in objects}
        edge_fragments: list[str] = []
        for rel in sorted(
            state.relationships,
            key=lambda item: (item.subject_id, item.predicate, item.object_id),
        ):
            if rel.subject_id == rel.object_id:
                continue
            src = object_positions.get(rel.subject_id)
            dst = object_positions.get(rel.object_id)
            src_obj = object_map.get(rel.subject_id)
            dst_obj = object_map.get(rel.object_id)
            if src is None or dst is None or src_obj is None or dst_obj is None:
                continue
            edge_fragments.append(
                self._render_edge(
                    src_x=src[0] + self.CARD_WIDTH // 2,
                    src_y=src[1] + src[2] // 2,
                    dst_x=dst[0] + self.CARD_WIDTH // 2,
                    dst_y=dst[1] + dst[2] // 2,
                    label=rel.predicate,
                )
            )

        body = "\n".join(
            [
                '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0 10 3.5 0 7" fill="#4b5563"/></marker></defs>',
                *edge_fragments,
                *node_fragments,
            ]
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{widths}" height="{heights}" '
            f'viewBox="0 0 {widths} {heights}" role="img" aria-label="Pepper memory graph">'
            '<rect width="100%" height="100%" fill="#f8fafc"/>'
            f"{body}</svg>"
        )

    def _render_node(
        self,
        label: str,
        object_id: int,
        attributes: list[str],
        x: int,
        y: int,
        height: int,
    ) -> str:
        title = html.escape(f"{label}_{object_id}")
        lines = [
            f'<text x="{x + 14}" y="{y + 24}" font-size="16" font-weight="700" fill="#111827">{title}</text>'
        ]
        if attributes:
            for idx, attribute in enumerate(attributes[:8], start=1):
                attr_text = html.escape(f"• {attribute}")
                line_y = y + 24 + idx * self.CARD_LINE_HEIGHT
                lines.append(
                    f'<text x="{x + 14}" y="{line_y}" font-size="13" fill="#374151">{attr_text}</text>'
                )
        else:
            lines.append(
                f'<text x="{x + 14}" y="{y + 44}" font-size="13" fill="#9ca3af">no attributes</text>'
            )
        lines_joined = "".join(lines)
        return (
            f'<g><rect x="{x}" y="{y}" rx="14" ry="14" width="{self.CARD_WIDTH}" height="{height}" '
            'fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>'
            f"{lines_joined}</g>"
        )

    def _render_edge(
        self,
        *,
        src_x: int,
        src_y: int,
        dst_x: int,
        dst_y: int,
        label: str,
    ) -> str:
        label_x = (src_x + dst_x) / 2
        label_y = (src_y + dst_y) / 2 - 6
        escaped = html.escape(label)
        return (
            f'<g><line x1="{src_x}" y1="{src_y}" x2="{dst_x}" y2="{dst_y}" '
            'stroke="#4b5563" stroke-width="2" marker-end="url(#arrow)"/>'
            f'<rect x="{label_x - 42}" y="{label_y - 14}" width="84" height="18" fill="#f8fafc"/>'
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="12" fill="#111827">{escaped}</text></g>'
        )

    def _empty_svg(self) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220" viewBox="0 0 640 220" '
            'role="img" aria-label="Pepper memory graph empty">'
            '<rect width="100%" height="100%" fill="#f8fafc"/>'
            '<text x="320" y="110" text-anchor="middle" font-size="22" fill="#64748b">'
            "Memory graph is empty"
            "</text></svg>"
        )
