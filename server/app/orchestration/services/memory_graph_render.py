import base64
import html
import io
import math

from PIL import Image

from app.schemas.scene import MemorySummary
from app.schemas.scene import SceneGraphRelation
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState


class MemoryGraphRenderService:
    """Builds a merged memory graph view and a lightweight SVG rendering."""

    MAX_RENDER_OBJECTS = 6
    COLUMNS = 3
    THUMB_WIDTH = 190
    THUMB_HEIGHT = 130
    NODE_WIDTH = 210
    NODE_PADDING = 10
    PILL_HEIGHT = 20
    PILL_GAP = 6
    MAX_ATTRIBUTE_PILLS = 5
    CELL_HEIGHT = 265
    PADDING_X = 40
    PADDING_Y = 40
    GAP_X = 70
    GAP_Y = 90

    def build_summary(
        self,
        state: SceneState,
        *,
        crop_map: dict[int, bytes | None] | None = None,
        render_limit: int | None = None,
        object_label_overrides: dict[int, str] | None = None,
        object_attribute_overrides: dict[int, list[str]] | None = None,
        relation_label_overrides: dict[tuple[int, str, int], str] | None = None,
    ) -> MemorySummary:
        labels, label_counts = self._labels_and_counts(
            state,
            object_label_overrides=object_label_overrides,
        )
        relations, attribute_counts, relationship_counts = self._build_relations(
            state,
            object_label_overrides=object_label_overrides,
            object_attribute_overrides=object_attribute_overrides,
            relation_label_overrides=relation_label_overrides,
        )
        limit = (
            self.MAX_RENDER_OBJECTS
            if render_limit is None
            else max(1, min(int(render_limit), self.MAX_RENDER_OBJECTS))
        )
        render_ids = self.select_render_object_ids(state, limit=limit)
        graph_svg = self.render_svg(
            state,
            crop_map=crop_map or {},
            render_object_ids=render_ids,
            object_label_overrides=object_label_overrides,
            object_attribute_overrides=object_attribute_overrides,
            relation_label_overrides=relation_label_overrides,
        )
        # TODO: think whether to send this -> will slow down communication, 
        # maybe save space and make pepper parse the counts and all.
        return MemorySummary(
            labels=labels,
            label_counts=label_counts,
            scene_graph=relations,
            graph_svg=graph_svg,
            timestamp=state.timestamp,
        )
    # TODO: probably get rid of _ids because it fucks it up 
    def build_text_description(self, 
                               state: SceneState) -> str:
        labels, label_counts = self._labels_and_counts(state)
        relations, attribute_counts, relationship_counts = self._build_relations(state)
        description = "Scene Description\n" + "\n".join([
            f"There is {label}. It is present {count} times."
        for label, count in label_counts.items()])
        description += "The objects are in these relations:\n" + "\n".join([
            f"{rel[0]} {rel[1]} {rel[2]}. This relation is present {count} times." for rel, count in relationship_counts.items() 
        ])
        return description


    def select_render_object_ids(
        self, state: SceneState, *, limit: int | None = None
    ) -> list[int]:
        objects = sorted(
            state.objects,
            key=lambda item: (item.last_seen, item.hits, item.id),
            reverse=True,
        )
        if limit is not None:
            objects = objects[:limit]
        return [obj.id for obj in objects]

    def _labels_and_counts(
        self,
        state: SceneState,
        *,
        object_label_overrides: dict[int, str] | None = None,
    ) -> tuple[list[str], dict[str, int]]:
        counts: dict[str, int] = {}
        for obj in state.objects:
            label = str(
                (object_label_overrides or {}).get(obj.id, obj.label)
            ).strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        labels = sorted(counts)
        return labels, counts

    def _build_relations(
        self,
        state: SceneState,
        *,
        object_label_overrides: dict[int, str] | None = None,
        object_attribute_overrides: dict[int, list[str]] | None = None,
        relation_label_overrides: dict[tuple[int, str, int], str] | None = None,
    ) -> tuple[
        list[SceneGraphRelation],
        dict[tuple[str, str, str], int],
        dict[tuple[str, str, str], int],
    ]:
        object_map = {obj.id: obj for obj in state.objects}
        relations: list[SceneGraphRelation] = []
        attribute_counts: dict[tuple[str, str, str]: int] = {}
        relationship_counts: dict[tuple[str, str, str]: int] = {}
        seen: set[tuple[str, str, str]] = set()

        for obj in sorted(state.objects, key=lambda item: item.id):
            label = str((object_label_overrides or {}).get(obj.id, obj.label)).strip()
            node_name = f"{label}_{obj.id}"
            attributes = (object_attribute_overrides or {}).get(obj.id, obj.attributes or [])
            for attribute in sorted(set(attributes or [])):
                key = (node_name, attribute, node_name)
                attribute_counts[key] = attribute_counts.get(key, 0) + 1
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
            sub_label = str(
                (object_label_overrides or {}).get(subject.id, subject.label)
            ).strip()
            obj_label = str(
                (object_label_overrides or {}).get(obj.id, obj.label)
            ).strip()
            predicate = str(
                (relation_label_overrides or {}).get(
                    (rel.subject_id, rel.predicate, rel.object_id), rel.predicate
                )
            ).strip()
            sub_name = f"{sub_label}_{subject.id}"
            obj_name = f"{obj_label}_{obj.id}"
            key = (sub_name, predicate, obj_name)
            relationship_counts[key] = relationship_counts.get(key, 0) + 1
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                SceneGraphRelation(sub=sub_name, rel=predicate, obj=obj_name)
            )
        return relations, attribute_counts, relationship_counts

    def render_svg(
        self,
        state: SceneState,
        *,
        crop_map: dict[int, bytes | None],
        render_object_ids: list[int],
        object_label_overrides: dict[int, str] | None = None,
        object_attribute_overrides: dict[int, list[str]] | None = None,
        relation_label_overrides: dict[tuple[int, str, int], str] | None = None,
    ) -> str:
        if not render_object_ids:
            return self._empty_svg()

        object_map = {obj.id: obj for obj in state.objects}
        objects = [
            object_map[obj_id] for obj_id in render_object_ids if obj_id in object_map
        ]
        if not objects:
            return self._empty_svg()

        rows = max(1, math.ceil(len(objects) / self.COLUMNS))
        width = (
            self.PADDING_X * 2
            + self.COLUMNS * self.NODE_WIDTH
            + (self.COLUMNS - 1) * self.GAP_X
        )
        height = self.PADDING_Y * 2 + rows * self.CELL_HEIGHT + (rows - 1) * self.GAP_Y

        positions: dict[int, tuple[int, int]] = {}
        edge_fragments: list[str] = []
        node_fragments: list[str] = []

        for index, obj in enumerate(objects):
            col = index % self.COLUMNS
            row = index // self.COLUMNS
            x = self.PADDING_X + col * (self.NODE_WIDTH + self.GAP_X)
            y = self.PADDING_Y + row * (self.CELL_HEIGHT + self.GAP_Y)
            positions[obj.id] = (x, y)

        for rel in sorted(
            state.relationships,
            key=lambda item: (item.subject_id, item.predicate, item.object_id),
        ):
            if rel.subject_id == rel.object_id:
                continue
            src = positions.get(rel.subject_id)
            dst = positions.get(rel.object_id)
            if src is None or dst is None:
                continue
            edge_fragments.append(
                self._render_edge(
                    src_x=src[0] + self.NODE_WIDTH / 2,
                    src_y=src[1] + self.THUMB_HEIGHT / 2,
                    dst_x=dst[0] + self.NODE_WIDTH / 2,
                    dst_y=dst[1] + self.THUMB_HEIGHT / 2,
                    label=str(
                        (relation_label_overrides or {}).get(
                            (rel.subject_id, rel.predicate, rel.object_id), rel.predicate
                        )
                    ).strip(),
                )
            )

        for obj in objects:
            x, y = positions[obj.id]
            node_fragments.append(
                self._render_node(
                    obj,
                    crop_bytes=crop_map.get(obj.id),
                    display_label=str(
                        (object_label_overrides or {}).get(obj.id, obj.label)
                    ).strip(),
                    display_attributes=(object_attribute_overrides or {}).get(
                        obj.id, obj.attributes or []
                    ),
                    x=x,
                    y=y,
                )
            )

        body = "\n".join(
            [
                '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0 10 3.5 0 7" fill="#475569"/></marker></defs>',
                *edge_fragments,
                *node_fragments,
            ]
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="Pepper memory graph">'
            '<rect width="100%" height="100%" fill="#f8fafc"/>'
            f"{body}</svg>"
        )

    def _render_node(
        self,
        obj: TrackedObjectState,
        *,
        crop_bytes: bytes | None,
        display_label: str,
        display_attributes: list[str],
        x: int,
        y: int,
    ) -> str:
        image_x = x + (self.NODE_WIDTH - self.THUMB_WIDTH) / 2
        image_y = y
        pills_y = image_y + self.THUMB_HEIGHT + 10
        pills = [self._title_pill(display_label, obj.id)]
        pills.extend(self._attribute_pills(display_attributes))

        image_fragment = self._render_thumbnail(crop_bytes, image_x, image_y)
        pill_fragments: list[str] = []
        current_y = pills_y
        for pill in pills:
            pill_fragments.append(
                self._render_pill(
                    pill,
                    x + self.NODE_PADDING,
                    current_y,
                    self.NODE_WIDTH - 2 * self.NODE_PADDING,
                )
            )
            current_y += self.PILL_HEIGHT + self.PILL_GAP

        return f'<g>{image_fragment}{"".join(pill_fragments)}</g>'

    def _title_pill(self, label: str, object_id: int) -> str:
        return f"{label}_{object_id}"

    def _attribute_pills(self, attributes: list[str] | None) -> list[str]:
        attributes = sorted(set(attributes or []))
        if not attributes:
            return ["no attributes"]
        visible = attributes[: self.MAX_ATTRIBUTE_PILLS]
        if len(attributes) > self.MAX_ATTRIBUTE_PILLS:
            visible.append(f"+{len(attributes) - self.MAX_ATTRIBUTE_PILLS} more")
        return visible

    def _render_thumbnail(
        self,
        crop_bytes: bytes | None,
        x: float,
        y: float,
    ) -> str:
        clip_id = f"clip-{int(x)}-{int(y)}"
        frame = (
            f'<rect x="{x}" y="{y}" width="{self.THUMB_WIDTH}" height="{self.THUMB_HEIGHT}" '
            'rx="16" ry="16" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>'
        )
        if crop_bytes:
            data_uri = self._thumbnail_data_uri(crop_bytes)
            if data_uri is not None:
                return (
                    f'<g><defs><clipPath id="{clip_id}">'
                    f'<rect x="{x}" y="{y}" width="{self.THUMB_WIDTH}" height="{self.THUMB_HEIGHT}" rx="16" ry="16"/></clipPath></defs>'
                    f"{frame}"
                    f'<image x="{x}" y="{y}" width="{self.THUMB_WIDTH}" height="{self.THUMB_HEIGHT}" '
                    f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})" href="{data_uri}"/>'
                    "</g>"
                )
        placeholder = (
            f"{frame}"
            f'<text x="{x + self.THUMB_WIDTH / 2}" y="{y + self.THUMB_HEIGHT / 2}" '
            'text-anchor="middle" font-size="14" fill="#94a3b8">no crop</text>'
        )
        return f"<g>{placeholder}</g>"

    def _render_pill(self, text: str, x: float, y: float, width: float) -> str:
        escaped = html.escape(text)
        return (
            f'<g><rect x="{x}" y="{y}" width="{width}" height="{self.PILL_HEIGHT}" '
            'rx="10" ry="10" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>'
            f'<text x="{x + 10}" y="{y + 14}" font-size="12" fill="#111827">{escaped}</text></g>'
        )

    def _render_edge(
        self,
        *,
        src_x: float,
        src_y: float,
        dst_x: float,
        dst_y: float,
        label: str,
    ) -> str:
        label_x = (src_x + dst_x) / 2
        label_y = (src_y + dst_y) / 2 - 6
        escaped = html.escape(label)
        return (
            f'<g><line x1="{src_x}" y1="{src_y}" x2="{dst_x}" y2="{dst_y}" '
            'stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>'
            f'<rect x="{label_x - 46}" y="{label_y - 14}" width="92" height="18" rx="8" ry="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="12" fill="#111827">{escaped}</text></g>'
        )

    def _thumbnail_data_uri(self, crop_bytes: bytes) -> str | None:
        try:
            image = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
            image.thumbnail((self.THUMB_WIDTH * 2, self.THUMB_HEIGHT * 2))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None

    def _empty_svg(self) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220" viewBox="0 0 640 220" '
            'role="img" aria-label="Pepper memory graph empty">'
            '<rect width="100%" height="100%" fill="#f8fafc"/>'
            '<text x="320" y="110" text-anchor="middle" font-size="22" fill="#64748b">'
            "Memory graph is empty"
            "</text></svg>"
        )
