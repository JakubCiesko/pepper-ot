import re
import time

from app.inference.types import SceneGraph
from app.schemas.scene import Relationship


class SceneMemoryStoreRelationsMixin:
    def insert_relation(self, rel: Relationship):
        if rel.subject_id not in self.objects_state:
            raise ValueError(f"Subject object id={rel.subject_id} does not exist")
        if rel.object_id not in self.objects_state:
            raise ValueError(f"Object object id={rel.object_id} does not exist")
        key = (rel.subject_id, rel.predicate, rel.object_id)
        if key in self.relations_state:
            raise ValueError("Relationship already exists")
        self.relations_state[key] = rel.model_copy(deep=True)

    def patch_relation(
        self,
        subject_id: int,
        predicate: str,
        object_id: int,
        updates: dict,
    ) -> Relationship:
        old_key = (subject_id, predicate, object_id)
        current = self.relations_state.get(old_key)
        if current is None:
            raise KeyError(
                f"Relationship ({subject_id}, {predicate}, {object_id}) does not exist"
            )
        new_subject_id = updates.get("subject_id", current.subject_id)
        new_predicate = updates.get("predicate", current.predicate)
        new_object_id = updates.get("object_id", current.object_id)

        if new_subject_id not in self.objects_state:
            raise ValueError(f"Subject object id={new_subject_id} does not exist")
        if new_object_id not in self.objects_state:
            raise ValueError(f"Object object id={new_object_id} does not exist")

        new_key = (new_subject_id, new_predicate, new_object_id)
        if new_key != old_key and new_key in self.relations_state:
            raise ValueError(
                f"Relationship ({new_subject_id}, {new_predicate}, {new_object_id}) already exists"
            )

        self.relations_state.pop(old_key, None)
        current.subject_id = new_subject_id
        current.predicate = new_predicate
        current.object_id = new_object_id

        for field in ("first_seen", "last_seen", "count"):
            if field in updates:
                setattr(current, field, updates[field])
        self.relations_state[new_key] = current
        return current

    def delete_relation(self, subject_id: int, predicate: str, object_id: int) -> bool:
        key = (subject_id, predicate, object_id)
        return self.relations_state.pop(key, None) is not None

    @staticmethod
    def _parse_id(value: str) -> int | None:
        try:
            return int(value)
        except Exception:
            match = re.search(r"(\d+)$", str(value))
            return int(match.group(1)) if match else None

    def update_scene_graph(self, scene_graph: SceneGraph):
        if scene_graph is None:
            return
        now = time.time()
        edges = getattr(scene_graph, "no_label_edges", None) or []
        for edge in edges:
            sub = self._parse_id(getattr(edge, "sub", ""))
            obj = self._parse_id(getattr(edge, "obj", ""))
            rel = getattr(edge, "rel", None)
            if sub is None or obj is None or not rel:
                continue
            if sub == obj:
                state = self.objects_state.get(sub)
                if state and rel not in state.attributes:
                    state.attributes.append(rel)
                continue
            key = (sub, rel, obj)
            existing = self.relations_state.get(key)
            if existing is None:
                self.relations_state[key] = Relationship(
                    subject_id=sub,
                    predicate=rel,
                    object_id=obj,
                    first_seen=now,
                    last_seen=now,
                    count=1,
                )
            else:
                existing.last_seen = now
                existing.count += 1
