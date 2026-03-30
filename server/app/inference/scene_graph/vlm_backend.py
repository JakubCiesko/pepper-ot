import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.prompting.renderer import PromptRenderContext
from app.core.prompting.renderer import render_prompt_template
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.inference.types import SceneGraphEdge
from app.providers.vlm import BaseVLMClient
from app.providers.vlm import build_vlm_client
from app.schemas.config import SceneGraphVLMConfig
from app.schemas.scene import SceneGraphRelation
from app.schemas.scene import SceneGraphStructuredResponse

logger = logging.getLogger(__name__)


# TODO: post filtering of invalid graphs
class VLMSceneGraphGenerator:
    def __init__(
        self,
        config: SceneGraphVLMConfig,
        predicates: list[str] | None = None,
        objects: dict[str, str] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        self.config = config
        self.predicates = predicates
        self.objects = objects
        self.system_prompt = system_prompt or ""
        self.user_prompt = user_prompt
        self.client: BaseVLMClient = build_vlm_client(config)

    def update_runtime(
        self,
        config: SceneGraphVLMConfig,
        predicates: list[str] | None,
        objects: dict[str, str] | None,
        system_prompt: str | None,
        user_prompt: str | None,
        rebuild_client: bool = False,
    ):
        self.config = config
        self.predicates = predicates
        self.objects = objects
        self.system_prompt = system_prompt or ""
        self.user_prompt = user_prompt
        if rebuild_client:
            self.client = build_vlm_client(config)
        else:
            self.client.update_runtime(config)

    @staticmethod
    def _serialize_image_bytes(image: Path | bytes | Image.Image | np.ndarray) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, Path):
            return image.read_bytes()
        if isinstance(image, Image.Image):
            with io.BytesIO() as buf:
                image.save(buf, format="JPEG")
                return buf.getvalue()
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype("uint8"))
            with io.BytesIO() as buf:
                pil_img.save(buf, format="JPEG")
                return buf.getvalue()
        raise TypeError(
            f"Unsupported input type: {type(image)}. Must be Path, bytes, PIL.Image, or np.ndarray."
        )

    @staticmethod
    def _extract_json_block(raw: str) -> str | None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        return None

    @staticmethod
    def _normalize_relations_payload(data):
        if isinstance(data, dict):
            for key in ["relationships", "scene_graph", "triplets", "relations"]:
                if key in data:
                    data = data[key]
                    break
            else:
                if all(k in data for k in ["sub", "rel", "obj"]):
                    data = [data]
        elif not isinstance(data, list):
            data = [data]
        return data if isinstance(data, list) else []

    def _parse_relations_json(self, raw: str) -> list[dict]:
        try:
            data = json.loads(raw)
            return self._normalize_relations_payload(data)
        except Exception:
            extracted = self._extract_json_block(raw)
            if extracted:
                try:
                    data = json.loads(extracted)
                    return self._normalize_relations_payload(data)
                except Exception:
                    return []
            return []

    async def _repair_json_output(self, image_bytes: bytes, raw: str) -> str:
        repair_system = (
            "You are a JSON repair engine. Return ONLY valid JSON with key "
            '"relationships" that is a list of {"sub","rel","obj"} objects.'
        )
        clipped = raw[:2000]
        repair_user = (
            "Fix the following output into valid JSON only. No extra text.\n\n"
            f"OUTPUT:\n{clipped}"
        )
        # TODO: check whether this works properly, not sending the image anymore
        repaired, _ = await self.client.infer(repair_system, repair_user, None)
        return repaired

    def _build_user_prompt(self, caption_text: str | None = None) -> str:
        render_context = PromptRenderContext(
            predicates=self.predicates, caption=caption_text
        )
        predicates_text = render_context.to_template_values().get("predicates", "")
        if self.user_prompt:
            return render_prompt_template(self.user_prompt, render_context)
        if self.predicates:
            return "Allowed predicates: " + predicates_text
        return "Focus on spatial, semantic, and functional relationships."

    def _build_system_prompt(self, caption_text: str | None = None) -> str:
        render_context = PromptRenderContext(
            predicates=self.predicates, caption=caption_text
        )
        return render_prompt_template(self.system_prompt, render_context)

    async def generate(
        self,
        image: Path | bytes | Image.Image | np.ndarray,
        detections: list[InferenceDetectionObject],
        caption_text: str | None = None,
    ) -> SceneGraph:
        image_bytes = self._serialize_image_bytes(image)
        system_prompt = self._build_system_prompt(caption_text)
        user_prompt = self._build_user_prompt(caption_text)
        output_schema: Any = SceneGraphStructuredResponse
        if self.config.structured_schema == "relationship_list":
            output_schema = list[SceneGraphRelation]
        try:
            raw, parsed = await self.client.infer(
                system_prompt,
                user_prompt,
                image_bytes,
                output_schema=output_schema,
            )
        except Exception as exc:
            logger.warning(
                "Structured VLM generation failed, falling back to raw mode: %s", exc
            )
            raw, parsed = await self.client.infer(
                system_prompt,
                user_prompt,
                image_bytes,
                output_schema=None,
            )
        # this is just for precommit to shutup
        relation_dicts: list[dict] = []
        if isinstance(parsed, SceneGraphStructuredResponse):
            relation_dicts = [rel.model_dump() for rel in parsed.relationships]
        elif isinstance(parsed, list):
            relation_dicts = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in parsed
            ]
        elif parsed is not None:
            relation_dicts = self._normalize_relations_payload(parsed)
        else:
            relation_dicts = self._parse_relations_json(raw)
        if not relation_dicts:
            logger.warning(
                "Failed to parse VLM output as JSON, attempting repair without picture"
            )
            repaired = await self._repair_json_output(image_bytes, raw)
            relation_dicts = self._parse_relations_json(repaired)
            if not relation_dicts:
                logger.warning("VLM repair failed, returning empty scene graph")
        logger.info(
            "VLM input: SYSTEM_PROMPT=[%s], USER_PROMPT=[%s]; VLM RAW OUTPUT=[%s]",
            system_prompt,
            user_prompt,
            raw,
        )
        scene_graph = SceneGraph.from_list(relation_dicts, raw=raw)
        scene_graph = self.filter_hallucinated_relations(scene_graph, detections)
        return scene_graph

    def filter_hallucinated_relations(
        self, scene_graph: SceneGraph, detections: list[InferenceDetectionObject] | None
    ) -> SceneGraph:
        if not detections:
            logger.info(
                "No detections (det=%s) passed to fix potential overgeneration."
                "Returning original scene graph",
                detections,
            )
            return scene_graph
        # only labels and ids appearing in detections can be part of the current scene graph
        id_to_label = {}
        logger.info(
            "Filtering scene graph based on current list of detections (%d dets)",
            len(detections),
        )
        for det in detections:
            if det.object_id is None:
                # can be but should not be
                continue
            obj_id = str(det.object_id)
            label = (det.label or "").strip() or "object"
            id_to_label[obj_id] = label
        # case: no obj ids, TODO: decide whether to return the original SG or empty SG... But should be really empty,
        #  because it is overgenerated then...
        if not id_to_label:
            logger.info(
                "No valid mapping of ids to labels for detections, returning empty scene graph."
            )
            return SceneGraph(edges=[], no_label_edges=[], raw=scene_graph.raw)
        # case: mapping works
        filtered_with_labels, filtered_without_labels = [], []
        source_edges = (
            scene_graph.no_label_edges or scene_graph.edges
        )  # prefering just the id-edges
        dropped_edges: list[tuple[str, str, str]] = (
            []
        )  # will be used for logging dropped edges
        for edge in source_edges:
            sub_id = SceneGraph._normalize_id(edge.sub)
            obj_id = SceneGraph._normalize_id(edge.obj)
            rel = str(edge.rel).strip()
            if not sub_id or not obj_id or not rel:
                dropped_edges.append((sub_id, rel, obj_id))
                continue
            if sub_id not in id_to_label or obj_id not in id_to_label:
                dropped_edges.append((sub_id, rel, obj_id))
                continue  # skip hallucinated edges and objects
            filtered_without_labels.append(
                SceneGraphEdge(sub=sub_id, obj=obj_id, rel=rel)
            )
            sub_label, obj_label = id_to_label[sub_id], id_to_label[obj_id]
            filtered_with_labels.append(
                SceneGraphEdge(
                    sub=f"{sub_label}_{sub_id}", obj=f"{obj_label}_{obj_id}", rel=rel
                )
            )

        logger.info(
            "Filtered scene graph removing overgeneration: # edges with labels %d -> %d, "
            "# edges without labels %d -> %d. First 5 dropped edges: %s",
            len(scene_graph.edges),
            len(filtered_with_labels),
            len(scene_graph.no_label_edges),
            len(filtered_without_labels),
            dropped_edges[:5],
        )
        # inherit raw scene graph
        return SceneGraph(
            edges=filtered_with_labels,
            no_label_edges=filtered_without_labels,
            raw=scene_graph.raw,
        )
