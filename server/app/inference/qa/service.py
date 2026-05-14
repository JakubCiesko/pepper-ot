import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.providers.llm.client import LLMClient
from app.schemas.config import ChatConfig

logger = logging.getLogger(__name__)


class _GeneratedQAPair(BaseModel):
    question: str
    answer: str


class _GeneratedQAPairs(BaseModel):
    items: list[_GeneratedQAPair] = Field(default_factory=list)


class SceneQAGenerationService:
    def __init__(self, llm_config: ChatConfig, *, pairs_per_update: int):
        self.llm = LLMClient(llm_config)
        self.pairs_per_update = max(1, int(pairs_per_update))
        self.system_prompt = (
            "You generate concise grounded question answer pairs from scene graph facts. "
            "Use only facts explicitly present in the graph triples. "
            "Do not hallucinate. "
            "Write in English."  # TODO: maybe passable language? But english is safe choice for all LLMs
        )

    def update_runtime(self, llm_config: ChatConfig, *, pairs_per_update: int):
        # TODO: double update for llm and then this
        self.llm.update_runtime(llm_config)
        self.pairs_per_update = max(1, int(pairs_per_update))

    @staticmethod
    def _normalize_item(item: Any) -> dict[str, str] | None:
        question = str(getattr(item, "question", "") or "").strip()
        answer = str(getattr(item, "answer", "") or "").strip()
        if not question or not answer:
            return None
        return {"question": question, "answer": answer}

    @staticmethod
    def _graph_to_text(
        scene_graph: SceneGraph,
        detections: list[InferenceDetectionObject],
        caption_text: str | None,
    ) -> str:
        detection_lines = [
            f"- {det.label}_{det.object_id if det.object_id is not None else idx + 1}"
            for idx, det in enumerate(detections)
        ]
        # graph_lines = [
        #     f"- {edge.get('sub', '')} | {edge.get('rel', '')} | {edge.get('obj', '')}"
        #     for edge in scene_graph.as_dict()[:120]
        # ]
        graph_lines = [
            (
                f"- {subj} | {edge.get('rel', '')} | {obj}"
                if (obj := edge.get("obj", "")) != (subj := edge.get("sub", ""))
                else f"- {subj} | {edge.get('rel', '')}"
            )
            for edge in scene_graph.as_dict()[:120]
        ]
        parts = [
            "Detected objects:",
            *(detection_lines or ["- none"]),
            "",
            "Scene graph facts:",
            *(graph_lines or ["- none"]),
        ]
        if caption_text:
            parts.extend(["", "Caption:", f"- {caption_text.strip()}"])
        return "\n".join(parts)

    async def generate(
        self,
        *,
        scene_graph: SceneGraph | None,
        detections: list[InferenceDetectionObject],
        caption_text: str | None,
    ) -> list[dict[str, str]]:
        if scene_graph is None or len(scene_graph) == 0:
            return []

        graph_description = self._graph_to_text(scene_graph, detections, caption_text)
        user_prompt = (
            f"Generate exactly {self.pairs_per_update} short question answer pairs.\n"
            "Return only structured output matching the schema.\n"
            "Questions should be useful for robot conversation.\n"
            "Answers must be factual and concise.\n\n"
            f"{graph_description}"
        )
        try:
            generated = await self.llm.generate_structured(
                self.system_prompt,
                user_prompt,
                output_schema=_GeneratedQAPairs,
            )
        except Exception as exc:
            logger.warning("QA generation stage failed: %s", exc)
            return []

        unique: dict[str, dict[str, str]] = {}
        for item in generated.items:
            normalized = self._normalize_item(item)
            if normalized is None:
                continue
            key = normalized["question"].strip().lower()
            unique[key] = normalized
            if len(unique) >= self.pairs_per_update:
                break
        return list(unique.values())
