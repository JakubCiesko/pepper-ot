import asyncio
import logging
from typing import Any

from PIL import Image

from app.inference.scene_graph.reltr_backend import RelTRSceneGraphGenerator
from app.inference.scene_graph.rules_backend import RuleSceneGraphGenerator
from app.inference.scene_graph.vlm_backend import VLMSceneGraphGenerator
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


class SceneGraphService:
    """Runs enabled scene graph backends and merges robot scene state."""

    def __init__(
        self,
        vlm_backend: VLMSceneGraphGenerator,
        rule_backend: RuleSceneGraphGenerator,
        reltr_backend: RelTRSceneGraphGenerator,
        parallel_execution: bool = False,
    ):
        self.vlm_backend = vlm_backend
        self.rule_backend = rule_backend
        self.reltr_backend = reltr_backend
        self.parallel_execution = parallel_execution

    async def generate(
        self,
        detections: list[InferenceDetectionObject],
        *,
        som_image=None,
        raw_image: Image.Image | None = None,
        caption_text: str | None = None,
        scene_state: SceneState | None = None,
    ) -> SceneGraph:
        if self.parallel_execution:
            return await self.generate_parallel(
                detections,
                som_image=som_image,
                raw_image=raw_image,
                caption_text=caption_text,
                scene_state=scene_state,
            )
        return await self.generate_sequential(
            detections,
            som_image=som_image,
            raw_image=raw_image,
            caption_text=caption_text,
            scene_state=scene_state,
        )

    async def generate_sequential(
        self,
        detections: list[InferenceDetectionObject],
        *,
        som_image=None,
        raw_image: Image.Image | None = None,
        caption_text: str | None = None,
        scene_state: SceneState | None = None,
    ) -> SceneGraph:
        vlm_image = som_image if som_image is not None else raw_image
        enabled_backends = self._enabled_backends()
        logger.info(
            "Generating scene graph with backends=%s merge_strategy='union+dedup' parallel_execution=false for %d detections",
            ",".join(enabled_backends) if enabled_backends else "none",
            len(detections),
        )
        graph = SceneGraph()

        if self.rule_backend.rules_config.enabled:
            rules_graph = self.rule_backend.generate(raw_image, detections)
            logger.debug("SGG RULES output: %s", rules_graph)
            graph = graph + rules_graph

        if self.reltr_backend.config.enabled:
            reltr_graph = await self.reltr_backend.generate(raw_image, detections)
            logger.debug("SGG RELTR output: %s", reltr_graph)
            graph = graph + reltr_graph

        if self.vlm_backend.config.enabled:
            vlm_graph = (
                SceneGraph()
                if vlm_image is None
                else await self.vlm_backend.generate(
                    vlm_image, detections, caption_text
                )
            )
            logger.debug("SGG VLM output: %s", vlm_graph)
            graph = graph + vlm_graph

        logger.debug("Merged SGG output: %s", graph)
        enhanced_graph = self.enhance_scene_graph_with_robot_data(
            graph, detections, scene_state
        )
        logger.debug("Final Robot-Enhanced Graph: %s", enhanced_graph)
        return enhanced_graph

    async def generate_parallel(
        self,
        detections: list[InferenceDetectionObject],
        *,
        som_image=None,
        raw_image: Image.Image | None = None,
        caption_text: str | None = None,
        scene_state: SceneState | None = None,
    ) -> SceneGraph:
        vlm_image = som_image if som_image is not None else raw_image
        enabled_backends = self._enabled_backends()
        logger.info(
            "Generating scene graph with backends=%s merge_strategy='union+dedup' parallel_execution=true for %d detections",
            ",".join(enabled_backends) if enabled_backends else "none",
            len(detections),
        )

        names: list[str] = []
        tasks: list[asyncio.Task[SceneGraph]] = []
        if self.rule_backend.rules_config.enabled:
            names.append("rules")
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        self.rule_backend.generate,
                        self._copy_image_for_parallel(raw_image),
                        detections,
                    )
                )
            )
        if self.reltr_backend.config.enabled:
            names.append("reltr")
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        self.reltr_backend.generate_sync,
                        self._copy_image_for_parallel(raw_image),
                        detections,
                    )
                )
            )
        if self.vlm_backend.config.enabled:
            names.append("vlm")
            if vlm_image is None:
                tasks.append(asyncio.create_task(self._empty_scene_graph()))
            else:
                tasks.append(
                    asyncio.create_task(
                        self.vlm_backend.generate(
                            self._copy_image_for_parallel(vlm_image),
                            detections,
                            caption_text,
                        )
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        graph = SceneGraph()
        for name, backend_graph in zip(names, results, strict=False):
            if isinstance(backend_graph, Exception):
                raise backend_graph
            logger.debug("SGG %s output: %s", name.upper(), backend_graph)
            graph = graph + backend_graph

        logger.debug("Merged SGG output: %s", graph)
        enhanced_graph = self.enhance_scene_graph_with_robot_data(
            graph, detections, scene_state
        )
        logger.debug("Final Robot-Enhanced Graph: %s", enhanced_graph)
        return enhanced_graph

    @staticmethod
    async def _empty_scene_graph() -> SceneGraph:
        return SceneGraph()

    @staticmethod
    def _copy_image_for_parallel(image: Any):
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.copy()
        if hasattr(image, "copy"):
            return image.copy()
        return image

    def _enabled_backends(self) -> list[str]:
        names: list[str] = []
        if self.rule_backend.rules_config.enabled:
            names.append("rules")
        if self.reltr_backend.config.enabled:
            names.append("reltr")
        if self.vlm_backend.config.enabled:
            names.append("vlm")
        return names

    def enhance_scene_graph_with_robot_data(
        self,
        graph: SceneGraph,
        detections: list[InferenceDetectionObject],
        scene_state: SceneState,
    ) -> SceneGraph:
        if not scene_state or not detections:
            return graph
        current_ids = {
            int(det.object_id) for det in detections if isinstance(det.object_id, int)
        }
        current_objects = [obj for obj in scene_state.objects if obj.id in current_ids]
        label_edges = [
            {
                "sub": f"{obj.label}_{obj.id}",
                "rel": attribute,
                "obj": f"{obj.label}_{obj.id}",
            }
            for obj in current_objects
            for attribute in obj.attributes
        ]
        robot_metadata_scene_graph = SceneGraph.from_list(label_edges)
        return graph + robot_metadata_scene_graph
