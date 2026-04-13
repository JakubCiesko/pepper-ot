from collections.abc import Awaitable
from collections.abc import Callable
import inspect
import logging

from app.core.prompting.renderer import PromptRenderContext
from app.core.prompting.renderer import render_prompt_template
from app.inference.memory.scene_memory import SceneMemory
from app.providers.llm.client import LLMClient
from app.schemas.config import ChatConfig
from app.schemas.scene import Relationship
from app.schemas.scene import SceneState
from app.schemas.scene import TrackedObjectState

logger = logging.getLogger(__name__)

SOCIAL_SALIENT_LABELS = [
    "person",
    "man",
    "woman",
    "human",
    "animal",
    "child",
    "robot",
    "dog",
    "cat",  # :)
]


class ChatService:
    def __init__(
        self,
        config: ChatConfig,
        memory: SceneMemory,
        system_prompt: str,
        user_prompt: str | None = None,
        object_system_prompt: str | None = None,
        object_user_prompt: str | None = None,
    ):
        self.memory = memory
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.object_system_prompt = object_system_prompt
        self.object_user_prompt = object_user_prompt
        self.llm = LLMClient(config)

    async def _get_scene_state(self) -> SceneState:
        state_or_awaitable = self.memory.scene_state()
        if inspect.isawaitable(state_or_awaitable):
            return await state_or_awaitable
        return state_or_awaitable

    # TODO: needs to check whether awaitable?
    async def _get_track_crop(self, object_id: int) -> bytes | None:
        getter = getattr(self.memory, "get_track_crop", None)
        if getter is None:
            return None
        crop_or_awaitable = getter(object_id)
        if inspect.isawaitable(crop_or_awaitable):
            crop_or_awaitable = await crop_or_awaitable
        if isinstance(crop_or_awaitable, bytearray):
            return bytes(crop_or_awaitable)
        if isinstance(crop_or_awaitable, bytes):
            return crop_or_awaitable
        return None

    async def _build_context_string(self) -> str:
        state = await self._get_scene_state()
        if not state.objects:
            return "You see nothing."

        ordered_objects = sorted(
            state.objects, key=self._object_salience_key, reverse=True
        )  # ordered by social-saliency, biggest first (maybe the opposite is better?)
        obj_id_to_label = {obj.id: obj.label for obj in ordered_objects}

        object_lines = []
        for obj in ordered_objects:
            attrs = ", ".join(obj.attributes) if obj.attributes else "no attributes"
            object_lines.append(f"- ID {obj.id}: {obj.label} ({attrs})")
        # TODO: what if memory fkcsup ? need to filter only presnet in obj_id_to_label?
        relation_lines = [
            f"- {obj_id_to_label[rel.subject_id]}_{rel.subject_id} {rel.predicate} {obj_id_to_label[rel.object_id]}_{rel.object_id}"
            for rel in state.relationships
            if rel.subject_id in obj_id_to_label and rel.object_id in obj_id_to_label
        ]

        parts = ["Objects:"] + object_lines
        if relation_lines:
            parts += ["Relationships:"] + relation_lines
        return "\n".join(parts)

    async def _latest_caption(self) -> str:
        state = await self._get_scene_state()
        captions = sorted(state.captions, key=lambda c: c.last_seen, reverse=True)
        if not captions:
            return "No caption available."
        return captions[0].text

    async def _recent_captions(self, limit: int = 5) -> str:
        state = await self._get_scene_state()
        captions = sorted(state.captions, key=lambda c: c.last_seen, reverse=True)[
            :limit
        ]
        if not captions:
            return "No recent captions."
        lines = [f"- {caption.text}" for caption in captions if caption.text]
        return "\n".join(lines) if lines else "No recent captions."

    @staticmethod
    def _person_social_salience(attributes: set[str], obj) -> float:
        score = 0.0
        if any(
            salient_label in obj.label.lower().strip()
            for salient_label in SOCIAL_SALIENT_LABELS
        ):
            score += 10.0
        if "is_waving" in attributes:
            score += 40.0
        if "is_looking_at_robot" in attributes:
            score += 30.0
        if "is_sitting" in attributes:
            score += 10.0
        if any(attribute.startswith("engagement_zone_1") for attribute in attributes):
            score += 20.0
        elif any(attribute.startswith("engagement_zone_2") for attribute in attributes):
            score += 10.0
        if obj.robot_distance is not None:
            score += max(0.0, 10.0 - (obj.robot_distance * 5.0))
        return score

    @classmethod
    def _object_salience_key(cls, obj) -> tuple[float, float, int]:
        attributes = set(obj.attributes or [])
        return (
            cls._person_social_salience(attributes, obj),
            obj.last_seen,
            obj.hits,
        )

    @staticmethod
    def _normalize_label(value: str | None) -> str:
        return (value or "").strip().lower()

    async def _base_prompt_values(self) -> dict[str, str]:
        world_context = await self._build_context_string()
        latest_caption = await self._latest_caption()
        captions_recent = await self._recent_captions()
        render_context = PromptRenderContext(
            context=world_context,
            caption=latest_caption,
            captions_recent=captions_recent,
        )
        return {
            key: str(value or "")
            for key, value in render_context.to_template_values().items()
        }

    async def compose_prompt(
        self,
        base: str,
        *,
        extra: dict[str, str] | None = None,
    ) -> str:
        values = await self._base_prompt_values()
        if extra:
            values.update({key: str(value or "") for key, value in extra.items()})
        rendered = render_prompt_template(base, values)
        return rendered or base

    @staticmethod
    def _format_history(history: list[tuple[str, str]] | None) -> str:
        if not history:
            return ""
        lines = []
        for role, text in history:
            role_name = "User" if role == "user" else "Assistant"
            lines.append(f"{role_name}: {text}")
        return "\n".join(lines)

    async def chat(
        self,
        user_query: str,
        *,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> str:
        system_prompt = await self.compose_prompt(self.system_prompt)
        logger.debug("Chat request received, system prompt: %s", system_prompt)
        history_text = self._format_history(conversation_history)
        if self.user_prompt:
            user_prompt = await self.compose_prompt(
                self.user_prompt,
                extra={
                    "query": user_query,
                    "history": history_text,
                },
            )
        elif history_text:
            user_prompt = (
                "Conversation so far:\n"
                f"{history_text}\n\n"
                "Current user message:\n"
                f"{user_query}"
            )
        else:
            user_prompt = user_query
        return await self.llm.generate_text(system_prompt, user_prompt)

    # TODO: think about better wording
    @staticmethod
    def _relation_line(
        rel: Relationship,
        *,
        current_object_id: int,
        object_by_id: dict[int, TrackedObjectState],
    ) -> str:
        if rel.subject_id == current_object_id:
            other_id = rel.object_id
            other_label = (
                object_by_id.get(other_id).label
                if other_id in object_by_id
                else "object"
            )
            return f"{rel.predicate} -> {other_label}_{other_id}"
        other_id = rel.subject_id
        other_label = (
            object_by_id.get(other_id).label if other_id in object_by_id else "object"
        )
        return f"<- {rel.predicate} from {other_label}_{other_id}"

    async def object_chat(
        self,
        user_query: str,
        *,
        object_label: str,
        conversation_history: list[tuple[str, str]] | None = None,
        max_instances: int | None = None,
        max_crop_fallbacks: int | None = None,
        caption_crop_callback: Callable[[bytes], Awaitable[str]] | None = None,
    ) -> tuple[str, list[int], list[int], str]:
        state = await self._get_scene_state()
        normalized_requested_label = self._normalize_label(object_label)

        matched_objects = [
            obj
            for obj in state.objects
            if self._normalize_label(obj.label) == normalized_requested_label
        ]
        if not matched_objects and normalized_requested_label:
            matched_objects = [
                obj
                for obj in state.objects
                if normalized_requested_label in self._normalize_label(obj.label)
                or self._normalize_label(obj.label) in normalized_requested_label
            ]

        matched_objects = sorted(
            matched_objects,
            key=self._object_salience_key,
            reverse=True,
        )
        if max_instances is not None:
            matched_objects = matched_objects[:max_instances]
        logger.info("ObjectChat: %d matched objects", len(matched_objects))
        source_object_ids = [obj.id for obj in matched_objects]
        object_by_id = {obj.id: obj for obj in state.objects}

        fallback_caption_by_object_id: dict[int, str] = {}
        crop_fallback_used_ids: list[int] = []
        # TODO: Think about better wording
        object_context_lines = [
            "Object-focused memory context:",
            f"Requested object label: {object_label}",
            f"Matched instances: {len(matched_objects)}",
        ]

        if not matched_objects:
            available_labels = sorted(
                {
                    self._normalize_label(obj.label)
                    for obj in state.objects
                    if self._normalize_label(obj.label)
                }
            )
            object_context_lines.append(
                "No objects matched the requested label in current memory."
            )
            if available_labels:
                object_context_lines.append(
                    "Available labels in memory: " + ", ".join(available_labels)
                )
        else:
            for obj in matched_objects:
                # TODO: think about more than one-hop away dist.
                obj_relations = [
                    rel
                    for rel in state.relationships
                    if rel.subject_id == obj.id or rel.object_id == obj.id
                ]
                attrs = list(obj.attributes or [])
                has_structured_facts = bool(attrs) or bool(obj_relations)
                # THIS!: if no facts about it, send to caption model to at least some info
                if (
                    not has_structured_facts
                    and caption_crop_callback is not None
                    and (
                        max_crop_fallbacks is None
                        or len(crop_fallback_used_ids) < max_crop_fallbacks
                    )
                ):
                    crop_bytes = await self._get_track_crop(obj.id)
                    if crop_bytes is not None:
                        try:
                            caption_text = (
                                await caption_crop_callback(crop_bytes)
                            ).strip()
                        except Exception as exc:
                            logger.warning(
                                "Object crop caption fallback failed for object_id=%s: %s",
                                obj.id,
                                exc,
                            )
                            caption_text = ""
                        if caption_text:
                            fallback_caption_by_object_id[obj.id] = caption_text
                            crop_fallback_used_ids.append(obj.id)

                object_context_lines.append(
                    f"- ID {obj.id}: {obj.label} (hits={obj.hits}, last_seen={obj.last_seen:.3f})"
                )
                if attrs:
                    object_context_lines.append(
                        "  attributes: " + ", ".join(sorted(set(attrs)))
                    )
                else:
                    object_context_lines.append("  attributes: none")

                if obj_relations:
                    relation_lines = [
                        self._relation_line(
                            rel,
                            current_object_id=obj.id,
                            object_by_id=object_by_id,
                        )
                        for rel in obj_relations
                    ]
                    object_context_lines.append(
                        "  relations: " + "; ".join(relation_lines)
                    )
                else:
                    object_context_lines.append("  relations: none")

                fallback_caption = fallback_caption_by_object_id.get(obj.id)
                if fallback_caption:
                    object_context_lines.append(
                        "  visual_fallback_description: " + fallback_caption
                    )

        object_context = "\n".join(object_context_lines)
        resolved_label = matched_objects[0].label if matched_objects else object_label

        history_text = self._format_history(conversation_history)
        base_prompt_values = await self._base_prompt_values()
        object_prompt_values = dict(base_prompt_values)
        object_prompt_values.update(
            {
                "query": user_query,
                "history": history_text,
                "object_label": object_label,
                "resolved_label": resolved_label,
                "matched_ids": (
                    ", ".join(str(object_id) for object_id in source_object_ids)
                    if source_object_ids
                    else "none"
                ),
                "matched_count": str(len(source_object_ids)),
                "scene_context": base_prompt_values.get("context", ""),
                "context": object_context,
                "object_context": object_context,
            }
        )
        system_prompt_template = self.object_system_prompt or self.system_prompt
        system_prompt = (
            render_prompt_template(
                system_prompt_template,
                object_prompt_values,
            )
            or system_prompt_template
        )

        if self.object_user_prompt:
            user_prompt = (
                render_prompt_template(
                    self.object_user_prompt,
                    object_prompt_values,
                )
                or self.object_user_prompt
            )
        elif history_text:
            user_prompt = (
                "Conversation so far:\n"
                f"{history_text}\n\n"
                f"{object_context}\n\n"
                "Current user message:\n"
                f"{user_query}"
            )
        else:
            user_prompt = f"{object_context}\n\nCurrent user message:\n{user_query}"

        response = await self.llm.generate_text(system_prompt, user_prompt)
        return response, source_object_ids, crop_fallback_used_ids, resolved_label
