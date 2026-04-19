import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from typing import Iterable

from app.providers.translation.google_trans import english_to_czech
from app.schemas.config import AppConfig
# from app.schemas.scene import MemorySummary
# from app.schemas.scene import SceneGraphRelation
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


class VocabularyTranslationService:
    """Persistent token-level translation for robot-facing vocabulary."""

    SUPPORTED_TARGET_LANGS = {"cs"}
    TOKEN_KINDS = ("label", "attribute", "relation")
    STATIC_FILES = {
        "label": "labels_cs.json",
        "attribute": "attributes_cs.json",
        "relation": "relations_cs.json",
    }
    USER_FILES = {
        "label": "labels_cs.user.json",
        "attribute": "attributes_cs.user.json",
        "relation": "relations_cs.user.json",
    }

    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        user_lexicon_dir: Path | None = None,
    ):
        self._base_dir = base_dir or Path(__file__).resolve().parent
        self._lexicon_dir = self._base_dir / "lexicons"
        self._user_lexicon_dir = user_lexicon_dir or (self._base_dir / "lexicons_user")
        self._lock = asyncio.Lock()

        self._static_maps: dict[str, dict[str, dict[str, str]]] = {
            "cs": {
                kind: self._load_lexicon(self._lexicon_dir / filename)
                for kind, filename in self.STATIC_FILES.items()
            }
        }
        self._ensure_user_storage()
        self._user_maps: dict[str, dict[str, dict[str, str]]] = {
            "cs": {
                kind: self._load_lexicon(self._user_lexicon_dir / filename)
                for kind, filename in self.USER_FILES.items()
            }
        }
        self._runtime_signature: tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ] | None = None

    def _load_lexicon(self, path: Path) -> dict[str, str]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            logger.exception("Failed to load vocabulary lexicon: %s", path)
            return {}
        if not isinstance(raw, dict):
            logger.warning("Invalid lexicon format for %s (expected object)", path)
            return {}
        return self._normalize_map(raw)

    def _normalize_map(self, mapping: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in mapping.items():
            source = self._normalize_token(key)
            target = str(value or "").strip()
            if not source or not target:
                continue
            out[source] = target
        return out

    def _ensure_user_storage(self):
        self._user_lexicon_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.USER_FILES.values():
            path = self._user_lexicon_dir / filename
            if path.exists():
                continue
            path.write_text("{}\n", encoding="utf-8")

    def _write_json_atomic(self, path: Path, payload: dict[str, str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def _kind(self, token_type: str) -> str:
        kind = str(token_type or "").strip().lower()
        if kind in {"label", "labels", "object", "objects"}:
            return "label"
        if kind in {"attribute", "attributes", "attr", "attrs"}:
            return "attribute"
        return "relation"

    def normalize_language(self, language: str | None) -> str:
        lang = str(language or "en").strip().lower()
        if lang in {"cs", "cz", "czc", "czech"}:
            return "cs"
        return "en"

    def _normalize_token(self, value: str | None) -> str:
        return str(value or "").strip().lower()

    def _prepare_for_translation(self, token: str) -> str:
        return token.replace("_", " ")

    def _effective_map(self, language: str, kind: str) -> dict[str, str]:
        # User map intentionally overrides static defaults for dashboard adjustability.
        merged = dict(self._static_maps.get(language, {}).get(kind, {}))
        merged.update(self._user_maps.get(language, {}).get(kind, {}))
        return merged

    def _extend_terms(self, target: set[str], values):
        if values is None:
            return
        if isinstance(values, str):
            cleaned = values.strip()
            if cleaned:
                target.add(cleaned)
            return
        if isinstance(values, dict):
            for key, value in values.items():
                self._extend_terms(target, key)
                self._extend_terms(target, value)
            return
        if isinstance(values, Iterable):
            for item in values:
                self._extend_terms(target, item)

    def _collect_terms(
        self,
        cfg: AppConfig,
        base_dir: Path | None,
    ) -> dict[str, set[str]]:
        labels: set[str] = set()
        relations: set[str] = set()
        attributes: set[str] = set()

        try:
            detection_ontology = (
                cfg.detection.resolve_ontology(base_dir)
                if base_dir is not None
                else cfg.detection.ontology
            )
        except Exception:
            detection_ontology = cfg.detection.ontology
        self._extend_terms(labels, detection_ontology)

        vlm_ontology = cfg.scene_graph.vlm.ontology
        try:
            predicates, objects = (
                vlm_ontology.resolve(base_dir)
                if base_dir is not None
                else (vlm_ontology.predicates, vlm_ontology.objects)
            )
        except Exception:
            predicates, objects = vlm_ontology.predicates, vlm_ontology.objects

        # Current ontology may mix predicate + attribute vocabulary.
        self._extend_terms(relations, predicates)
        self._extend_terms(attributes, predicates)
        self._extend_terms(labels, objects)

        for rule in cfg.scene_graph.rules.rule_list:
            predicate = str(rule.predicate or "").strip()
            if predicate:
                relations.add(predicate)

        return {"label": labels, "relation": relations, "attribute": attributes}

    async def _bulk_translate_to_cs(self, terms: list[str]) -> dict[str, str]:
        if not terms:
            return {}
        prepared = [self._prepare_for_translation(term) for term in terms]
        try:
            translated, _ok = await english_to_czech.translate(
                prepared,
                source_lang="en",
                target_lang="cs",
                run_checks=False,
                max_retries=1,
            )
        except Exception:
            logger.exception("Bulk translation to cs failed")
            return {}
        if isinstance(translated, str):
            translated = [translated]
        if not isinstance(translated, list):
            return {}
        out: dict[str, str] = {}
        for source, target in zip(terms, translated, strict=False):
            source_clean = str(source or "").strip()
            if not source_clean:
                continue
            target_clean = str(target or "").strip() or source_clean
            out[source_clean] = target_clean
        return out

    async def _persist_user_entries(
        self,
        language: str,
        kind: str,
        entries: dict[str, str],
    ):
        if not entries:
            return
        language = self.normalize_language(language)
        if language not in self.SUPPORTED_TARGET_LANGS:
            return
        normalized_entries = self._normalize_map(entries)
        if not normalized_entries:
            return
        async with self._lock:
            target_map = self._user_maps.setdefault(language, {}).setdefault(kind, {})
            target_map.update(normalized_entries)
            path = self._user_lexicon_dir / self.USER_FILES[kind]
            self._write_json_atomic(path, target_map)

    async def _translate_missing_and_persist(
        self,
        language: str,
        kind: str,
        terms: list[str],
    ) -> dict[str, str]:
        language = self.normalize_language(language)
        if language != "cs":
            return {}
        translated = await self._bulk_translate_to_cs(terms)
        normalized: dict[str, str] = {}
        for source in terms:
            key = self._normalize_token(source)
            if not key:
                continue
            value = str(translated.get(source, source) or source).strip() or source
            normalized[key] = value
        await self._persist_user_entries(language, kind, normalized)
        return normalized

    async def warm_from_config(self, cfg: AppConfig, base_dir: Path | None):
        terms = self._collect_terms(cfg, base_dir)
        signature = (
            tuple(sorted(self._normalize_token(term) for term in terms["label"] if term)),
            tuple(
                sorted(
                    self._normalize_token(term) for term in terms["attribute"] if term
                )
            ),
            tuple(
                sorted(
                    self._normalize_token(term) for term in terms["relation"] if term
                )
            ),
        )
        if signature == self._runtime_signature:
            return

        language = "cs"
        for kind in self.TOKEN_KINDS:
            effective = self._effective_map(language, kind)
            missing: list[str] = []
            for raw in sorted(terms[kind]):
                cleaned = str(raw or "").strip()
                if not cleaned:
                    continue
                key = self._normalize_token(cleaned)
                if key in effective:
                    continue
                missing.append(cleaned)
            await self._translate_missing_and_persist(language, kind, missing)

        self._runtime_signature = signature

    async def translate_token(
        self,
        token: str,
        *,
        token_type: str,
        language: str | None,
    ) -> str:
        original = str(token or "").strip()
        if not original:
            return original
        lang = self.normalize_language(language)
        if lang not in self.SUPPORTED_TARGET_LANGS:
            return original
        kind = self._kind(token_type)
        key = self._normalize_token(original)
        if not key:
            return original

        effective = self._effective_map(lang, kind)
        if key in effective:
            return effective[key]

        await self._translate_missing_and_persist(lang, kind, [original])
        return self._effective_map(lang, kind).get(key, original)

    # async def translate_node_reference(
    #     self, node_reference: str, language: str | None
    # ) -> str:
    #     text = str(node_reference or "").strip()
    #     if not text:
    #         return text
    #     if "_" in text:
    #         prefix, suffix = text.rsplit("_", 1)
    #         if suffix.isdigit():
    #             translated_prefix = await self.translate_token(
    #                 prefix,
    #                 token_type="label",
    #                 language=language,
    #             )
    #             return f"{translated_prefix}_{suffix}"
    #     return await self.translate_token(
    #         text,
    #         token_type="label",
    #         language=language,
    #     )

    # async def localize_memory_summary(
    #     self,
    #     summary: MemorySummary,
    #     *,
    #     language: str | None,
    # ) -> MemorySummary:
    #     target_language = self.normalize_language(language)
    #     if target_language == "en":
    #         return summary

    #     translated_labels: list[str] = []
    #     seen_labels: set[str] = set()
    #     for label in summary.labels or []:
    #         translated = await self.translate_token(
    #             label,
    #             token_type="label",
    #             language=target_language,
    #         )
    #         if translated and translated not in seen_labels:
    #             translated_labels.append(translated)
    #             seen_labels.add(translated)

    #     translated_label_counts: dict[str, int] = {}
    #     for label, count in (summary.label_counts or {}).items():
    #         translated = await self.translate_token(
    #             label,
    #             token_type="label",
    #             language=target_language,
    #         )
    #         if not translated:
    #             continue
    #         translated_label_counts[translated] = translated_label_counts.get(
    #             translated, 0
    #         ) + int(count)

    #     translated_graph: list[SceneGraphRelation] = []
    #     for edge in summary.scene_graph or []:
    #         token_type = "attribute" if edge.sub == edge.obj else "relation"
    #         translated_graph.append(
    #             SceneGraphRelation(
    #                 sub=await self.translate_node_reference(edge.sub, target_language),
    #                 rel=await self.translate_token(
    #                     edge.rel,
    #                     token_type=token_type,
    #                     language=target_language,
    #                 ),
    #                 obj=await self.translate_node_reference(edge.obj, target_language),
    #             )
    #         )

    #     return MemorySummary(
    #         timestamp=summary.timestamp,
    #         labels=translated_labels,
    #         label_counts=translated_label_counts,
    #         scene_graph=translated_graph,
    #         graph_svg=summary.graph_svg,
    #         pregenerated_qa=summary.pregenerated_qa,
    #     )

    async def build_memory_display_overrides(
        self,
        state: SceneState,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        target_language = self.normalize_language(language)
        if target_language == "en":
            return {
                "object_labels": {},
                "object_attributes": {},
                "relation_labels": {},
            }

        request_cache: dict[tuple[str, str], str] = {}

        async def resolve(kind: str, token: str) -> str:
            normalized = self._normalize_token(token)
            cache_key = (kind, normalized)
            if cache_key in request_cache:
                return request_cache[cache_key]
            value = await self.translate_token(
                token,
                token_type=kind,
                language=target_language,
            )
            request_cache[cache_key] = value
            return value

        object_labels: dict[int, str] = {}
        object_attributes: dict[int, list[str]] = {}
        relation_labels: dict[tuple[int, str, int], str] = {}

        for obj in state.objects:
            object_labels[obj.id] = await resolve("label", obj.label)
            translated_attributes: list[str] = []
            seen: set[str] = set()
            for attr in obj.attributes or []:
                translated_attr = await resolve("attribute", attr)
                if translated_attr in seen:
                    continue
                seen.add(translated_attr)
                translated_attributes.append(translated_attr)
            object_attributes[obj.id] = translated_attributes

        for rel in state.relationships:
            relation_labels[(rel.subject_id, rel.predicate, rel.object_id)] = (
                await resolve("relation", rel.predicate)
            )

        return {
            "object_labels": object_labels,
            "object_attributes": object_attributes,
            "relation_labels": relation_labels,
        }

    def get_dashboard_payload(self) -> dict[str, Any]:
        language = "cs"
        labels_map = self._effective_map(language, "label")
        attributes_map = self._effective_map(language, "attribute")
        relations_map = self._effective_map(language, "relation")
        return {
            "active": {
                "labels": {language: dict(sorted(labels_map.items()))},
                "attributes": {language: dict(sorted(attributes_map.items()))},
                "relations": {language: dict(sorted(relations_map.items()))},
            },
            "meta": {
                "language": language,
                "counts": {
                    "labels": len(labels_map),
                    "attributes": len(attributes_map),
                    "relations": len(relations_map),
                },
                "user_counts": {
                    "labels": len(self._user_maps.get(language, {}).get("label", {})),
                    "attributes": len(
                        self._user_maps.get(language, {}).get("attribute", {})
                    ),
                    "relations": len(
                        self._user_maps.get(language, {}).get("relation", {})
                    ),
                },
                "files": {
                    "labels": str(self._user_lexicon_dir / self.USER_FILES["label"]),
                    "attributes": str(
                        self._user_lexicon_dir / self.USER_FILES["attribute"]
                    ),
                    "relations": str(
                        self._user_lexicon_dir / self.USER_FILES["relation"]
                    ),
                },
            },
        }

    def _extract_cs_map(self, section: dict[str, Any], plural_key: str) -> dict[str, str]:
        if not isinstance(section, dict):
            raise ValueError(f"translations.{plural_key} must be an object")
        if "cs" in section:
            cs_map = section.get("cs")
            if not isinstance(cs_map, dict):
                raise ValueError(f"translations.{plural_key}.cs must be an object")
            return self._normalize_map(cs_map)
        return self._normalize_map(section)

    async def apply_dashboard_patch(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("translations patch must be an object")

        normalized: dict[str, dict[str, str]] = {}
        if "labels" in payload:
            normalized["label"] = self._extract_cs_map(payload["labels"], "labels")
        if "attributes" in payload:
            normalized["attribute"] = self._extract_cs_map(
                payload["attributes"], "attributes"
            )
        if "relations" in payload:
            normalized["relation"] = self._extract_cs_map(
                payload["relations"], "relations"
            )
        if not normalized:
            raise ValueError(
                "translations patch must contain labels and/or attributes and/or relations"
            )

        async with self._lock:
            for kind, values in normalized.items():
                self._user_maps.setdefault("cs", {})[kind] = values
                path = self._user_lexicon_dir / self.USER_FILES[kind]
                self._write_json_atomic(path, values)
            self._runtime_signature = None
        return self.get_dashboard_payload()["meta"]


vocabulary_translator = VocabularyTranslationService()
