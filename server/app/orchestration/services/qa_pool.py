from __future__ import annotations

import asyncio
import logging
import threading
import time

# TODO: this is done in the outside-facing parts of the code, thing whether not put to the system interla parts.
from app.providers.translation import enforce_output_language

logger = logging.getLogger(__name__)


class QAPoolService:
    def __init__(self, *, max_entries: int = 200):
        self._lock = threading.RLock()
        self._items: list[dict[str, object]] = []
        self._max_entries = max(1, int(max_entries))

    @staticmethod
    def _normalize_question(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _sanitize_text(value: object) -> str:
        return str(value or "").strip()

    def set_max_entries(self, max_entries: int):
        max_entries = max(1, int(max_entries))
        with self._lock:
            self._max_entries = max_entries
            if len(self._items) > self._max_entries:
                self._items = self._items[-self._max_entries :]

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self):
        with self._lock:
            self._items = []

    def _upsert_locked(self, entry: dict[str, object]):
        question_en = self._sanitize_text(entry.get("question_en"))
        answer_en = self._sanitize_text(entry.get("answer_en"))
        if not question_en or not answer_en:
            return
        normalized = self._normalize_question(question_en)
        existing_idx: int | None = None
        for idx, item in enumerate(self._items):
            if self._normalize_question(str(item.get("question_en", ""))) == normalized:
                existing_idx = idx
                break
        payload = {
            "question_en": question_en,
            "answer_en": answer_en,
            "question_cs": self._sanitize_text(entry.get("question_cs")),
            "answer_cs": self._sanitize_text(entry.get("answer_cs")),
            "created_at": float(entry.get("created_at") or time.time()),
            "frame_id": self._sanitize_text(entry.get("frame_id")) or None,
            "scan_id": self._sanitize_text(entry.get("scan_id")) or None,
            "source": self._sanitize_text(entry.get("source"))
            or "pipeline_scene_graph",
        }
        if existing_idx is not None:
            self._items.pop(existing_idx)
        self._items.append(payload)
        if len(self._items) > self._max_entries:
            self._items = self._items[-self._max_entries :]

    def ingest_generated_pairs(
        self,
        pairs: list[dict[str, str]],
        *,
        frame_id: str | None = None,
        scan_id: str | None = None,
        source: str = "pipeline_scene_graph",
    ):
        if not pairs:
            return
        created_at = time.time()
        with self._lock:
            for pair in pairs:
                self._upsert_locked(
                    {
                        "question_en": pair.get("question", ""),
                        "answer_en": pair.get("answer", ""),
                        "question_cs": pair.get("question_cs", ""),
                        "answer_cs": pair.get("answer_cs", ""),
                        "created_at": created_at,
                        "frame_id": frame_id,
                        "scan_id": scan_id,
                        "source": source,
                    }
                )

    def replace_items(self, items: list[dict[str, object]]):
        with self._lock:
            self._items = []
            for item in items:
                self._upsert_locked(item)

    def snapshot_bilingual(
        self, *, limit: int | None = None
    ) -> list[dict[str, object]]:
        with self._lock:
            items = list(self._items)
        items.reverse()
        if limit is not None:
            items = items[: max(0, int(limit))]
        return [dict(item) for item in items]

    async def snapshot_pairs(
        self, *, language: str = "english", limit: int | None = None
    ) -> list[dict[str, str]]:
        normalized_language = str(language or "english").strip().lower()
        if normalized_language in {"cs", "czech"}:
            return await self._snapshot_czech(limit=limit)
        return self._snapshot_english(limit=limit)

    def _snapshot_english(self, *, limit: int | None = None) -> list[dict[str, str]]:
        bilingual = self.snapshot_bilingual(limit=limit)
        return [
            {
                "question": str(item.get("question_en", "")).strip(),
                "answer": str(item.get("answer_en", "")).strip(),
            }
            for item in bilingual
            if str(item.get("question_en", "")).strip()
            and str(item.get("answer_en", "")).strip()
        ]

    async def _snapshot_czech(
        self, *, limit: int | None = None
    ) -> list[dict[str, str]]:
        bilingual = self.snapshot_bilingual(limit=limit)
        prepared: list[dict[str, str]] = []
        translation_tasks: list[tuple[int, str, asyncio.Task[object]]] = []

        for item in bilingual:
            q_en = str(item.get("question_en", "")).strip()
            a_en = str(item.get("answer_en", "")).strip()
            if not q_en or not a_en:
                continue

            q_cs = str(item.get("question_cs", "")).strip()
            a_cs = str(item.get("answer_cs", "")).strip()

            entry_idx = len(prepared)
            prepared.append(
                {
                    "question_en": q_en,
                    "answer_en": a_en,
                    "question_cs": q_cs,
                    "answer_cs": a_cs,
                }
            )

            if not q_cs:
                translation_tasks.append(
                    (
                        entry_idx,
                        "question_cs",
                        asyncio.create_task(enforce_output_language(q_en, "czech")),
                    )
                )
            if not a_cs:
                translation_tasks.append(
                    (
                        entry_idx,
                        "answer_cs",
                        asyncio.create_task(enforce_output_language(a_en, "czech")),
                    )
                )

        if translation_tasks:
            results = await asyncio.gather(
                *(task for _, _, task in translation_tasks),
                return_exceptions=True,
            )
            for (entry_idx, target_key, _), result in zip(
                translation_tasks, results, strict=True
            ):
                entry = prepared[entry_idx]
                fallback = (
                    entry["question_en"]
                    if target_key == "question_cs"
                    else entry["answer_en"]
                )
                if isinstance(result, Exception):
                    if target_key == "question_cs":
                        logger.debug("Czech QA question translation failed: %s", result)
                    else:
                        logger.debug("Czech QA answer translation failed: %s", result)
                    entry[target_key] = fallback
                else:
                    entry[target_key] = str(result).strip()

        out: list[dict[str, str]] = []
        to_cache: list[tuple[str, str, str, str]] = []
        for entry in prepared:
            q_en = entry["question_en"]
            a_en = entry["answer_en"]
            q_cs = entry["question_cs"]
            a_cs = entry["answer_cs"]
            out.append({"question": q_cs, "answer": a_cs})
            to_cache.append((q_en, q_cs, a_en, a_cs))

        if to_cache:
            with self._lock:
                for q_en, q_cs, a_en, a_cs in to_cache:
                    key = self._normalize_question(q_en)
                    for item in self._items:
                        if (
                            self._normalize_question(str(item.get("question_en", "")))
                            != key
                        ):
                            continue
                        if not str(item.get("question_cs", "")).strip():
                            item["question_cs"] = q_cs
                        if not str(item.get("answer_cs", "")).strip():
                            item["answer_cs"] = a_cs
                        if not str(item.get("answer_en", "")).strip():
                            item["answer_en"] = a_en
                        break
        return out
