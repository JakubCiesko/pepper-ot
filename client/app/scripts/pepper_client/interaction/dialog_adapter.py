from pepper_client.interaction import speech_policy
from pepper_client.utils import text as text_utils


class DialogAdapter(object):
    """Thin ALDialog wrapper for runtime dynamic concept refresh."""

    def __init__(self, services, config, logger):
        self.dialog = services.ALDialog
        self.tts = services.ALTextToSpeech
        self.config = config
        self.logger = logger

    def update_config(self, config):
        self.config = config

    def resolve_dialog_language(self, lang_code=None):
        _, runtime_language = speech_policy.resolve_language_state(
            self.config,
            requested=lang_code,
            tts=self.tts,
            logger=self.logger,
        )
        return speech_policy.dialog_language_for_runtime(runtime_language)

    def set_dynamic_concept(self, name, values, language=None):
        if self.dialog is None:
            self.logger.info(
                "ALDialog unavailable, skipping concept update for %s", name)
            return False
        language = self.resolve_dialog_language(language)
        # TODO: here is the problem
        cleaned = self._clean_values(values)
        # TODO: do i need to clear them?
        if not cleaned:
            return self.clear_dynamic_concept(name, language)

        # ALDialog API can differ across NAOqi bindings, so we try common shapes.
        # TODO: look at this properly
        attempts = [
            (name, language, cleaned),
            (name, language, [[value] for value in cleaned]),
            (name, cleaned, language),
            (name, [[value] for value in cleaned], language),
        ]
        for args in attempts:
            try:
                self.dialog.setConcept(*args)
                self.logger.info(
                    "Updated dynamic concept %s (%s entries, lang=%s), %s = %s",
                    name,
                    len(cleaned),
                    language,
                    name,
                    args,
                )
                return True
            except Exception as exc:
                self.logger.info(
                    "setConcept attempt failed for %s args_shape=%s: %s",
                    name,
                    len(args),
                    exc,
                )
        self.logger.warning("Failed to update dynamic concept %s", name)
        return False

    def clear_dynamic_concept(self, name, language=None):
        if self.dialog is None:
            return False
        language = self.resolve_dialog_language(language)
        for payload in ([], [[]]):
            try:
                self.dialog.setConcept(name, language, payload)
                self.logger.info("Cleared dynamic concept %s", name)
                return True
            except Exception:
                continue
        self.logger.warning("Failed to clear dynamic concept %s", name)
        return False

    def refresh_memory_concepts(self,
                                labels,
                                attributes,
                                relations,
                                cached_questions=None):
        if self.dialog is None:
            return False
        dialog_cfg = self.config.get("dialog", {})
        if not dialog_cfg.get("enable_dynamic_memory_concepts", True):
            self.logger.info("Dynamic concept refresh disabled by config")
            return False

        labels = self._cap(labels, dialog_cfg.get("memory_objects_max"))
        attributes = self._cap(attributes,
                               dialog_cfg.get("memory_attributes_max"))
        relations = self._cap(relations,
                              dialog_cfg.get("memory_relations_max"))
        cached_questions = self._cap(
            cached_questions or [],
            dialog_cfg.get("memory_cached_questions_max"),
        )

        ok_objects = self.set_dynamic_concept("memory_objects", labels)
        ok_attrs = self.set_dynamic_concept("memory_attributes", attributes)
        ok_rels = self.set_dynamic_concept("memory_relations", relations)
        ok_qa = self.set_dynamic_concept("memory_cached_questions",
                                         cached_questions)
        self.logger.info(
            "Refreshed memory concepts objects=%s attrs=%s rels=%s questions=%s",
            len(labels),
            len(attributes),
            len(relations),
            len(cached_questions),
        )
        return bool(ok_objects or ok_attrs or ok_rels or ok_qa)

    def _clean_values(self, values):
        if values is None:
            return []
        seen = set()
        cleaned = []
        for value in list(values):
            text = text_utils.clean_text(value)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

    def _cap(self, values, max_items):
        cleaned = self._clean_values(values)
        try:
            max_items = int(max_items)
        except Exception:
            max_items = 0
        if max_items <= 0:
            return cleaned
        return cleaned[:max_items]
