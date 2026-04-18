import threading

from pepper_client.utils.error_policy import SpeechError
from pepper_client.interaction import speech_policy
from pepper_client.utils import text as text_utils


class SpeechAdapter(object):
    LANGUAGE_MAP = {
        "en": "English",
        "cs": "Czech",
    }

    def __init__(self, services, logger):
        self.tts = services.ALTextToSpeech
        self.animated = services.ALAnimatedSpeech
        self.logger = logger
        self._lock = threading.RLock()
        self._last_tts_language = None

    def say(self, text, lang_code=None):
        text = text_utils.clean_text(text)
        if not text:
            self.logger.info("Skipping empty speech payload")
            return
        if self.tts is None and self.animated is None:
            raise SpeechError("No speech service available")
        with self._lock:
            self._apply_language(lang_code)
            if isinstance(text, unicode):
                text = text.encode("utf-8")
            self.logger.info("Speaking text: %s", text)
            # animated is primary
            if self.animated is not None:
                try:
                    self.animated.say(text)
                    return
                except Exception as exc:
                    self.logger.info(
                        "Animated speech unavailable, falling back to TTS: %s", exc
                    )
            if self.tts is None:
                raise SpeechError("No text to speech service available")
            try:
                self.tts.say(text)
            except Exception as exc:
                raise SpeechError(str(exc))

    def stop(self):
        with self._lock:
            if self.tts is not None and hasattr(self.tts, "stopAll"):
                try:
                    self.tts.stopAll()
                except Exception:
                    pass
            if self.animated is not None and hasattr(self.animated, "stopAll"):
                try:
                    self.animated.stopAll()
                except Exception:
                    pass

    def _apply_language(self, lang_code):
        if self.tts is None:
            return
        lang_code = speech_policy.language_code(lang_code)
        target = self.LANGUAGE_MAP.get(lang_code)
        if not target or target == self._last_tts_language:
            return
        try:
            available = self.tts.getAvailableLanguages()
        except Exception:
            available = []
        if target not in available:
            self.logger.info("TTS language %s not available", target)
            return
        try:
            self.tts.setLanguage(target)
            self._last_tts_language = target
            self.logger.info("TTS language set to %s", target)
        except Exception as exc:
            self.logger.info("Failed to set TTS language to %s: %s", target, exc)
