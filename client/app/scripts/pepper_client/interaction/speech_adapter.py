import threading

from pepper_client.interaction import speech_policy
from pepper_client.utils import text as text_utils
from pepper_client.utils.error_policy import SpeechError


class SpeechAdapter(object):

    def __init__(self, services, config, logger):
        self.tts = services.ALTextToSpeech
        self.animated = services.ALAnimatedSpeech
        self.config = config
        self.logger = logger
        self._lock = threading.RLock()

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
                        "Animated speech unavailable, falling back to TTS: %s",
                        exc)
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
        mode, _ = speech_policy.resolve_language_state(
            self.config,
            requested=lang_code,
            tts=self.tts,
            logger=self.logger,
        )
        target = speech_policy.tts_language_for_mode(mode)
        if mode == "auto" or not target:
            return
        try:
            available = self.tts.getAvailableLanguages()
        except Exception:
            available = []
        if target not in available:
            self.logger.info("TTS language %s not available", target)
            return
        try:
            current = self.tts.getLanguage()
        except Exception:
            current = None
        if current == target:
            return
        try:
            #TODO: never do this?
            self.tts.setLanguage(target)
            self.logger.info("TTS language set to %s", target)
        except Exception as exc:
            self.logger.info("Failed to set TTS language to %s: %s", target,
                             exc)
