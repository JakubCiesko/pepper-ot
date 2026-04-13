import logging

from googletrans import Translator

# TODO: add translate gemma? otherwise enforce translation via prompt engineering
# I think translation is best to be kept as the last layer, and the overall server logic should run in english (no translation of detection classes, and english-only ontologies)

logger = logging.getLogger(__name__)


class TranslationService:
    """TranslationService used (mainly) for simple czech-to-english and english-to-czech translation.
    More details and language codes: https://py-googletrans.readthedocs.io/en/latest/"""

    #    URLS = [
    #        "translate.google.com",
    #        "translate.google.cz",
    #        "translate.google.sk"
    #    ]
    DEFAULT_LANGS = {
        "src": "en",
        "dest": "cs",
    }

    def __init__(self, source_lang: str | None = None, target_lang: str | None = None):
        logger.info(
            "Initializing TranslationService(src_lang=%s, target_lang=%s)",
            source_lang,
            target_lang,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = Translator(
            # service_urls=self.URLS
        )
        logger.info("Initialized TranslationService(%s, %s)", source_lang, target_lang)

    async def detect_language(self, text: str | list[str]) -> list[str]:
        result = await self.translator.detect(text)
        if isinstance(result, list):
            return [r.lang for r in result]
        return [result.lang]

    async def run_translation_checks(
        self, translation_output: str | list[str], dest_lang: str
    ) -> bool:
        """Sometimes the translation can fail, in that case it is better to retry the translation"""
        langs = await self.detect_language(translation_output)
        return all(lang == dest_lang for lang in langs)

    async def enforce_language(
        self,
        text: str | list[str],
        language: str | None = None,
        return_languages: bool = False,
    ) -> tuple[str | list[str], list[str] | None]:
        logger.debug("Enforcing language %s, on text %s", language, text)
        language = language or self.target_lang or self.DEFAULT_LANGS["dest"]
        languages = await self.detect_language(text)
        normalized_languages = [expand_language_code(lang) for lang in languages]
        if all(lang == language for lang in languages):
            logger.info(
                "Text %s.. already in language %s. Bypassing translation. Returning text",
                text[:10],
                language,
            )
            return text, normalized_languages if return_languages else text
        # otherwise some texts are not translated, find out which
        wrong_language_text_indices = [
            i for i, lang in enumerate(languages) if lang != language
        ]
        # enforce list
        texts = [text] if isinstance(text, str) else text
        wrong_texts = [texts[i] for i in wrong_language_text_indices]
        fixed, _ = await self.translate(wrong_texts, "auto", language, run_checks=True)
        for fix, i in zip(fixed, wrong_language_text_indices, strict=True):
            texts[i] = fix
        if len(texts) > 1:
            return texts, normalized_languages if return_languages else texts
        return texts[0], normalized_languages if return_languages else texts[0]

    async def translate(
        self,
        text: str | list[str],
        source_lang: str | None = None,
        target_lang: str | None = None,
        run_checks: bool = True,
        max_retries: int = 2,
    ) -> tuple[str | list[str], bool]:
        # if nothing passed or initialized, default fallback to en->cs translation
        logger.debug(
            "Running translation from %s to %s languages for text: %s",
            source_lang,
            target_lang,
            text,
        )
        src = (
            (self.source_lang or self.DEFAULT_LANGS["src"])
            if source_lang is None
            else source_lang
        )
        dest = (
            (self.target_lang or self.DEFAULT_LANGS["dest"])
            if target_lang is None
            else target_lang
        )
        retries = max(0, int(max_retries))
        last_output: str | list[str] = text
        for _attempt in range(retries + 1):
            translation = await self.translator.translate(text, src=src, dest=dest)
            translation_output = (
                [trans.text for trans in translation]
                if isinstance(translation, list)
                else translation.text
            )
            last_output = translation_output
            if not run_checks:
                return translation_output, True
            translation_ok = await self.run_translation_checks(translation_output, dest)
            if translation_ok:
                return translation_output, True
        return last_output, False


def expand_language_code(language: str) -> str:
    lang = language.strip().lower()
    if lang == "en":
        return "english"
    if lang == "cs":
        return "czech"
    return language


# TODO: user might speak czech -> make it english -> pass to llm -> llm output enforce czech again


async def enforce_output_language(
    text: str, output_language: str | None, return_languages: bool = False
) -> str | tuple[str, str]:
    mode = (output_language or "default").strip().lower()
    if mode == "default":
        return text if not return_languages else text, [output_language]
    if mode == "english":
        logger.info("Enforcing english for text: %s", text[:25] + "...")
        translated = await czech_to_english.enforce_language(
            text, return_languages=return_languages
        )
        return (
            translated if isinstance(translated, str) else translated[0]
            if not return_languages
            else translated
        )
    if mode == "czech":
        logger.info("Enforcing czech for text: %s", text[:25] + "...")
        translated = await english_to_czech.enforce_language(
            text, return_languages=return_languages
        )
        return (
            translated if isinstance(translated, str) else translated[0]
            if not return_languages
            else translated
        )
    return text if not return_languages else text, [output_language]


english_to_czech = TranslationService(source_lang="en", target_lang="cs")
czech_to_english = TranslationService(source_lang="cs", target_lang="en")
