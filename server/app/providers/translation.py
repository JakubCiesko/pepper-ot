from googletrans import Translator

# TODO: add translate gemma? otherwise enforce translation via prompt engineering
# I think translation is best to be kept as the last layer, and the overall server logic should run in english (no translation of detection classes, and english-only ontologies)


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
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = Translator(
            # service_urls=self.URLS
        )

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
        self, text: str | list[str], language: str | None = None
    ) -> str | list[str]:
        language = language or self.target_lang or self.DEFAULT_LANGS["dest"]
        languages = await self.detect_language(text)
        if all(lang == language for lang in languages):
            return text
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
        return texts if len(texts) > 1 else texts[0]

    async def translate(
        self,
        text: str | list[str],
        source_lang: str | None = None,
        target_lang: str | None = None,
        run_checks: bool = True,
    ) -> tuple[list[str], bool]:
        # if nothing passed or initialized, default fallback to en->cs translation
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
        translation = await self.translator.translate(text, src=src, dest=dest)
        translation_output = (
            [trans.text for trans in translation]
            if isinstance(translation, list)
            else translation.text
        )
        translation_ok = (
            await self.run_translation_checks(translation_output, dest)
            if run_checks
            else True
        )
        if run_checks and not translation_ok:
            translation_output, translation_ok = await self.translate(
                text, source_lang, target_lang, run_checks
            )
        return translation_output, translation_ok


english_to_czech = TranslationService(source_lang="en", target_lang="cs")
czech_to_english = TranslationService(source_lang="cs", target_lang="en")
