# -*- coding: utf-8 -*-
class PepperClientError(Exception):
    pass


class CameraCaptureError(PepperClientError):
    pass


class ServerUnavailableError(PepperClientError):
    pass


class ServerTimeoutError(PepperClientError):
    pass


class MalformedResponseError(PepperClientError):
    pass


class SpeechError(PepperClientError):
    pass

_MESSAGES = {
    "busy": {
        "en": "I am still processing the previous request.",
        "cs": "Stále zpracovávám předchozí požadavek.",
    },
    "camera": {
        "en": "I could not capture an image right now.",
        "cs": "Bohužel nepodařilo jse mi pořídiť záběr.",
    },
    "server_unavailable": {
        "en": "I cannot reach my vision server right now.",
        "cs": "Teď se mi nedaří spojit se serverem.",
    },
    "server_timeout": {
        "en": "The server response took too long.",
        "cs": "Server neodpovídá už dlouho.",
    },
    "malformed": {
        "en": "I received an invalid response from the server.",
        "cs": "Server vrátil neplatnou odpověď.",
    },
    "unexpected": {
        "en": "Something went wrong while I was processing that.",
        "cs": "Při zpracování došlo k chybě.",
    },
}


def fallback_message(kind, lang_code):
    lang = (lang_code or "en").strip().lower()
    if lang not in ("en", "cs"):
        lang = "en"
    return _MESSAGES.get(kind, _MESSAGES["unexpected"]).get(lang)
