# -*- coding: utf-8 -*-
import re
import random

TEXT_TYPES = (str, bytes, unicode)

_ACK = {
    "look": {
        "en": ["Let me look.", "I'll take a look.", "Checking now."],
        "cs": ["Podívám se.", "Chvilku počkej, podívám se.", "Mrknu na to."],
    },
    "scan": {
        "en": ["I will scan around me.", "Scanning now.", "Let me see what's around."],
        "cs": ["Rozhlednu se kolem sebe.", "Teď skenuju.", "Podívám se, co je kolem.", "Tohle chvilku potrvá. Jdu na to."],
    },
    "ask": {
        "en": ["Let me think about that.", "I need a moment to consider.", "Thinking"],
        "cs": ["Podívám se na to.", "Chvilku přemýšlím.", "Přemýšlím", "Hm",],
    },
    "reset": {
        "en": ["I will start a new conversation.", "Resetting conversation.", "Let's start over."],
        "cs": ["Zacnu novou konverzaci.", "Resetuji konverzaci.", "Začneme znovu."],
    },
    "dashboard": {
        "en": ["I will show the dashboard.", "Opening the dashboard.", "Displaying the dashboard."],
        "cs": ["Ukážu dashboard.", "Otevírám dashboard.", "Zobrazuji dashboard."],
    },
    "greet": {
        "en": ["Hello!", "Hi there!", "Greetings!"],
        "cs": ["Ahoj!", "Dobrý den!", "Zdravím!"],
    },
    "thanks": {
        "en": ["You're welcome!", "No problem!", "Glad to help!"],
        "cs": ["Není zač!", "Rádo se stalo!", "Těší mě, že mohu pomoci!"],
    },
    "farewell": {
        "en": ["Goodbye!", "See you later!", "Take care!"],
        "cs": ["Nashledanou!", "Uvidíme se později!", "Měj se hezky!"],
    },
    "confirm": {
        "en": ["Understood.", "Got it.", "Acknowledged."],
        "cs": ["Rozumím.", "Chápu.", "Beru na vědomí."],
    },
    "error": {
        "en": ["Something went wrong.", "I encountered an error.", "Oops, an error occurred."],
        "cs": ["Něco se pokazilo.", "Došlo k chybě.", "Jejda, nastala chyba."],
    },
}

_GENERIC = {
    "scan_complete": {
        "en": [
            "My view is updated.",
            "Scanning complete.",
            "I have finished scanning.",
            "The environment is now observed."
        ],
        "cs": [
            "Můj pohled je aktuální.",
            "Skenování dokončeno.",
            "Dokončil jsem skenování.",
            "Okolí bylo prozkoumáno."
        ],
    },
    "language_updated": {
        "en": [
            "Output language updated.",
            "The language has been changed.",
            "Language settings updated.",
            "System output language is now set."
        ],
        "cs": [
            "Výstupní jazyk byl změněn.",
            "Jazyk byl aktualizován.",
            "Nastavení jazyka bylo upraveno.",
            "Od teď používám nový jazyk."
        ],
    },
    "system_ready": {
        "en": [
            "System is ready.",
            "All systems operational.",
            "I am ready to proceed.",
            "I am online and ready."
        ],
        "cs": [
            "Systém je připraven.",
            "Všechny systémy jsou funkční.",
            "Jsem připraven pokračovat.",
            "Jsem onlajn a připraven."
        ],
    },
    "action_confirmed": {
        "en": [
            "Action confirmed.",
            "Understood and confirmed.",
            "I will proceed as requested.",
            "Confirmed, executing now."
        ],
        "cs": [
            "Akce potvrzena.",
            "Rozumím a potvrzuji.",
            "Budou provedeny požadované kroky.",
            "Potvrzeno, nyní provádím."
        ],
    },
    "waiting": {
        "en": [
            "Please wait a moment.",
            "Hold on, processing.",
            "I am working on it.",
            "Give me a second, please."
        ],
        "cs": [
            "Počkejte prosím chvíli.",
            "Chvilku počkejte, teďkom pracuji.",
            "Pracuji na tom.",
            "Dejte mi prosím vteřinu."
        ],
    },
    "error_occurred": {
        "en": [
            "An error occurred.",
            "Something went wrong.",
            "I encountered a problem.",
            "Oops, there was an issue."
        ],
        "cs": [
            "Došlo k chybě.",
            "Něco se pokazilo.",
            "Narazil jsem na problém.",
            "Jejda, nastal problém."
        ],
    },
    "task_completed": {
        "en": [
            "Task completed successfully.",
            "All steps finished.",
            "I have completed the task.",
            "Task execution finished."
        ],
        "cs": [
            "Úkol byl úspěšně dokončen.",
            "Všechny kroky jsou hotové.",
            "Dokončil jsem úkol.",
            "Provádění úkolu dokončeno."
        ],
    },
    "idle": {
        "en": [
            "I am idle.",
            "Standing by.",
            "Awaiting instructions.",
            "I am ready for the next task."
        ],
        "cs": [
            "Jsem nečinný.",
            "Čekám.",
            "Čekám na instrukce.",
            "Jsem připraven na další úkol."
        ],
    },
}

_WHITESPACE_RE = re.compile(r"\s+")


def language_code(value, default="en"):
    lang = str(value or default).strip().lower()
    if lang.startswith("cs"):
        return "cs"
    return "en"


def pick_random(pick_from, kind, lang_code):
    lang = language_code(lang_code)
    options = pick_from.get(kind, {}).get(lang, [])
    if len(options) > 0:
        return random.choice(options)
    return ""

def acknowledgement(kind, lang_code):
    return pick_random(_ACK, kind, lang_code)

def generic_message(kind, lang_code):
    return pick_random(_GENERIC, kind, lang_code)


def clean_text(text, max_chars=None):
    if text is None:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except Exception:
            text = str(text)
    else:
        try:
            text = text.encode("utf-8")
        except Exception:
            text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    return text


def sanitize_query(query, max_chars):
    return clean_text(query, max_chars=max_chars)
