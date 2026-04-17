import re


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text, max_chars=None):
    if text is None:
        return ""

    if isinstance(text, str):
        try:
            text = text.decode("utf-8")
        except Exception:
            text = text.decode("latin-1", "ignore")

    elif not isinstance(text, unicode):
        text = unicode(text)

    text = text.replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    # deffensive text output
    return text or ""


def sanitize_query(query, max_chars):
    return clean_text(query, max_chars=max_chars)
