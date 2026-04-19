import re

_WHITESPACE_RE = re.compile(r"\s+")
_WHITESPACE_RE_UNICODE = re.compile(ur"\s+")


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
        text = text[:max_chars - 1].rstrip() + "..."
    # deffensive text output
    return text or ""


def sanitize_query(query, max_chars):
    return clean_text(query, max_chars=max_chars)


def clean_text_unicode(text, max_chars=None):
    if text is None:
        return u""

    # Normalize to unicode
    if isinstance(text, str):
        try:
            text = text.decode("utf-8")
        except Exception:
            text = text.decode("latin-1", "ignore")
    elif not isinstance(text, unicode):
        text = unicode(text)

    # All operations use unicode literals
    text = text.replace(u"\n", u" ").replace(u"\r", u" ")
    text = _WHITESPACE_RE_UNICODE.sub(u" ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars - 1].rstrip() + u"..."

    return text or u""
