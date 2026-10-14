"""Content projection (MODEL.md §8.4): addresses extracted from free text by syntax only —
URL, e-mail, IBAN — in a canonical form per kind. No model, no meaning: a token is what the
grammar below matches, nothing else.

MODEL.md §8.14: a token with no scheme and no ``www.`` is a domain only if its last label is a
top-level domain of the Public Suffix List, is not a common file extension, and the token is not
one of the caller's own file names (``feedback.xlsx``, ``notes.md`` and ``run.py`` are files)."""
from __future__ import annotations

import re

from axor_core.policy.tlds import TLDS

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
# a URL: a scheme, or www., or a dotted host ending in an alphabetic TLD; an optional path
_URL = re.compile(
    r"(?:(?:https?://)|(?:www\.))[^\s<>\"')]+"
    r"|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>\"')]*)?")
_TRAIL = ".,;:!?)'\"]}>"
_EXPLICIT = re.compile(r"^(?:https?://|www\.)", re.I)
# file extensions: a bare name ending in one is a file, even where the extension is also a TLD
FILE_EXTENSIONS = frozenset({
    "doc", "docx", "odt", "rtf", "txt", "md", "pdf", "tex",
    "xls", "xlsx", "ods", "csv", "tsv",
    "ppt", "pptx", "odp", "key",
    "json", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "log",
    "html", "htm", "css", "js", "ts", "py", "rb", "go", "rs", "java", "c", "cpp", "h", "sh", "sql",
    "zip", "tar", "gz", "tgz", "rar", "7z",
    "png", "jpg", "jpeg", "gif", "svg", "bmp", "webp", "heic",
    "mp3", "mp4", "mov", "avi", "mkv", "wav", "m4a",
    "eml", "msg", "ics", "vcf", "bak", "tmp",
})


def canonical(kind: str, token: str) -> str:
    t = token.strip().rstrip(_TRAIL)
    if kind == "email":
        return t.lower()
    if kind == "iban":
        return t.replace(" ", "").upper()
    t = re.sub(r"^https?://", "", t, flags=re.I)
    t = re.sub(r"^www\.", "", t, flags=re.I)
    host, _, path = t.partition("/")
    return (host.lower() + ("/" + path if path else "")).rstrip("/")


def is_domain(token: str, own_names: frozenset[str] = frozenset()) -> bool:
    """A bare (no scheme, no ``www.``) host token is a domain (§8.14): a PSL top-level domain
    as its last label, not a file extension, not one of the own file names."""
    host = token.split("/", 1)[0].lower().rstrip(".")
    if host in own_names or token.lower().rstrip("/") in own_names:
        return False
    tld = host.rsplit(".", 1)[-1]
    return tld in TLDS and tld not in FILE_EXTENSIONS


def extract(text: object, own_names: "frozenset[str] | set[str]" = frozenset()) -> set[tuple[str, str]]:
    """{(kind, canonical)} for every URL / e-mail / IBAN in ``text`` (strings in lists and
    dicts included). ``own_names``: the user's own file names, never domains (§8.14)."""
    out: set[tuple[str, str]] = set()
    own = frozenset(n.lower() for n in own_names)
    if isinstance(text, dict):
        for v in text.values():
            out |= extract(v, own)
        return out
    if isinstance(text, (list, tuple, set, frozenset)):
        for v in text:
            out |= extract(v, own)
        return out
    if not isinstance(text, str) or not text:
        return out
    emails = list(_EMAIL.finditer(text))
    for m in emails:
        out.add(("email", canonical("email", m.group(0))))
    masked = text
    for m in emails:  # an e-mail's domain is not a URL
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    for m in _IBAN.finditer(masked):
        out.add(("iban", canonical("iban", m.group(0))))
    for m in _URL.finditer(masked):
        raw = m.group(0)
        c = canonical("url", raw)
        if not c or "." not in c.split("/")[0]:
            continue
        if not _EXPLICIT.match(raw) and not is_domain(raw.rstrip(_TRAIL), own):
            continue  # a bare token that is not a domain (§8.14): a file name, a version, …
        out.add(("url", c))
    return out
