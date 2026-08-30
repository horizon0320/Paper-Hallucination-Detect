from __future__ import annotations

import re

from paperaudit.models import Reference


_YEAR = r"(?:19|20)\d{2}[a-z]?"
_NAME = r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’´¨\-]*"
_AUTHOR = rf"{_NAME}(?:\s+{_NAME})?(?:\s+(?:et\s+al\.|and\s+{_NAME}(?:\s+{_NAME})?|&\s*{_NAME}(?:\s+{_NAME})?))?"
_CITATION_RE = re.compile(
    rf"(?P<author>{_AUTHOR})(?P<separator>,\s*|\s+)(?P<opening>\(?\s*)"
    rf"(?P<year>{_YEAR})(?P<closing>\s*\))?"
)
_AUTHOR_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on",
    "or", "the", "to", "via", "with", "using", "section", "table", "figure",
}


def _author_key(author: str) -> str:
    first = re.split(r"\s+(?:et\s+al\.|and|&)\s*", author, maxsplit=1, flags=re.IGNORECASE)[0]
    return first.split()[0].casefold()


def extract_intext_citations(text: str) -> list[Reference]:
    """Extract unique author-year citations when a PDF lacks full references.

    This deliberately returns citation-only records: an author-year key does
    not contain enough information to assert a definitive work title. Such
    records are marked ``citation_only`` and are surfaced for bibliography
    repair/manual review instead of being silently dropped or misclassified.
    """
    seen: set[tuple[str, int]] = set()
    refs: list[Reference] = []
    for match in _CITATION_RE.finditer(text):
        author = re.sub(r"\s+", " ", match.group("author")).strip()
        if author.casefold() in _AUTHOR_STOPWORDS:
            continue
        separator = match.group("separator") or ""
        opening = match.group("opening") or ""
        explicit_multi = bool(re.search(r"(?:et\s+al\.|\band\b|&)", author, re.IGNORECASE))
        if not ("," in separator or opening.strip().startswith("(") or explicit_multi):
            continue
        year = int(match.group("year")[:4])
        key = (_author_key(author), year)
        if key in seen:
            continue
        seen.add(key)
        refs.append(Reference(
            raw=f"{author}, {year}",
            authors=[author],
            year=year,
            citation_only=True,
        ))
    return refs
