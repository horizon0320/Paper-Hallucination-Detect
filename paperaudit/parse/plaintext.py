from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING

from paperaudit.models import Reference

if TYPE_CHECKING:
    pass

_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_URL_RE = re.compile(r"https?://[^\s,;>\"')]+")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})(?:[a-z])?\b", re.IGNORECASE)
_PUBLISHER_YEAR_TAIL_RE = re.compile(
    r"^(?:IEEE(?:\s+Computer\s+Society)?|ACM|Springer(?:\s+Nature)?|Elsevier|Wiley),\s*"
    r"(?:19|20)\d{2}[a-z]?\.$", re.IGNORECASE
)


def _normalize_pdf_text(value: str) -> str:
    """Repair common PDF glyph artifacts before parsing bibliographic fields."""
    text = unicodedata.normalize("NFKC", value or "")
    # Some PDF fonts export accents as a spacing glyph before the letter
    # (``Herv´e J´egou``) instead of a combining mark. Move the mark onto the
    # following vowel, then compose it so reports show the real author name.
    for spacing, combining in (("´", "\u0301"), ("`", "\u0300"), ("¨", "\u0308"), ("ˆ", "\u0302")):
        # NFKC may turn the spacing accent into whitespace + a combining mark.
        text = re.sub(
            rf"\s*{re.escape(combining)}([A-Za-z])",
            lambda m, mark=combining: unicodedata.normalize("NFC", m.group(1) + mark),
            text,
        )
        text = re.sub(
            rf"{re.escape(spacing)}([A-Za-z])",
            lambda m, mark=combining: unicodedata.normalize("NFC", m.group(1) + mark),
            text,
        )
    return unicodedata.normalize("NFC", text)


def _citation_year_matches(chunk: str) -> list[re.Match[str]]:
    """Return year-like tokens while excluding page numbers such as ``pp. 2082–2086``."""
    matches: list[re.Match[str]] = []
    for match in _YEAR_RE.finditer(chunk):
        before = chunk[max(0, match.start() - 12):match.start()]
        after = chunk[match.end():match.end() + 3]
        if re.search(r"\bpp?\.?\s*$", before, re.IGNORECASE):
            continue
        if re.match(r"\s*[-–—]", after) or re.search(r"[-–—]\s*$", before):
            continue
        matches.append(match)
    return matches

_NUMBERED_RE = re.compile(r"^\s*(\[?\d{1,3}\]?\.?\)?\s+|[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s+)")
# Author-year references are common in AAAI, ACL, Springer and similar
# proceedings. The author list can contain semicolons, "and", initials and
# punctuation, so match a bounded prefix up to the first publication year
# rather than assuming one surname + one initial.
_AUTHOR_YEAR_RE = re.compile(
    # Allow multi-word surnames ("Van Swol", "Da San Martino") and
    # semicolon-separated author lists used by AAAI proceedings.
    r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ\-\'’]+\s+)*[A-Z][a-zA-ZÀ-ÿ\-\'’]+,\s*[^0-9\n]{0,180}"
    r"(?<!\()\b(?:(?:19|20)\d{2}[a-z]?\b|\[\s*n\.\s*d\.\s*\])"
    r"\s*[.\,:)]"
)


class Style(str, Enum):
    IEEE = "IEEE"
    APA = "APA"
    VANCOUVER = "VANCOUVER"
    MLA = "MLA"
    CHICAGO = "CHICAGO"
    LNCS = "LNCS"
    AUTHOR_YEAR = "AUTHOR_YEAR"
    UNKNOWN = "UNKNOWN"


def _fix_hyphenated_breaks(text: str) -> str:
    # Join words split at a line ending (``authenti-\ncation``), but preserve
    # literal hyphens inside wrapped URLs (``authscan-automatic-\nextraction``).
    # PDF text extraction commonly inserts the newline after a real URL
    # hyphen, so blindly deleting it corrupts the cited landing-page address.
    pattern = re.compile(r"(\w)-(\n\s*)(\w)")

    def replace(match: re.Match[str]) -> str:
        line_start = text.rfind("\n", 0, match.start()) + 1
        prefix = text[line_start:match.start()].casefold()
        return f"{match.group(1)}-{match.group(3)}" if "http" in prefix or "://" in prefix else f"{match.group(1)}{match.group(3)}"

    return pattern.sub(replace, text)


def _cleanup_chunk(chunk: str) -> str:
    chunk = _fix_hyphenated_breaks(chunk)
    chunk = re.sub(r"\s+", " ", chunk).strip()
    # PyMuPDF can expose a running page footer as an isolated 4-digit token
    # inside a reference when a column continues on the next page (e.g.
    # "Detection by 1256 Visiting ..."). Restrict removal to this distinctive
    # context so legitimate 18xx publication years remain intact.
    chunk = re.sub(r"(?<=\bby)\s+1[0-8]\d{2}(?=\s+[A-Z])", " ", chunk)
    chunk = re.sub(r"\s{2,}", " ", chunk).strip()
    chunk = _NUMBERED_RE.sub("", chunk, count=1).strip()
    return chunk


def split(text: str) -> list[str]:
    text = _fix_hyphenated_breaks(text)
    lines = text.splitlines()

    for strategy in [
        _split_numbered,
        _split_author_year,
        _split_blank_lines,
        _split_hanging_indent,
    ]:
        chunks = strategy(lines)
        chunks = _split_embedded_organization_refs(chunks)
        # PDF extraction may place an entry's author list on one line and its
        # title on the next line.  The author-only line can look like a new
        # author-year entry to the boundary heuristic, leaving a bogus
        # author-less title record behind.  Repair this layout before field
        # extraction; the rule is based on bibliographic shape, not a venue or
        # paper-specific string.
        chunks = _merge_wrapped_author_chunks(chunks)
        cleaned = [_cleanup_chunk(c) for c in chunks if _cleanup_chunk(c)]
        if len(cleaned) >= 2:
            return cleaned

    # Even when no line/number/blank-line strategy yields multiple chunks,
    # adjacent web references can still be flattened onto one extracted line.
    # Apply the organization-boundary pass to this final fallback as well.
    embedded = _split_embedded_organization_refs([text])
    cleaned = [_cleanup_chunk(c) for c in embedded if _cleanup_chunk(c)]
    return [c for c in cleaned if c]


def _split_embedded_organization_refs(chunks: list[str]) -> list[str]:
    """Split adjacent references flattened onto one PDF line.

    PDF extraction can concatenate a completed DOI/URL reference with the
    next web or author-year entry.  The next entry may begin with an
    organization label (``The White House.``), a personal author list
    (``Four Flynn, Helen King, Anca Dragan.``), or a compact organization-year
    form (``Nagabu, 2024. URL ...``).  Keep the left-hand completion signal
    strict so ordinary title punctuation is not split.
    """
    result: list[str] = []
    for chunk in chunks:
        boundaries: list[int] = []
        # Candidate sentence boundaries.  Inspect the complete left context
        # and a bounded right window instead of encoding every citation style
        # in one lookbehind-heavy regex.
        for match in re.finditer(r"(?<=\.)\s+(?=[A-Z])", chunk):
            left = chunk[:match.start()].rstrip()
            right = chunk[match.end():].lstrip()
            if not right:
                continue
            # A completed bibliographic record normally has a URL/DOI and a
            # publication year near its end.  This prevents splitting normal
            # title sentences such as ``Models. In Proceedings ...``.
            left_tail = left[-360:]
            completed = bool(
                re.search(r"https?\s*:\s*//|10\.\d{4,}/|\b(?:19|20)\d{2}[a-z]?\b", left_tail, re.IGNORECASE)
            )
            if not completed:
                continue

            first_sentence = re.split(r"\.\s+", right, maxsplit=1)[0].strip()
            # Article titles are also often title-cased (e.g. ``America's AI
            # Action Plan``), so a generic title-case check would split in the
            # middle of a web citation.  Embedded organization labels in the
            # affected layouts use the explicit ``The <Organization>.`` form;
            # retain compact non-``The`` labels through ``compact_org_year``.
            organization_start = bool(
                re.match(r"^The\s+", first_sentence, re.IGNORECASE)
                and _looks_like_organization_line(first_sentence + ".")
            )
            # Multi-author prefix followed by a title.  Require at least one
            # comma/``and`` separator and a sentence boundary so prose is not
            # mistaken for a new record.
            author_start = bool(
                (first_sentence.count(",") >= 1 or re.search(r"\band\b", first_sentence, re.IGNORECASE))
                and _looks_like_author_start(first_sentence + ".")
            )
            compact_org_year = bool(
                re.match(r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’&\- ]{1,90},\s*(?:19|20)\d{2}[a-z]?\.", right)
                and not _PUBLISHER_YEAR_TAIL_RE.match(right.strip())
            )
            # The new record must itself look complete (a URL within a bounded
            # span), except for compact author-year citations that may be
            # completed by a URL after the year.
            has_url = bool(re.search(r"https?\s*:\s*//", right[:360], re.IGNORECASE))
            if (organization_start or author_start or compact_org_year) and (has_url or compact_org_year):
                boundaries.append(match.start())

        if not boundaries:
            result.append(chunk)
            continue
        start = 0
        parts: list[str] = []
        for boundary in boundaries:
            part = chunk[start:boundary].strip()
            if part:
                parts.append(part)
            start = boundary
        tail = chunk[start:].strip()
        if tail:
            parts.append(tail)
        result.extend(parts or [chunk])
    return result


def _merge_wrapped_author_chunks(chunks: list[str]) -> list[str]:
    """Merge an author-only chunk with the following title chunk.

    Some two-column PDFs flatten a bibliography entry like this::

        A. Author, B. Author, and C. Author.
        Paper title. In Conference..., 2024.

    The first line satisfies the natural-name boundary heuristic, although it
    is not a complete reference.  Merge only when the first chunk has a clear
    multi-author shape, contains no year, and the following chunk starts with
    title-like text rather than another author list.  This preserves genuine
    author-only records (for example a web citation) and ordinary adjacent
    references.
    """
    if len(chunks) < 2:
        return chunks

    merged: list[str] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if index + 1 < len(chunks):
            following = chunks[index + 1]
            current_text = current.strip()
            following_text = following.strip()
            if _is_wrapped_author_only_chunk(current_text) and following_text:
                first_following_line = following.splitlines()[0].strip() if following else ""
                # A title/venue continuation normally starts with a word,
                # colon, quote, or parenthesis, not a second author list.
                if first_following_line and (
                    ":" in first_following_line
                    or not _looks_like_author_start(first_following_line)
                ):
                    merged.append(f"{current} {following}")
                    index += 2
                    continue
        merged.append(current)
        index += 1
    return merged


def _is_wrapped_author_only_chunk(value: str) -> bool:
    """Identify a chunk containing only a wrapped personal author list."""
    value = value.strip()
    if not value or len(value) > 500 or not value.endswith("."):
        return False
    # Standalone organization labels (common for news/web citations) use the
    # same wrapped-entry shape: label on one line, article title on the next.
    if _looks_like_organization_line(value):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", value):
        return False
    if re.search(r"[:\"']|https?://|\b(?:In|Proceedings|Journal|Conference)\b", value, re.IGNORECASE):
        return False
    # Require multiple author separators so a short title sentence is not
    # mistaken for an author-only fragment.
    if value.count(",") < 2 and not re.search(r"\band\b", value, re.IGNORECASE):
        return False
    if not _looks_like_author_start(value):
        return False
    # Every comma/and-delimited segment should look like a person's name;
    # this rejects prose sentences that happen to contain several commas.
    author_text = re.sub(r"\s+and\s+", ",", value[:-1], flags=re.IGNORECASE)
    parts = [part.strip() for part in author_text.split(",") if part.strip()]
    if len(parts) < 2:
        return False
    for part in parts:
        tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’´¨`\-]*\.?", part)
        if not 2 <= len(tokens) <= 5:
            return False
        if any(token.rstrip(".").casefold() in _AUTHOR_START_STOPWORDS for token in tokens):
            return False
    return True


def _split_numbered(lines: list[str]) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _NUMBERED_RE.match(line):
            if current:
                chunks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        chunks.append(current)
    if len(chunks) < 2:
        return []
    return [" ".join(c) for c in chunks]


def _split_blank_lines(lines: list[str]) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                chunks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        chunks.append(current)
    if len(chunks) < 2:
        return []
    return [" ".join(c) for c in chunks]


def _split_hanging_indent(lines: list[str]) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and current:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append(current)
    if len(chunks) < 2:
        return []
    return [" ".join(c) for c in chunks]


def _split_author_year(lines: list[str]) -> list[str]:
    # Explicit APA-style ``(YYYY)`` entries should be handled by the blank-line
    # splitter; the boundary heuristic below is intended for natural-name
    # references whose year appears at the end.
    if any(re.search(r"\((?:19|20)\d{2}\)", line) for line in lines):
        return []
    chunks: list[list[str]] = []
    current: list[str] = []
    current_has_title_boundary = False
    for index, line in enumerate(lines):
        # PDF line wrapping frequently puts the publication year on the next
        # line (e.g. "Greifeneder, ..." followed by "2020."). Inspect a
        # short look-ahead window while keeping the match anchored to the
        # beginning of the candidate line.
        # Author lists can wrap across many lines before the publication year
        # (especially long NeurIPS/technical-report author lists). A wider
        # bounded look-ahead prevents adjacent entries from being merged.
        lookahead = " ".join(lines[index:index + 12])
        previous = lines[index - 1].strip() if index else ""
        continuation_suffix = (",", ";", ":", "-", "and", "or", "in", "of", "to", "with")
        starts_new_reference = not previous.casefold().endswith(continuation_suffix)
        aaai_match = _AUTHOR_YEAR_RE.match(lookahead.strip())
        aaai_single_author = re.match(
            r"^\s*[A-Z][a-zA-ZÀ-ÿ\-\'’]+,\s*[A-Z](?:\.|\s)", line
        )
        aaai_author_year = bool(
            aaai_match
            and (";" in line or aaai_single_author is not None)
        )
        plain_author_year = _looks_like_plain_author_year_start(lines, index)
        # Natural-name references normally begin after a completed previous
        # entry. This guards against wrapped author/title/venue lines that
        # happen to contain commas and a year in the look-ahead window.
        previous_boundary = (
            not previous
            or previous.endswith((".", ")", ":"))
            or previous.casefold() == "preprint"
            or previous.isdigit()
        )
        # A long wrapped author list may place the year more than twelve lines
        # below its first line. If the preceding line already ends an entry,
        # a name-shaped line is still a reliable new-reference boundary.
        group_author_boundary = bool(
            re.match(r"^(?:OpenAI|Llama\s+Team)\b", line.strip())
            or (line.strip() == "Llama" and index + 1 < len(lines)
                and lines[index + 1].strip().startswith("Team"))
        )
        # Web/news citations often use an organization name on its own line
        # (for example ``The Guardian.``) followed by the article title.  When
        # the previous entry already ended with a URL or publication year,
        # treat this compact organization line as a new record boundary.
        organization_boundary = bool(
            previous_boundary
            and _looks_like_organization_line(line)
            and (
                "http" in previous.casefold()
                or "/" in previous
                or re.search(r"\b(?:19|20)\d{2}\b", previous)
            )
        )
        author_boundary = previous_boundary and (
            ("," in line and _looks_like_author_start(line)) or group_author_boundary
        )
        # A wrapped author list may itself begin with a name-shaped line (for
        # example ``Xiao, ... et al.``). Do not split it until the current
        # chunk has crossed its author→title boundary.
        can_start_plain_reference = plain_author_year and previous_boundary
        if starts_new_reference and (
            (aaai_author_year and previous_boundary)
            or can_start_plain_reference
            or author_boundary
            or organization_boundary
        ):
            if current:
                chunks.append(current)
            current = [line]
            current_has_title_boundary = _has_author_title_boundary(line)
        elif current:
            # If the next line still looks like a name-shaped continuation,
            # keep it in the author list (wrapped ``... et al.`` is common).
            # Otherwise the first non-author line establishes the title
            # boundary even when the author list and title are on separate
            # lines.
            if (
                line.strip()
                and not current_has_title_boundary
                and not _looks_like_author_start(line)
            ):
                current_has_title_boundary = True
            current.append(line)
    if current:
        chunks.append(current)
    if len(chunks) < 2:
        return []
    return [" ".join(c) for c in chunks]


def _looks_like_plain_author_year_start(lines: list[str], index: int) -> bool:
    """Detect bibliography entries such as ``A. Author, B. Author. Title...``.

    arXiv/NeurIPS-style references use natural-name author lists rather than
    AAAI's semicolon syntax and put the publication year near the end. A
    candidate must have author-list punctuation and a year within a short
    window; page headers and ordinary title continuations fail these checks.
    """
    value = lines[index].strip()
    if not value or value.casefold() in {"preprint", "references"} or value.isdigit():
        return False
    if not re.match(r"^[A-ZÀ-ÿ][^\n]{2,180}", value):
        return False
    # Wrapped continuation lines often begin with a venue or a single surname
    # (e.g. ``Security, 13(11):...`` or ``Herout. Imd2020...``). A new
    # natural-name reference normally starts with a given name + family name,
    # an initial + family name, or an organization name followed by a comma or
    # period. This shape check is deliberately independent of venue/year so it
    # remains useful across citation styles and publisher templates.
    if not _looks_like_author_start(value):
        return False
    window = " ".join(lines[index:index + 12])
    # APA has an explicit ``(YYYY)`` marker and should remain on the blank-line
    # strategy so its style detector can distinguish it from natural-name
    # author-year formats.
    if re.search(r"\((?:19|20)\d{2}\)", window):
        return False
    if not re.search(r"\b(?:19|20)\d{2}[a-z]?\b", window, re.IGNORECASE):
        return False
    # A reference's first line normally has commas or an explicit "and" in
    # the author list. Avoid treating title/venue continuation lines as starts.
    return (
        "," in value
        or re.search(r"\band\b", value, re.IGNORECASE) is not None
        or _has_author_title_boundary(value)
        or value.endswith(".")
    )




def _has_author_title_boundary(value: str) -> bool:
    """Return whether a line contains a likely author-list/title boundary."""
    # Do not treat the period after a publication year as an author boundary.
    return re.search(r"(?<!\d)\.\s+(?=[A-Za-zÀ-ÿ])", value) is not None


_AUTHOR_START_STOPWORDS = {
    "a", "an", "the", "under", "in", "and", "or", "of", "for", "using",
    "proceedings", "international", "advances", "journal", "ieee", "acm",
    "information", "computer", "pattern", "security", "conference", "image",
    "review", "language", "learning", "detection", "manipulation", "models",
    "can", "towards", "toward", "from", "on", "with", "et",
    "released", "accessed", "available", "retrieved", "as", "mandated",
}


def _looks_like_organization_line(value: str) -> bool:
    """Recognize a short standalone publisher/news organization line."""
    value = value.strip()
    if not value or len(value) > 100 or "," in value or ":" in value:
        return False
    if not value.endswith(".") or re.search(r"\b(?:19|20)\d{2}\b|https?://", value):
        return False
    words = value[:-1].split()
    if not 1 <= len(words) <= 8:
        return False
    # A two-token capitalized line is much more often a personal author
    # (``David Tan. Title...``) than an organization. Do not let the web
    # citation branch swallow the actual title in that common format.
    if len(words) == 2 and all(re.match(r"^[A-Z][a-zÀ-ÿ'’\-]+$", word) for word in words):
        return False
    # A standalone label is title-cased or all-caps; sentence prose and
    # ordinary wrapped paper titles generally contain lowercase starters.
    return all(re.match(r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’&\-]*$", word) for word in words)


def _looks_like_author_start(value: str) -> bool:
    """Recognize likely author-list starts while rejecting venue/title lines.

    PyMuPDF flattens typography and indentation, so a wrapped venue line can
    look like an author-year reference. Requiring a name-shaped first pair of
    tokens catches natural-name and initial-based author lists without relying
    on a particular publisher's punctuation.
    """
    # PDF text extraction may retain combining accent marks as standalone
    # glyphs (``Nikoli´c``, ``L¨udke``) or use a grave accent in a surname.
    tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’´¨`\-]*\.?", value)
    if len(tokens) < 2:
        # Permit a small set of recognizable group-author forms such as
        # ``OpenAI.`` while rejecting one-word venue continuations.
        return bool(re.match(r"^(?:OpenAI|Gemma)(?:\s+Team)?[,.]\s+", value))
    first, second = tokens[0], tokens[1]
    if len(first.rstrip(".")) > 1 and first.casefold().rstrip(".") in _AUTHOR_START_STOPWORDS:
        return False
    if len(second.rstrip(".")) > 1 and second.casefold().rstrip(".") in _AUTHOR_START_STOPWORDS:
        return False
    first_is_initial = bool(re.fullmatch(r"[A-Z]", first.rstrip(".")))
    second_is_initial = bool(re.fullmatch(r"[A-Z]", second.rstrip(".")))
    def clean_name(token: str) -> str:
        return token.rstrip(".").replace("´", "").replace("¨", "").replace("`", "")

    first_clean = clean_name(first)
    second_clean = clean_name(second)
    first_is_name = bool(re.fullmatch(r"[A-ZÀ-ÿ][a-zà-ÿ'’\-]+", first_clean))
    second_is_name = bool(re.fullmatch(r"[A-ZÀ-ÿ][a-zà-ÿ'’\-]+", second_clean))
    if first_clean.casefold() in {"openai", "gemma", "meta", "anthropic"}:
        return True
    if first_is_initial:
        return second_is_name and not second_is_initial
    # A given name may be followed by a middle initial (``Timothy K Ng``).
    return first_is_name and (second_is_name or second_is_initial)


def _looks_like_plain_author_year_chunk(chunk: str) -> bool:
    """Style-detection variant that rejects APA/IEEE year and quote syntax."""
    if any(mark in chunk for mark in ('"', "'", '“', '”')):
        return False
    year = re.search(r"(?<!\()\b(?:19|20)\d{2}[a-z]?\b", chunk, re.IGNORECASE)
    if not year or (year.start() > 0 and chunk[year.start() - 1] == '('):
        return False
    prefix = chunk[:year.start()]
    return ("," in prefix or re.search(r"\band\b", prefix, re.IGNORECASE) is not None) and re.search(
        r"\.\s+(?=[A-ZÀ-ÿ])", prefix
    ) is not None


# IEEE: quoted title + year; cleanup strips [N] so we match on quotes + year
_IEEE_RE = re.compile(r'["“][^"”]{5,}["”],?\s+in\s+|["“][^"”]{5,}[,."”]\s+[A-Z].*\d{4}')
_IEEE_QUOTED_RE = re.compile(r'["“][^"”]{5,}["”].*\d{4}')
_APA_RE = re.compile(r"[A-Z][a-z]+,\s+[A-Z]\.\s+\(\d{4}\)")
_VANCOUVER_RE = re.compile(r"\d{4};\d+[\s(]")
_MLA_RE = re.compile(r'"[^"]+"\s+.*vol\.\s+\d+.*\(\d{4}\)')
_CHICAGO_RE = re.compile(r"\(\w+:\s*\w+,\s*\d{4}\)")
# LNCS/Springer: "Lastname, I., ...: Title. Venue (Year)"
_LNCS_RE = re.compile(r'^[A-Z][a-zA-ZÀ-ÿ\-]+,\s+[A-Z]\.')


def detect_style(chunks: list[str]) -> Style:
    samples = chunks[:5]
    scores: dict[Style, int] = {s: 0 for s in Style}
    author_year_hits = 0

    for chunk in samples:
        group_author_year = bool(
            re.match(r"^(.+?)\.\s+.+\b(?:19|20)\d{2}\b", chunk)
            and _looks_like_group_author(chunk.split(". ", 1)[0])
        )
        if _AUTHOR_YEAR_RE.match(chunk) or _looks_like_plain_author_year_chunk(chunk) or group_author_year:
            # Keep APA's explicit ``(YYYY)`` marker more specific when both
            # heuristics happen to match.
            scores[Style.AUTHOR_YEAR] += 1
            author_year_hits += 1
        if _IEEE_RE.search(chunk) or _IEEE_QUOTED_RE.search(chunk):
            scores[Style.IEEE] += 2
        if _APA_RE.search(chunk):
            scores[Style.APA] += 2
        if _VANCOUVER_RE.search(chunk):
            scores[Style.VANCOUVER] += 2
        if _MLA_RE.search(chunk):
            scores[Style.MLA] += 2
        if _CHICAGO_RE.search(chunk):
            scores[Style.CHICAGO] += 2
        if _LNCS_RE.match(chunk) and ': ' in chunk:
            scores[Style.LNCS] += 2

        # secondary signals
        if re.search(r'["“][^"”]{5,}["”]', chunk):
            scores[Style.IEEE] += 1
        if re.search(r'\(\d{4}\)', chunk):
            scores[Style.APA] += 1
        if re.search(r'\d{4};\d+\(\d+\):\d+', chunk):
            scores[Style.VANCOUVER] += 1

    # AAAI's semicolon-delimited author-year format can contain colons in
    # titles, which otherwise makes the generic LNCS heuristic win. Require
    # a small majority of clear author-year starts before selecting it.
    if author_year_hits >= 2 and author_year_hits * 2 >= len(samples):
        return Style.AUTHOR_YEAR

    best_style = max(scores, key=lambda s: scores[s])
    if scores[best_style] >= 2:
        return best_style
    return Style.UNKNOWN


def extract_fields(chunk: str, style: Style = Style.UNKNOWN) -> Reference:
    chunk = _normalize_pdf_text(chunk)
    # PDF text extraction may insert spaces into URL punctuation (``https: //``
    # or ``arxiv. org``). Normalize those artifacts before identifier and URL
    # detection so preprint records remain resolvable.
    chunk = re.sub(r"\b(https?)\s*:\s*//", r"\1://", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"\barxiv\.\s+org", "arxiv.org", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"\barxiv\.org/\s+abs/", "arxiv.org/abs/", chunk, flags=re.IGNORECASE)
    # URL paths can wrap after a slash in extracted PDF text. Join a
    # hyphenated/path-like continuation while leaving ordinary prose after a
    # landing-page URL untouched.
    chunk = re.sub(
        r"(https?://[^\s]+/)\s+([A-Za-z0-9][A-Za-z0-9._/-]*-[A-Za-z0-9._/-]*)",
        r"\1\2",
        chunk,
        flags=re.IGNORECASE,
    )
    # Publisher PDF extraction often wraps DOI URLs after ``arXiv.`` or a
    # slash. Collapse only this DOI-specific break so ordinary word spacing
    # remains untouched and identifiers such as ``10.48550/arXiv.2404.04475``
    # can be recovered reliably.
    chunk = re.sub(r"(10\.\d{4,}/arXiv\.)\s+", r"\1", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"(10\.\d{4,}/)\s+(?=[A-Za-z])", r"\1", chunk)
    # DOI suffixes in two-column PDFs can wrap after a period, e.g.
    # ``10.5281/zenodo. 15753192``. Join the numeric continuation while
    # leaving normal prose spacing unchanged.
    chunk = re.sub(r"(10\.\d{4,}/[A-Za-z0-9._-]+\.)\s+(\d{3,})", r"\1\2", chunk)
    doi = None
    m = _DOI_RE.search(chunk)
    if m:
        doi = m.group().rstrip(".,;)")

    urls = [u.rstrip(".,;)>\"'") for u in _URL_RE.findall(chunk)]
    urls = [u for u in urls if "doi.org" not in u]
    # Drop truncated URLs that have no meaningful path (e.g. "https://arxiv" from pdfminer line breaks)
    from urllib.parse import urlparse
    urls = [u for u in urls if urlparse(u).path.rstrip("/")]

    arxiv_id = None
    for u in urls + [chunk]:
        m2 = re.search(r"arxiv\.org/abs/([^\s/,;]+)", u)
        if m2:
            arxiv_id = m2.group(1).rstrip(".,;)")
            break
    if not arxiv_id:
        m3 = re.search(r"\barXiv:\s*(\d{4}\.\d+)", chunk, re.IGNORECASE)
        if m3:
            arxiv_id = m3.group(1).rstrip(".,;)")
    # Recover arXiv ID from pdfminer-broken URLs like "arxiv. org/abs/2308.07201"
    if not arxiv_id:
        m4 = re.search(r'arxiv[.\s]+org/abs/([^\s,;)]+)', chunk, re.IGNORECASE)
        if m4:
            arxiv_id = m4.group(1).strip().rstrip(".,;)")
    if not arxiv_id:
        m5 = re.search(r'10\.48550/arXiv\.(\d{4}\.\d+)', chunk, re.IGNORECASE)
        if m5:
            arxiv_id = m5.group(1)

    year = None
    year_matches = [
        m for m in _citation_year_matches(chunk)
        if not re.search(r"arxiv\s*[:.]\s*$", chunk[max(0, m.start() - 12):m.start()], re.IGNORECASE)
    ]
    if year_matches:
        # Natural-name arXiv/NeurIPS references put the publication year at
        # the end, while AAAI's semicolon style puts it immediately after the
        # author list. Ignore years embedded in arXiv IDs such as 1909.09436.
        # IEEE and numbered proceedings commonly put the publication year at
        # the end; after page-number filtering, prefer the last remaining year
        # when style detection is uncertain.
        ym = year_matches[-1] if (style == Style.AUTHOR_YEAR and ";" not in chunk) or len(year_matches) > 1 else year_matches[0]
        year = int(ym.group(1))

    title = None
    authors: list[str] = []
    venue = None

    if style == Style.IEEE:
        title, authors = _extract_ieee(chunk)
    elif style == Style.AUTHOR_YEAR:
        title, authors, venue = _extract_author_year(chunk)
    elif style == Style.APA:
        title, authors = _extract_apa(chunk)
    elif style == Style.VANCOUVER:
        title, authors = _extract_vancouver(chunk)
    elif style == Style.LNCS:
        title, authors, venue = _extract_lncs(chunk)
        if not title:
            title, fallback_authors = _extract_generic(chunk)
            if not authors:
                authors = fallback_authors
    else:
        # Web citations often use a natural-name format but are not numerous
        # enough for global style detection to classify as AUTHOR_YEAR. Reuse
        # the author-year parser when a URL is present so the article title is
        # stopped before publisher/organization metadata (e.g. ``Center for
        # Technology... January 2024. URL``).
        if urls and re.search(r"\b(?:URL|https?://|January|February|March|April|May|June|July|August|September|October|November|December)\b", chunk, re.IGNORECASE):
            web_title, web_authors, web_venue = _extract_author_year(chunk)
            if web_title and web_authors:
                title, authors, venue = web_title, web_authors, web_venue
            else:
                title, authors = _extract_generic(chunk)
        else:
            title, authors = _extract_generic(chunk)

    title = _normalize_compact_title(title)

    return Reference(
        raw=chunk,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        urls=urls,
    )


def _extract_ieee(chunk: str) -> tuple[str | None, list[str]]:
    m = re.search(r'["“]([^"”]{10,})["”]', chunk)
    title = m.group(1).strip().rstrip(",.") if m else None

    quote_start = re.search(r'["“]', chunk)
    authors_part = chunk[:quote_start.start()] if quote_start else ""
    authors = _parse_author_list(authors_part)
    return title, authors


def _normalize_compact_title(value: str | None) -> str | None:
    """Repair common PDF text-layer omissions in title punctuation/spacing."""
    if not value:
        return value
    value = unicodedata.normalize("NFKC", value)
    # Restore spaces removed after punctuation and between camel-case words.
    value = re.sub(r"(?<=[,:;])(?=[A-Za-z])", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    # A few PDF encoders remove spaces even when both neighboring words are
    # lowercase (``Spatialgroup-wiseenhance``). Split only well-known
    # technical title morphemes to avoid changing legitimate identifiers.
    for left, right in (
        ("spatial", "group"), ("group", "wise"), ("wise", "enhance"),
        ("enhancing", "semantic"), ("semantic", "feature"),
        ("feature", "learning"), ("real", "time"), ("object", "detection"),
    ):
        value = re.sub(rf"(?i)\b({left})(?={right}\b)", r"\1 ", value)
    return re.sub(r"\s+", " ", value).strip()


def _extract_apa(chunk: str) -> tuple[str | None, list[str]]:
    # Match both (2025) and (n.d.) year tokens
    _year_tok = r"(?:\d{4}|n\.d\.)"
    m = re.match(
        rf"^((?:[A-Z][a-zA-ZÀ-ÿ\-\.']+,?\s+[A-Z]\.?,?\s*(?:&\s*)?)+)\s*\({_year_tok}\)\.\s*([^.]+\.)",
        chunk,
    )
    if m:
        authors = _parse_author_list(m.group(1))
        title = m.group(2).strip().rstrip(".")
        return title, authors

    title_m = re.search(rf"\({_year_tok}\)\.\s+([^.]+)\.", chunk)
    title = title_m.group(1) if title_m else None
    return title, []


def _extract_vancouver(chunk: str) -> tuple[str | None, list[str]]:
    m = re.match(r"^(?:\d+\.\s+)?((?:[A-Z][a-zA-ZÀ-ÿ\-\.']+\s+[A-Z]{1,3},?\s*)+)\.\s+([^.]+)\.", chunk)
    if m:
        authors = _parse_author_list(m.group(1))
        title = m.group(2).strip()
        return title, authors
    return None, []


def _extract_lncs(chunk: str) -> tuple[str | None, list[str], str | None]:
    """LNCS/Springer: 'Lastname, I., ...: Title. Venue (Year)'
    Returns (title, authors, venue).
    """
    idx = chunk.find(': ')
    if idx < 0:
        return None, [], None
    authors_text = chunk[:idx]
    rest = chunk[idx + 2:].strip()
    # Verify the left side has at least one "Lastname, I." author pattern
    if not re.search(r'[A-Z][a-z]+,\s+[A-Z]\.', authors_text):
        return None, [], None
    authors = _parse_author_list(authors_text)

    # Title ends at the first ". " followed by an uppercase letter or "("
    title_m = re.match(r'^(.+?)\.\s+(?=[A-Z(])', rest)
    if title_m:
        raw_title = title_m.group(1).strip()
        # Venue is everything after the title period up to the year "(YYYY)"
        after_title = rest[title_m.end():].strip()
        venue_m = re.match(r'^(?:In:\s*)?(.+?)(?:\.\s*(?:pp\.\s*[\d–-]+|vol\.\s*\d)|\s*\(\d{4}\)|$)', after_title, re.IGNORECASE)
        venue = venue_m.group(1).strip().rstrip('.,').strip() if venue_m else None
    else:
        raw_title = rest.split('. ')[0].strip() or None
        venue = None

    # Clean up title artifacts: strip URLs first so the trailing-year pattern
    # then sees "(2024)" at the end (e.g. "Title (2024), https://..." → "Title")
    title = raw_title
    if title:
        title = re.sub(r',?\s*https?://\S+', '', title).strip()    # URLs first
        title = re.sub(r'\s*\(\d{4}\),?\s*$', '', title).strip()  # then trailing "(YYYY)"
        title = title.rstrip('.,').strip() or None

    return title, authors, venue


def _extract_generic(chunk: str) -> tuple[str | None, list[str]]:
    m = re.search(r'"([^"]{10,})"', chunk)
    if m:
        return m.group(1), []

    # Try LNCS/Springer colon-separated format as fallback
    lncs_title, lncs_authors, _venue = _extract_lncs(chunk)
    if lncs_title:
        return lncs_title, lncs_authors

    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    for sentence in sentences[1:3]:
        stripped = sentence.strip()
        if len(stripped) > 20 and not re.match(r"^[A-Z]{2,}", stripped):
            return stripped.rstrip("."), []

    return None, []


def _extract_author_year(chunk: str) -> tuple[str | None, list[str], str | None]:
    """Extract AAAI/ACL and arXiv natural-name author-year references."""
    # Organization-authored web/news records use a sentence-style prefix,
    # rather than the ``Organization, 2025. URL ...`` form handled below:
    # ``The White House. America's AI Action Plan. https://...``.  Parse the
    # organization and article title before applying year/venue heuristics;
    # otherwise the first year in a URL path can make the trailing prose look
    # like the title and shift the organization into the author field.
    web_org = re.match(
        r"^\s*(?P<org>[^.]{2,100})\.\s+(?P<title>.+?)\s+https?://",
        chunk,
        re.IGNORECASE | re.DOTALL,
    )
    if web_org and _looks_like_organization_line(web_org.group("org") + "."):
        title = web_org.group("title").strip().rstrip(" .,:;") or None
        org = web_org.group("org").strip()
        return title, [org], None

    # Organization/web citations commonly have no article title in the
    # bibliography, e.g. ``Anthropic, 2025. URL https://...``.  Do not turn
    # the literal ``URL`` marker into the title or include the year in the
    # organization name.
    org_url = re.match(
        r"^\s*(?P<org>[^.]{2,100}?),\s*(?:19|20)\d{2}\.?\s+(?:URL\s*:?[ \t]*)?https?://",
        chunk,
        re.IGNORECASE,
    )
    if org_url:
        return None, _parse_author_list(org_url.group("org")), None

    year_m = re.search(r"\b((?:19|20)\d{2})(?:[a-z])?\b", chunk, re.IGNORECASE)
    if not year_m:
        return _extract_generic(chunk)[0], [], None

    # AAAI places the year before the title; arXiv/NeurIPS styles place it at
    # the end. Detect the former by a year followed by a sentence boundary.
    if re.search(r";\s*(?:and\s+)?[A-Z][a-zÀ-ÿ'’\-]+", chunk[:year_m.start()]) and re.match(
        r"^[^\n]{0,220}\b(?:19|20)\d{2}[a-z]?\b\s*[.:)]", chunk, re.IGNORECASE
    ):
        authors_text = chunk[:year_m.start()].strip(" .,:;")
        authors = _parse_author_list(authors_text)
        rest = chunk[year_m.end():].lstrip(" .,:;")
    else:
        # Find the end of the author sentence. Initials contain periods, so
        # require a following capitalized title-like phrase and a long enough
        # comma-rich prefix before accepting the boundary.
        boundary = None
        # Repository/Zenodo records often use a project slug as the title and
        # therefore begin with a lowercase token (for example
        # ``Yu, X. et al. deepseek-ai/DeepSeek-V3: v1.0.0. Zenodo ...``).
        # The general capitalized-title lookahead intentionally rejects this,
        # so recognize the bounded ``et al. <slug>`` form explicitly.
        slug_boundary = re.search(
            r"\.\s+(?=(?:deepseek-ai|openai|google|meta|anthropic)[/:-])",
            chunk,
            re.IGNORECASE,
        )
        if slug_boundary and ("," in chunk[:slug_boundary.start()] or "et al" in chunk[:slug_boundary.start()].casefold()):
            boundary = slug_boundary
        for m in re.finditer(r"\.\s+(?=[A-ZÀ-ÿ][^\n]{3,140})", chunk):
            if boundary is not None:
                break
            prefix = chunk[:m.start()]
            after = chunk[m.end():].lstrip()
            # Initial-based author lists contain many periods (``Y. Cao, B.
            # Yang``). If the text after a candidate period still begins with
            # another comma-separated name, this is not the author/title
            # boundary.
            if _looks_like_author_continuation(after) or (
                re.search(r"(?:^|[\s,])(?:and\s+)?[A-Z]$", prefix)
                and re.match(r"^[A-Z]\.\s+[A-Z][a-zà-ÿ]", after)
            ):
                continue
            # A final initial before the period (``and Z. Zhang``) is still
            # part of the author list when the next text is another
            # name-shaped continuation.  If the next text is a title (for
            # example ``Iwasawa, Y. Large language models...``), the period
            # is the actual author/title boundary.
            if re.search(r"(?:^|[\s,])(?:and\s+)?[A-Z]$", prefix) and re.match(
                r"^(?:[A-Z][a-zà-ÿ'’\-]+|and),\s+(?:(?:[A-Z]\.?|and)\s+)?(?:[A-Z]\.\s+)?[A-ZÀ-ÿ][a-zà-ÿ'’\-]+",
                after,
            ):
                continue
            if (
                prefix.count(",") >= 1
                or re.search(r"\band\b", prefix, re.IGNORECASE)
                or _looks_like_author_start(prefix.strip())
                or _looks_like_group_author(prefix.strip())
            ):
                boundary = m
                break
        if boundary is None:
            return _extract_generic(chunk)[0], [], None
        authors_text = chunk[:boundary.start()].strip(" .,:;")
        authors = _parse_author_list(authors_text)
        rest = chunk[boundary.end():].strip()

    # Stop the title at common venue/metadata markers; otherwise use the first
    # sentence boundary. This keeps search queries focused on the work title.
    venue_markers = re.search(
        r"\s+(?=(?:arXiv\s+preprint|Preprint\s+at|In\s+Proceedings|In\s+Advances\s+in|"
            r"In\s+International|In\s+\d+(?:st|nd|rd|th)\b|"
            r"(?:19|20)\d{2}\s+\d+(?:st|nd|rd|th)\s+International\b|"
            r"Advances\s+in|Under\s+review|Transactions\s+of|ACM\s+Computing|"
            r"Journal\s+of\b|Foundations\s+and\s+Trends\b|Center\s+for\b|"
            r"LongCat\s+LLM|Zenodo|URL\b|https?://|;\s+https?://))",
        rest,
        re.IGNORECASE,
    )
    title_part = rest[:venue_markers.start()].strip() if venue_markers else rest
    if not venue_markers:
        sentence = re.search(r"[.!?]\s+(?=[A-ZÀ-ÿ])", rest)
        if sentence:
            title_part = rest[:sentence.start()].strip()
    title = title_part.strip(" .;") or None
    venue = rest[venue_markers.start():].strip().rstrip(".") if venue_markers else None
    if venue:
        # Web records frequently append publication date and a literal URL
        # marker after the organization name. Keep only the carrier/source
        # name in the venue field; the URL is already captured separately.
        venue = re.sub(
            r"\s*,?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}.*$",
            "",
            venue,
            flags=re.IGNORECASE,
        )
        venue = re.sub(r"\s*,?\s*URL\s*:?.*$", "", venue, flags=re.IGNORECASE).strip(" ,.;") or None
    # A URL introduced by a literal ``URL`` marker is an identifier, not a
    # publication venue. Keep it in ``urls`` and leave venue unknown.
    if venue and re.match(r"^(?:URL\s*:?[ \t]*)?https?://", venue, re.IGNORECASE):
        venue = None
    # A page footer can appear inside a title after a column break.
    if title:
        title = re.sub(r"(?<=\bby)\s+1[0-8]\d{2}(?=\s+[A-Z])", " ", title)
        # Some author-year styles put the publication year directly after the
        # title (``Mistral 7b, 2023`` / ``Qwen2.5 technical report, 2025``)
        # before a URL or venue. The year is already captured as metadata and
        # should not become part of the searchable title.
        title = re.sub(r"[,;.]\s*(?:19|20)\d{2}\s*$", "", title)
        title = re.sub(r",?\s+volume\s+\d+\s*$", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s{2,}", " ", title).strip(" .") or None
    return title, authors, venue


def _looks_like_group_author(value: str) -> bool:
    """Recognize organization/group authors followed by a title.

    Nature and similar publisher styles use entries such as ``OpenAI.`` or
    ``Chinese Mathematical Society (CMS).`` without a personal surname and
    initials.  Keep this deliberately conservative so a normal title sentence
    is not mistaken for an author prefix.
    """
    value = value.strip(" .,:;")
    if not value or len(value) > 100 or "," in value:
        return False
    if re.search(r"\b(?:In|Preprint|Proceedings|Journal|Conference)\b", value, re.IGNORECASE):
        return False
    return bool(
        re.match(
            r"^(?:OpenAI(?:\s+et\s+al\.?)?|Aider|DeepSeek(?:-AI)?(?:\s+et\s+al\.?)?|Google|Meta|Anthropic|"
            r"Chinese Mathematical Society(?:\s*\([^)]*\))?|"
            r"Mathematical Association of America)(?:\s+Team)?$",
            value,
            re.IGNORECASE,
        )
    )


def _looks_like_author_continuation(value: str) -> bool:
    """Return whether text after a period still looks like an author list."""
    value = value.lstrip()
    # ``T. Gu, S.`` / ``B. et al.`` / ``S., Reid`` are wrapped initials and
    # surnames, not the beginning of a paper title.
    # A final given-name initial is often followed by another author on a
    # wrapped line (``G. J., and Wong, E.`` or ``F. and Goodman, J.``).
    # Keep such fragments in the author list instead of starting the title
    # with the trailing initials. Lowercase surname particles are supported
    # as well (``S., de las Casas, D.``).
    initial_then_author = re.compile(
        r"^[A-Z]\.\s*,?\s+"
        r"(?:(?:and|et\s+al\.?)\s+)?"
        r"(?:(?:de|del|der|den|van|von|da|das|las|la)\s+)*"
        r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’´¨`\-]+"
        r"(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’´¨`\-]+)*\s*(?:,|\.)",
        re.IGNORECASE,
    )
    initial_et_al = re.compile(r"^[A-Z]\.\s*,?\s+et\s+al\b", re.IGNORECASE)
    return bool(
        re.match(r"^et\s+al\b", value, re.IGNORECASE)
        or re.match(r"^(?:[A-Z]\.)+\s+et\s+al\b", value, re.IGNORECASE)
        or initial_et_al.match(value)
        or initial_then_author.match(value)
        # A name written as ``A. Vedaldi`` can be mistaken for a title
        # boundary at the period after the initial (``... K. Lenc. Matconvnet``).
        # Require a multi-letter surname here so the real boundary after a
        # final initial (``... Goodman, J. An empirical...``) is preserved.
        or re.match(r"^[A-ZÀ-ÿ][a-zà-ÿ'’´¨`\-]{2,}\.\s+[A-ZÀ-ÿ]", value)
        or re.match(r"^[A-Z]\.,\s*[A-ZÀ-ÿ][a-zà-ÿ'’\-]+", value)
        or re.match(r"^[A-Z]\.[, ]+\s*[A-ZÀ-ÿ][a-zà-ÿ'’\-]+,", value)
        or re.match(r"^[A-ZÀ-ÿ][a-zà-ÿ'’\-]+,\s+(?:[A-Z]\.?\s*)+", value)
    )


def _parse_author_list(text: str) -> list[str]:
    text = text.strip().rstrip(".,")
    if not text:
        return []
    # PDF text extraction may remove spaces after commas, around ``and``, or
    # inside initial blocks (``Y.Li,X.Li,andJ.Yang``). Restore these separators
    # before applying the author split rules below.
    text = re.sub(r",(?=[A-Z])", ", ", text)
    text = re.sub(r",\s*and(?=\s|[A-Z])", ", and ", text, flags=re.IGNORECASE)
    text = re.sub(r"\band(?=[A-Z])", "and ", text)
    text = re.sub(r"\b([A-Z])\.([A-Z][a-zà-öø-ÿ'’\-]+)", r"\1. \2", text)
    # Some PDF text layers drop the comma between adjacent abbreviated
    # authors, producing strings such as ``K. R. HeGirshick`` for the IEEE
    # citation ``K. He, R. Girshick``.  Recover the two initials/surnames when
    # a glued CamelCase surname clearly contains two name segments.  Keep the
    # rule narrow so normal names such as ``J. D. Kaplan`` are untouched.
    text = re.sub(
        r"\b([A-Z])\.\s+([A-Z])\.\s+"
        r"([A-Z][a-zà-öø-ÿ'’\-]{1,})([A-Z][a-zà-öø-ÿ'’\-]{2,})\b",
        r"\1. \3, \2. \4",
        text,
    )
    # A compact multi-initial form can lose the separator in the same way,
    # e.g. ``C.H. ShenChen`` for ``C.H. Shen, H. Chen``.  Preserve the first
    # author's initials and derive the second author's initial from the glued
    # surname's second capitalized segment.
    def _split_compact_glued(match: re.Match[str]) -> str:
        initials, first_surname, second_surname = match.groups()
        # In ``C.H. ShenChen``, the trailing ``H.`` is the given-name initial
        # for the second author (Hao Chen); use the final initial from the
        # compact block rather than the surname's first letter (C.).
        initial_tokens = re.findall(r"[A-Z]", initials)
        second_initial = initial_tokens[-1] if initial_tokens else second_surname[0]
        return f"{initials.strip()} {first_surname}, {second_initial}. {second_surname}"

    text = re.sub(
        r"\b((?:[A-Z]\.\s*){2,})"
        r"\s*"
        r"([A-Z][a-zà-öø-ÿ'’\-]{1,})([A-Z][a-zà-öø-ÿ'’\-]{2,})\b",
        _split_compact_glued,
        text,
    )
    # Split on ", " only before a Lastname (uppercase + lowercase letter), NOT before
    # initials like "H.W." (uppercase + period) or "et al." (lowercase).
    # This prevents "Chung, H.W." from splitting into ["Chung", "H.W"].
    # PDF extraction can preserve combining accents as standalone glyphs
    # (e.g. ``J´egou`` or ``G¨unnemann``). Include those code points in the
    # surname lookahead so ``Sablayrolles, A., J´egou, H`` is split into two
    # authors while ``Chung, H.W.`` remains a single author.
    parts = re.split(
        r"(?i:\s+and\s+)|;\s*|,\s+(?=[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’´¨`\-])|"
        # Initial-based authors are common in IEEE/NeurIPS references and
        # were previously glued to the preceding author (``Shi, J Zico
        # Kolter`` or ``Liu, M.A.P. Chamikara``). Split only when the comma
        # is followed by an initial block and a surname-like word, which does
        # not break conventional ``Lastname, J.`` formatting.
        r",\s+(?=(?:[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ.\-]*\s+)+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’´¨`\-]+)|"
        # Surname particles may be chained (``de las Casas`` / ``van den
        # Berg``), so consume the complete particle prefix before requiring
        # the capitalized surname.  A single-particle lookahead incorrectly
        # kept ``Chaplot, D. S., de las Casas, D`` as one author.
        r",\s+(?=(?:(?:de|del|der|den|van|von|da|das|las|la)\s+){1,3}[A-ZÀ-ÖØ-Ý])",
        text,
    )
    authors = []
    for p in parts:
        p = re.sub(r"^and\s+", "", p.strip(), flags=re.IGNORECASE).rstrip(",.")
        if p and len(p) > 1 and p.lower().replace(" ", "") not in ("etal", "etal.", "others"):
            authors.append(p)
    # Preserve the complete author list. Truncating at ten names causes valid
    # long-author references (for example Mistral 7B) to look inconsistent
    # against index records and can incorrectly trigger a hallucination flag.
    return authors


async def _llm_extract(chunk: str, llm_client: object) -> Reference:
    import json as _json

    prompt = f"""\
Extract bibliographic fields from the reference text below. Return ONLY a JSON object with these keys \
(use null when unknown): title, authors (list of strings), year (integer), venue, doi, arxiv_id, urls (list).

The text may be garbled or have OCR artifacts — do your best to recover the real values.

Reference text:
{chunk}

JSON:"""
    try:
        response = await llm_client.complete(prompt)  # type: ignore[union-attr]
        # Strip markdown code fences if the model wraps the response
        text = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
        data = _json.loads(text)
        return Reference(
            raw=chunk,
            title=data.get("title"),
            authors=data.get("authors") or [],
            year=data.get("year"),
            venue=data.get("venue"),
            doi=data.get("doi"),
            arxiv_id=data.get("arxiv_id"),
            urls=data.get("urls") or [],
        )
    except Exception:
        return Reference(raw=chunk)


async def enrich_with_llm(refs: list[Reference], llm_client: object) -> list[Reference]:
    """Async post-processing: use LLM to recover fields for refs that regex couldn't parse.

    Only processes refs with no title, DOI, or arXiv ID (UNRESOLVABLE candidates).
    Runs extractions concurrently in batches to avoid overwhelming the LLM.
    """
    import asyncio as _asyncio

    async def _try_enrich(ref: Reference) -> Reference:
        if ref.title or ref.doi or ref.arxiv_id or not ref.raw.strip():
            return ref
        enriched = await _llm_extract(ref.raw, llm_client)
        # Only replace if LLM actually found something useful
        return enriched if (enriched.title or enriched.doi or enriched.arxiv_id) else ref

    return list(await _asyncio.gather(*[_try_enrich(r) for r in refs]))


def parse_plaintext(text: str, style_override: Style | None = None) -> list[Reference]:
    chunks = split(text)
    if not chunks:
        return []
    style = style_override or detect_style(chunks)
    return [extract_fields(chunk, style) for chunk in chunks]
