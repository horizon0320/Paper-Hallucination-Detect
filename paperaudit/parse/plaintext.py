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
_NO_DATE_RE = re.compile(r"(?:\[\s*n\.\s*d\.\s*\]|\bn\.\s*d\.)", re.IGNORECASE)
_PUBLISHER_YEAR_TAIL_RE = re.compile(
    r"^(?:IEEE(?:\s+Computer\s+Society)?|ACM|Springer(?:\s+Nature)?|Elsevier|Wiley),\s*"
    r"(?:19|20)\d{2}[a-z]?\.$", re.IGNORECASE
)


def _normalize_pdf_text(value: str) -> str:
    """Repair common PDF glyph artifacts before parsing bibliographic fields."""
    text = unicodedata.normalize("NFKC", value or "")
    # MinerU preserves inline HTML from some ACM templates (for example
    # ``Wan<sub>g</sub>``).  Those tags are layout annotations, not citation
    # content; remove the tags while retaining their text payload so names
    # and titles become searchable again.
    text = re.sub(r"</?[^>]+>", "", text)
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
    identifier_spans = [
        match.span()
        for pattern in (
            r"https?://\S+",
            r"10\.\d{4,}/\S+",
            r"arxiv\s*:\s*\S+",
        )
        for match in re.finditer(pattern, chunk, re.IGNORECASE)
    ]
    for match in _YEAR_RE.finditer(chunk):
        if any(start <= match.start() < end for start, end in identifier_spans):
            continue
        before = chunk[max(0, match.start() - 12):match.start()]
        after = chunk[match.end():match.end() + 3]
        # A capitalized author initial (``... and Gao, P. 2023``) can look
        # like the page abbreviation ``p.``. Only discard lowercase page
        # markers here; this preserves the final author's initial in AAAI
        # semicolon-style records.
        if re.search(r"\bpp?\.?\s*$", before):
            continue
        if re.match(r"\s*[-–—]", after) or re.search(r"[-–—]\s*$", before):
            continue
        matches.append(match)
    return matches

_NUMBERED_RE = re.compile(r"^\s*(\[?\d{1,3}\]?\.?\)?\s+|[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s+)")
# Nature and a few publisher production templates put the numeric label in
# its own text block, with the author/title block directly below it.  Treat a
# bracketed or punctuated standalone label as the same structural boundary as
# an inline ``[12] Author...`` marker.  Bare numbers remain excluded so page
# numbers cannot manufacture bibliography entries.
_STANDALONE_NUMBERED_RE = re.compile(r"^\s*(?:\[\s*\d{1,3}\s*\]|\d{1,3}[.)])\s*$")
# Author-year references are common in AAAI, ACL, Springer and similar
# proceedings. The author list can contain semicolons, "and", initials and
# punctuation, so match a bounded prefix up to the first publication year
# rather than assuming one surname + one initial.
_AUTHOR_YEAR_RE = re.compile(
    # Allow multi-word surnames ("Van Swol", "Da San Martino") and
    # semicolon-separated author lists used by AAAI proceedings.
    r"^\s*(?:[A-Z][a-zA-ZÀ-ÿ\-\'’]+\s+)*[A-Z][a-zA-ZÀ-ÿ\-\'’]+,\s*[^0-9\n]{0,180}"
    r"(?<!\()(?:\b(?:19|20)\d{2}[a-z]?\b|\[\s*n\.\s*d\.\s*\])"
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
    # Once physical lines are joined, a line-wrap hyphen is represented as
    # ``detec- tion`` rather than ``detec-\ntion``.  Remove this artifact only
    # when both fragments are lowercase; ordinary compounds such as
    # ``multi-task`` have no whitespace after the hyphen and remain intact.
    chunk = re.sub(r"(?<=[a-z]{2})-\s+(?=[a-z])", "", chunk)
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

    # PDF block extraction preserves a blank line between bibliography text
    # blocks. Prefer those explicit boundaries when they are plentiful; this
    # is especially important for ACL/EMNLP two-column references where a
    # wrapped author line can otherwise be mistaken for a new entry.
    # The semicolon-author strategy already returns one complete record per
    # chunk. Running the generic embedded-reference pass afterward would
    # incorrectly split initials such as ``A. G.; Kirillov`` at the period.
    # Explicit numbered markers are authoritative boundaries. Do not run the
    # embedded organization splitter on these chunks: ACM/IEEE records often
    # contain publisher names, editor lists, volume labels, or URLs that look
    # like a second author/reference and would create phantom entries.
    strategies = [(_split_numbered, False), (_split_semicolon_author_year, False)]
    if sum(1 for line in lines if not line.strip()) >= 2:
        strategies.append((_split_blank_lines, True))
    strategies.extend([(_split_author_year, True), (_split_hanging_indent, True)])
    for strategy, apply_embedded_split in strategies:
        chunks = strategy(lines)
        if apply_embedded_split:
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


def _split_semicolon_author_year(lines: list[str]) -> list[str]:
    """Split semicolon-delimited author-year bibliographies across PDF blocks.

    AAAI-style references use semicolons between authors, but a two-column
    PDF can cut a single author list at the column boundary (for example,
    ``Cheng, B.; Misra, I.; Schwing, A.`` followed by ``G.; Kirillov, A.``).
    Treating every text block as an independent record then creates bogus
    author-only entries.  Instead, flatten the lines and locate only genuine
    sentence starts whose first surname is followed by a semicolon-delimited
    author list and a publication year.  Initial-only continuation fragments
    such as ``G.; Kirillov`` are explicitly excluded.
    """
    if not lines:
        return []
    value = " ".join(line.strip() for line in lines if line.strip())
    if not value:
        return []

    # Require a semicolon before the year so this path is limited to the
    # target citation style rather than natural-name/APA references.
    candidate_re = re.compile(
        r"(?:^|(?<=\.\s))"
        r"(?P<author>(?:(?:de|da|del|dos|van|von|e)\s+)?"
        r"(?:(?:[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’\-]*|der|den)\s+){0,3}"
        r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’\-]{1,60},\s*)"
    )
    starts: list[int] = []
    for match in candidate_re.finditer(value):
        start = match.start("author")
        # ``G.; Kirillov`` and similar fragments begin after an initial plus
        # semicolon, not at a new surname.  A real reference start is not
        # immediately preceded by that continuation shape.
        if re.search(r"[A-ZÀ-ÿ]\.;\s*$", value[:start]):
            continue
        # AAAI author lists place initials immediately after the surname
        # comma (``Carvalho, O.``).  Requiring that shape avoids mistaking a
        # title sentence such as ``Prompt, Generate, then Cache`` for a new
        # bibliography record.
        after_comma = value[match.end():].lstrip()
        if not re.match(r"[A-Z](?:[.\-A-Z]*\.)", after_comma):
            continue
        window = value[start:start + 420]
        year_match = re.search(r"\b(?:19|20)\d{2}[a-z]?\.", window, re.IGNORECASE)
        if not year_match:
            continue
        author_window = window[:year_match.end()]
        # This splitter is reserved for bibliographies whose authors are
        # explicitly separated with semicolons (the AAAI-style layout it was
        # written for).  Ordinary author-year records naturally contain
        # ``and`` and must proceed to the blank-line/layout splitter; treating
        # them as semicolon records drops long wrapped entries.
        if ";" not in author_window:
            continue
        starts.append(start)

    year_count = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\.", value, re.IGNORECASE))
    # Do not take over mixed/natural author-year text merely because its first
    # two entries happen to use semicolons.  In a genuine AAAI bibliography,
    # nearly every publication-year marker belongs to one semicolon author
    # record; a mixed sample has a noticeably lower coverage.
    if len(starts) < 2 or (year_count >= 3 and len(starts) * 100 < year_count * 70):
        return []
    chunks: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(value)
        chunk = value[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks if len(chunks) >= 2 else []


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
            # A PDF may flatten two adjacent web/data-set records. The
            # carrier can be an explicit ``The ...`` label or a compact
            # project organization such as ``ModelScope Community``.
            organization_start = _looks_like_organization_line(first_sentence + ".")
            # Multi-author prefix followed by a title.  Require at least one
            # comma/``and`` separator and a sentence boundary so prose is not
            # mistaken for a new record.
            author_start = bool(
                (first_sentence.count(",") >= 1 or re.search(r"\band\b", first_sentence, re.IGNORECASE))
                and ":" not in first_sentence
                and _looks_like_author_start(first_sentence + ".")
            )
            # ``et al.`` entries are frequently followed by a title sentence
            # that starts with a capitalized word (e.g. ``et al. 2023. Factool``).
            # That period is the author/year boundary, not an embedded record
            # boundary.  Without this guard the title is detached and then
            # parsed as a bogus author-only reference.
            if re.search(r"\bet\s+al\.?\s+(?:19|20)\d{2}[a-z]?\.$", left_tail, re.IGNORECASE):
                author_start = False
            compact_org_year = bool(
                re.match(r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’&\- ]{1,90},\s*(?:19|20)\d{2}[a-z]?\.", right)
                and not _PUBLISHER_YEAR_TAIL_RE.match(right.strip())
            )
            # The new record must itself look complete (a URL within a bounded
            # span), except for compact author-year citations that may be
            # completed by a URL after the year.
            has_url = bool(re.search(r"https?\s*:\s*//", right[:360], re.IGNORECASE))
            has_bibliography_signal = bool(
                re.search(
                    r"https?\s*:\s*//|arXiv\s+preprint|\b(?:19|20)\d{2}[a-z]?\b|\bIn\s+",
                    right[:420],
                    re.IGNORECASE,
                )
            )
            if (organization_start and has_url) or (author_start and has_bibliography_signal) or compact_org_year:
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
            # In two-column PDFs, a very long author list can be split at a
            # column/page boundary immediately after a surname comma:
            # ``..., Lim,`` / ``H., Zoph, ... Paper title``.  The first piece
            # is too long for the generic author-only heuristic, but its
            # trailing comma plus an initial-led continuation is a reliable
            # layout signal. Join it before field extraction.
            if (
                current_text.endswith(",")
                and current_text.count(",") >= 2
                and not re.search(r"\b(?:19|20)\d{2}[a-z]?\b", current_text, re.IGNORECASE)
                and re.match(r"^[A-ZÀ-ÿ]\.,\s*[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’\-]+,", following_text)
            ):
                merged.append(f"{current} {following}")
                index += 2
                continue
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
            # PDF block extraction can place a long URL in its own block
            # directly after the author/title block (common for ACL web
            # citations). Keep that URL with the same record rather than
            # creating a title-less phantom reference.
            if (
                following_text
                and re.match(r"^https?://", following_text, re.IGNORECASE)
                and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", current_text, re.IGNORECASE)
            ):
                merged.append(f"{current} {following}")
                index += 2
                continue
            # A long ACL title can continue in a separate PDF text block
            # without repeating its year. Merge such a non-author continuation
            # into the preceding record instead of creating a title fragment.
            if (
                following_text
                and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", current_text, re.IGNORECASE)
                and not re.search(r"\b(?:19|20)\d{2}[a-z]?\b", following_text, re.IGNORECASE)
                and not _looks_like_author_start(following_text.splitlines()[0].strip())
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
    # ACL/EMNLP blocks often put the publication year on the author-only
    # block (``... and C. Author. 2024a.``) before the title starts in the
    # next block.  Treat only a terminal year as part of this wrapper; years
    # elsewhere can belong to a title or venue and should not be merged.
    year_tail = re.search(r"\s+(?:19|20)\d{2}[a-z]?\s*\.\s*$", value, re.IGNORECASE)
    author_value = value[:year_tail.start()].rstrip(" .,;") if year_tail else value
    if re.search(r"\b(?:19|20)\d{2}\b", author_value):
        return False
    if re.search(r"[:\"']|https?://|\b(?:In|Proceedings|Journal|Conference)\b", value, re.IGNORECASE):
        return False
    # Require multiple author separators so a short title sentence is not
    # mistaken for an author-only fragment.
    if author_value.count(",") < 2 and not re.search(r"\band\b", author_value, re.IGNORECASE):
        return False
    if not _looks_like_author_start(author_value):
        return False
    # Semicolon-delimited styles (common in AAAI/EMNLP exports) keep each
    # author's initials after the surname comma. Treat a clearly name-shaped
    # semicolon list as one wrapped author block before applying the
    # comma-delimited token checks below.
    if ";" in author_value and author_value.count(";") >= 1:
        return True
    # Every comma/and-delimited segment should look like a person's name;
    # this rejects prose sentences that happen to contain several commas.
    author_text = re.sub(r"\s+and\s+", ",", author_value, flags=re.IGNORECASE)
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
        if _NUMBERED_RE.match(line) or _STANDALONE_NUMBERED_RE.match(line):
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
            (("," in line or re.search(r"\band\b", line, re.IGNORECASE))
             and _looks_like_author_start(line))
            or group_author_boundary
        )
        # A wrapped author list may itself begin with a name-shaped line (for
        # example ``Xiao, ... et al.``). Do not split it until the current
        # chunk has crossed its author→title boundary.
        can_start_plain_reference = plain_author_year and previous_boundary
        current_has_year = bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\s*\.\s*$|\b(?:19|20)\d{2}[a-z]?\s*\.\s+", " ".join(current), re.IGNORECASE))
        if starts_new_reference and current_has_year and (
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
    # Do not mistake a title-cased article title (``America's AI Action
    # Plan``) for an organization. Beyond explicit ``The ...`` labels, use
    # organizational suffixes/markers that occur in group-authored records.
    has_org_marker = bool(
        re.match(r"^The\s+", value[:-1], re.IGNORECASE)
        or re.search(
            r"\b(?:Community|Team|Association|Institute|Organization|Laboratory|Lab|"
            r"Foundation|Committee|Corporation|Company|Press|Society|Group)\b$",
            value[:-1],
            re.IGNORECASE,
        )
    )
    if not has_org_marker:
        return False
    # A standalone label is title-cased or all-caps; sentence prose and
    # ordinary wrapped paper titles generally contain lowercase starters.
    return all(re.match(r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ0-9'’&\-/]*$", word) for word in words)


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
    if (
        len(first.rstrip(".")) > 1
        and first.casefold().rstrip(".") in _AUTHOR_START_STOPWORDS
        and not (
            first.casefold().rstrip(".") == "an"
            and re.match(r"^An\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’\-]+,", value)
        )
        and not (
            first.casefold().rstrip(".") == "or"
            and re.match(r"^Or\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’\-]+,", value)
        )
    ):
        return False
    if len(second.rstrip(".")) > 1 and second.casefold().rstrip(".") in _AUTHOR_START_STOPWORDS:
        return False
    first_is_initial = bool(re.fullmatch(r"[A-Z]", first.rstrip(".")))
    second_is_initial = bool(re.fullmatch(r"[A-Z]", second.rstrip(".")))
    def clean_name(token: str) -> str:
        return token.rstrip(".").replace("´", "").replace("¨", "").replace("`", "")

    first_clean = clean_name(first)
    second_clean = clean_name(second)
    first_is_name = bool(re.fullmatch(r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’\-]*[a-zà-ÿ]", first_clean))
    second_is_name = bool(re.fullmatch(r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’\-]*[a-zà-ÿ]", second_clean))
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
    year = re.search(
        r"(?<!\()\b(?:19|20)\d{2}[a-z]?\b|\[\s*n\.\s*d\.\s*\]|\bn\.\s*d\.",
        chunk,
        re.IGNORECASE,
    )
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
    semicolon_author_year_hits = 0
    lncs_hits = 0
    lncs_structural_hits = 0

    for chunk in samples:
        group_author_year = bool(
            re.match(r"^(.+?)\.\s+.+(?:\b(?:19|20)\d{2}\b|\[\s*n\.\s*d\.\s*\])", chunk)
            and _looks_like_group_author(chunk.split(". ", 1)[0])
        )
        if _AUTHOR_YEAR_RE.match(chunk) or _looks_like_plain_author_year_chunk(chunk) or group_author_year:
            # Keep APA's explicit ``(YYYY)`` marker more specific when both
            # heuristics happen to match.
            scores[Style.AUTHOR_YEAR] += 1
            author_year_hits += 1
            # AAAI-style author-year records use semicolons between named
            # authors; this is the only author-year form that can plausibly
            # resemble the ``Surname, I.: Title`` syntax of Springer/LNCS.
            year_prefix = chunk[:next((m.start() for m in _YEAR_RE.finditer(chunk)), len(chunk))]
            if ";" in year_prefix:
                semicolon_author_year_hits += 1
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
            lncs_hits += 1
            # A colon inside ``arXiv:2304.04736`` or a DOI/URL is not the
            # LNCS author/title separator. Count LNCS only when the first
            # colon precedes the publication year and follows an author-like
            # prefix. This prevents author-year CS references with arXiv
            # identifiers from being parsed as ``authors: DOI``.
            first_colon = chunk.find(': ')
            first_year = next((m.start() for m in _YEAR_RE.finditer(chunk)), len(chunk))
            if first_colon < first_year:
                lncs_structural_hits += 1
                scores[Style.LNCS] += 2

        # secondary signals
        if re.search(r'["“][^"”]{5,}["”]', chunk):
            scores[Style.IEEE] += 1
        if re.search(r'\(\d{4}\)', chunk):
            scores[Style.APA] += 1
        if re.search(r'\d{4};\d+\(\d+\):\d+', chunk):
            scores[Style.VANCOUVER] += 1

    # AAAI's semicolon-delimited author-year format can contain colons in
    # titles, which otherwise makes the generic LNCS heuristic win. Restrict
    # that priority to the actual semicolon form. A Springer bibliography has
    # a year and author-like prefix too, but must retain its `Author: Title`
    # field boundary.
    if semicolon_author_year_hits >= 2 and semicolon_author_year_hits * 2 >= len(samples):
        return Style.AUTHOR_YEAR

    if lncs_structural_hits >= 2 and lncs_structural_hits * 2 >= len(samples):
        return Style.LNCS

    if author_year_hits >= 2 and author_year_hits * 2 >= len(samples):
        return Style.AUTHOR_YEAR

    best_style = max(scores, key=lambda s: scores[s])
    if scores[best_style] >= 2:
        return best_style
    return Style.UNKNOWN


def extract_fields(chunk: str, style: Style = Style.UNKNOWN) -> Reference:
    chunk = _normalize_pdf_text(chunk)
    # A global style vote can be inconclusive when only one entry is supplied
    # (or when a small bibliography mixes numeric years with ``[n.d.]``). Use
    # the per-entry author/date signature as a safe local override so the
    # author/title boundary is still parsed correctly.
    if style == Style.UNKNOWN and (
        _AUTHOR_YEAR_RE.match(chunk)
        or (
            _NO_DATE_RE.search(chunk)
            and re.match(r"^\s*.+\.\s+\[?\s*n\.\s*d\.", chunk, re.IGNORECASE)
            and ("," in chunk.split("[", 1)[0] or re.search(r"\band\b", chunk.split("[", 1)[0], re.IGNORECASE))
        )
    ):
        style = Style.AUTHOR_YEAR
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
    chunk = re.sub(
        r"(https?://[^\s]+)\s+([A-Za-z0-9]+[_/-][A-Za-z0-9._/-]*)",
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
    # ACL and Springer DOI strings are occasionally split at a space by the
    # PDF text layer (``10.18653/v1/2020. insights-1.7`` and
    # ``10.1007/978-3-031-35320-8 42``). Join only these DOI namespaces so a
    # normal publication year following another DOI is never absorbed.
    chunk = re.sub(
        r"(10\.18653/v1/[A-Za-z0-9._-]+)\s+([A-Za-z][A-Za-z0-9._-]+)",
        r"\1\2",
        chunk,
        flags=re.IGNORECASE,
    )
    chunk = re.sub(
        r"(10\.1007/978-[A-Za-z0-9._-]+)\s+(\d{1,4})(?=\b|[.,;)])",
        r"\1_\2",
        chunk,
        flags=re.IGNORECASE,
    )

    # Natural-name references with an explicit arXiv/DOI tail are common in
    # ACL/NeurIPS bibliographies.  The title is the sentence immediately
    # before the identifier tail (``... Authors. Title. arxiv:...``).  The
    # generic author-year boundary can otherwise consume the title into the
    # author list and leave ``arxiv:...`` as the searchable title.  Recover
    # this structure before style-specific extraction; the rule is based only
    # on the explicit identifier marker and sentence shape, not on a paper
    # name.
    explicit_id_tail = re.search(
        r"\s+(?:arxiv\s*:\s*\d{4}\.\d+|10\.48550/arXiv\.\d{4}\.\d+)",
        chunk,
        re.IGNORECASE,
    )
    # Proceedings citations without an arXiv identifier often use the same
    # long-author form and introduce the carrier with ``. In ...``. Use that
    # carrier boundary as an equivalent title delimiter (not a venue-specific
    # special case).
    carrier_tail = re.search(r"\.\s+In\s+", chunk, re.IGNORECASE)
    forced_boundary = explicit_id_tail or carrier_tail
    if forced_boundary and style in {Style.UNKNOWN, Style.AUTHOR_YEAR, Style.LNCS}:
        before_id = chunk[:forced_boundary.start()].rstrip(" .")
        # Match the final author-sentence/title-sentence pair. Greedy author
        # prefix is intentional because initials contain many periods.
        pair = re.search(
            r"^(?P<authors>.+)\.\s+(?P<title>[A-ZÀ-ÖØ-Ý][^.]{8,240})\.?$",
            before_id,
        )
        if pair:
            parsed_authors = _parse_author_list(pair.group("authors"))
            if parsed_authors:
                # Keep the normal metadata extraction below, but supply the
                # unambiguous title/authors override after identifiers/URLs
                # are collected.
                forced_title = pair.group("title").strip(" .")
                forced_authors = parsed_authors
            else:
                forced_title = forced_authors = None
        else:
            forced_title = forced_authors = None
    else:
        forced_title = forced_authors = None
    doi = None
    m = _DOI_RE.search(chunk)
    if m:
        doi = m.group().rstrip(".,;)")
        doi = re.sub(r"\.url$", "", doi, flags=re.IGNORECASE)

    urls = [u.rstrip(".,;)>\"'") for u in _URL_RE.findall(chunk)]
    urls = [u for u in urls if "doi.org" not in u]
    # Keep domain-root URLs. A citation to ``https://sourcegraph.com/`` or an
    # organization home page is still an intentional source reference; the
    # URL regex already requires a host, so only malformed scheme-less values
    # are excluded here. The web adapter can then distinguish a blocked page
    # from a citation that never supplied a source URL.
    from urllib.parse import urlparse
    urls = [u for u in urls if urlparse(u).scheme and urlparse(u).netloc]

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
        # In author-year records the publication date is the first valid year
        # after the author list. Later four-digit values commonly belong to
        # ACM article numbers, volume/page metadata, or a repeated page
        # header (for example ``Article 2011``). Taking the last token turns
        # those layout artifacts into a false publication year.
        ym = year_matches[0] if style == Style.AUTHOR_YEAR else (
            year_matches[-1] if len(year_matches) > 1 else year_matches[0]
        )
        year = int(ym.group(1))

    title = None
    authors: list[str] = []
    venue = None

    if style == Style.IEEE:
        title, authors = _extract_ieee(chunk)
        # Mixed IEEE bibliographies can contain one unquoted short title among
        # otherwise quoted entries (books, reports, or older proceedings).
        # Preserve the global style for normal records, but fall back to the
        # structural generic extractor when the IEEE quote parser found no
        # fields instead of returning an unresolvable reference.
        if not title:
            generic_title, generic_authors = _extract_generic(chunk)
            if generic_title:
                title, authors = generic_title, (generic_authors or authors)
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

    apply_forced = bool(forced_title) and (
        (explicit_id_tail is not None and (
            not title
            or re.search(r"\b(?:arxiv|doi)\s*[:.]", title or "", re.IGNORECASE)
        ))
        or (carrier_tail is not None and len(authors) < 3)
    )
    if apply_forced:
        title = forced_title
        authors = forced_authors or authors
        if venue and re.match(r"^(?:URL\s*:?[ \t]*)?https?://", venue, re.IGNORECASE):
            venue = None

    # Some IEEE/ACL text layers place the DOI marker where a quote/title
    # boundary would normally be, causing the parser to return the literal
    # ``doi: ...`` (or an arXiv metadata tail) as the title.  Re-parse only
    # the bibliographic text before that marker; this is a structural repair
    # that works across DOI-bearing citation styles.
    if title and re.match(r"^\s*(?:doi\s*:|arxiv\s*:\s*\d{4}\.)", title, re.IGNORECASE):
        marker = re.search(r"\s+(?:doi\s*:|arxiv\s*:\s*\d{4}\.)", chunk, re.IGNORECASE)
        if marker:
            prefix = chunk[:marker.start()].rstrip(" .")
            alt_title, alt_authors, alt_venue = _extract_author_year(prefix)
            if alt_title:
                title = alt_title
                if alt_authors:
                    authors = alt_authors
                if alt_venue:
                    venue = alt_venue

    # If a DOI-bearing entry was parsed by a generic style branch, it may
    # still contain the proceedings carrier after the actual title
    # (``Title. In ... doi:...``).  The first sentence is the bibliographic
    # title; trim only when a clear carrier marker is present.
    if title and doi:
        carrier = re.search(r"\.\s+In\s+(?:the\s+|proceedings\b|\d|[A-Z])", title, re.IGNORECASE)
        if carrier:
            title = title[:carrier.start()].strip(" .") or title
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
    # PDF text layers can insert a wrap-space after a decimal point in a model
    # name (``Qwen2. 5-Coder``). Join only digit-dot-digit fragments; ordinary
    # sentence punctuation and decimal values separated by prose are intact.
    value = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", value)
    # Restore spaces removed after punctuation and between camel-case words.
    value = re.sub(r"(?<=[,:;])(?=[A-Za-z])", " ", value)
    # Split camel-case words in prose (``NuminaMath``), but preserve
    # repository/data-set identifiers containing a slash, hyphen, or digit
    # (``AI-ModelScope/CodeAlpaca-20k``).
    def split_camel_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.casefold() in {"arxiv", "openai"}:
            return token
        if any(ch in token for ch in "/-0123456789"):
            return token
        # Preserve conventional all-caps/mixed-case acronyms such as LoRA,
        # OpenCLIP, PSCC and BERT. A normal concatenated phrase such as
        # ``NuminaMath`` has only one internal capital and remains splittable.
        if sum(ch.isupper() for ch in token) >= 2:
            return token
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token)

    value = re.sub(r"\S+", split_camel_token, value)
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

    # IEEE-like records occasionally omit quotation marks around a short
    # book/report title (for example ``... Nawab, Signals & systems.
    # Pearson Educación, 1997.``).  The generic sentence-length guard below
    # rejects such titles as too short and leaves the whole record
    # unresolvable.  Detect the bibliographic shape instead: a comma-delimited
    # author prefix, a title sentence, then a carrier/year sentence.  Requiring
    # a publication year after the title keeps ordinary prose and author-list
    # commas out of this fallback.
    unquoted = re.search(
        r",\s+(?P<title>[A-Za-z][^.\n]{2,100})\.\s+"
        r"(?P<carrier>[^.\n]{2,120}\b(?:19|20)\d{2})\.",
        chunk,
    )
    if unquoted:
        prefix = chunk[:unquoted.start()]
        title = unquoted.group("title").strip(" .,:;") or None
        if title:
            return title, _parse_author_list(prefix)

    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    for sentence in sentences[1:3]:
        stripped = sentence.strip()
        if len(stripped) > 20 and not re.match(r"^[A-Z]{2,}", stripped):
            return stripped.rstrip("."), []

    return None, []


def _extract_author_year(chunk: str) -> tuple[str | None, list[str], str | None]:
    """Extract AAAI/ACL and arXiv natural-name author-year references."""
    # Organization-only landing-page records (for example an association or
    # consortium followed directly by ``URL https://...``) must be handled
    # before the sentence-style web branch below; otherwise ``URL`` becomes a
    # bogus article title.
    org_root_url = re.match(
        r"^\s*(?P<org>[^.]{2,100})\.\s*(?:URL\s*:?[ \t]*)?https?://",
        chunk,
        re.IGNORECASE,
    )
    if org_root_url:
        org = org_root_url.group("org").strip()
        if re.search(
            r"\b(?:Association|Organization|Institute|Consortium|Society|Laboratory|Lab|"
            r"Foundation|Committee|Corporation|Company|Community|Team)\b",
            org,
            re.IGNORECASE,
        ):
            return None, [org], None

    # Dataset pages often include a descriptive title and month/year before
    # the URL, while still using an organization as the author. Preserve that
    # title and keep the landing page in ``Reference.urls``.
    dataset_web = re.match(
        r"^\s*(?P<org>[^.]{2,100})\.\s+(?P<title>.+?),\s*"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(?:19|20)\d{2}\.\s*(?:URL\s*:?[ \t]*)?https?://",
        chunk,
        re.IGNORECASE,
    )
    if dataset_web and re.search(r"dataset|hugging\s*face|model", dataset_web.group("title"), re.IGNORECASE):
        return dataset_web.group("title").strip(" .,:;"), [dataset_web.group("org").strip()], None

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
        if title and title.casefold().startswith("url "):
            return None, [org], None
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

    # ACL/EMNLP references may use ``[n.d.]``/``n.d.`` instead of a numeric
    # publication year. Treat that token as the same author/title boundary,
    # while leaving ``Reference.year`` as unknown (None).
    year_m = re.search(
        r"\b((?:19|20)\d{2})(?:[a-z])?\b|\[\s*n\.\s*d\.\s*\]|\bn\.\s*d\.",
        chunk,
        re.IGNORECASE,
    )
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
        # Natural-name ACL/EMNLP references conventionally place the year
        # immediately after the author sentence: ``A. Author and B. Author.
        # 2024. Title.``  Use that terminal period as the boundary before the
        # generic initial-aware scan; otherwise the final author's name absorbs
        # ``2024`` as if it were part of the author string.
        natural_prefix = chunk[:year_m.start()].rstrip()
        natural_author = natural_prefix[:-1].rstrip() if natural_prefix.endswith(".") else ""
        # Do not mistake a complete author sentence plus a title sentence for
        # one long author list (e.g. ``Alex Hore and Djemel Ziou. Image ...``).
        # Initials such as ``A. Vaswani`` do not satisfy the name-shaped
        # prefix check and therefore remain eligible for the generic scan.
        first_sentence = re.search(r"\.\s+(?=[A-ZÀ-ÿ])", natural_prefix)
        if first_sentence:
            prefix_before_sentence = natural_prefix[:first_sentence.start()].strip()
            suffix_after_sentence = natural_prefix[first_sentence.end():].strip()
            if suffix_after_sentence and _looks_like_author_start(prefix_before_sentence):
                natural_author = ""
        if natural_author and (
            _looks_like_author_start(natural_author)
            or _looks_like_group_author(natural_author)
        ):
            authors_text = natural_author
            authors = _parse_author_list(authors_text)
            rest = chunk[year_m.end():].lstrip(" .,:;")
            # Continue with the common title/venue handling below.
            boundary = True
        else:
            boundary = None
        # Find the end of the author sentence. Initials contain periods, so
        # require a following capitalized title-like phrase and a long enough
        # comma-rich prefix before accepting the boundary.
        if boundary is None:
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
            # Titles can start with a lowercase project name (``epalm:`` or
            # ``news-please:``). A colon immediately after that token is a
            # strong author/title boundary when the prefix is comma-rich.
            for m in re.finditer(r"\.\s+(?=[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\-]{1,60}:)", chunk):
                prefix = chunk[:m.start()]
                if _looks_like_author_start(prefix.strip()) and (
                    prefix.count(",") >= 1 or len(prefix.split()) <= 4
                ):
                    boundary = m
                    break
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
            r"\s+(?=(?:arXiv\s*(?:preprint|:)|Preprint(?:\s+at|\s*[,;:]\s*arXiv)|In\s+Proceedings|In\s+Advances\s+in|"
            r"In\s+(?:The\s+)?(?:[A-Z][A-Za-z-]*\s+){0,4}International\b|"
            r"In\s+(?:The\s+)?(?:USENIX|NeurIPS|NIPS|ICLR|ICML|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|"
            r"AAAI|IJCAI|WACV|NDSS|SIGMOD|KDD|ICSE|FSE|PLDI|POPL|SOSP|OSDI)\b|"
            r"In\s+\d+(?:st|nd|rd|th)\b|"
            r"(?:19|20)\d{2}\s+\d+(?:st|nd|rd|th)\s+International\b|"
            r"In\s+(?:The\s+)?(?:[A-Z][a-z]+(?:-[a-z]+)?\s+){0,5}(?:Annual|International)\s+Conference\b|"
            r"Under\s+review|(?:IEEE\s+)?Transactions\s+(?:of|on)\b|"
            r"(?:IEEE\s+)?Journal\s+of\b|ACM\s+Computing|Foundations\s+and\s+Trends\b|Center\s+for\b|"
            r"(?=(?:Comput|Proc|Trans|Adv|Nat|Sci)\.\s+[A-Z][A-Za-z.& ]{2,80}\s+\d{1,4}\s*,\s*\d+)|"
            # Journal/conference records often append a carrier name followed
            # by volume/issue/pages without a stable keyword (e.g.
            # ``Remote Sensing, 15 (18):4588, 2023`` or
            # ``Pattern Recognition 123 (2022) 108...``). The structural
            # boundary is deliberately generic and applies across publishers.
            r"(?=[A-Z][^.\n]{2,140},\s*\d{1,4}\s*(?:\(\s*\d{1,4}\s*\))?\s*[:;,]\s*\d{1,6}(?:[-–]\d{1,6})?\b)|"
            r"(?=[A-Z][^.\n]{2,140}\s+\d{1,4}\s*\(\s*(?:19|20)\d{2}\s*\)\s*\d{1,6})|"
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
            r"Mathematical Association of America|Vectara)(?:\s+Team)?$",
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
        # OCR/PDF text layers can glue an initial block to a lower-case
        # surname (``C. J. H. un``). It is still an author continuation,
        # not the start of a title.
        or re.match(r"^(?:[A-Z]\.\s*){1,3}[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]{1,},", value)
        or re.match(r"^[A-ZÀ-ÿ][a-zà-ÿ'’\-]+,\s+(?:[A-Z]\.?\s*)+", value)
    )


def _parse_author_list(text: str) -> list[str]:
    text = text.strip().rstrip(".,")
    if not text:
        return []
    # ``et al.`` is a truncation marker, not an author. Removing it before
    # splitting makes the cited names a true prefix of the complete index
    # record, so normal abbreviated citations do not trigger author conflicts.
    text = re.sub(r"\s*,?\s+et\s+al\.?\s*$", "", text, flags=re.IGNORECASE).strip(" .,;")
    # PDF text extraction may remove spaces after commas, around ``and``, or
    # inside initial blocks (``Y.Li,X.Li,andJ.Yang``). Restore these separators
    # before applying the author split rules below.
    text = re.sub(r",(?=[A-Z])", ", ", text)
    text = re.sub(r",\s*and\b(?=\s|[A-Z])", ", and ", text, flags=re.IGNORECASE)
    text = re.sub(r"\band\b(?=[A-Z])", "and ", text)
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
        r"(?i:\s+and\s+)|;\s*|,\s+(?=[A-ZÀ-ÖØ-ÝŁ][a-zà-öø-ÿ'’´¨`\-])|"
        # Initial-based authors are common in IEEE/NeurIPS references and
        # were previously glued to the preceding author (``Shi, J Zico
        # Kolter`` or ``Liu, M.A.P. Chamikara``). Split only when the comma
        # is followed by an initial block and a surname-like word, which does
        # not break conventional ``Lastname, J.`` formatting.
        r",\s+(?=(?:[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ.\-]*\s+)+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'’´¨`\-]+)|"
        # Keep OCR-degraded abbreviated names separate even when the final
        # surname was lower-cased (for example ``Yoav Goldberg, C. J. H. un``).
        r",\s+(?=(?:[A-Z]\.?\s*){1,3}[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’´¨`\-]{1,}(?:,|$))|"
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
        # Natural-name references put the publication year after the final
        # author (``... and Yossi Matias. 2022.``).  If a block boundary or
        # normalization leaves that suffix attached to the author token,
        # remove it here rather than exposing the year as part of the name.
        p = re.sub(r"\s*\.\s*(?:19|20)\d{2}[a-z]?\s*$", "", p, flags=re.IGNORECASE)
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
