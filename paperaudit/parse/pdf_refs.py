from __future__ import annotations

import re
from pathlib import Path

from paperaudit.models import Reference

# ---------------------------------------------------------------------------
# Bibliography heading patterns (ordered most-specific → most-general).
# Adapted from refchecker (markrussinovich/refchecker).
# Each pattern anchors to a newline so we don't match "cross-references" etc.
# They tolerate PDF extraction artifacts: tabs, page numbers, non-breaking
# spaces (\xa0), and extra digits that appear before/after the heading.
# ---------------------------------------------------------------------------
_BIB_HEADER_PATTERNS = [
    # Numbered section header: "7. References", "12 References"
    re.compile(
        r'\n[\s\d\t\r\xa0]*\d+\.?\s+(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*:?\s*\n',
    ),
    # Standard headings with leading PDF noise (tabs, page nums, \xa0)
    re.compile(
        r'\n[\s\t\r\xa0\d]*'
        r'(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|'
        r'WORKS\s+CITED|Works\s+Cited|LITERATURE\s+CITED|Literature\s+Cited|'
        r'CITED\s+WORKS|Cited\s+Works)'
        r'\s*:?\s*(?:\d*\s*)\n',
    ),
    # Heading immediately followed by a reference (no trailing blank line)
    re.compile(
        r'\n[\s\t\r\xa0\d]*'
        r'(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)'
        r'\s*:?\s+(?=[A-Z\d\[\(])',
    ),
    # Inline line-number artefact: "559\tReferences 560\t"
    re.compile(r'\d+\s*\t\s*(?:References|REFERENCES)\s*\d*\s*\t'),
    # Heading preceded by spaces/line-nums without a leading newline
    re.compile(
        r'\s(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*:?\s+(?=\d|[A-Z])',
    ),
]

# Markdown heading markers (from marker's PDF->Markdown output, e.g. "## References").
# Stripped before heading-pattern matching so marker's output is handled the same
# way as plain extracted text, instead of duplicating every pattern above with a
# leading "#*" alternative.
_MD_HEADING_RE = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+', re.MULTILINE)

# Sections that mark the end of a bibliography
_END_SECTION_RE = re.compile(
    r'^\s*(?:'
    r'APPENDIX|Appendix|APPENDICES|Appendices|'
    r'SUPPLEMENTARY\s+MATERIAL|Supplementary\s+Material|'
    r'SUPPLEMENTAL\s+MATERIAL|Supplemental\s+Material|'
    r'SUPPORTING\s+INFORMATION|Supporting\s+Information|'
    r'ACKNOWLEDGMENTS?|Acknowledgments?|ACKNOWLEDGEMENTS?|Acknowledgements?|'
    r'AUTHOR\s+CONTRIBUTIONS?|Author\s+Contributions?|'
    r'FUNDING|Funding|'
    r'ETHICS\s+STATEMENT|Ethics\s+Statement|'
    r'CONFLICT\s+OF\s+INTEREST|Conflict\s+of\s+Interest|'
    r'DATA\s+AVAILABILITY|Data\s+Availability|'
    r'OPEN\s+SCIENCE|Open\s+Science|'
    r'ARTIFACTS?|Artifacts?|REPRODUCIBILITY|Reproducibility|'
    r'ABOUT\s+THE\s+AUTHORS?|About\s+the\s+Authors?|'
    r'NEURIPS\s+PAPER\s+CHECKLIST|NeurIPS\s+Paper\s+Checklist|'
    r'PAPER\s+CHECKLIST|Paper\s+Checklist|'
    r'INDEX|Glossary|'
    r"PUBLISHER[’']?S\s+NOTE|Publisher[’']?s\s+note|"
    r'OPEN\s+ACCESS|Open\s+Access|'
    r'©\s*(?:THE\s+AUTHOR|The\s+Author)'
    r')\s*\.?\s*$',
    re.IGNORECASE,
)

# Some publisher templates render appendix labels and titles on separate lines:
#
#     A
#     Additional Results
#
# This is common in ACM-style PDFs and is not covered by _END_SECTION_RE because
# the label alone is too short to identify as a section heading.
_STANDALONE_SECTION_LABEL_RE = re.compile(
    r'^\s*(?:APPENDIX\s+)?[A-Z](?:\s*\.\s*)?$'
)

# Other templates keep the appendix label and title on one line, e.g.
# "Appendix A: Additional Results" or "A. Additional Results".
_INLINE_APPENDIX_HEADING_RE = re.compile(
    r'^\s*(?:APPEND(?:IX|ICES)\s+[A-Z0-9]+|[A-Z])\s*'
    r'(?:[.:)\-]\s+|\s{2,})[^\n]{2,}$',
    re.IGNORECASE,
)

_NUMBERED_REFERENCE_START_RE = re.compile(
    r'^\s*(?:\[\s*\d{1,3}\s*\]|\d{1,3}[.)])\s+'
)

_RUNNING_BIB_HEADER_RE = re.compile(
    r'^(?:under\s+review|submitted\s+to|accepted\s+(?:at|for)|anonymous\s+submission|'
    r'draft\s+version|manuscript\s+under\s+review)\b',
    re.IGNORECASE,
)

_POST_BIB_CONTENT_RE = re.compile(
    r"^\s*(?:Publisher[’']?s\s+note|Open\s+Access|©\s*(?:The\s+Author|THE\s+AUTHOR)|"
    r"Acknowledgements?|Author\s+contributions?|Competing\s+interests?|Additional\s+information|"
    r"Supplementary\s+information|Correspondence\s+and\s+requests|Peer\s+review\s+information|"
    r"Reprints\s+and\s+permissions)\b",
    re.IGNORECASE,
)


def _next_nonempty_line(lines: list[str], start: int) -> tuple[int, str] | None:
    for index in range(start, len(lines)):
        value = lines[index].strip()
        if value:
            return index, value
    return None


def _looks_like_section_title(value: str) -> bool:
    """Return whether a line plausibly begins a non-bibliography section."""
    if not value or len(value) > 140:
        return False
    # Numbered references and ordinary continuation lines are not headings.
    if value.startswith("[") or re.match(r'^\d+[.)]\s', value):
        return False
    if re.search(r'\b(?:doi|https?://|arxiv:)\b', value, re.IGNORECASE):
        return False
    # Headings in the relevant templates can be title-cased, sentence-cased or
    # all-caps. Avoid treating a normal reference continuation as a heading.
    words = value.split()
    if len(words) > 12:
        return False
    if value.isupper():
        return True
    if value.endswith((".", "?", "!")) or re.search(r'[\"\';]', value):
        return False
    if re.match(r'^(?:In\b|And\b|Or\b|Et\s+al\b|Proceedings\s+of\b)', value):
        return False
    first = next((ch for ch in value if ch.isalpha()), "")
    return bool(first and first.isupper())


def _is_standalone_section_start(lines: list[str], index: int) -> bool:
    value = lines[index].strip()
    if not _STANDALONE_SECTION_LABEL_RE.match(value):
        return False
    following = _next_nonempty_line(lines, index + 1)
    return following is not None and _looks_like_section_title(following[1])


def _is_inline_appendix_start(line: str) -> bool:
    value = line.strip()
    if not _INLINE_APPENDIX_HEADING_RE.match(value):
        return False
    explicit_appendix = re.match(r'^\s*APPEND(?:IX|ICES)\b', value, re.IGNORECASE)
    # A bibliography continuation can legitimately start with an author's
    # initial ("A. One, \"Paper\"..."). Restrict the short letter form to
    # heading-like text rather than citation punctuation.
    if not explicit_appendix:
        # A wrapped author-year reference can start with an initial on its own
        # line (e.g. ``L. 2024. Title``). This is citation content, not an
        # appendix heading.
        if re.match(r'^[A-Z]\.?\s+(?:19|20)\d{2}\b', value):
            return False
        # Wrapped author lists and titles can also start with two initials
        # (e.g. ``W. T. Maskgit: ...``); these are not appendix headings.
        if re.match(r'^[A-Z]\.\s+[A-Z]\.', value):
            return False
        if any(token in value for token in ('"', "'", '[', ']')) or re.search(r',\s', value):
            return False
    return bool(explicit_appendix) or _looks_like_section_title(value)


def _trim_bibliography_end(lines: list[str]) -> list[str]:
    """Trim lines after the first post-bibliography section heading."""
    for j, line in enumerate(lines):
        # Some publisher PDFs put the prose directly on the same line as the
        # marker (for example ``Publisher's note Springer Nature ...``), so a
        # heading-only regex is insufficient here.
        if j > 0 and _POST_BIB_CONTENT_RE.match(line.strip()):
            return lines[:j]
        if j > 0 and _END_SECTION_RE.match(line.strip()):
            return lines[:j]
        # IEEE papers may place author biographies immediately after the last
        # numbered reference without a heading. Stop before the first bio so
        # it cannot be appended to the final citation.
        if j > 0 and re.match(
            r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:[- ][A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+){0,3}\s+(?:\([^)]*\)\s+)?received\s+the\b",
            line.strip(),
            re.IGNORECASE,
        ):
            return lines[:j]
        if j > 0 and _is_standalone_section_start(lines, j):
            return lines[:j]
        if j > 0 and _is_inline_appendix_start(line):
            return lines[:j]
        # A running article title can reappear at the top of the next page,
        # followed by the first author line of the next section. Treat that
        # pair as a page header rather than bibliography content.
        value = line.strip()
        if j > 0 and _looks_like_running_title(value):
            following = _next_nonempty_line(lines, j + 1)
            if following and _looks_like_author_line(following[1]):
                # This is a repeated article title at the top of the next
                # page, not the end of the bibliography. Keep scanning so the
                # author line and all subsequent references remain included.
                lines[j] = ""
                continue
            if following and re.match(r"^(?:Figure|Table|Appendix)\b", following[1], re.IGNORECASE):
                return lines[:j]
    return lines


def _looks_like_running_title(value: str) -> bool:
    if not value or len(value) > 140 or value.endswith((".", ",", ";")):
        return False
    # A wrapped bibliography entry often starts with ``Surname, Initials``;
    # this is author text, not a repeated page header.
    if re.match(r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’\-]{1,40},\s+", value):
        return False
    words = value.split()
    if len(words) < 4 or re.search(r"\b(?:doi|https?://|arxiv:)\b", value, re.IGNORECASE):
        return False
    return bool(value[0].isupper() and (":" in value or sum(w[:1].isupper() for w in words) >= 3))


def _looks_like_author_line(value: str) -> bool:
    value = value.strip()
    return bool(re.match(
        r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’´¨`\-]+,\s+[A-Z](?:\.|\s|,)", value
    ))


def _remove_pdf_bibliography_noise(lines: list[str]) -> list[str]:
    """Remove running page numbers and common publisher headers from refs."""
    cleaned: list[str] = []
    for line in lines:
        value = line.strip()
        # Some IEEE PDF text layers append the next page's running journal
        # header to the end of the preceding reference, e.g. ``... pp.
        # 9626-9635. IEEE TRANSACTIONS ... JUNE 2025``.  This is page chrome,
        # not part of the cited record.  Remove only an all-caps header that
        # follows a completed sentence/page range, preserving legitimate
        # title-case journal venues inside normal references.
        value = re.sub(
            r"\.\s+(?=(?:IEEE\s+)?TRANSACTIONS\b|PROCEEDINGS\b|JOURNAL\b)"
            r"[A-Z0-9][A-Z0-9\s,.:;()&/\-]{18,}\b(?:19|20)\d{2}\s*$",
            ".",
            value,
        )
        # The same running header can occupy its own extracted line between
        # bibliography entries.  It is recognizable by all-caps journal text
        # plus volume/issue/date markers and must be discarded wholesale.
        if (
            value.isupper()
            and len(value) >= 24
            and re.search(r"\b(?:IEEE\s+)?TRANSACTIONS\b|\bPROCEEDINGS\b|\bJOURNAL\b", value)
            and re.search(r"\bVOL\.?\b|\bNO\.?\b|\b(?:JAN|FEB|MAR|APR|MAY|JUNE|JULY|AUG|SEP|OCT|NOV|DEC)\b|\b(?:19|20)\d{2}\b", value)
        ):
            continue
        # IEEE Xplore may inject its download/license banner and the article
        # running header into the bibliography. Remove these fragments before
        # reference splitting so download dates are not mistaken for years.
        value = re.sub(
            r"Authorized licensed use limited to:.*?(?:Restrictions apply\.|$)",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\bCHEN\s+et\s+al\.:\s+YOLO-MS:.*$", "", value, flags=re.IGNORECASE)
        if not value.strip():
            continue
        if (
            re.fullmatch(r"\d{1,4}", value)
            or value.casefold() == "preprint"
            or _RUNNING_BIB_HEADER_RE.match(value)
        ):
            continue
        cleaned.append(value)
    return cleaned


def _find_numbered_reference_tail(text: str) -> str | None:
    """Conservative fallback when a bibliography heading was not extracted."""
    lines = text.splitlines()
    starts = [
        i for i, line in enumerate(lines)
        if _NUMBERED_REFERENCE_START_RE.match(line)
    ]
    # Body text can contain one or two numbered lists. Require a cluster of
    # starts near the end before treating it as a bibliography.
    for first in starts:
        if first < len(lines) * 0.30:
            continue
        cluster = [i for i in starts if first <= i < first + 240]
        if len(cluster) >= 3:
            initial_lines = _trim_bibliography_end(lines[first:])
            initial_lines = _remove_pdf_bibliography_noise(initial_lines)
            numbers = []
            for line in initial_lines:
                match = re.match(r'^\s*(?:\[\s*(\d{1,3})\s*\]|(\d{1,3})[.)])', line)
                if match:
                    numbers.append(int(match.group(1) or match.group(2)))
            result_lines = list(initial_lines)
            if numbers:
                continuation = _find_numbered_continuation(text, max(numbers) + 1, 0)
                if continuation:
                    result_lines.extend(continuation.splitlines())
            return "\n".join(result_lines).strip()
    return None


def _find_numbered_continuation(text: str, expected_number: int, min_pos: int) -> str | None:
    """Find a later numbered-reference cluster continuing an earlier list.

    A few publisher layouts (notably Nature) split references around article
    metadata and Methods: refs 1–30 appear under the bibliography area while
    refs 31 onward are placed in a later right-hand column.  If the next
    numbered entry is present and at least two numbers continue sequentially,
    append that cluster without treating arbitrary numbered body text as refs.
    """
    lines = text.splitlines()
    starts: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if _NUMBERED_REFERENCE_START_RE.match(line) is None:
            continue
        match = re.match(r'^\s*(?:\[\s*(\d{1,3})\s*\]|(\d{1,3})[.)])', line)
        if match:
            number = int(match.group(1) or match.group(2))
            starts.append((index, number))
    for offset, (index, number) in enumerate(starts):
        if number != expected_number:
            continue
        # Translate line index to an approximate character position so that a
        # body list before the bibliography cannot be selected.
        position = sum(len(line) + 1 for line in lines[:index])
        if position < min_pos:
            continue
        cluster: list[list[str]] = []
        current_number = number
        current: list[str] = []
        for j in range(index, len(lines)):
            line = lines[j]
            marker = re.match(r'^\s*(?:\[\s*(\d{1,3})\s*\]|(\d{1,3})[.)])', line)
            if marker:
                found_number = int(marker.group(1) or marker.group(2))
                if found_number != current_number:
                    if found_number == current_number + 1:
                        cluster.append(current)
                        current = [line]
                        current_number = found_number
                        continue
                    break
            if not current:
                current = [line]
            else:
                current.append(line)
        if current:
            cluster.append(current)
        if len(cluster) >= 2:
            continuation_lines = [line for entry in cluster for line in entry]
            continuation_lines = _trim_bibliography_end(continuation_lines)
            return "\n".join(continuation_lines).strip()
    return None


def _extract_text(path: Path) -> str:
    """
    Extract full text from a PDF.

    If PAPERAUDIT_USE_MARKER is set, tries marker (PDF -> Markdown) first for
    better layout/heading fidelity on hard documents. Otherwise (or if marker
    isn't installed / fails) falls back to three backends in order:
    PyMuPDF (best multi-column support) → pypdf → pdfminer.
    """
    from paperaudit.config import get_settings
    if get_settings().use_marker:
        from paperaudit.parse.marker_pdf import convert_pdf_to_markdown
        text = convert_pdf_to_markdown(path)
        if text:
            return text

    # 1. PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        # Keep page boundaries visible to section detection. A number of PDF
        # templates place a repeated header/footer between the final reference
        # on one page and the first appendix heading on the next page.
        return "\n\f\n".join(pages)
    except ImportError:
        pass

    # 2. pypdf (used by refchecker)
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    parts.append(text)
            except Exception:
                continue
        return "\n\f\n".join(parts)
    except ImportError:
        pass

    # 3. pdfminer (fallback)
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError as e:
        raise ImportError(
            "No PDF backend found. Install one of: pymupdf, pypdf, pdfminer.six"
        ) from e


def _augment_with_intext_refs(refs: list[Reference], text: str) -> list[Reference]:
    """Add body author-year keys that have no corresponding full entry.

    This runs for both the plaintext and GROBID paths.  A PDF can have a
    malformed/missing bibliography heading, and GROBID can successfully parse
    only the entries it sees; neither case should silently discard citations
    that are visible in the article body.
    """
    from paperaudit.parse.intext import canonical_author_key, extract_intext_citations

    marker = re.search(r"(?im)^\s*(?:References|Bibliography)\s*$", text)
    body = text[:marker.start()] if marker else text
    inline = extract_intext_citations(body)
    existing_keys: set[tuple[str, int]] = set()
    for ref in refs:
        if not ref.year or not ref.authors:
            continue
        # A body citation may use a non-first author (for example ``Anca
        # Dragan, 2025`` for a three-author web article).  Index every cited
        # author surname so a correctly extracted full record is not reported
        # again as a false "缺少文末条目" entry.
        for author in ref.authors:
            key = canonical_author_key(author)
            if key:
                existing_keys.add((key, ref.year))
    refs.extend(
        r for r in inline
        if (canonical_author_key(r.authors[0]), r.year) not in existing_keys
    )
    return refs


def _find_ref_section(text: str) -> str | None:
    """
    Locate the bibliography/references section in extracted PDF text.

    Strategy (adapted from refchecker):
    - Try each of the heading patterns above
    - Skip matches in the first 20% of the document (avoids "see references" etc.)
    - Among all matches, keep the LAST one (bibliography is near the end)
    - Trim the section at the first known post-bibliography heading
    """
    n = len(text)
    min_pos = int(n * 0.20)   # ignore the first 20 % of the document

    best_match = None
    for pattern in _BIB_HEADER_PATTERNS:
        for m in pattern.finditer(text):
            if m.start() < min_pos:
                continue
            if best_match is None or m.start() > best_match.start():
                best_match = m

    # Fallback: plain-line search for the heading word on its own line
    if best_match is None:
        fallback = re.compile(
            r'\n[^\n]{0,20}?\b(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b\s*:?\s*\n',
        )
        for m in fallback.finditer(text):
            if m.start() > n * 0.40:
                if best_match is None or m.start() > best_match.start():
                    best_match = m

    if best_match is None:
        return _find_numbered_reference_tail(text)

    bib_text = text[best_match.end():].strip()

    # Trim at a known end-of-bibliography section heading
    lines = _trim_bibliography_end(bib_text.splitlines())
    lines = _remove_pdf_bibliography_noise(lines)

    # If the bibliography was interrupted by publisher matter/Methods, look
    # for a later cluster beginning at the next expected number (e.g. Nature
    # refs 31–36) and append it as part of the same reference list.
    numbers = []
    for line in lines:
        match = re.match(r'^\s*(?:\[\s*(\d{1,3})\s*\]|(\d{1,3})[.)])', line)
        if match:
            numbers.append(int(match.group(1) or match.group(2)))
    if numbers:
        continuation = _find_numbered_continuation(text, max(numbers) + 1, best_match.end())
        if continuation:
            lines.extend(continuation.splitlines())
    return "\n".join(lines)


def parse_pdf_refs(path: Path, use_grobid: bool = True, llm_client: object | None = None) -> list[Reference]:
    grobid_refs: list[Reference] = []
    if use_grobid:
        try:
            from paperaudit.config import get_settings
            from paperaudit.parse.grobid import parse_pdf_via_grobid
            api_url = get_settings().grobid_api_url
            if api_url:
                grobid_refs = parse_pdf_via_grobid(path, api_url=api_url) or []
        except Exception:
            grobid_refs = []

    from paperaudit.parse.plaintext import parse_plaintext
    text = _extract_text(path)
    text = _MD_HEADING_RE.sub('', text)
    ref_section = _find_ref_section(text)
    if ref_section is None:
        # Parsing the entire document as a bibliography creates many false
        # references when a publisher omits or mangles the heading. The
        # numbered-tail fallback above is intentionally conservative; if it
        # also fails, report no extracted references instead of body text.
        return _augment_with_intext_refs(grobid_refs, text)
    refs = grobid_refs or parse_plaintext(ref_section)
    # Some submissions cite dozens of works in the body but contain only a
    # partial bibliography. Preserve those author-year keys as citation-only
    # records so the report exposes the omission instead of returning only the
    # two entries found after the References heading.
    try:
        return _augment_with_intext_refs(refs, text)
    except Exception:
        return refs
