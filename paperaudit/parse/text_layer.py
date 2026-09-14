"""Recover characters that MinerU's Markdown drops for some PDF fonts.

MinerU supplies the layout structure this project relies on (headings, list
items, per-entry boundaries), but its text re-serialization is lossy for PDFs
that use heavy kerning or ligature fonts.  The ACM acmart template is the
clearest example: it wraps many letters in sub/sup spans and, inside
bibliography entries, silently discards the letters those spans carried, so
"large language models" reaches the parser as "lar e lan ua e models" and
"Evaluating" as "Evaluatin".  The same flattening can lose a line-final
hyphen together with its break space, turning "LLM- based" into "LLMbased".

The PDF own text layer is unaffected, so this module keeps MinerU block
structure and re-reads each block characters from the embedded text layer
by bounding box.  Both repairs are gated by character order: a block is
replaced when the text layer holds the same characters in the same order
plus extra ones (letters were dropped), or when it holds exactly the same
characters and differs only in punctuation and whitespace placement
(hyphens and break spaces were lost).  Those gates mean the repair can
restore omitted glyphs and dashes but can never invent, reorder, or adopt
neighbouring text.  Scanned pages have no text layer, and any block whose
MinerU output already matches the text layer is left untouched, so OCR
input and well-behaved PDFs keep MinerU output verbatim.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


_TAG_RE = re.compile(r"</?[^>]+>")
_SPACE_RE = re.compile(r"\s+")

# Blocks shorter than this carry too little evidence to justify a swap: a
# two letter heading or a bare page number could otherwise match an
# unrelated region of the page.
_MIN_EVIDENCE = 12


def _norm(value: str) -> str:
    """Reduce text to comparable characters, ignoring punctuation and tags."""
    text = unicodedata.normalize("NFKC", value or "")
    text = _TAG_RE.sub("", text)
    return re.sub(r"[^0-9a-z]+", "", text.casefold())


def _plain(value: str) -> str:
    """Collapse tags and whitespace runs while keeping punctuation."""
    text = unicodedata.normalize("NFKC", value or "")
    text = _TAG_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    iterator = iter(haystack)
    return all(character in iterator for character in needle)


def _mineru_block_text(block: dict) -> str:
    """Join MinerU span text for one block or list item, line by line."""
    pieces: list[str] = []
    for line in block.get("lines") or []:
        spans = line.get("spans") or []
        value = "".join(
            str(span.get("content", ""))
            for span in spans
            if isinstance(span, dict)
        )
        if value.strip():
            pieces.append(_SPACE_RE.sub(" ", value).strip())
    return " ".join(pieces).strip()


def _pdf_text_lines(page) -> list[tuple[list[float], str]]:
    rows: list[tuple[list[float], str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            value = "".join(str(span.get("text", "")) for span in spans)
            if value.strip():
                bbox = line.get("bbox")
                if bbox and len(bbox) == 4:
                    rows.append((list(bbox), value.strip()))
    return rows


def _text_in_region(rows, bbox: list[float], tol: float = 1.5) -> str:
    """Collect PDF lines whose centre falls inside a MinerU block bbox.

    Using centres keeps a two-column bibliography honest: a line belongs to
    the block that actually contains it, so a neighbouring column can never
    be pulled in by a bounding box that merely overlaps the gutter.
    """
    x0, y0, x1, y1 = bbox
    picked: list[tuple[float, float, str]] = []
    for (bx0, by0, bx1, by1), value in rows:
        centre_x = (bx0 + bx1) / 2.0
        centre_y = (by0 + by1) / 2.0
        if x0 - tol <= centre_x <= x1 + tol and y0 - tol <= centre_y <= y1 + tol:
            picked.append((by0, bx0, value))
    picked.sort()
    return " ".join(value for _y, _x, value in picked).strip()


def collect_repairs(pdf_path: Path, middle_path: Path) -> list[tuple[str, str]]:
    """Return (mineru_text, pdf_text) pairs for lossy MinerU blocks.

    Propagates read and import failures so callers can treat the whole
    repair as optional and fall back to MinerU own output.
    """
    import fitz  # PyMuPDF

    data = json.loads(middle_path.read_text(encoding="utf-8", errors="replace"))
    pages = data.get("pdf_info")
    if not isinstance(pages, list):
        return []

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page_info in enumerate(pages):
            if page_index >= doc.page_count or not isinstance(page_info, dict):
                continue
            rows = _pdf_text_lines(doc[page_index])
            if not rows:
                continue  # scanned page: MinerU OCR text is the only evidence
            for block in page_info.get("para_blocks") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("image", "table", "code"):
                    continue
                # A MinerU list keeps one bibliography entry per item, and
                # each item carries its own bbox, so repairs stay accurate.
                units = block.get("items") if block.get("type") == "list" else [block]
                for unit in units or []:
                    if not isinstance(unit, dict):
                        continue
                    if unit.get("type") in ("image", "table"):
                        continue
                    bbox = unit.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    mineru_text = _mineru_block_text(unit)
                    key = _norm(mineru_text)
                    if len(key) < _MIN_EVIDENCE or key in seen:
                        continue
                    pdf_text = _text_in_region(rows, list(bbox))
                    candidate = _norm(pdf_text)
                    if not candidate:
                        continue
                    # Letters were dropped: the text layer keeps the same
                    # characters in order and adds the missing ones.
                    restored_letters = (
                        candidate != key
                        and len(candidate) > len(key)
                        and _is_subsequence(key, candidate)
                    )
                    # Same characters, different punctuation: MinerU
                    # flattened a line-final hyphen and its break space.
                    restored_punctuation = (
                        candidate == key
                        and _plain(mineru_text) != _plain(pdf_text)
                    )
                    if not (restored_letters or restored_punctuation):
                        continue
                    seen.add(key)
                    pairs.append((mineru_text, pdf_text))
    return pairs


def apply_repairs(markdown: str, pairs: list[tuple[str, str]]) -> tuple[str, int]:
    """Replace Markdown lines whose text matches a lossy MinerU block."""
    if not pairs:
        return markdown, 0
    lookup: dict[str, str] = {}
    for mineru_text, pdf_text in pairs:
        lookup.setdefault(_norm(mineru_text), pdf_text)
    heading_mark = chr(35)
    replaced = 0
    output: list[str] = []
    for line in markdown.split("\n"):
        body = line.lstrip(heading_mark).strip()
        replacement = lookup.get(_norm(body)) if body else None
        if replacement is None:
            output.append(line)
            continue
        depth = len(line) - len(line.lstrip(heading_mark))
        output.append(f"{heading_mark * depth} {replacement}" if depth else replacement)
        replaced += 1
    return "\n".join(output), replaced


def recover_pdf_text_layer(path: Path, markdown: str, output_dir: Path) -> tuple[str, int]:
    """Repair MinerU Markdown with the PDF text layer when MinerU lost glyphs."""
    middles = sorted(
        (p for p in output_dir.rglob("*_middle.json") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not middles:
        return markdown, 0
    try:
        pairs = collect_repairs(path, middles[0])
        return apply_repairs(markdown, pairs)
    except Exception:
        # The text layer is an enhancement: any failure (missing PyMuPDF,
        # an unreadable intermediate, an unexpected MinerU schema) must
        # leave the original MinerU output usable rather than aborting.
        return markdown, 0
