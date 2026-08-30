from __future__ import annotations

import re
import asyncio
import unicodedata
import xml.etree.ElementTree as ET

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter

_BASE = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _entry_to_candidate(entry: ET.Element) -> Candidate:
    def text(tag: str) -> str | None:
        el = entry.find(f"atom:{tag}", _NS)
        return el.text.strip() if el is not None and el.text else None

    title = text("title")
    if title:
        title = re.sub(r"\s+", " ", title)

    year = None
    published = text("published")
    if published:
        m = re.match(r"(\d{4})", published)
        if m:
            year = int(m.group(1))

    authors = []
    for author in entry.findall("atom:author", _NS):
        name_el = author.find("atom:name", _NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    arxiv_id = None
    id_el = entry.find("atom:id", _NS)
    if id_el is not None and id_el.text:
        m = re.search(r"arxiv\.org/abs/([^\s/v]+)", id_el.text)
        if m:
            arxiv_id = m.group(1)

    doi = None
    doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
    if doi_el is not None and doi_el.text:
        doi = doi_el.text.strip()

    return Candidate(
        title=title,
        authors=authors,
        year=year,
        venue="arXiv",
        doi=doi,
        arxiv_id=arxiv_id,
        source="arxiv",
    )


def _author_query_parts(ref: Reference) -> list[str]:
    """Return conservative arXiv author clauses from a parsed citation.

    arXiv's title index is not a bibliography resolver: a title can change
    between a preprint and a conference version, and PDF extraction can lose
    punctuation or a complete line.  Author clauses give us an independent
    route to the preprint without treating every paper by an author as a hit.
    """
    parts: list[str] = []
    for author in ref.authors:
        value = " ".join(str(author or "").split()).strip()
        if not value:
            continue
        # Keep the query short and avoid sending citation suffixes such as
        # ``et al.`` to the arXiv parser.  Both ``Last, First`` and ``First
        # Last`` are accepted by arXiv's author search.
        value = re.sub(r"\bet\s+al\.?\b", "", value, flags=re.IGNORECASE).strip(" ,")
        if value and value not in parts:
            parts.append(value)
    return parts[:4]


def _title_keywords(title: str) -> list[str]:
    """Extract useful ASCII title terms for author-constrained searches."""
    stop = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
        "is", "of", "on", "or", "the", "to", "using", "via", "with",
        "against", "attack", "attacks", "defense", "defending", "based",
        "approach", "method", "model", "models", "learning", "neural",
        "robust", "robustness",
    }
    words = re.findall(r"[A-Za-z0-9]+", title)
    return [w for w in words if len(w) > 2 and w.casefold() not in stop][:6]


class ArXivAdapter(SourceAdapter):
    name = "arxiv"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=get_settings().request_timeout)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def lookup_doi(self, doi: str) -> Candidate | None:
        # Handle arXiv-issued DOIs: 10.48550/arXiv.XXXX.XXXXX
        m = re.search(r"10\.48550/arXiv\.(\d+\.\d+)", doi, re.IGNORECASE)
        if m:
            return await self.lookup_arxiv_id(m.group(1))
        return None

    async def lookup_arxiv_id(self, arxiv_id: str) -> Candidate | None:
        client = await self._get_client()
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        # Explicit IDs are definitive identifiers.  arXiv occasionally
        # responds with transient 429/5xx errors; do not turn one such API
        # hiccup into NOT_FOUND.  Retry a few times with a short backoff before
        # returning no result, while keeping the request budget bounded.
        for attempt in range(3):
            try:
                r = await client.get(
                    _BASE,
                    params={"id_list": clean_id, "max_results": "1"},
                    headers={"User-Agent": "Paper-audit/0.5 (citation verification)"},
                )
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    entries = root.findall("atom:entry", _NS)
                    if entries:
                        return _entry_to_candidate(entries[0])
                if r.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        delay = min(float(retry_after), 3.0) if retry_after else 0.75 * (attempt + 1)
                    except ValueError:
                        delay = 0.75 * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                break
            except (httpx.HTTPError, ET.ParseError):
                if attempt < 2:
                    await asyncio.sleep(0.75 * (attempt + 1))
                    continue
                break
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        if ref.arxiv_id:
            c = await self.lookup_arxiv_id(ref.arxiv_id)
            return [c] if c else []
        if not ref.title and not ref.authors:
            return []
        client = await self._get_client()
        # First try the exact title. Quoting the title keeps this pass strict;
        # an unquoted ``ti:`` query is interpreted as a loose token search and
        # can return unrelated records. If the exact pass returns nothing, a
        # second token-based pass handles punctuation, hyphenation, and PDF
        # line-break differences that are common in extracted references.
        # PDF extraction may retain typographic ligatures (``Certiﬁed``).
        # Normalize compatibility characters before querying arXiv so exact
        # title searches do not miss the canonical ASCII record.
        clean_title = unicodedata.normalize("NFKC", ref.title or "")
        clean_title = re.sub(r'["\\]+', " ", clean_title).strip()
        title_queries: list[str] = [f'ti:"{clean_title}"'] if clean_title else []
        # arXiv's title parser is sensitive to punctuation and occasionally
        # fails an otherwise valid exact-title query. Use the meaningful title
        # words as a conservative second pass; the normal record-linkage
        # scorer still decides whether any returned candidate is acceptable.
        words = re.findall(r"[A-Za-z0-9]+", clean_title)
        if len(words) >= 4:
            meaningful = [w for w in words if len(w) > 2]
            if len(meaningful) >= 3:
                title_queries.append("ti:(" + " AND ".join(meaningful[:12]) + ")")

        # A title-only lookup misses legitimate preprints when the cited
        # version uses a different title (or when extraction truncated it).
        # Search each of the first few cited authors with several meaningful
        # title terms.  The normal record-linkage scorer still decides whether
        # a returned item is actually the same work, so this broadens recall
        # without making an author-only result authoritative.
        keywords = _title_keywords(clean_title)
        author_queries: list[str] = []
        for author in _author_query_parts(ref):
            escaped_author = author.replace('"', ' ')
            if keywords:
                # Later title terms are usually more discriminative (e.g.
                # ``spectral regularization`` rather than ``defense against``).
                # Keep at most three to avoid over-constraining arXiv's parser.
                query_keywords = keywords[-3:]
                author_queries.append(
                    f'au:"{escaped_author}" AND ' + " AND ".join(f"all:{word}" for word in query_keywords)
                )
            else:
                author_queries.append(f'au:"{escaped_author}"')

        seen_ids: set[str] = set()
        candidates: list[Candidate] = []
        async def fetch(query: str) -> None:
            try:
                r = await client.get(
                    _BASE, params={"search_query": query, "max_results": "5"}
                )
                if r.status_code != 200:
                    return
                root = ET.fromstring(r.text)
                for entry in root.findall("atom:entry", _NS):
                    candidate = _entry_to_candidate(entry)
                    key = candidate.arxiv_id or (candidate.title or "").casefold()
                    if key and key in seen_ids:
                        continue
                    if key:
                        seen_ids.add(key)
                    candidates.append(candidate)
            except (httpx.HTTPError, ET.ParseError):
                return

        # Prefer the strict title routes.  Only when both title queries miss
        # do we spend additional requests on the author+keyword route.  This
        # keeps normal references inexpensive while recovering preprints whose
        # title changed between arXiv and the cited conference version.
        for query in title_queries:
            await fetch(query)
            if candidates:
                break
        if not candidates:
            for query in author_queries:
                await fetch(query)
        return candidates
