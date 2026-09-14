from __future__ import annotations

import re
import asyncio
import unicodedata
import xml.etree.ElementTree as ET
import weakref
import shutil

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter, SourceUnavailableError

_BASE = "https://export.arxiv.org/api/query"
# arXiv's export host is normally the most reliable API endpoint, but it can
# intermittently fail TLS negotiation or return an empty response from a
# regional proxy. Keep the canonical host as a read-only second route.
_BASE_ALIASES = (_BASE, "https://arxiv.org/api/query")
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_USER_AGENT = "Paper-audit/0.5 (citation verification)"
_ARXIV_CONCURRENCY_LIMIT = 3
_LOOP_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _get_arxiv_semaphore() -> asyncio.Semaphore:
    """Return an arXiv limiter scoped to the active event loop.

    The public export API is sensitive to bursts.  Serialize requests within
    a batch so every citation receives a meaningful lookup instead of several
    concurrent connection resets being interpreted as empty results.
    """
    loop = asyncio.get_running_loop()
    semaphore = _LOOP_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_ARXIV_CONCURRENCY_LIMIT)
        _LOOP_SEMAPHORES[loop] = semaphore
    return semaphore


# arXiv allows clients at most one API request every three seconds, and a
# tripped quota is shared by the whole host, so one 429 parks the batch.
_ARXIV_MIN_INTERVAL_SECONDS = 3.0
_ARXIV_QUOTA_COOLDOWN_SECONDS = 20.0
_ARXIV_RATE_LIMIT_BACKOFF_SECONDS = 5.0
_ARXIV_RATE_LIMIT_ATTEMPTS = 2
_LOOP_NEXT_ARXIV_REQUEST: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, float] = weakref.WeakKeyDictionary()


async def _wait_for_arxiv_slot() -> None:
    """Pace arXiv requests globally within the active verification batch.

    The export endpoint publishes a per-client request budget.  With several
    citations in flight, each issuing an exact-title query plus a token query,
    the burst overruns that budget and arXiv answers HTTP 429 with an empty
    body.  Spacing requests keeps every citation's lookup meaningful instead
    of trading quota for a batch of silent rate-limit responses.

    The budget is also shared: once one request is told to back off, every
    other request from this host meets the same answer.  _trip_arxiv_quota
    parks the whole batch for a cooldown window, which replaces dozens of
    independently doomed retries with one pause and leaves quota for the
    lookups that run afterwards.
    """
    loop = asyncio.get_running_loop()
    next_allowed = _LOOP_NEXT_ARXIV_REQUEST.get(loop, 0.0)
    now = loop.time()
    if next_allowed > now:
        await asyncio.sleep(next_allowed - now)
        now = loop.time()
    _LOOP_NEXT_ARXIV_REQUEST[loop] = now + _ARXIV_MIN_INTERVAL_SECONDS


def _trip_arxiv_quota(seconds: float) -> None:
    """Push the batch-wide arXiv schedule out by a quota cooldown."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    if deadline > _LOOP_NEXT_ARXIV_REQUEST.get(loop, 0.0):
        _LOOP_NEXT_ARXIV_REQUEST[loop] = deadline


def _is_atom_feed(text: str | None) -> bool:
    """Return whether a response body is a genuine Atom document.

    The export endpoint answers HTTP 429 with an empty body, and a 200 that
    carries a truncated page or an HTML error block is equally unusable.
    Only a parseable feed counts as the registry actually answering, even
    when that feed legitimately holds zero entries.
    """
    if not text or "<feed" not in text:
        return False
    try:
        ET.fromstring(text)
    except ET.ParseError:
        return False
    return True


def _feed_entries(text: str) -> list[ET.Element]:
    """Parse an Atom feed already validated by _is_atom_feed()."""
    root = ET.fromstring(text)
    return root.findall("atom:entry", _NS)


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
        self._prefer_curl = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # arXiv is rate-limited and occasionally stalls on a regional
            # proxy. Keep its per-request timeout bounded independently from
            # slower publisher lookups so a full batch remains interactive.
            timeout = min(float(get_settings().request_timeout), 8.0)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def _fetch_atom(self, endpoint: str, params: dict[str, str]) -> tuple[str | None, str]:
        """Query one arXiv route, reporting whether the service answered.

        Returns (atom_text, "ok") when arXiv returned a parseable feed and
        (None, reason) otherwise.  The split exists because arXiv answers
        HTTP 429 with an empty body: the previous code treated any non-200
        exactly like a legitimate empty feed, so a throttled batch reported
        arxiv=no_match for most citations and the verdict engine read that
        silence as evidence that a real preprint does not exist.

        The quota is per host and shared by both transports, so every
        request -- in-process or through the system HTTPS client -- first
        takes a paced slot, and a 429 waits out the window and retries on
        the same route.  Switching transports stays reserved for a client
        that cannot reach arXiv at all, because the other stack merely
        consumes the same budget.
        """
        reason = "no route attempted"
        for attempt in range(_ARXIV_RATE_LIMIT_ATTEMPTS):
            if self._prefer_curl:
                await _wait_for_arxiv_slot()
                text = await self._curl_query(endpoint, params)
                if _is_atom_feed(text):
                    return text, "ok"
                reason = "curl transport failed"
                self._prefer_curl = False
                continue
            await _wait_for_arxiv_slot()
            try:
                client = await self._get_client()
                r = await client.get(
                    endpoint,
                    params=params,
                    headers={"User-Agent": _USER_AGENT},
                )
            except httpx.HTTPError as exc:
                reason = type(exc).__name__
                await _wait_for_arxiv_slot()
                text = await self._curl_query(endpoint, params)
                if _is_atom_feed(text):
                    self._prefer_curl = True
                    return text, "ok"
                continue
            if _is_atom_feed(r.text):
                return r.text, "ok"
            reason = "HTTP " + str(r.status_code)
            if r.status_code != 429:
                break
            # Park the whole batch instead of retrying only this request: the
            # next paced slot lands past the cooldown, so no other citation
            # burns the same exhausted quota.
            _trip_arxiv_quota(_ARXIV_QUOTA_COOLDOWN_SECONDS)
        return None, reason

    async def _curl_query(self, endpoint: str, params: dict[str, str]) -> str | None:
        """Fallback to the OS HTTPS client for a public arXiv Atom query.

        In some local Python runtimes, arXiv's Atom API intermittently stalls
        despite the same canonical request succeeding with the system HTTPS
        stack. This preserves the API request and XML parser; it is not HTML
        scraping and is used only after the in-process client cannot return a
        usable response.
        """
        curl = shutil.which("curl")
        if not curl:
            return None
        command = [
            curl, "--silent", "--show-error", "--location", "--connect-timeout", "4",
            "--max-time", "12", "--get", endpoint,
        ]
        for key, value in params.items():
            command.extend(["--data-urlencode", f"{key}={value}"])
        command.extend(["-A", "Paper-audit/0.5 (citation verification)"])
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=14.0)
            if process.returncode != 0 or not stdout.strip():
                return None
            return stdout.decode("utf-8", errors="replace")
        except (asyncio.TimeoutError, OSError):
            return None

    async def lookup_doi(self, doi: str) -> Candidate | None:
        # Handle arXiv-issued DOIs: 10.48550/arXiv.XXXX.XXXXX
        m = re.search(r"10\.48550/arXiv\.(\d+\.\d+)", doi, re.IGNORECASE)
        if m:
            return await self.lookup_arxiv_id(m.group(1))
        return None

    async def lookup_arxiv_id(self, arxiv_id: str) -> Candidate | None:
        async with _get_arxiv_semaphore():
            return await self._lookup_arxiv_id_limited(arxiv_id)

    async def _lookup_arxiv_id_limited(self, arxiv_id: str) -> Candidate | None:
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        params = {"id_list": clean_id, "max_results": "1"}
        # An explicit identifier is definitive, so a route that cannot answer
        # must not be reported as "this paper is not on arXiv".  Try every
        # configured endpoint and both transports, and raise when none of
        # them actually replied.
        answered = False
        for endpoint in _BASE_ALIASES:
            text, outcome = await self._fetch_atom(endpoint, params)
            if outcome == "ok":
                answered = True
                entries = _feed_entries(text)
                if entries:
                    return _entry_to_candidate(entries[0])
        if not answered:
            raise SourceUnavailableError(
                "arXiv identifier lookup unavailable"
            )
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        async with _get_arxiv_semaphore():
            return await self._search_limited(ref)

    async def _search_limited(self, ref: Reference) -> list[Candidate]:
        if ref.arxiv_id:
            c = await self._lookup_arxiv_id_limited(ref.arxiv_id)
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
        any_answered = False
        # Only the strict title routes can establish that arXiv holds no copy of
        # the cited work. Author-constrained queries are a deliberate recall
        # fallback, so an empty author route says nothing about existence.
        title_routes_answered = False

        def add_entries(response_text: str) -> bool:
            """Parse an Atom payload and append its distinct records.

            A valid empty feed is a genuine non-match, so callers need the
            entry count rather than an HTTP status; that is also why
            _fetch_atom validates the body before this is ever reached.
            """
            for entry in _feed_entries(response_text):
                candidate = _entry_to_candidate(entry)
                key = candidate.arxiv_id or (candidate.title or "").casefold()
                if key and key in seen_ids:
                    continue
                if key:
                    seen_ids.add(key)
                candidates.append(candidate)
            return bool(candidates)

        async def fetch(query: str, *, strict_title: bool = False) -> None:
            nonlocal any_answered, title_routes_answered
            for endpoint in _BASE_ALIASES:
                params = {"search_query": query, "max_results": "5"}
                text, outcome = await self._fetch_atom(endpoint, params)
                if outcome != "ok":
                    continue
                any_answered = True
                if strict_title:
                    title_routes_answered = True
                add_entries(text)
                return

        # Prefer the strict title routes.  Only when both title queries miss
        # do we spend additional requests on the author+keyword route.  This
        # keeps normal references inexpensive while recovering preprints whose
        # title changed between arXiv and the cited conference version.
        for query in title_queries:
            await fetch(query, strict_title=True)
            if candidates:
                break
        if not candidates:
            for query in author_queries:
                await fetch(query)
                if candidates:
                    break
        # Every route failed to answer.  Returning [] here would tell the
        # verdict engine that arXiv holds no copy of the cited work, which is
        # how a 429 quota window used to become a fabricated-citation finding.
        if not candidates and (not any_answered or not title_routes_answered):
            raise SourceUnavailableError(
                "arXiv search unavailable after all routes"
            )
        return candidates
