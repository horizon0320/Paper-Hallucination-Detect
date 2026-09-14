from __future__ import annotations

import asyncio
import re
import urllib.parse
import os
import weakref

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter, SourceUnavailableError

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,venue,externalIds,abstract"

# Semantic Scholar documents an introductory API-key limit of 1 request per
# second. Unauthenticated traffic is a shared pool and may be throttled more
# aggressively, so keep one process-wide (per event loop) queue at 1.1 RPS.
# The delay can be increased without a code change when a network is noisy.
_SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS = float(
    os.getenv("PAPERAUDIT_SEMANTIC_SCHOLAR_MIN_INTERVAL", "1.1")
)
_SEMANTIC_SCHOLAR_CONCURRENCY_LIMIT = 1
_LOOP_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()
_LOOP_NEXT_REQUEST: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, float] = weakref.WeakKeyDictionary()


def _get_semantic_scholar_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _LOOP_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_SEMANTIC_SCHOLAR_CONCURRENCY_LIMIT)
        _LOOP_SEMAPHORES[loop] = semaphore
    return semaphore


async def _wait_for_semantic_scholar_slot() -> None:
    loop = asyncio.get_running_loop()
    next_allowed = _LOOP_NEXT_REQUEST.get(loop, 0.0)
    now = loop.time()
    if next_allowed > now:
        await asyncio.sleep(next_allowed - now)
        now = loop.time()
    _LOOP_NEXT_REQUEST[loop] = now + _SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS


def _paper_to_candidate(paper: dict) -> Candidate:
    authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
    ext = paper.get("externalIds", {})
    doi = ext.get("DOI")
    arxiv_id = ext.get("ArXiv")

    return Candidate(
        title=paper.get("title"),
        authors=authors,
        year=paper.get("year"),
        venue=paper.get("venue"),
        doi=doi,
        arxiv_id=arxiv_id,
        source="semantic_scholar",
    )


class SemanticScholarAdapter(SourceAdapter):
    name = "semantic_scholar"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            settings = get_settings()
            headers = {}
            if getattr(settings, "semantic_scholar_api_key", None):
                headers["x-api-key"] = settings.semantic_scholar_api_key
            self._client = httpx.AsyncClient(
                timeout=settings.request_timeout,
                headers=headers,
            )
        return self._client

    async def _get(self, url: str, params: dict) -> dict | None:
        """GET with 429 retry, raising when the registry never answered.

        A parsed JSON body is returned even when it holds zero records,
        because that is a real answer from the registry.  A transport
        failure, a non-200 status, or an exhausted retry budget is not:
        returning None there used to make ``search()`` report an empty
        candidate list, which the existence check read as Semantic Scholar
        holding no copy of a real paper.  Semantic Scholar throttles
        unauthenticated traffic hard, so this turned routine rate limiting
        into false NOT_FOUND verdicts.
        """
        client = await self._get_client()
        reason = "no attempt"
        async with _get_semantic_scholar_semaphore():
            for attempt in range(3):
                await _wait_for_semantic_scholar_slot()
                try:
                    r = await client.get(url, params=params)
                except httpx.HTTPError as exc:
                    raise SourceUnavailableError(type(exc).__name__) from exc
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    # A 404 on an identifier route is the registry answering
                    # that it holds no such record, which is evidence; an error
                    # status is not.
                    return None
                reason = "HTTP " + str(r.status_code)
                if r.status_code == 429:
                    retry_after = r.headers.get("retry-after", "")
                    try:
                        server_delay = float(retry_after)
                    except (TypeError, ValueError):
                        server_delay = 0.0
                    await asyncio.sleep(max(2 ** attempt, server_delay, _SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS))
                    continue
                break
        raise SourceUnavailableError(reason)

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def lookup_doi(self, doi: str) -> Candidate | None:
        data = await self._get(f"{_BASE}/paper/DOI:{doi}", {"fields": _FIELDS})
        return _paper_to_candidate(data) if data and data.get("title") else None

    async def lookup_arxiv_id(self, arxiv_id: str) -> Candidate | None:
        clean = re.sub(r"v\d+$", "", arxiv_id.strip())
        data = await self._get(f"{_BASE}/paper/arXiv:{clean}", {"fields": _FIELDS})
        return _paper_to_candidate(data) if data and data.get("title") else None

    async def lookup_url(self, url: str) -> Candidate | None:
        encoded = urllib.parse.quote(url, safe="")
        data = await self._get(f"{_BASE}/paper/URL:{encoded}", {"fields": _FIELDS})
        return _paper_to_candidate(data) if data and data.get("title") else None

    async def search(self, ref: Reference) -> list[Candidate]:
        if not ref.title:
            return []
        data = await self._get(
            f"{_BASE}/paper/search",
            {"query": ref.title, "fields": _FIELDS, "limit": "5"},
        )
        if data:
            return [_paper_to_candidate(p) for p in data.get("data", [])]
        return []
