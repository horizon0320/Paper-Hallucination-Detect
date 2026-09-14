from __future__ import annotations

import asyncio
import re
import unicodedata

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter, SourceUnavailableError

_BASE = "https://api.openalex.org"


class OpenAlexUnavailableError(SourceUnavailableError):
    """OpenAlex answered with an error rather than with zero records.

    Inherits the shared registry-availability contract so a throttle or outage
    can never be read as evidence that a cited paper does not exist.
    """


def _work_to_candidate(work: dict) -> Candidate:
    authors = []
    for a in work.get("authorships", []):
        name = a.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    doi = work.get("doi")
    if doi:
        doi = re.sub(r"^https?://doi\.org/", "", doi)

    ids = work.get("ids", {})
    arxiv_id = None
    arxiv_url = ids.get("arxiv", "")
    if arxiv_url:
        m = re.search(r"arxiv\.org/abs/([^\s/]+)", arxiv_url)
        if m:
            arxiv_id = m.group(1)

    venue = None
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    venue = src.get("display_name")

    pub_year = work.get("publication_year")

    return Candidate(
        title=work.get("title"),
        authors=authors,
        year=pub_year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        source="openalex",
    )


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # The desktop environment's ambient HTTPS proxy can respond with
            # a quota/rate-limit page for OpenAlex while the public endpoint
            # itself is reachable.  Avoid treating that proxy response as an
            # ordinary empty search result; direct public API access yields
            # the real bibliographic record.
            self._client = httpx.AsyncClient(
                timeout=get_settings().request_timeout,
                trust_env=False,
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    def _mailto_params(self) -> dict:
        return {"mailto": get_settings().mailto}

    async def lookup_doi(self, doi: str) -> Candidate | None:
        client = await self._get_client()
        url = f"{_BASE}/works/https://doi.org/{doi}"
        try:
            r = await client.get(url, params=self._mailto_params())
            if r.status_code == 200:
                return _work_to_candidate(r.json())
        except httpx.HTTPError:
            pass
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        if not ref.title:
            return []
        client = await self._get_client()
        params = {
            **self._mailto_params(),
            "search": unicodedata.normalize("NFKC", ref.title),
            "per-page": "5",
        }
        if ref.year:
            # A work is frequently indexed a year apart from the date the
            # citation uses: an arXiv preprint posted in 2018 is cited as its
            # 2019 proceedings version (Carlini et al., Secret Sharer).
            # Restricting the query to the cited year removes that record
            # before scoring can compare it, producing a false NOT_FOUND for
            # a real paper.  Widen the window by one year either side and let
            # the candidate scorer judge the year.
            params["filter"] = "publication_year:{0}|{1}|{2}".format(
                ref.year - 1, ref.year, ref.year + 1
            )
        # Throttling is transient: retry with the same backoff schedule the
        # Semantic Scholar adapter uses before giving up on the record.
        for attempt in range(3):
            try:
                r = await client.get(f"{_BASE}/works", params=params)
            except httpx.HTTPError as exc:
                raise OpenAlexUnavailableError(type(exc).__name__) from exc
            if r.status_code == 200:
                results = r.json().get("results", [])
                return [_work_to_candidate(w) for w in results]
            if r.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            raise OpenAlexUnavailableError(f"HTTP {r.status_code}")
        raise OpenAlexUnavailableError("rate limited after retries")
