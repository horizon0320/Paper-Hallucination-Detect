from __future__ import annotations

import re
import unicodedata

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter

_BASE = "https://api.crossref.org"
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def _item_to_candidate(item: dict) -> Candidate:
    authors = []
    for a in item.get("author", []):
        family = a.get("family", "")
        given = a.get("given", "")
        name = f"{given} {family}".strip() if given else family
        if name:
            authors.append(name)

    doi = item.get("DOI")

    pub_year = None
    for date_key in ("published-print", "published-online", "issued"):
        dp = item.get(date_key, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            pub_year = int(dp[0][0])
            break

    title_list = item.get("title", [])
    title = title_list[0] if title_list else None

    venue_list = item.get("container-title", [])
    venue = venue_list[0] if venue_list else None

    return Candidate(
        title=title,
        authors=authors,
        year=pub_year,
        venue=venue,
        doi=doi,
        source="crossref",
    )


class CrossrefAdapter(SourceAdapter):
    name = "crossref"

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

    def _mailto_params(self) -> dict:
        return {"mailto": get_settings().mailto}

    async def lookup_doi(self, doi: str) -> Candidate | None:
        client = await self._get_client()
        url = f"{_BASE}/works/{doi}"
        try:
            r = await client.get(url, params=self._mailto_params())
            if r.status_code == 200:
                item = r.json().get("message", {})
                return _item_to_candidate(item)
        except httpx.HTTPError:
            pass
        return None

    async def lookup_url(self, url: str) -> Candidate | None:
        """Resolve publisher landing pages (including IEEE Xplore) via DOI metadata."""
        client = await self._get_client()
        try:
            r = await client.get(url, follow_redirects=True)
            if r.status_code >= 400:
                return None
            # Publisher pages commonly expose citation_doi/DC.identifier meta
            # tags even when the DOI is not visible in the page title.
            html = r.text[:2_000_000]
            for pattern in (
                r'name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)',
                r'name=["\']DC\.identifier["\'][^>]+content=["\']([^"\']+)',
                r'property=["\']og:url["\'][^>]+content=["\']([^"\']+)',
            ):
                m = re.search(pattern, html, re.IGNORECASE)
                if not m:
                    continue
                doi = _DOI_RE.search(m.group(1))
                if doi:
                    candidate = await self.lookup_doi(doi.group(0).rstrip(".,;)") )
                    if candidate:
                        return candidate
            # Last resort: locate a DOI anywhere in the page and validate it
            # through Crossref rather than trusting page text as metadata.
            m = _DOI_RE.search(html)
            if m:
                return await self.lookup_doi(m.group(0).rstrip(".,;)") )
        except httpx.HTTPError:
            pass
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        if not ref.title:
            return []
        client = await self._get_client()
        params = {
            **self._mailto_params(),
            "query.title": unicodedata.normalize("NFKC", ref.title),
            "rows": "5",
        }
        if ref.authors:
            params["query.author"] = " ".join(ref.authors[:2])
        try:
            r = await client.get(f"{_BASE}/works", params=params)
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                return [_item_to_candidate(i) for i in items]
        except httpx.HTTPError:
            pass
        return []
