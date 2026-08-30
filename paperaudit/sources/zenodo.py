from __future__ import annotations

import re

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter


_BASE = "https://zenodo.org/api/records"
_DOI_RE = re.compile(r"10\.5281/zenodo\.(\d+)", re.IGNORECASE)


def _record_to_candidate(record: dict) -> Candidate | None:
    metadata = record.get("metadata") or {}
    title = metadata.get("title")
    if not title:
        return None
    creators = metadata.get("creators") or []
    authors = [str(c.get("name", "")).strip() for c in creators if c.get("name")]
    publication_date = str(metadata.get("publication_date") or "")
    year = int(publication_date[:4]) if publication_date[:4].isdigit() else None
    doi = record.get("doi") or metadata.get("doi") or record.get("conceptdoi")
    return Candidate(
        title=title,
        authors=authors,
        year=year,
        venue="Zenodo",
        doi=doi,
        source="zenodo",
    )


class ZenodoAdapter(SourceAdapter):
    name = "zenodo"

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
        match = _DOI_RE.search(doi)
        if not match:
            return None
        client = await self._get_client()
        try:
            response = await client.get(f"{_BASE}/{match.group(1)}", follow_redirects=True)
            if response.status_code == 200:
                candidate = _record_to_candidate(response.json())
                # Zenodo concept DOIs resolve to a versioned record DOI. Keep
                # the DOI supplied by the citation for exact identifier
                # matching while the title/year/authors come from the record.
                if candidate:
                    candidate.doi = doi.rstrip("/")
                return candidate
        except httpx.HTTPError:
            pass
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        # Zenodo is used as an identifier-backed repository source. Searching
        # its free-text API is noisy and is intentionally left to Crossref,
        # OpenAlex, Semantic Scholar and arXiv.
        return []
