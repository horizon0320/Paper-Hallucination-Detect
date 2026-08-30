from __future__ import annotations

import html
import re

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter


_META_RE = re.compile(
    r'<meta\s+(?=[^>]*\b(?:property|name)=["\'](?P<key>[^"\']+)["\'])[^>]*\bcontent=["\'](?P<value>[^"\']*)["\'][^>]*>',
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _meta(html_text: str, *keys: str) -> str | None:
    wanted = {key.casefold() for key in keys}
    for match in _META_RE.finditer(html_text):
        if match.group("key").casefold() in wanted:
            return html.unescape(match.group("value")).strip()
    return None


def _clean_title(value: str | None) -> str | None:
    if not value:
        return None
    value = _TAG_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    # Publisher page titles commonly append the site/section name.
    value = re.split(r"\s+[-|]\s+(?:NDSS Symposium|IEEE Xplore|ACM Digital Library)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    # PDF/report landing pages sometimes expose a filename instead of a
    # human-readable title, e.g. ``B_Jan24_Generative_AI_and_Copyright...``.
    value = re.sub(r"^[A-Z]_?Jan\d+_", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+-\s+Centre?\s+for\s+Technology.*$", "", value, flags=re.IGNORECASE)
    value = value.replace("_", " ")
    return value.strip(" .") or None


def _authors_from_page(html_text: str) -> list[str]:
    description = _meta(html_text, "og:description", "description") or ""
    match = re.search(r"Author\(s\)\s*:\s*(.+?)(?:\s+Download\s*:|\s+Date\s*:|$)", description, re.IGNORECASE)
    if not match:
        match = re.search(r"Author\(s\)\s*:\s*(.+?)(?:</p>|<br|\n)", html_text, re.IGNORECASE)
    if not match:
        return []
    value = _TAG_RE.sub(" ", html.unescape(match.group(1)))
    value = re.sub(r"\s+", " ", value).strip(" .")
    parts = re.split(r"\s*,\s*|\s+and\s+", value, flags=re.IGNORECASE)
    return [part.strip(" .").removeprefix("and ").strip() for part in parts if part.strip(" .")]


class WebPageAdapter(SourceAdapter):
    """Extract bibliographic metadata from a cited landing page.

    This is deliberately a metadata fallback, not a general web search. It
    allows conference/publisher pages that are absent from Crossref/OpenAlex
    to provide independent evidence when the citation already contains a URL.
    """

    name = "web"

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
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        return []

    async def lookup_url(self, url: str) -> Candidate | None:
        client = await self._get_client()
        try:
            response = await client.get(url, follow_redirects=True)
            if response.status_code >= 400:
                return None
            text = response.text[:2_000_000]
            title = _clean_title(_meta(text, "og:title", "twitter:title"))
            if not title:
                match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
                title = _clean_title(match.group(1) if match else None)
            authors = _authors_from_page(text)
            date_text = _meta(text, "article:published_time", "citation_publication_date", "date")
            if not date_text:
                visible = _TAG_RE.sub(" ", text)
                date_text = _meta(text, "og:description", "description") or visible
                date_match = re.search(r"\bDate\s*:\s*[^\d]{0,20}(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})", visible, re.IGNORECASE)
                if date_match:
                    date_text = date_match.group(1)
            year_match = _YEAR_RE.search(date_text)
            year = int(year_match.group(1)) if year_match else None
            doi = _meta(text, "citation_doi")
            return Candidate(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                urls=[str(response.url)],
                source=self.name,
            ) if title or authors else None
        except (httpx.HTTPError, UnicodeError):
            return None
