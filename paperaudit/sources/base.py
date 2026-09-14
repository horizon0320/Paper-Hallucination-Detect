from __future__ import annotations

from abc import ABC, abstractmethod

from paperaudit.models import Candidate, Reference


class SourceUnavailableError(RuntimeError):
    """Raised when a registry could not answer, as opposed to holding no match.

    An empty candidate list is evidence: the registry genuinely has no record
    for the citation, which is how a fabricated reference is detected. A 429, a
    5xx, or a transport failure is not evidence at all, because the request never
    produced data. Adapters that collapse the two turn an outage or a throttle
    into a false NOT_FOUND verdict for a real paper, so every registry signals
    unavailability by raising this instead of returning an empty list.
    """


class SourceAdapter(ABC):
    name: str = "unknown"

    @abstractmethod
    async def lookup_doi(self, doi: str) -> Candidate | None:
        """Resolve a DOI to a candidate record."""
        ...

    @abstractmethod
    async def search(self, ref: Reference) -> list[Candidate]:
        """Search by title/author/year; return ranked candidates."""
        ...

    async def lookup_url(self, url: str) -> Candidate | None:
        """Resolve a landing-page URL to a candidate record. Optional."""
        return None
