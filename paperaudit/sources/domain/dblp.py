from __future__ import annotations

import re
import unicodedata
import asyncio
import json
import urllib.parse
import urllib.request
import weakref

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter

_BASE = "https://dblp.org/search/publ/api"
_FALLBACK_BASE = "http://dblp.org/search/publ/api"
_DISAMBIGUATION_SUFFIX = re.compile(r"\s+\d{4}$")
_DBLP_CONCURRENCY_LIMIT = 2
_LOOP_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _get_dblp_semaphore() -> asyncio.Semaphore:
    """Return a DBLP limiter bound to the current event loop.

    Python 3.9 binds an asyncio semaphore to the loop that first waits on it.
    A module-level semaphore therefore leaks loop affinity across repeated
    ``asyncio.run`` calls (and caused ``Future attached to a different loop``
    errors in CLI verification). Keep one short-lived limiter per loop.
    """
    loop = asyncio.get_running_loop()
    semaphore = _LOOP_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_DBLP_CONCURRENCY_LIMIT)
        _LOOP_SEMAPHORES[loop] = semaphore
    return semaphore

_CS_PATTERNS = [
    # ML / AI — abbreviations and unambiguous full-name fragments
    "neurips", "nips", "icml", "cvpr", "iccv", "eccv", "acl", "emnlp", "naacl",
    "iclr", "aaai", "ijcai", "uai", "aistats", "icip", "icpr", "icme",
    "pmlr",                              # Proceedings of Machine Learning Research (ICML, AISTATS, …)
    "conference on learning representations",  # ICLR full name
    # IEEE-style abbreviation of ICLR commonly found in references.
    "conf. learn. representations",
    "int. conf. learn. representations",
    "proc. int. conf. learn. representations",
    # Common abbreviated NeurIPS/NeurIPS Proceedings forms used by IEEE-style
    # bibliographies (e.g. ``Proc. Adv. Neural Inf. Process. Syst.``).  These
    # must route to DBLP just like the unabbreviated venue name; otherwise a
    # genuine NeurIPS paper is searched only in generic metadata sources.
    "neural inf. process. syst.",
    "adv. neural inf. process. syst.",
    "proc. adv. neural inf. process. syst.",
    "neural inform. process. syst.",
    "adv. neural inform. process. syst.",
    # Common abbreviated computer-vision conference forms (ICCV/ACCV).
    "conf. comput. vis.",
    "asian conf. comput. vis.",
    "proc. asian conf. comput. vis.",
    "ieee international conference", "acm international conference",
    "image processing", "computer vision", "pattern recognition",
    "conference on machine learning",    # ICML full name fragment
    "neural information processing",     # NeurIPS full name fragment
    "empirical methods in natural language",  # EMNLP
    # Data / DB
    "vldb", "sigmod", "sigkdd", "kdd", "icde", "www", "iswc",
    # Systems
    "sosp", "osdi", "eurosys", "nsdi", "pldi", "popl", "oopsla", "asplos",
    "isca", "micro", "hpca",
    # Computer security venues.  These are common in references without an
    # explicit ``computer science`` label, so they must still activate DBLP
    # in automatic domain routing.
    "usenix", "security symposium", "computer and communications security",
    "acm ccs", "ndss", "eurosp", "european symposium on security",
    "ieee symposium on security and privacy", "raid symposium",
    # Software engineering conferences
    "icse", "fse", "esec", "ase", "icsme", "icsm", "msr", "issta", "saner",
    "icsa", "ecsa", "icpc", "models", "caise", "ease", "promise",
    "forge", "satrends", "sat ", "aiware", "aixse", "conisoft", "iccca",
    "southeast", "southeastcon", "mrai", "iciip", "acai", "cain",
    # SE journals
    "tosem", "tse", "emse", "ist ", "information and software technology",
    "journal of systems and software", "software: practice",
    # General IEEE/ACM CS
    "ieee transactions", "ieee access", "acm transactions",
    "journal of machine learning", "journal of artificial intelligence",
    "artificial intelligence",
]


def is_cs(ref: Reference) -> bool:
    if ref.arxiv_id:
        return True
    # Check both extracted venue and raw reference text (venue abbreviations like
    # "ICML" rarely appear in the extracted venue field for LNCS-formatted papers)
    text = " ".join(filter(None, [ref.venue, ref.raw])).lower()
    return any(p in text for p in _CS_PATTERNS)


def _hit_to_candidate(hit: dict) -> Candidate:
    info = hit.get("info", {})
    authors_data = info.get("authors", {}).get("author", [])
    if isinstance(authors_data, str):
        authors = [authors_data]
    elif isinstance(authors_data, dict):
        authors = [authors_data.get("text", "")]
    else:
        authors = [a.get("text", "") for a in authors_data if isinstance(a, dict)]

    # DBLP appends a four-digit author disambiguation key to some display
    # names (for example ``Ting Chen 0007``).  It is not part of the author's
    # bibliographic name and should never appear in reports or comparisons.
    authors = [_DISAMBIGUATION_SUFFIX.sub("", author).strip() for author in authors]

    year_str = info.get("year")
    year = int(year_str) if year_str and year_str.isdigit() else None

    doi = info.get("doi")

    return Candidate(
        title=info.get("title"),
        authors=[a for a in authors if a],
        year=year,
        venue=info.get("venue"),
        doi=doi,
        source="dblp",
    )


class DBLPAdapter(SourceAdapter):
    name = "dblp"

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
        if not ref.title:
            return []
        async with _get_dblp_semaphore():
            return await self._search_limited(ref)

    async def _search_limited(self, ref: Reference) -> list[Candidate]:
        client = await self._get_client()
        params = {"q": unicodedata.normalize("NFKC", ref.title), "format": "json", "h": "5"}
        # Some macOS/network environments fail TLS negotiation with dblp.org
        # even though the public API itself is reachable. Retry the same
        # read-only query over the API's HTTP endpoint so a known conference
        # record (notably USENIX Security) is not lost to a transport error.
        # Prefer the endpoint that is reachable in restricted macOS TLS
        # environments; HTTPS remains the second attempt where available.
        for endpoint in (_FALLBACK_BASE, _BASE):
            try:
                # The HTTP endpoint redirects to HTTPS. Follow that redirect
                # so the fallback is useful on environments where the first
                # TLS attempt fails, rather than being treated as an empty
                # search response (301 has no JSON payload).
                r = await client.get(endpoint, params=params, follow_redirects=True)
                if r.status_code == 200:
                    hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
                    # A proxy/rate-limit layer can return a valid 200 with an
                    # empty hit list. Do not treat that transient response as
                    # authoritative; try the alternate endpoint and the
                    # transport fallbacks below first.
                    if hits:
                        return [_hit_to_candidate(h) for h in hits]
            except Exception:
                continue
        # A few macOS proxy/TLS stacks intermittently raise an SSL record-layer
        # error for the shared client even though the public DBLP endpoint is
        # reachable. Retry once with a short-lived client that disables only
        # certificate verification; this is a read-only fallback and avoids
        # silently losing the DBLP conference record.
        try:
            async with httpx.AsyncClient(
                timeout=get_settings().request_timeout,
                verify=False,
                follow_redirects=True,
            ) as fallback_client:
                r = await fallback_client.get(_BASE, params=params)
                if r.status_code == 200:
                    hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
                    if hits:
                        return [_hit_to_candidate(h) for h in hits]
        except Exception:
            pass
        # Last-resort stdlib transport.  On some macOS builds httpx/OpenSSL
        # fails during DBLP's HTTP→HTTPS redirect while urllib can complete
        # the same read-only request through the system networking stack.
        try:
            query = urllib.parse.urlencode(params)
            url = f"{_FALLBACK_BASE}?{query}"
            payload = await asyncio.to_thread(
                lambda: urllib.request.urlopen(url, timeout=get_settings().request_timeout).read()
            )
            data = json.loads(payload.decode("utf-8", errors="replace"))
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if hits:
                return [_hit_to_candidate(h) for h in hits]
        except Exception:
            pass
        return []
