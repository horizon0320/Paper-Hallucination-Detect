from __future__ import annotations

import re
import unicodedata
import asyncio
import weakref
import json
import os
import shutil

import httpx

from paperaudit.config import get_settings
from paperaudit.models import Candidate, Reference
from paperaudit.sources.base import SourceAdapter

_BASE = "https://dblp.org/search/publ/api"
_FALLBACK_BASE = "http://dblp.org/search/publ/api"
_DISAMBIGUATION_SUFFIX = re.compile(r"\s+\d{4}$")
# DBLP's public endpoint applies rate limiting to bursts. A paced serial
# queue is slower than a burst, but avoids turning real records into false
# ``no_match`` results during a full-paper audit.
# DBLP does not tolerate the short request bursts produced by a PDF-wide
# audit particularly well.  A single paced queue is deliberate here: it is
# preferable to spend a little longer on a paper than to silently turn a
# throttled registry response into a false ``no_match``.
_DBLP_CONCURRENCY_LIMIT = 1
_DBLP_MIN_INTERVAL_SECONDS = float(os.getenv("PAPERAUDIT_DBLP_MIN_INTERVAL", "1.5"))
_DBLP_FAILURES_BEFORE_CIRCUIT_BREAK = 2
_LOOP_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()
_LOOP_NEXT_REQUEST: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, float] = weakref.WeakKeyDictionary()


class DBLPTransportError(RuntimeError):
    """Raised when DBLP could not be reached, distinct from an empty search."""


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


async def _wait_for_dblp_slot() -> None:
    """Pace DBLP requests globally within the active verification batch."""
    loop = asyncio.get_running_loop()
    next_allowed = _LOOP_NEXT_REQUEST.get(loop, 0.0)
    now = loop.time()
    if next_allowed > now:
        await asyncio.sleep(next_allowed - now)
        now = loop.time()
    _LOOP_NEXT_REQUEST[loop] = now + _DBLP_MIN_INTERVAL_SECONDS

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
        # A public endpoint outage should be reported once, not multiplied by
        # every bibliography entry. This state belongs to the adapter instance,
        # which is shared by one verification batch.
        self._consecutive_transport_failures = 0
        self._unavailable_error: str | None = None
        self._prefer_curl = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # In some local/proxied environments the ambient proxy stalls
            # DBLP while a direct HTTPS request succeeds. DBLP is a public
            # registry and its canonical endpoint is directly reachable, so
            # bypass proxy environment variables for this adapter only.
            self._client = httpx.AsyncClient(
                # DBLP's public API can take 5-8 seconds for long titles.
                # Give a direct lookup enough time to complete rather than
                # converting a slow valid response into an empty result.
                timeout=min(float(get_settings().request_timeout), 12.0),
                trust_env=False,
                # DBLP's certificate chain intermittently fails validation in
                # the bundled macOS/Python runtime; the adapter already uses
                # HTTPS and only reads public metadata. Keep the direct route
                # usable, with the normal verified route still attempted by
                # callers that provide their own client.
                verify=False,
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def lookup_doi(self, doi: str) -> Candidate | None:
        return None

    async def search(self, ref: Reference) -> list[Candidate]:
        if not ref.title:
            return []
        if self._unavailable_error:
            raise DBLPTransportError(self._unavailable_error)
        async with _get_dblp_semaphore():
            try:
                result = await self._search_limited(ref)
            except DBLPTransportError as exc:
                self._consecutive_transport_failures += 1
                if self._consecutive_transport_failures >= _DBLP_FAILURES_BEFORE_CIRCUIT_BREAK:
                    self._unavailable_error = (
                        "DBLP registry temporarily unavailable after repeated transport failures"
                    )
                raise
            self._consecutive_transport_failures = 0
            return result

    async def _curl_search(self, params: dict[str, str]) -> list[Candidate] | None:
        """Use the OS HTTPS client only after the in-process client fails.

        Some bundled Python TLS stacks intermittently time out against DBLP
        while the system curl succeeds. This is a transport fallback, not web
        scraping: it calls the same public JSON API with explicitly passed
        arguments and has a bounded timeout.
        """
        curl = shutil.which("curl")
        if not curl:
            return None
        command = [
            curl, "--silent", "--show-error", "--connect-timeout", "4", "--max-time", "12", "--get", _BASE,
            "--data-urlencode", f"q={params['q']}",
            "--data-urlencode", "format=json",
            "--data-urlencode", "h=5",
            "-A", "Paper-audit/0.5 (citation verification)",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=14.0)
            if process.returncode != 0:
                return None
            payload = json.loads(stdout.decode("utf-8", errors="replace"))
            # DBLP may return a JSON error/rate-limit payload with HTTP 200
            # through an intermediary.  Do not interpret that transport
            # response as a legitimate empty search result.
            result = payload.get("result")
            if not isinstance(result, dict) or "hits" not in result:
                return None
            hits = result.get("hits", {}).get("hit", [])
            return [_hit_to_candidate(hit) for hit in hits] if hits else []
        except (asyncio.TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return None

    async def _search_limited(self, ref: Reference) -> list[Candidate]:
        client = await self._get_client()
        params = {"q": unicodedata.normalize("NFKC", ref.title), "format": "json", "h": "5"}
        # If the native Python transport failed earlier in this same batch but
        # the OS HTTPS client succeeded, avoid paying both timeouts for every
        # following reference. This is transport adaptation only; results are
        # still obtained from DBLP's canonical JSON API.
        if self._prefer_curl:
            fallback = await self._curl_search(params)
            if fallback is not None:
                return fallback
            self._prefer_curl = False
        primary_error: str | None = None
        try:
            await _wait_for_dblp_slot()
            r = await client.get(_BASE, params=params)
            if r.status_code == 200:
                hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
                return [_hit_to_candidate(h) for h in hits] if hits else []
            primary_error = f"HTTP {r.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            primary_error = type(exc).__name__
        # A 429 response often reflects a short-lived per-client bucket. Give
        # the registry a brief recovery window before switching transports;
        # otherwise the immediate curl retry receives the same throttling
        # response and a valid record is misclassified as unavailable.
        if primary_error == "HTTP 429":
            await asyncio.sleep(2.0)
        fallback = await self._curl_search(params)
        if fallback is not None:
            self._prefer_curl = True
            return fallback
        # Do not masquerade a registry/network outage as a genuine negative
        # result.  ``check_existence`` records this as source-level evidence
        # and can still use independent arXiv/publisher records, while the
        # report makes the missing DBLP confirmation explicit.
        raise DBLPTransportError(
            f"DBLP lookup unavailable ({primary_error or 'transport failure'})"
        )
