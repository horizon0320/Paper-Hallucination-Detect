from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from typing import Any

from rapidfuzz import fuzz

from paperaudit.config import get_settings
from paperaudit.models import Candidate, CheckCost, CheckResult, CheckStatus, Reference, Verdict
from paperaudit.sources.base import SourceAdapter

_TITLE_PASS_THRESHOLD = 85.0
_TITLE_WARN_THRESHOLD = 70.0
_DOI_IN_URL = re.compile(r'\b(10\.\d{4,}/[^\s"<>\]{}\\^`|]+)')
# Bump whenever matching, parsing, or source-routing semantics change.  Older
# cached verdicts can otherwise mask a newly available arXiv/DBLP/web record
# and make the report appear unchanged after a logic fix.
_CACHE_VERSION = "record-linkage-v7-version-aware-web-citations"
# A source adapter may contain several transport fallbacks. Bound the whole
# operation so one unavailable registry cannot stall a batch indefinitely.
_MAX_SOURCE_OPERATION_SECONDS = 12.0
_AUTHOR_GROUP = re.compile(r"^[A-ZÀ-ÖØ-Ý](?:\.\s*|\s+)[A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+)*$")


def _surname(name: str) -> str:
    """Extract surname from 'Last, First' or 'First Last' format."""
    name = name.strip()
    if "," in name:
        return name.split(",")[0].strip().lower()
    parts = name.split()
    return parts[-1].lower() if parts else ""


# German and Nordic family names are transliterated two ways in bibliographic
# records: a registry may hold Schoelkopf where the PDF prints Schölkopf.
# Stripping the diacritic alone gives scholkopf, a different token, so an
# otherwise exact record was reported as an author disagreement.
_TRANSLITERATIONS = {
    "Ä": "AE", "ä": "ae",
    "Ö": "OE", "ö": "oe",
    "Ü": "UE", "ü": "ue",
    "ß": "ss",
    "Å": "AA", "å": "aa",
    "Ø": "OE", "ø": "oe",
}


def _normalize_name_token(value: str) -> str:
    """Normalize PDF/API name variants before comparing author overlap."""
    folded = "".join(_TRANSLITERATIONS.get(ch, ch) for ch in value)
    decomposed = unicodedata.normalize("NFKD", folded)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in without_marks.casefold() if ch.isalnum())


def _author_parts(name: str) -> list[str]:
    """Split PDF-collapsed author groups without breaking ``Last, First``."""
    value = " ".join(str(name or "").split()).strip()
    if not value:
        return []
    pieces = [piece.strip() for piece in value.split(",")]
    # PDF extraction sometimes stores several abbreviated authors as one list
    # item: ``Y. He, Y. Cao, B. Yang``.  Split that form, but preserve the
    # conventional ``Last, First`` representation used by bibliographic APIs.
    if len(pieces) > 1 and all(_AUTHOR_GROUP.match(piece) for piece in pieces):
        return pieces
    return [value]


def _author_signature(name: str) -> tuple[str, str]:
    """Return normalized (surname, given-name initial) for author matching."""
    value = " ".join(str(name or "").split()).strip()
    if not value:
        return "", ""
    if "," in value:
        surname, given = (part.strip() for part in value.split(",", 1))
    else:
        tokens = value.split()
        surname, given = (tokens[-1], " ".join(tokens[:-1])) if tokens else ("", "")
    # Family names with particles are formatted inconsistently across PDF
    # citations and indexes (``van den Berg, R.`` vs ``Rianne van den Berg``).
    # Compare the terminal family-name token while retaining the full given
    # name/initial for disambiguation.
    surname_parts = surname.split()
    if len(surname_parts) > 1:
        surname = surname_parts[-1]
    surname = _normalize_name_token(surname)
    given_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", given)
    initial = _normalize_name_token(given_tokens[0])[:1] if given_tokens else ""
    return surname, initial


def _author_sequence(authors: list[str]) -> list[tuple[str, str]]:
    return [
        signature
        for author in authors
        for signature in (_author_signature(part) for part in _author_parts(author))
        if signature[0]
    ]


def _authors_equivalent(ref_authors: list[str], cand_authors: list[str]) -> bool:
    """Compare author lists after normalizing initials and PDF formatting."""
    return _author_sequence(ref_authors) == _author_sequence(cand_authors)


def _is_ordered_subsequence(shorter: list, longer: list) -> bool:
    """Return whether every item of ``shorter`` occurs in ``longer`` in order.

    A bibliography index may legitimately add entries that the citation never
    listed (a group author before the person list, or a middle name the PDF
    dropped), so prefix equality is too strict for author compatibility.
    """
    iterator = iter(longer)
    return all(any(item == candidate for candidate in iterator) for item in shorter)


def _no_registry_answered(source_results: list[dict]) -> bool:
    """Return whether every queried adapter failed instead of returning records.

    NOT_FOUND asserts that the scholarly record holds no such paper, which
    requires somebody to have actually looked.  When every source reports a
    throttle, a bot interstitial, or a transport failure, no lookup produced
    evidence and the honest verdict is undetermined.  Collapsing that state
    into a negative is what turned rate-limited batches into fabricated-
    reference findings.
    """
    errors = [item for item in source_results if item["status"] == "error"]
    if not source_results or len(errors) != len(source_results):
        # At least one adapter answered, so a genuine negative is on record.
        return False
    return True


# Registry adapters that index computer-science literature as a whole,
# rather than resolving a publisher record or a landing page.
_CITATION_INDEX_SOURCES = ("arxiv", "dblp", "openalex", "semantic_scholar")


def _citation_indexes_answered(source_results: list[dict]) -> bool:
    """Return whether the queried bibliography indexes support a negative.

    Crossref and the generic web adapter resolve publisher records and
    landing pages, so an empty result there does not establish that a
    computer-science work is absent from the literature.  arXiv, DBLP,
    OpenAlex, and Semantic Scholar are the indexes that hold it, and each has
    blind spots: arXiv misses venues whose authors never post preprints, and
    a publisher-only record can be absent from an aggregator.  A negative
    from a single index is therefore too thin to call a citation fabricated,
    so when several indexes were queried NOT_FOUND requires corroboration from
    at least two.  Under an outage -- throttle responses and bot
    interstitials -- the audit says it could not see instead of asserting what
    is unproven.
    """
    indexes = [
        item for item in source_results if item["source"] in _CITATION_INDEX_SOURCES
    ]
    if not indexes:
        # The batch queried no citation index at all, so nothing here can
        # corroborate or refute the citation; the caller's own
        # all-sources-failed rule is the only meaningful check.
        return True
    answered = [item for item in indexes if item["status"] != "error"]
    if any(item["status"] == "matched" for item in answered):
        # A candidate already cleared the linkage gate, so this is a positive
        # identification rather than a bare negative.
        return True
    if len(indexes) < 2:
        # Only one index is configured: its empty result is the complete
        # evidence available and the operator asked for that judgment.
        return bool(answered)
    return len(answered) >= 2


_ARXIV_PREPRINT_CARRIER = re.compile(
    r"arxiv\s*(?:preprint)?\s*\[?\s*(?:arxiv:\s*)?\d{4}\.\d{4,5}",
    re.IGNORECASE,
)


def _cites_arxiv_preprint_version(ref: Reference) -> bool:
    """Return whether the bibliography names an arXiv preprint as its carrier.

    ``arXiv preprint arXiv:2306.11698 (2024)`` is the standard ACM/USENIX
    rendering of a preprint citation, and the trailing year is the revision the
    author read rather than the submission date arXiv reports for v1.  When the
    selected record is that same preprint, the year pair describes two
    revisions of one carrier and cannot support a metadata mismatch; without
    this, correctly cited preprints were reported as inconsistent.
    """
    if not ref.arxiv_id:
        return False
    carrier = " ".join(filter(None, [ref.venue, ref.raw]))
    return ref.arxiv_id in carrier.replace(" ", "") or bool(
        _ARXIV_PREPRINT_CARRIER.search(carrier)
    )


_TRUNCATED_CITATION = re.compile(r"\bet\s+al\.?", re.IGNORECASE)


def _citation_authors_truncated(ref: Reference) -> bool:
    """Return whether the bibliography abbreviates its author list.

    ``et al.`` means the citation itself is an incomplete author list, so no
    index can be asked to agree with it name for name: the registry may name
    authors the citation abbreviated, and a preprint and its proceedings
    record may order them differently.  Requiring full compatibility here
    reported correctly cited papers as metadata mismatches; an actually
    substituted name still fails the overlap test below.
    """
    return bool(_TRUNCATED_CITATION.search(ref.raw or ""))


def _author_token_universe(authors: list[str]) -> set[str]:
    """Return every meaningful name token an index exposes for a list.

    Some registries merge a group author into the following person when they
    tokenize an author string, so ``CodeGemma Team, Heri Zhao`` arrives as the
    single entry ``CodeGemma Team Heri Zhao``.  Comparing only the trailing
    surname of each entry then loses the group author and reports a name
    conflict for the correct record.  Keeping every multi-character token
    lets a cited surname be accounted for wherever the index filed it.
    """
    tokens: set[str] = set()
    for author in authors:
        for part in _author_parts(author):
            for word in re.findall(r"[A-Za-z[\u00c0-\u00ff]']+", str(part or "")):
                token = _normalize_name_token(word)
                if len(token) > 1:
                    tokens.add(token)
    return tokens


def _authors_compatible(ref_authors: list[str], cand_authors: list[str]) -> bool:
    """Allow normal citation truncation while rejecting substitutions.

    A reference may list only the first authors followed by ``et al.`` while
    an index exposes the complete author list. Treat matching prefixes as
    compatible after abbreviation normalization; only an actual name conflict
    should become a metadata mismatch.
    """
    ref_seq = _author_sequence(ref_authors)
    cand_seq = _author_sequence(cand_authors)
    if not ref_seq or not cand_seq:
        return True
    if ref_seq == cand_seq:
        return True
    shorter, longer = (ref_seq, cand_seq) if len(ref_seq) < len(cand_seq) else (cand_seq, ref_seq)
    # An index list may add entries the citation never named: a group author
    # before the person list (``GLM-4.5 Team, Aohan Zeng, ...``), or a
    # middle author a PDF dropped.  Require only that the shorter list occur
    # in order inside the longer one, so every cited name is still accounted
    # for; a substituted or missing name breaks the ordered match and is
    # still reported.
    if _is_ordered_subsequence(shorter, longer):
        return True
    # Merged group authors (see _author_token_universe) are the remaining
    # legitimate reason a cited name is absent from the trailing-surname
    # sequence, so accept a cited surname that the index filed anywhere else
    # inside one of its entries.  A genuinely invented name still fails:
    # its surname token appears nowhere in the record.
    cand_tokens = _author_token_universe(cand_authors)
    ref_tokens = _author_token_universe(ref_authors)
    cited_surnames = {surname for surname, _ in ref_seq}
    index_surnames = {surname for surname, _ in cand_seq}
    if cited_surnames and cited_surnames <= (index_surnames | cand_tokens):
        return True
    if index_surnames and index_surnames <= (cited_surnames | ref_tokens) \
            and len(cand_seq) <= len(ref_seq):
        return True
    # Preprints and their eventual proceedings versions occasionally reorder
    # the same contributors. That is not an author-list substitution and must
    # not be reported as a likely hallucination.  Continue to reject actual
    # missing/replaced names (and initial conflicts) below.
    if len(ref_seq) == len(cand_seq):
        return sorted(ref_seq) == sorted(cand_seq)
    return False


def _author_overlap(ref_authors: list[str], cand_authors: list[str]) -> float:
    if not ref_authors or not cand_authors:
        return 0.5
    ref_parts = [part for author in ref_authors for part in _author_parts(author)]
    cand_parts = [part for author in cand_authors for part in _author_parts(author)]
    ref_signatures = [_author_signature(part) for part in ref_parts]
    cand_signatures = [_author_signature(part) for part in cand_parts]
    ref_signatures = [sig for sig in ref_signatures if sig[0]]
    cand_signatures = [sig for sig in cand_signatures if sig[0]]
    ref_surnames = {surname for surname, _initial in ref_signatures}
    cand_surnames = {surname for surname, _initial in cand_signatures}
    # Match exact normalized surnames first, then tolerate OCR/PDF artifacts
    # such as inserted apostrophes or minor transliteration differences.
    intersection = ref_surnames & cand_surnames
    unmatched_ref = ref_surnames - intersection
    unmatched_cand = cand_surnames - intersection
    fuzzy_matches = 0
    for ref_name in unmatched_ref:
        best = max((fuzz.ratio(ref_name, cand_name) for cand_name in unmatched_cand), default=0.0)
        if best >= 86.0:
            fuzzy_matches += 1
    matched_count = len(intersection) + fuzzy_matches
    # When both records expose a given-name initial, require it only to reject
    # an otherwise identical surname with a conflicting initial.  A missing
    # initial (e.g. ``He`` vs ``Yiran He``) remains a valid match.
    for ref_surname, ref_initial in ref_signatures:
        if ref_surname not in cand_surnames or not ref_initial:
            continue
        candidate_initials = {initial for surname, initial in cand_signatures if surname == ref_surname and initial}
        if candidate_initials and ref_initial not in candidate_initials:
            matched_count -= 1
    # Use subset recall when cited list is much shorter than found list
    # (covers et al. truncation and papers with many authors).
    if len(ref_surnames) < len(cand_surnames) * 0.5:
        return matched_count / len(ref_surnames) if ref_surnames else 0.0
    # Some indexes expose only a shortened author list. When the candidate is
    # substantially shorter than the citation, score against the names that
    # are actually present instead of penalizing omitted authors.
    if len(cand_surnames) < len(ref_surnames) * 0.75:
        return matched_count / len(cand_surnames) if cand_surnames else 0.0
    union = ref_surnames | cand_surnames
    return matched_count / len(union) if union else 0.0


def _year_score(ref_year: int | None, cand_year: int | None) -> float:
    if ref_year is None or cand_year is None:
        return 0.5
    diff = abs(ref_year - cand_year)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    return 0.0


def _venue_score(ref_venue: str | None, candidate_venue: str | None) -> float:
    """Compare long citation carriers with compact CS registry labels."""
    if not ref_venue or not candidate_venue:
        return 0.5

    def normalize(value: str) -> str:
        text = unicodedata.normalize("NFKC", value).casefold()
        # DBLP typically emits compact venue labels while PDF bibliographies
        # spell out the carrier. These aliases are bibliographic equivalences,
        # not title-specific rules.
        text = re.sub(
            r"international conference on (?:learning representations|representation learning)",
            "iclr",
            text,
        )
        text = re.sub(r"(?:advances in |conference on )?neural information processing systems", "neurips", text)
        text = re.sub(r"international conference on machine learning", "icml", text)
        text = re.sub(r"conference on computer vision and pattern recognition", "cvpr", text)
        text = re.sub(r"association for computational linguistics", "acl", text)
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    cited, found = normalize(ref_venue), normalize(candidate_venue)
    known = {"iclr", "neurips", "icml", "cvpr", "acl", "emnlp", "naacl", "aaai", "ijcai", "iccv", "eccv"}
    if any(alias in cited.split() and alias in found.split() for alias in known):
        return 1.0
    return fuzz.token_sort_ratio(cited, found) / 100.0


def _score_candidate(ref: Reference, cand: Candidate) -> tuple[float, dict[str, Any]]:
    evidence: dict[str, Any] = {}

    # An identifier is strong evidence, but a malformed citation can pair a
    # real ID with the wrong title (for example a stale/wrong arXiv number).
    # Only treat it as definitive when the cited title is absent or agrees;
    # otherwise fall through to normal title/author linkage so the mismatch is
    # not silently verified as the identified work.
    exact_arxiv = bool(ref.arxiv_id and cand.arxiv_id and ref.arxiv_id == cand.arxiv_id)
    exact_doi = bool(ref.doi and cand.doi and ref.doi.lower().rstrip("/") == cand.doi.lower().rstrip("/"))
    identifier_match = exact_arxiv or exact_doi
    if identifier_match and (not ref.title or not cand.title):
        evidence.update({"title_score": 1.0, "exact_arxiv_match": exact_arxiv, "exact_doi_match": exact_doi, "composite": 1.0})
        return 1.0, evidence

    title_score = 0.0
    if ref.title and cand.title:
        # ``token_set_ratio`` can score a short unrelated title as 1.0 when
        # its words are a subset of a longer title (e.g. "A framework for
        # object modeling" vs. Pix2seq).  Token-sort similarity preserves
        # tolerance for punctuation/order while penalizing missing words.
        ref_title = unicodedata.normalize("NFKC", ref.title)
        cand_title = unicodedata.normalize("NFKC", cand.title)
        title_score = fuzz.token_sort_ratio(ref_title.lower(), cand_title.lower()) / 100.0
    evidence["title_score"] = round(title_score, 3)

    author_score = _author_overlap(ref.authors, cand.authors)
    evidence["author_score"] = round(author_score, 3)

    year_score = _year_score(ref.year, cand.year)
    evidence["year_score"] = round(year_score, 3)

    venue_score = _venue_score(ref.venue, cand.venue)
    evidence["venue_score"] = round(venue_score, 3)
    # Expose identifier agreement only when the cited metadata also supports
    # the record; a reused/wrong identifier must not look like a confirmed
    # match to downstream verdict/report code.
    if title_score >= 0.85 or not ref.title or not cand.title:
        evidence.update({"exact_arxiv_match": exact_arxiv, "exact_doi_match": exact_doi})

    if identifier_match and title_score >= 0.85:
        # Keep identifier matches dominant, but let the best author-aligned
        # record win when multiple sources expose the same identifier with
        # different author normalization/completeness.
        evidence["composite"] = round(0.80 + 0.20 * author_score, 3)
        return evidence["composite"], evidence

    if title_score == 0.0:
        composite = 0.0
    else:
        composite = (
            title_score * 0.55
            + author_score * 0.25
            + year_score * 0.15
            + venue_score * 0.05
        )
    evidence["composite"] = round(composite, 3)
    return composite, evidence


def _check_metadata_consistency(
    ref: Reference,
    best: Candidate,
    venue_score: float | None = None,
    identifier_match: bool = False,
) -> list[str]:
    mismatches = []
    # A preprint and its eventual proceedings version may be one year apart.
    # That difference is material only when the citation itself names the
    # same publication venue as the selected proceedings record: in that
    # case, mixing the preprint year with the conference carrier is a
    # metadata/version inconsistency, not an author hallucination.
    venue_aligned = bool(venue_score is not None and venue_score >= 0.75)
    # An explicit arXiv identifier pins which *work* is cited, never which
    # revision.  When the bibliography itself names the preprint as the carrier
    # (``arXiv preprint arXiv:2306.11698 (2024)``), the cited year is the
    # revision the author read while every index reports the v1 submission
    # date, and some indexes file the eventual proceedings year instead.
    # That one-year drift describes revisions of one carrier, not an
    # inconsistency, and flagging it turned correctly cited preprints into
    # metadata mismatches.
    preprint_revisions = bool(
        ref.year
        and best.year
        and identifier_match
        and abs(ref.year - best.year) <= 1
        and _cites_arxiv_preprint_version(ref),
    )
    if ref.year and best.year and not preprint_revisions and (
        abs(ref.year - best.year) > 2
        or (ref.year != best.year and venue_aligned)
        or (ref.year != best.year and identifier_match)
    ):
        mismatches.append(f"year: cited={ref.year}, found={best.year}")
    if ref.doi and best.doi and ref.doi.lower().strip("/") != best.doi.lower().strip("/"):
        mismatches.append(f"doi: cited={ref.doi}, found={best.doi}")
    # One or two cited names can be an intentional ``et al.``/truncated
    # citation.  Enforce full-list consistency once the citation exposes at
    # least three author entries, which is enough to catch substitutions in
    # long hallucinated lists without penalizing normal short citations.
    if (
        ref.authors
        and best.authors
        and len(_author_sequence(ref.authors)) >= 3
        and not _authors_compatible(ref.authors, best.authors)
    ):
        overlap = _author_overlap(ref.authors, best.authors)
        # A citation that abbreviates its own author list cannot support a
        # strict name-for-name comparison, so only a weak overlap -- what an
        # invented list actually produces -- stays a mismatch.
        if _citation_authors_truncated(ref) and overlap >= 0.5:
            return mismatches
        mismatches.append(
            f"authors: author list differs after abbreviation normalization "
            f"(overlap={overlap:.2f}), cited={ref.authors[:2]}, found={best.authors[:2]}"
        )
    return mismatches


def _strong_candidates(
    ref: Reference, candidates: list[tuple[float, dict[str, Any], Candidate]]
) -> list[tuple[float, dict[str, Any], Candidate]]:
    """Candidates strong enough to participate in cross-source consensus."""
    return [
        item for item in candidates
        if item[1].get("title_score", 0.0) >= 0.85
        and item[1].get("author_score", 0.0) >= 0.3
    ]


def _record_carrier(candidate: Candidate) -> str:
    """Return the publication carrier a record reports, for corroboration.

    arXiv and the aggregators that mirror it expose the same v1 submission date
    under labels such as ``arXiv``, ``arXiv.org``, and ``CoRR 2024``.  Two
    sources that both read that one date are a single piece of evidence about
    the year, so they must not corroborate each other into a metadata
    mismatch; only a genuinely different carrier (a proceedings or journal
    record) counts separately.
    """
    label = " ".join(str(candidate.venue or "").split()).casefold()
    source = str(candidate.source or "").casefold()
    if (
        "arxiv" in label
        or "corr" in label
        or "preprint" in label
        or "openreview" in label
        or source == "arxiv"
    ):
        return "preprint"
    return " ".join([source, label]).strip()


def _candidate_summaries(
    candidates: list[tuple[float, dict[str, Any], Candidate]],
) -> list[dict[str, Any]]:
    """Expose compact, JSON-safe evidence from each independent source."""
    summaries: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, int | None]] = set()
    for score, ev, cand in sorted(candidates, key=lambda x: x[0], reverse=True):
        key = (cand.source, cand.title, cand.year)
        if key in seen:
            continue
        seen.add(key)
        summaries.append({
            "source": cand.source,
            "title": cand.title,
            "year": cand.year,
            "authors": cand.authors[:10],
            "doi": cand.doi,
            "title_score": ev.get("title_score", 0.0),
            "author_score": ev.get("author_score", 0.0),
            "composite": ev.get("composite", round(score, 3)),
            "via": ev.get("via"),
        })
    return summaries


async def check_existence(
    ref: Reference,
    sources: list[SourceAdapter],
    domain_sources: list[SourceAdapter] | None = None,
    use_cache: bool = True,
) -> CheckResult:
    start = time.monotonic()
    api_calls = 0
    candidates: list[tuple[float, dict, Candidate]] = []
    evidence: dict[str, Any] = {}

    source_timeout = min(
        max(float(get_settings().request_timeout), 1.0),
        _MAX_SOURCE_OPERATION_SECONDS,
    )

    async def _bounded(src: SourceAdapter, awaitable):
        # DBLP and arXiv have their own batch-wide rate limiters. Applying a
        # wall-clock timeout while a record is merely waiting in that queue
        # turns a valid queued lookup into a false ``no_match``. Their network
        # clients still enforce request-level timeouts once admitted.
        if src.name in {"arxiv", "dblp"}:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=source_timeout)

    # Domain adapters are part of the primary evidence set.  A CS paper may
    # appear as an arXiv preprint, a conference record, and a later indexed
    # publication; waiting until the generic search fails would hide useful
    # version evidence and can make a correct record look inconsistent.
    all_sources: list[SourceAdapter] = []
    seen_source_names: set[str] = set()
    for src in [*(sources or []), *(domain_sources or [])]:
        if src.name in seen_source_names:
            continue
        seen_source_names.add(src.name)
        all_sources.append(src)
    evidence["queried_sources"] = [src.name for src in all_sources]

    if use_cache:
        from paperaudit.cache import get_cache
        cache = get_cache()
        raw_key = ref.doi or ref.arxiv_id or (f"{ref.title}|{ref.year}" if ref.title else None)
        cache_key = f"{_CACHE_VERSION}:{raw_key}" if raw_key else None
        if cache_key:
            cached = cache.get("existence", cache_key)
            if cached is not None:
                elapsed = (time.monotonic() - start) * 1000
                return CheckResult(
                    name="existence",
                    status=CheckStatus(cached["status"]),
                    confidence=cached["confidence"],
                    evidence={**cached["evidence"], "from_cache": True},
                    cost=CheckCost(api_calls=0, elapsed_ms=elapsed),
                )

    # Extract DOI from URLs if not already present (e.g. https://journal.org/10.1234/xyz)
    effective_doi = ref.doi
    if not effective_doi and ref.urls:
        for url in ref.urls:
            m = _DOI_IN_URL.search(url)
            if m:
                effective_doi = m.group(1).rstrip(".,;)")
                evidence["doi_extracted_from_url"] = effective_doi
                break

    if effective_doi:
        async def _lookup_doi(src: SourceAdapter):
            try:
                return src, await _bounded(src, src.lookup_doi(effective_doi)), None
            except Exception as exc:
                return src, None, exc

        doi_results = await asyncio.gather(*(_lookup_doi(src) for src in all_sources))
        api_calls += len(all_sources)
        for src, cand, error in doi_results:
            if error is not None:
                evidence[f"{src.name}_error"] = str(error)
            if cand:
                score, ev = _score_candidate(ref, cand)
                candidates.append((score, {**ev, "source": src.name, "via": "doi_lookup"}, cand))

    if ref.arxiv_id:
        # An explicit arXiv identifier is an independent record key and can
        # be resolved alongside title searches.  The same work can have an
        # arXiv preprint and a later conference/journal record, so this is
        # parallel evidence rather than a fallback lookup.
        async def _lookup_arxiv(src: SourceAdapter):
            if not hasattr(src, "lookup_arxiv_id"):
                return src, None, None
            try:
                return src, await _bounded(src, src.lookup_arxiv_id(ref.arxiv_id)), None
            except Exception as exc:
                return src, None, exc

        arxiv_results = await asyncio.gather(*(_lookup_arxiv(src) for src in all_sources))
        api_calls += sum(1 for src in all_sources if hasattr(src, "lookup_arxiv_id"))
        arxiv_id_resolved_sources: set[str] = set()
        for src, cand, error in arxiv_results:
            if error is not None:
                evidence[f"{src.name}_arxiv_error"] = str(error)
            if cand:
                score, ev = _score_candidate(ref, cand)
                candidates.append((score, {**ev, "source": src.name, "via": "arxiv_lookup"}, cand))
                arxiv_id_resolved_sources.add(src.name)
    else:
        arxiv_id_resolved_sources = set()

    # Always collect title/author records from every configured adapter. This
    # is deliberate record linkage, rather than a fallback after NOT_FOUND.
    # Search the independent registries concurrently: one slow/blocked API
    # should not add its timeout to every other source in the batch.
    async def _search_source(src: SourceAdapter):
        try:
            return src, await _bounded(src, src.search(ref)), None
        except Exception as exc:
            return src, [], exc

    # A successful explicit arXiv-ID lookup is already the source's strongest
    # possible title/author record.  Repeating `search()` for that adapter
    # issued the same public API request a second time for every preprint,
    # increasing rate-limit failures in full-paper audits.  Keep all other
    # independent source searches, and retry arXiv search only when the ID
    # lookup itself produced no record.
    search_sources = [
        src for src in all_sources
        if not (ref.arxiv_id and src.name == "arxiv" and src.name in arxiv_id_resolved_sources)
    ]
    search_results = await asyncio.gather(*(_search_source(src) for src in search_sources))
    api_calls += len(search_sources)
    for src, results, error in search_results:
        if error is not None:
            evidence[f"{src.name}_error"] = str(error)
        for cand in results:
            score, ev = _score_candidate(ref, cand)
            candidates.append((score, {**ev, "source": src.name, "via": "search"}, cand))

    # URL identifiers are also part of the primary record-linkage pass.  A
    # landing page can identify a work even when title search is noisy or the
    # publisher record is not indexed by the generic sources.
    if ref.urls:
        async def _lookup_url(src: SourceAdapter, url: str):
            try:
                return src, await _bounded(src, src.lookup_url(url)), None
            except Exception as exc:
                return src, None, exc

        url_results = await asyncio.gather(
            *(_lookup_url(src, url) for src in all_sources for url in ref.urls)
        )
        api_calls += len(all_sources) * len(ref.urls)
        for src, cand, error in url_results:
            if error is not None:
                evidence[f"{src.name}_url_error"] = str(error)
            if cand:
                score, ev = _score_candidate(ref, cand)
                candidates.append((score, {**ev, "source": src.name, "via": "url_lookup"}, cand))

    # Preserve a per-source audit trail even when no candidate survives the
    # matching thresholds. This lets the HTML report distinguish "no match"
    # from a source/API error and gives reviewers a reproducible search trail.
    candidate_counts: dict[str, int] = {}
    for _score, _ev, candidate in candidates:
        candidate_counts[candidate.source] = candidate_counts.get(candidate.source, 0) + 1
    source_results = []
    for src in all_sources:
        error = None
        for key in (f"{src.name}_error", f"{src.name}_arxiv_error", f"{src.name}_url_error"):
            if key in evidence:
                error = evidence[key]
                break
        count = candidate_counts.get(src.name, 0)
        source_results.append({
            "source": src.name,
            "status": "error" if error else ("matched" if count else "no_match"),
            "candidate_count": count,
            "error": error,
        })
    evidence["source_results"] = source_results

    elapsed = (time.monotonic() - start) * 1000

    if not candidates:
        # If every configured source failed at the transport/API layer, the
        # citation is unresolved rather than evidence that the work does not
        # exist. This distinction prevents outages and bot protection from
        # being reported as a fabricated reference.
        if _no_registry_answered(source_results):
            result = CheckResult(
                name="existence",
                status=CheckStatus.ERROR,
                confidence=0.0,
                evidence={**evidence, "verdict": "source_unavailable", "candidates_checked": 0},
                cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
            )
        elif ref.urls:
            # A cited URL can describe a legitimate software project, policy
            # page, dataset, or other web source that is not expected to have
            # an academic database record. With no high-confidence metadata
            # result, do not call it "not found": we can only say that no
            # scholarly record was linked. Preserve it as a review item.
            result = CheckResult(
                name="existence",
                status=CheckStatus.WARN,
                confidence=0.0,
                evidence={
                    **evidence,
                    "verdict": "web_source_unresolved",
                    "candidates_checked": 0,
                    "web_source_unresolved": True,
                },
                cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
            )
        elif _citation_indexes_answered(source_results):
            result = CheckResult(
                name="existence",
                status=CheckStatus.FAIL,
                confidence=0.9 if (ref.title or ref.doi) else 0.5,
                evidence={**evidence, "verdict": "not_found", "candidates_checked": 0},
                cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
            )
        else:
            result = CheckResult(
                name="existence",
                status=CheckStatus.ERROR,
                confidence=0.0,
                evidence={**evidence, "verdict": "source_unavailable", "candidates_checked": 0},
                cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
            )
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        # An exact DOI/arXiv identifier whose authoritative title disagrees
        # with the citation is stronger evidence than a failed lexical search:
        # it indicates a wrong identifier or a hallucinated title. Surface it
        # explicitly instead of downgrading the entry to NOT_FOUND.
        identifier_conflicts = [
            item for item in candidates
            if ((ref.arxiv_id and item[2].arxiv_id and ref.arxiv_id == item[2].arxiv_id)
                or (ref.doi and item[2].doi and ref.doi.lower().rstrip("/") == item[2].doi.lower().rstrip("/")))
            and item[1].get("title_score", 0.0) < 0.65
            and ref.title
            and item[2].title
        ]
        if identifier_conflicts:
            best_score, best_ev, best_cand = max(identifier_conflicts, key=lambda item: item[0])
            evidence.update(best_ev)
            evidence.update({
                "best_candidate_title": best_cand.title,
                "best_candidate_source": best_cand.source,
                "best_candidate_year": best_cand.year,
                "best_candidate_doi": best_cand.doi,
                "best_candidate_venue": best_cand.venue,
                "best_candidate_authors": best_cand.authors,
                "total_candidates": len(candidates),
                "mismatches": [
                    f"title: cited={ref.title}, found={best_cand.title}",
                ],
            })
            result = CheckResult(
                name="existence",
                status=CheckStatus.WARN,
                confidence=0.95,
                evidence=evidence,
                cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
            )
            if use_cache:
                from paperaudit.cache import get_cache
                cache = get_cache()
                raw_key = ref.doi or ref.arxiv_id or (f"{ref.title}|{ref.year}" if ref.title else None)
                cache_key = f"{_CACHE_VERSION}:{raw_key}" if raw_key else None
                if cache_key:
                    cache.set("existence", cache_key, {
                        "status": result.status.value,
                        "confidence": result.confidence,
                        "evidence": result.evidence,
                    })
            return result
        # Search APIs routinely return lexical near-misses (especially for
        # short titles). Never promote a candidate with weak title *and*
        # author evidence to the selected record; doing so made unrelated
        # works such as ALBERT/DPIC appear as if they were the cited paper.
        # Keep all raw candidates in the audit trail, but link only records
        # with a meaningful title match plus either author overlap or an exact
        # identifier whose metadata agrees with the title.
        linkable = [
            item for item in candidates
            if item[1].get("title_score", 0.0) >= 0.65
            and (
                item[1].get("author_score", 0.0) >= 0.2
                or item[1].get("exact_arxiv_match")
                or item[1].get("exact_doi_match")
            )
        ]
        if not linkable:
            evidence["candidate_sources"] = _candidate_summaries(candidates)
            evidence["total_candidates"] = len(candidates)
            # Weak lexical near-misses say nothing about a web-identified
            # source. A tool, dataset, or documentation page cited by URL is
            # not expected to hold a scholarly record, so this branch must
            # apply the same exemption as the no-candidate branch; previously
            # it returned FAIL here and the verdict engine reported a
            # legitimate citation as not found.
            if ref.urls:
                result = CheckResult(
                    name="existence",
                    status=CheckStatus.WARN,
                    confidence=0.0,
                    evidence={
                        **evidence,
                        "verdict": "web_source_unresolved",
                        "candidates_checked": len(candidates),
                        "web_source_unresolved": True,
                    },
                    cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
                )
            elif _citation_indexes_answered(source_results):
                result = CheckResult(
                    name="existence",
                    status=CheckStatus.FAIL,
                    confidence=0.7,
                    evidence={**evidence, "verdict": "not_found", "candidates_checked": len(candidates)},
                    cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
                )
            else:
                result = CheckResult(
                    name="existence",
                    status=CheckStatus.ERROR,
                    confidence=0.0,
                    evidence={
                        **evidence,
                        "verdict": "source_unavailable",
                        "candidates_checked": len(candidates),
                    },
                    cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
                )
            if use_cache:
                from paperaudit.cache import get_cache
                cache = get_cache()
                raw_key = ref.doi or ref.arxiv_id or (f"{ref.title}|{ref.year}" if ref.title else None)
                cache_key = f"{_CACHE_VERSION}:{raw_key}" if raw_key else None
                if cache_key:
                    cache.set("existence", cache_key, {
                        "status": result.status.value,
                        "confidence": result.confidence,
                        "evidence": result.evidence,
                    })
            return result
        # Select from linkable candidates only; low-quality records remain
        # visible above for manual/LLM review but cannot drive a verdict.
        best_score, best_ev, best_cand = linkable[0]
        strong = _strong_candidates(ref, linkable)
        strong_sources = sorted({item[2].source for item in strong})
        evidence["candidate_sources"] = _candidate_summaries(candidates)
        evidence["independent_source_count"] = len(strong_sources)

        # Prefer a venue-aligned version with the strongest complete metadata.
        # A preprint year can legitimately differ from the venue year, so
        # choosing a lower-quality year match first can manufacture an author
        # conflict when the cited venue/version has an exact author record.
        year_matches = [item for item in strong if ref.year and item[2].year == ref.year]
        venue_matches = [
            item for item in strong
            if ref.venue and item[1].get("venue_score", 0.0) >= 0.75
        ]
        if venue_matches:
            # Several registries expose the same work once as the preprint
            # year and once as the proceedings year.  Among venue-aligned
            # records, prefer the one agreeing with the cited year before
            # falling back to composite score; otherwise the preprint record
            # wins on author completeness and the report claims a year
            # mismatch for a citation that matches the proceedings record.
            best_score, best_ev, best_cand = max(
                venue_matches,
                key=lambda item: (
                    item[1].get("author_score", 0.0),
                    1 if (ref.year and item[2].year == ref.year) else 0,
                    item[0],
                ),
            )
        elif year_matches:
            best_score, best_ev, best_cand = max(year_matches, key=lambda x: x[0])

        evidence.update(best_ev)
        evidence["best_candidate_title"] = best_cand.title
        evidence["best_candidate_source"] = best_cand.source
        evidence["best_candidate_year"] = best_cand.year
        evidence["best_candidate_doi"] = best_cand.doi
        evidence["best_candidate_venue"] = best_cand.venue
        evidence["best_candidate_authors"] = best_cand.authors
        evidence["total_candidates"] = len(candidates)

        title_score = best_ev.get("title_score", 0.0)

        # High title match always warrants a look, even if other fields diverge
        if title_score >= 0.85 or best_score * 100 >= _TITLE_PASS_THRESHOLD:
            mismatches = _check_metadata_consistency(
                ref,
                best_cand,
                best_ev.get("venue_score"),
                bool(best_ev.get("exact_arxiv_match") or best_ev.get("exact_doi_match")),
            )
            # A single source's year disagreement is not enough to claim a
            # metadata mismatch. Crossref in particular can contain duplicate
            # or later-version records for seminal papers; require either a
            # matching candidate or at least two independent sources to support
            # the conflicting year before surfacing a mismatch verdict.
            if mismatches and ref.year and any(m.startswith("year:") for m in mismatches):
                conflicting_sources = {
                    item[2].source for item in strong
                    if item[2].year is not None and item[2].year != ref.year
                }
                venue_aligned = bool(best_ev.get("venue_score", 0.0) >= 0.75)
                identifier_match = bool(
                    best_ev.get("exact_arxiv_match") or best_ev.get("exact_doi_match")
                )
                # The corroboration rule is the same whether or not the carrier
                # agrees: proceedings volumes are indexed under both the volume
                # year and the online-first year, so one index's disagreement
                # cannot establish that the citation's year is wrong.  A real
                # year error is visible as the same conflicting year reported by
                # two independent registries.
                conflicting_year = best_cand.year
                # Only distinct carriers corroborate.  A preprint date is
                # one fact reported by several registries, so it cannot
                # agree with itself into a metadata error.
                agreeing = {
                    _record_carrier(item[2]) for item in strong
                    if item[2].year == conflicting_year
                    and item[2].source
                    and _record_carrier(item[2]) != "preprint"
                }
                if year_matches and not venue_aligned and not identifier_match:
                    mismatches = [m for m in mismatches if not m.startswith("year:")]
                elif (
                    len(agreeing) < 2
                    and not identifier_match
                    and not (ref.doi and best_cand.doi)
                ):
                    mismatches = [m for m in mismatches if not m.startswith("year:")]
                    evidence["year_conflict_unconfirmed"] = {
                        "cited_year": ref.year,
                        "conflicting_years": sorted({item[2].year for item in strong if item[2].year}),
                        "supporting_sources": [],
                        "conflicting_sources": sorted(conflicting_sources),
                    }
            if mismatches:
                evidence["mismatches"] = mismatches
                status = CheckStatus.WARN
                confidence = min(max(title_score, best_score), 0.85)
            else:
                status = CheckStatus.PASS
                confidence = max(title_score, best_score)
            if evidence.get("year_conflict_unconfirmed"):
                confidence = min(confidence, 0.85)
        elif best_score * 100 >= _TITLE_WARN_THRESHOLD:
            status = CheckStatus.WARN
            confidence = best_score * 0.7
        else:
            status = CheckStatus.FAIL
            confidence = 0.7

        # Do not label a web citation as non-existent solely because generic
        # academic search returned loose, low-score candidates. The cited URL
        # still identifies a source type that may be a tool, organization, or
        # policy page rather than a scholarly work.
        if status == CheckStatus.FAIL and ref.urls:
            status = CheckStatus.WARN
            confidence = 0.0
            evidence["verdict"] = "web_source_unresolved"
            evidence["web_source_unresolved"] = True

        result = CheckResult(
            name="existence",
            status=status,
            confidence=confidence,
            evidence=evidence,
            cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
        )

    if use_cache:
        from paperaudit.cache import get_cache
        cache = get_cache()
        raw_key = ref.doi or ref.arxiv_id or (f"{ref.title}|{ref.year}" if ref.title else None)
        cache_key = f"{_CACHE_VERSION}:{raw_key}" if raw_key else None
        if cache_key:
            cache.set("existence", cache_key, {
                "status": result.status.value,
                "confidence": result.confidence,
                "evidence": result.evidence,
            })

    return result
