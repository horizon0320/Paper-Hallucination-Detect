from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from rapidfuzz import fuzz

from paperaudit.models import Candidate, CheckCost, CheckResult, CheckStatus, Reference, Verdict
from paperaudit.sources.base import SourceAdapter

_TITLE_PASS_THRESHOLD = 85.0
_TITLE_WARN_THRESHOLD = 70.0
_DOI_IN_URL = re.compile(r'\b(10\.\d{4,}/[^\s"<>\]{}\\^`|]+)')
# Bump whenever matching, parsing, or source-routing semantics change.  Older
# cached verdicts can otherwise mask a newly available arXiv/DBLP/web record
# and make the report appear unchanged after a logic fix.
_CACHE_VERSION = "record-linkage-v6-per-reference-dblp-arxiv"
_AUTHOR_GROUP = re.compile(r"^[A-ZÀ-ÖØ-Ý](?:\.\s*|\s+)[A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+)*$")


def _surname(name: str) -> str:
    """Extract surname from 'Last, First' or 'First Last' format."""
    name = name.strip()
    if "," in name:
        return name.split(",")[0].strip().lower()
    parts = name.split()
    return parts[-1].lower() if parts else ""


def _normalize_name_token(value: str) -> str:
    """Normalize PDF/API name variants before comparing author overlap."""
    decomposed = unicodedata.normalize("NFKD", value)
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
    return longer[:len(shorter)] == shorter


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

    venue_score = 0.5
    if ref.venue and cand.venue:
        venue_score = fuzz.token_sort_ratio(ref.venue.lower(), cand.venue.lower()) / 100.0
    evidence["venue_score"] = round(venue_score, 3)

    if identifier_match and title_score >= 0.85:
        evidence.update({"exact_arxiv_match": exact_arxiv, "exact_doi_match": exact_doi})
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


def _check_metadata_consistency(ref: Reference, best: Candidate) -> list[str]:
    mismatches = []
    if ref.year and best.year and abs(ref.year - best.year) > 2:
        mismatches.append(f"year: cited={ref.year}, found={best.year}")
    if ref.doi and best.doi and ref.doi.lower().strip("/") != best.doi.lower().strip("/"):
        mismatches.append(f"doi: cited={ref.doi}, found={best.doi}")
    # One or two cited names can be an intentional ``et al.``/truncated
    # citation.  Enforce full-list consistency once the citation exposes at
    # least three author entries, which is enough to catch substitutions in
    # long hallucinated lists without penalizing normal short citations.
    if ref.authors and best.authors and len(_author_sequence(ref.authors)) >= 3 and not _authors_compatible(ref.authors, best.authors):
        overlap = _author_overlap(ref.authors, best.authors)
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
        for src in all_sources:
            try:
                cand = await src.lookup_doi(effective_doi)
                api_calls += 1
                if cand:
                    score, ev = _score_candidate(ref, cand)
                    candidates.append((score, {**ev, "source": src.name, "via": "doi_lookup"}, cand))
            except Exception as e:
                evidence[f"{src.name}_error"] = str(e)

    if ref.arxiv_id:
        # An explicit arXiv identifier is an independent record key and can
        # be resolved alongside title searches.  The same work can have an
        # arXiv preprint and a later conference/journal record, so this is
        # parallel evidence rather than a fallback lookup.
        for src in all_sources:
            if not hasattr(src, "lookup_arxiv_id"):
                continue
            try:
                cand = await src.lookup_arxiv_id(ref.arxiv_id)
                api_calls += 1
                if cand:
                    score, ev = _score_candidate(ref, cand)
                    candidates.append((score, {**ev, "source": src.name, "via": "arxiv_lookup"}, cand))
            except Exception as e:
                evidence[f"{src.name}_arxiv_error"] = str(e)

    # Always collect title/author records from every configured adapter.  This
    # is deliberate record linkage, rather than a fallback after NOT_FOUND.
    for src in all_sources:
        try:
            results = await src.search(ref)
            api_calls += 1
            for cand in results:
                score, ev = _score_candidate(ref, cand)
                candidates.append((score, {**ev, "source": src.name, "via": "search"}, cand))
        except Exception as e:
            evidence[f"{src.name}_error"] = str(e)

    # URL identifiers are also part of the primary record-linkage pass.  A
    # landing page can identify a work even when title search is noisy or the
    # publisher record is not indexed by the generic sources.
    if ref.urls:
        for src in all_sources:
            for url in ref.urls:
                try:
                    cand = await src.lookup_url(url)
                    api_calls += 1
                    if cand:
                        score, ev = _score_candidate(ref, cand)
                        candidates.append((score, {**ev, "source": src.name, "via": "url_lookup"}, cand))
                except Exception as e:
                    evidence[f"{src.name}_url_error"] = str(e)

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
        result = CheckResult(
            name="existence",
            status=CheckStatus.FAIL,
            confidence=0.9 if (ref.title or ref.doi) else 0.5,
            evidence={**evidence, "verdict": "not_found", "candidates_checked": 0},
            cost=CheckCost(api_calls=api_calls, elapsed_ms=elapsed),
        )
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ev, best_cand = candidates[0]
        strong = _strong_candidates(ref, candidates)
        strong_sources = sorted({item[2].source for item in strong})
        evidence["candidate_sources"] = _candidate_summaries(candidates)
        evidence["independent_source_count"] = len(strong_sources)

        # Prefer a high-quality candidate whose year agrees with the citation
        # over a same-title record from one source carrying a bad/late year.
        year_matches = [item for item in strong if ref.year and item[2].year == ref.year]
        if year_matches:
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
            mismatches = _check_metadata_consistency(ref, best_cand)
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
                if year_matches:
                    mismatches = [m for m in mismatches if not m.startswith("year:")]
                elif len(conflicting_sources) < 2:
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
