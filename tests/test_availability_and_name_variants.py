"""Guarantees for the registry-availability and citation-variant fixes.

Two failure classes produced wrong verdicts on correctly cited papers: a
registry that never answered being read as a negative result, and a
bibliography-format variant (transliterated surname, merged group author,
preprint revision year, BibTeX protection braces) being read as a mismatch.
"""

import asyncio

import httpx
import pytest
import respx

from paperaudit.checks import existence as existence_module
from paperaudit.checks.existence import (
    _authors_compatible,
    _check_metadata_consistency,
    _citation_indexes_answered,
    _normalize_name_token,
)
from paperaudit.models import Candidate, Reference
from paperaudit.parse.plaintext import _normalize_pdf_text
from paperaudit.sources.arxiv import ArXivAdapter
from paperaudit.sources.base import SourceUnavailableError
from paperaudit.sources.crossref import CrossrefAdapter
from paperaudit.sources.openalex import OpenAlexAdapter


async def _no_curl(*args, **kwargs):
    raise AssertionError("curl fallback must not run while the API answers")


@pytest.fixture
def fast_arxiv(monkeypatch):
    """Keep quota tests from waiting out a real cooldown window."""
    monkeypatch.setattr(existence_module, "_MAX_SOURCE_OPERATION_SECONDS", 30.0)
    from paperaudit.sources import arxiv as arxiv_module

    monkeypatch.setattr(arxiv_module, "_ARXIV_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(arxiv_module, "_ARXIV_QUOTA_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(arxiv_module, "_ARXIV_RATE_LIMIT_BACKOFF_SECONDS", 0.0)


class TestAuthorNameVariants:
    def test_german_transliteration_is_not_an_author_conflict(self):
        assert _normalize_name_token("Schölkopf") == _normalize_name_token("Schoelkopf")
        assert _authors_compatible(
            ["Bernhard Schölkopf", "Klaus-Robert Müller"],
            ["Bernhard Schoelkopf", "Klaus-Robert Mueller"],
        )

    def test_merged_group_author_is_not_an_author_conflict(self):
        # Some registries fold a group author into the following person.
        assert _authors_compatible(
            ["CodeGemma Team", "Heri Zhao", "Jeffrey Hui", "Joshua Howland"],
            ["CodeGemma Team Heri Zhao", "Jeffrey Hui", "Joshua Howland"],
        )

    def test_substituted_surname_is_still_rejected(self):
        assert not _authors_compatible(
            ["Asha Mehta", "Bruno Katz", "Chen Liu"],
            ["Asha Mehta", "Bruno Katz", "Invented Person"],
        )

    def test_et_al_truncation_does_not_force_full_list_agreement(self):
        ref = Reference(
            title="Code Llama",
            year=2023,
            arxiv_id="2308.12950",
            authors=["Baptiste Roziere", "Jonas Gehring", "Fabian Gloeckle", "Tal Remez"],
            raw="Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Tal Remez, et al. 2023.",
        )
        cand = Candidate(
            title="Code Llama",
            year=2023,
            arxiv_id="2308.12950",
            source="arxiv",
            authors=["Baptiste Rozière", "Jonas Gehring", "Fabian Gloeckle"],
        )
        assert _check_metadata_consistency(ref, cand, venue_score=0.3, identifier_match=True) == []

    def test_invented_author_list_is_still_a_mismatch(self):
        ref = Reference(
            title="Some Work",
            year=2023,
            authors=["Alicia Voss", "Marco Ito", "Petra Novak"],
            raw="Alicia Voss, Marco Ito, Petra Novak, et al. 2023.",
        )
        cand = Candidate(
            title="Some Work",
            year=2023,
            source="arxiv",
            authors=["Alicia Voss", "Boris Kalman", "Cheng Wu"],
        )
        mismatches = _check_metadata_consistency(ref, cand, venue_score=0.3)
        assert any(m.startswith("authors:") for m in mismatches)


class TestPreprintRevisionYear:
    def test_cited_preprint_year_is_not_a_mismatch(self):
        ref = Reference(
            title="DecodingTrust",
            year=2024,
            arxiv_id="2306.11698",
            venue="arXiv preprint arXiv:2306.11698 (2024)",
        )
        cand = Candidate(
            title="DecodingTrust", year=2023, arxiv_id="2306.11698", source="arxiv", venue="arXiv"
        )
        assert _check_metadata_consistency(ref, cand, venue_score=0.2, identifier_match=True) == []

    def test_preprint_year_against_proceedings_record_is_not_a_mismatch(self):
        ref = Reference(
            title="DecodingTrust",
            year=2024,
            arxiv_id="2306.11698",
            venue="arXiv preprint arXiv:2306.11698 (2024)",
        )
        cand = Candidate(
            title="DecodingTrust",
            year=2023,
            arxiv_id="2306.11698",
            source="semantic_scholar",
            venue="Neural Information Processing Systems",
        )
        assert _check_metadata_consistency(ref, cand, venue_score=0.1, identifier_match=True) == []

    def test_identifier_year_conflict_without_preprint_carrier_stays(self):
        ref = Reference(title="Example Work", year=2024, arxiv_id="2308.12950")
        cand = Candidate(title="Example Work", year=2023, arxiv_id="2308.12950", source="arxiv")
        assert "year: cited=2024, found=2023" in _check_metadata_consistency(
            ref, cand, venue_score=0.2, identifier_match=True
        )


class TestBibliographyArtifacts:
    def test_bibtex_protection_braces_are_removed(self):
        assert (
            _normalize_pdf_text("{CodexLeaks}: Privacy leaks from code models")
            == "CodexLeaks: Privacy leaks from code models"
        )
        assert _normalize_pdf_text("A {Mixed-Methods} study") == "A Mixed-Methods study"

    def test_inline_markup_does_not_swallow_word_boundaries(self):
        assert _normalize_pdf_text("<i>Deep</i>Think") == "DeepThink"


class TestSourceAvailability:
    def test_two_answering_indexes_corroborate_a_negative(self):
        results = [
            {"source": "crossref", "status": "no_match", "candidate_count": 0, "error": None},
            {"source": "arxiv", "status": "no_match", "candidate_count": 0, "error": None},
            {"source": "dblp", "status": "error", "candidate_count": 0, "error": "bot check"},
            {"source": "openalex", "status": "error", "candidate_count": 0, "error": "HTTP 429"},
            {"source": "semantic_scholar", "status": "no_match", "candidate_count": 0, "error": None},
        ]
        assert _citation_indexes_answered(results)

    def test_one_answering_index_among_failures_is_not_enough(self):
        results = [
            {"source": "crossref", "status": "matched", "candidate_count": 5, "error": None},
            {"source": "arxiv", "status": "error", "candidate_count": 0, "error": "HTTP 429"},
            {"source": "dblp", "status": "error", "candidate_count": 0, "error": "bot check"},
            {"source": "openalex", "status": "error", "candidate_count": 0, "error": "HTTP 429"},
            {"source": "semantic_scholar", "status": "error", "candidate_count": 0, "error": "HTTP 429"},
        ]
        assert not _citation_indexes_answered(results)

    def test_matched_candidate_needs_no_corroboration(self):
        results = [
            {"source": "arxiv", "status": "matched", "candidate_count": 1, "error": None},
            {"source": "dblp", "status": "error", "candidate_count": 0, "error": "bot check"},
        ]
        assert _citation_indexes_answered(results)

    @pytest.mark.asyncio
    async def test_arxiv_rate_limit_raises_instead_of_empty_result(self, fast_arxiv):
        async with respx.mock:
            respx.get("https://export.arxiv.org/api/query").mock(
                return_value=httpx.Response(429, text="")
            )
            respx.get("https://arxiv.org/api/query").mock(
                return_value=httpx.Response(429, text="")
            )
            adapter = ArXivAdapter()
            adapter._curl_query = _no_curl
            with pytest.raises(SourceUnavailableError):
                await adapter.search(Reference(title="A real preprint", year=2024))
            await adapter.close()

    @pytest.mark.asyncio
    async def test_arxiv_empty_feed_is_a_genuine_negative(self, fast_arxiv):
        empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>'
        async with respx.mock:
            respx.get("https://export.arxiv.org/api/query").mock(
                return_value=httpx.Response(200, text=empty)
            )
            respx.get("https://arxiv.org/api/query").mock(
                return_value=httpx.Response(200, text=empty)
            )
            adapter = ArXivAdapter()
            adapter._curl_query = _no_curl
            results = await adapter.search(Reference(title="Nowhere Found", year=2024))
            assert results == []
            await adapter.close()

    @pytest.mark.asyncio
    async def test_openalex_error_status_raises(self):
        async with respx.mock:
            respx.get("https://api.openalex.org/works").mock(
                return_value=httpx.Response(429, json={"error": "budget"})
            )
            adapter = OpenAlexAdapter()
            with pytest.raises(SourceUnavailableError):
                await adapter.search(Reference(title="Attention", year=2017))
            await adapter.close()

    @pytest.mark.asyncio
    async def test_crossref_server_error_raises(self):
        async with respx.mock:
            respx.get("https://api.crossref.org/works").mock(
                return_value=httpx.Response(500, text="upstream failure")
            )
            adapter = CrossrefAdapter()
            with pytest.raises(SourceUnavailableError):
                await adapter.search(Reference(title="Attention", year=2017))
            await adapter.close()

    @pytest.mark.asyncio
    async def test_crossref_empty_result_is_a_genuine_negative(self):
        async with respx.mock:
            respx.get("https://api.crossref.org/works").mock(
                return_value=httpx.Response(200, json={"message": {"items": []}})
            )
            adapter = CrossrefAdapter()
            results = await adapter.search(Reference(title="Nowhere Found"))
            assert results == []
            await adapter.close()
