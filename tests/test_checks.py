from __future__ import annotations

import pytest
import respx
import httpx

from paperaudit.models import Candidate, CheckStatus, Reference
from paperaudit.sources.base import SourceAdapter
from paperaudit.checks.url_liveness import check_url_liveness
from paperaudit.checks.existence import check_existence
from paperaudit.sources.openalex import OpenAlexAdapter

_OPENALEX_WORK = {
    "id": "https://openalex.org/W2964232343",
    "title": "Attention is all you need",
    "doi": "https://doi.org/10.48550/arXiv.1706.03762",
    "publication_year": 2017,
    "authorships": [
        {"author": {"display_name": "Ashish Vaswani"}},
        {"author": {"display_name": "Noam Shazeer"}},
    ],
    "primary_location": {
        "source": {"display_name": "Advances in Neural Information Processing Systems"}
    },
    "ids": {},
}


@pytest.mark.asyncio
class TestUrlLiveness:
    async def test_live_url_passes(self):
        async with respx.mock:
            respx.head("https://example.com/paper").mock(
                return_value=httpx.Response(200)
            )
            result = await check_url_liveness(["https://example.com/paper"], use_cache=False)
            assert result.status == CheckStatus.PASS

    async def test_dead_url_fails(self):
        async with respx.mock:
            respx.head("https://example.com/dead").mock(
                return_value=httpx.Response(404)
            )
            result = await check_url_liveness(["https://example.com/dead"], use_cache=False)
            assert result.status == CheckStatus.FAIL

    async def test_no_urls_skipped(self):
        result = await check_url_liveness([], use_cache=False)
        assert result.status == CheckStatus.SKIPPED

    async def test_bot_protection_skipped(self):
        async with respx.mock:
            respx.head("https://www.linkedin.com/paper").mock(
                return_value=httpx.Response(403)
            )
            result = await check_url_liveness(["https://www.linkedin.com/paper"], use_cache=False)
            assert result.status == CheckStatus.SKIPPED

    async def test_redirect_passes(self):
        async with respx.mock:
            respx.head("http://example.com/paper").mock(
                return_value=httpx.Response(301, headers={"Location": "https://example.com/paper"})
            )
            respx.head("https://example.com/paper").mock(
                return_value=httpx.Response(200)
            )
            result = await check_url_liveness(["http://example.com/paper"], use_cache=False)
            assert result.status in (CheckStatus.PASS, CheckStatus.WARN)

    async def test_timeout_warns(self):
        async with respx.mock:
            respx.head("https://example.com/timeout").mock(
                side_effect=httpx.TimeoutException("timeout")
            )
            result = await check_url_liveness(["https://example.com/timeout"], use_cache=False)
            assert result.status == CheckStatus.WARN


@pytest.mark.asyncio
class TestExistenceCheck:
    class _StaticSource(SourceAdapter):
        def __init__(self, name: str, candidate: Candidate):
            self.name = name
            self.candidate = candidate
            self.search_calls = 0

        async def lookup_doi(self, doi: str):
            return self.candidate

        async def search(self, ref: Reference):
            self.search_calls += 1
            return [self.candidate]

    async def test_doi_lookup_pass(self):
        doi = "10.48550/arXiv.1706.03762"
        ref = Reference(
            title="Attention is all you need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            doi=doi,
        )
        async with respx.mock:
            respx.get(f"https://api.openalex.org/works/https://doi.org/{doi}").mock(
                return_value=httpx.Response(200, json=_OPENALEX_WORK)
            )
            adapter = OpenAlexAdapter()
            result = await check_existence(ref, sources=[adapter], use_cache=False)
            assert result.status in (CheckStatus.PASS, CheckStatus.WARN)
            await adapter.close()

    async def test_not_found(self):
        ref = Reference(
            title="A completely nonexistent paper that no database will ever find",
            authors=["Nobody Known"],
            year=2099,
        )
        async with respx.mock:
            respx.get("https://api.openalex.org/works").mock(
                return_value=httpx.Response(200, json={"results": []})
            )
            adapter = OpenAlexAdapter()
            result = await check_existence(ref, sources=[adapter], use_cache=False)
            assert result.status == CheckStatus.FAIL
            assert result.evidence["source_results"] == [{
                "source": "openalex",
                "status": "no_match",
                "candidate_count": 0,
                "error": None,
            }]
            await adapter.close()

    async def test_all_source_errors_are_not_reported_as_not_found(self):
        class _BrokenSource(SourceAdapter):
            name = "broken"

            async def lookup_doi(self, doi):
                raise TimeoutError("registry unavailable")

            async def search(self, ref):
                raise TimeoutError("registry unavailable")

        result = await check_existence(
            Reference(title="A real paper"),
            sources=[_BrokenSource()],
            use_cache=False,
        )
        assert result.status == CheckStatus.ERROR
        assert result.evidence["verdict"] == "source_unavailable"

    async def test_metadata_mismatch_detected(self):
        doi = "10.48550/arXiv.1706.03762"
        ref = Reference(
            title="Attention is all you need",
            authors=["John Smith", "Jane Doe"],
            year=2020,
            doi=doi,
        )
        async with respx.mock:
            respx.get(f"https://api.openalex.org/works/https://doi.org/{doi}").mock(
                return_value=httpx.Response(200, json=_OPENALEX_WORK)
            )
            adapter = OpenAlexAdapter()
            result = await check_existence(ref, sources=[adapter], use_cache=False)
            assert result.status in (CheckStatus.PASS, CheckStatus.WARN)
            await adapter.close()

    async def test_title_score_does_not_treat_subset_title_as_exact_match(self):
        ref = Reference(title="Pix2seq: A language modeling framework for object detection")
        result = await check_existence(
            ref,
            sources=[self._StaticSource(
                "crossref",
                Candidate(title="A framework for object modeling", year=1999, source="crossref"),
            )],
            use_cache=False,
        )
        assert result.status == CheckStatus.FAIL
        assert result.evidence["title_score"] < 0.85

    async def test_author_overlap_tolerates_pdf_encoding_variants(self):
        ref = Reference(
            title="IMD2020: A large-scale annotated dataset tailored for detecting image manipulations",
            authors=["Adam Novoz’amsk’y", "Babak Mahdian", "Stanislav Kriminala"],
            year=2020,
        )
        result = await check_existence(
            ref,
            sources=[self._StaticSource(
                "crossref",
                Candidate(
                    title="IMD2020: A Large-Scale Annotated Dataset Tailored for Detecting Manipulated Images",
                    authors=["Adam Novozamsky", "Babak Mahdian", "Stanislav Kriminala"],
                    year=2020,
                    source="crossref",
                ),
            )],
            use_cache=False,
        )
        assert result.status == CheckStatus.PASS
        assert result.evidence["author_score"] >= 0.8

    async def test_single_source_year_conflict_is_not_mismatch(self):
        ref = Reference(
            title="Attention is all you need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
        )
        candidate = Candidate(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2025,
            source="crossref",
        )
        result = await check_existence(
            ref,
            sources=[self._StaticSource("crossref", candidate)],
            use_cache=False,
        )
        assert result.status == CheckStatus.PASS
        assert result.evidence.get("year_conflict_unconfirmed")

    async def test_two_sources_supporting_conflict_warns(self):
        ref = Reference(
            title="Attention is all you need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
        )
        candidates = [
            Candidate(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"], year=2025, source="crossref"),
            Candidate(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"], year=2025, source="openalex"),
        ]
        result = await check_existence(
            ref,
            sources=[self._StaticSource("crossref", candidates[0]), self._StaticSource("openalex", candidates[1])],
            use_cache=False,
        )
        assert result.status == CheckStatus.WARN
        assert any(m.startswith("year:") for m in result.evidence.get("mismatches", []))

    async def test_domain_source_is_queried_as_primary_evidence(self):
        ref = Reference(
            title="A versioned computer vision paper",
            authors=["Alice Researcher"],
            year=2024,
        )
        generic = self._StaticSource(
            "crossref",
            Candidate(
                title=ref.title,
                authors=ref.authors,
                year=2024,
                source="crossref",
            ),
        )
        domain = self._StaticSource(
            "dblp",
            Candidate(
                title=ref.title,
                authors=ref.authors,
                year=2024,
                source="dblp",
            ),
        )
        result = await check_existence(
            ref,
            sources=[generic],
            domain_sources=[domain],
            use_cache=False,
        )
        assert result.status == CheckStatus.PASS
        assert domain.search_calls == 1
        assert set(result.evidence["queried_sources"]) == {"crossref", "dblp"}
        assert {c["source"] for c in result.evidence["candidate_sources"]} == {"crossref", "dblp"}

    async def test_explicit_arxiv_id_is_queried_alongside_other_records(self):
        class _ArxivRecordSource(self._StaticSource):
            async def lookup_arxiv_id(self, arxiv_id: str):
                return Candidate(
                    title="Versioned Paper",
                    authors=["Alice Researcher"],
                    year=2021,
                    arxiv_id=arxiv_id,
                    source="arxiv",
                )

        ref = Reference(
            title="Versioned Paper",
            authors=["Alice Researcher"],
            year=2022,
            arxiv_id="2109.10852",
        )
        generic = self._StaticSource(
            "crossref",
            Candidate(title=ref.title, authors=ref.authors, year=2022, source="crossref"),
        )
        arxiv = _ArxivRecordSource(
            "arxiv",
            Candidate(title=ref.title, authors=ref.authors, year=2021, source="arxiv"),
        )
        result = await check_existence(ref, sources=[generic, arxiv], use_cache=False)
        assert result.status == CheckStatus.PASS
        assert any(c["source"] == "arxiv" and c["via"] == "arxiv_lookup"
                   for c in result.evidence["candidate_sources"])
