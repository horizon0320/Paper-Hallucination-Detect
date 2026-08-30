from __future__ import annotations

import pytest
import respx
import httpx

from paperaudit.models import Candidate, Reference, Verdict, CheckStatus
from paperaudit.core.engine import _build_enriched, _build_domain_sources, verify_one, verify_many
from paperaudit.core.cascade import VerifyOptions
from paperaudit.sources.openalex import OpenAlexAdapter
from paperaudit.checks.existence import _score_candidate

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
class TestEngine:
    async def test_wrong_arxiv_id_does_not_override_conflicting_title(self):
        ref = Reference(title="NuminaMath: A new frontier for LLM mathematical reasoning", arxiv_id="2406.00762")
        cand = Candidate(
            title="Mechanisms of unstable blowup in a quadratic nonlinear Schrödinger equation",
            authors=["Jonathan Jaquette"],
            arxiv_id="2406.00762",
            source="arxiv",
        )
        score, evidence = _score_candidate(ref, cand)
        assert score < 0.85
        assert not evidence.get("exact_arxiv_match")

    async def test_auto_mode_always_includes_dblp_for_titled_reference(self):
        # Venue parsing can fail even for an unmistakable CS paper.  DBLP is
        # therefore a per-reference evidence pass, not a venue-classification
        # fallback.
        ref = Reference(title="Deformable DETR: Deformable Transformers for End-to-End Object Detection")
        sources = _build_domain_sources(ref, VerifyOptions(domain_mode="auto"))
        assert {source.name for source in sources} >= {"dblp", "arxiv"}
        for source in sources:
            await source.close()

    async def test_unconfirmed_conflict_does_not_propagate_candidate_metadata(self):
        ref = Reference(title="Attention is all you need", year=2017)
        evidence = {
            "year_conflict_unconfirmed": {
                "cited_year": 2017,
                "conflicting_sources": ["crossref"],
            },
            "best_candidate_doi": "10.65215/ysbyhc05",
            "best_candidate_year": 2025,
        }
        assert _build_enriched(ref, evidence) is None

    def _opts(self, sources=None, llm_client=None, check_url=False, check_relevance=False):
        return VerifyOptions(
            sources=sources or [],
            llm_client=llm_client,
            check_url=check_url,
            check_relevance=check_relevance,
            use_cache=False,
            domain_mode="none",
        )

    async def test_verified_with_matching_doi(self):
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
            opts = self._opts(sources=[adapter])
            report = await verify_one(ref, opts)
            assert report.overall_verdict == Verdict.VERIFIED
            await adapter.close()

    async def test_not_found_for_unknown_paper(self):
        ref = Reference(
            title="Completely Made Up Paper That Does Not Exist",
            authors=["Fictional Author"],
            year=2099,
        )
        async with respx.mock:
            respx.get("https://api.openalex.org/works").mock(
                return_value=httpx.Response(200, json={"results": []})
            )
            adapter = OpenAlexAdapter()
            opts = self._opts(sources=[adapter])
            report = await verify_one(ref, opts)
            assert report.overall_verdict == Verdict.NOT_FOUND
            await adapter.close()

    async def test_unresolvable_empty_ref(self):
        ref = Reference()
        opts = self._opts()
        report = await verify_one(ref, opts)
        assert report.overall_verdict == Verdict.UNRESOLVABLE

    async def test_verify_many(self):
        refs = [
            Reference(title="Nonexistent A", authors=["Nobody"], year=2099),
            Reference(title="Nonexistent B", authors=["Anybody"], year=2099),
        ]
        async with respx.mock:
            respx.get("https://api.openalex.org/works").mock(
                return_value=httpx.Response(200, json={"results": []})
            )
            adapter = OpenAlexAdapter()
            opts = self._opts(sources=[adapter])
            reports = await verify_many(refs, opts)
            assert len(reports) == 2
            assert all(r.overall_verdict == Verdict.NOT_FOUND for r in reports)
            await adapter.close()

    async def test_verdict_display_not_found_no_fake(self):
        ref = Reference(title="fake paper", authors=["nobody"], year=2099)
        async with respx.mock:
            respx.get("https://api.openalex.org/works").mock(
                return_value=httpx.Response(200, json={"results": []})
            )
            adapter = OpenAlexAdapter()
            opts = self._opts(sources=[adapter])
            report = await verify_one(ref, opts)
            display = report.verdict_display()
            assert "fake" not in display.lower()
            assert "manual review" in display.lower() or "verify" in display.lower()
            await adapter.close()

    async def test_dead_url_verdict(self):
        ref = Reference(
            title="Attention is all you need",
            authors=["Ashish Vaswani"],
            year=2017,
            doi="10.48550/arXiv.1706.03762",
            urls=["https://example.com/dead-paper"],
        )
        async with respx.mock:
            respx.get(
                "https://api.openalex.org/works/https://doi.org/10.48550/arXiv.1706.03762"
            ).mock(return_value=httpx.Response(200, json=_OPENALEX_WORK))
            respx.head("https://example.com/dead-paper").mock(
                return_value=httpx.Response(404)
            )
            adapter = OpenAlexAdapter()
            opts = self._opts(sources=[adapter], check_url=True)
            report = await verify_one(ref, opts)
            assert report.overall_verdict == Verdict.DEAD_URL
            await adapter.close()
