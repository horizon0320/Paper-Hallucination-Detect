from __future__ import annotations

import asyncio
import json
import pytest
import respx
import httpx

from paperaudit.models import Reference
from paperaudit.sources.openalex import OpenAlexAdapter
from paperaudit.sources.crossref import CrossrefAdapter
from paperaudit.sources.semantic_scholar import SemanticScholarAdapter
from paperaudit.sources.arxiv import ArXivAdapter
from paperaudit.sources.zenodo import ZenodoAdapter
from paperaudit.sources.webpage import WebPageAdapter
from paperaudit.sources.domain.dblp import DBLPAdapter, is_cs


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

_CROSSREF_ITEM = {
    "DOI": "10.48550/arXiv.1706.03762",
    "title": ["Attention is all you need"],
    "author": [
        {"given": "Ashish", "family": "Vaswani"},
        {"given": "Noam", "family": "Shazeer"},
    ],
    "published-print": {"date-parts": [[2017]]},
    "container-title": ["Advances in Neural Information Processing Systems"],
}

_SS_PAPER = {
    "paperId": "abc123",
    "title": "Attention is all you need",
    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
    "year": 2017,
    "venue": "NeurIPS",
    "externalIds": {"DOI": "10.48550/arXiv.1706.03762"},
    "abstract": "We propose a new simple network architecture...",
}


@pytest.mark.asyncio
class TestOpenAlexAdapter:
    async def test_lookup_doi_success(self):
        doi = "10.48550/arXiv.1706.03762"
        async with respx.mock:
            respx.get(f"https://api.openalex.org/works/https://doi.org/{doi}").mock(
                return_value=httpx.Response(200, json=_OPENALEX_WORK)
            )
            adapter = OpenAlexAdapter()
            cand = await adapter.lookup_doi(doi)
            assert cand is not None
            assert cand.title == "Attention is all you need"
            assert cand.year == 2017
            assert any("Vaswani" in a for a in cand.authors)
            await adapter.close()

    async def test_lookup_doi_not_found(self):
        doi = "10.9999/nonexistent"
        async with respx.mock:
            respx.get(f"https://api.openalex.org/works/https://doi.org/{doi}").mock(
                return_value=httpx.Response(404, json={"error": "not found"})
            )
            adapter = OpenAlexAdapter()
            cand = await adapter.lookup_doi(doi)
            assert cand is None
            await adapter.close()

    async def test_search_returns_candidates(self):
        ref = Reference(title="Attention is all you need", year=2017)
        async with respx.mock:
            respx.get("https://api.openalex.org/works").mock(
                return_value=httpx.Response(200, json={"results": [_OPENALEX_WORK]})
            )
            adapter = OpenAlexAdapter()
            results = await adapter.search(ref)
            assert len(results) >= 1
            assert results[0].title == "Attention is all you need"
            await adapter.close()

    async def test_search_empty_title_returns_nothing(self):
        ref = Reference()
        adapter = OpenAlexAdapter()
        results = await adapter.search(ref)
        assert results == []
        await adapter.close()


@pytest.mark.asyncio
class TestCrossrefAdapter:
    async def test_lookup_doi_success(self):
        doi = "10.48550/arXiv.1706.03762"
        async with respx.mock:
            respx.get(f"https://api.crossref.org/works/{doi}").mock(
                return_value=httpx.Response(200, json={"message": _CROSSREF_ITEM})
            )
            adapter = CrossrefAdapter()
            cand = await adapter.lookup_doi(doi)
            assert cand is not None
            assert cand.doi == doi
            assert cand.year == 2017
            await adapter.close()


@pytest.mark.asyncio
class TestWebPageAdapter:
    async def test_lookup_url_extracts_ndss_metadata(self):
        landing = "https://example.org/authscan"
        page = """
        <html><head>
          <title>AUTHSCAN: Automatic Extraction of Web Authentication Protocols from Implementations - NDSS Symposium</title>
          <meta property="og:title" content="AUTHSCAN: Automatic Extraction of Web Authentication Protocols from Implementations - NDSS Symposium">
          <meta property="og:description" content="Author(s): Guangdong Bai, Jike Lei, Guozhu Meng, and Jin Song Dong Download: Paper (PDF) Date: 23 Apr 2013">
        </head><body></body></html>
        """
        async with respx.mock:
            respx.get(landing).mock(return_value=httpx.Response(200, text=page))
            adapter = WebPageAdapter()
            cand = await adapter.lookup_url(landing)
            assert cand is not None
            assert cand.title == "AUTHSCAN: Automatic Extraction of Web Authentication Protocols from Implementations"
            assert cand.year == 2013
            assert cand.authors[:2] == ["Guangdong Bai", "Jike Lei"]
            await adapter.close()

    async def test_lookup_url_cleans_filename_style_page_title(self):
        landing = "https://example.org/report"
        page = "<html><head><title>B_Jan24_Generative_AI_and_Copyright_Infringement - Centre for Technology, Robotics, Artificial Intelligence and the Law</title></head></html>"
        async with respx.mock:
            respx.get(landing).mock(return_value=httpx.Response(200, text=page))
            adapter = WebPageAdapter()
            cand = await adapter.lookup_url(landing)
            assert cand is not None
            assert cand.title == "Generative AI and Copyright Infringement"
            await adapter.close()

    async def test_search(self):
        ref = Reference(title="Attention is all you need")
        async with respx.mock:
            respx.get("https://api.crossref.org/works").mock(
                return_value=httpx.Response(200, json={"message": {"items": [_CROSSREF_ITEM]}})
            )
            adapter = CrossrefAdapter()
            results = await adapter.search(ref)
            assert len(results) >= 1
            await adapter.close()

    async def test_lookup_ieee_landing_page_via_embedded_doi(self):
        doi = "10.1109/ICIP.2016.7532339"
        landing = "https://ieeexplore.ieee.org/document/7532339"
        page = '<meta name="citation_doi" content="10.1109/ICIP.2016.7532339">'
        item = {**_CROSSREF_ITEM, "DOI": doi, "title": ["COVERAGE — A novel database for copy-move forgery detection"]}
        async with respx.mock:
            respx.get(landing).mock(return_value=httpx.Response(200, text=page))
            respx.get(f"https://api.crossref.org/works/{doi}").mock(
                return_value=httpx.Response(200, json={"message": item})
            )
            adapter = CrossrefAdapter()
            cand = await adapter.lookup_url(landing)
            assert cand is not None
            assert cand.doi == doi
            await adapter.close()


@pytest.mark.asyncio
class TestSemanticScholarAdapter:
    async def test_lookup_doi(self):
        doi = "10.48550/arXiv.1706.03762"
        async with respx.mock:
            respx.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}").mock(
                return_value=httpx.Response(200, json=_SS_PAPER)
            )
            adapter = SemanticScholarAdapter()
            cand = await adapter.lookup_doi(doi)
            assert cand is not None
            assert "Attention" in (cand.title or "")
            await adapter.close()

    async def test_search(self):
        ref = Reference(title="Attention is all you need")
        async with respx.mock:
            respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
                return_value=httpx.Response(200, json={"data": [_SS_PAPER]})
            )
            adapter = SemanticScholarAdapter()
            results = await adapter.search(ref)
            assert len(results) >= 1
            await adapter.close()


@pytest.mark.asyncio
class TestArXivAdapter:
    async def test_search_quotes_title_for_exact_match(self):
        ref = Reference(title="Pix2seq: A language modeling framework for object detection", year=2022)
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2109.10852v2</id>
    <title>Pix2seq: A Language Modeling Framework for Object Detection</title>
    <published>2021-09-22T00:00:00Z</published>
    <author><name>Ting Chen</name></author>
    <author><name>Saurabh Saxena</name></author>
  </entry>
</feed>"""
        async with respx.mock:
            route = respx.get("https://export.arxiv.org/api/query").mock(
                return_value=httpx.Response(200, text=xml)
            )
            adapter = ArXivAdapter()
            results = await adapter.search(ref)
            assert len(results) == 1
            assert results[0].arxiv_id == "2109.10852"
            assert "ti:\"Pix2seq: A language modeling framework for object detection\"" == route.calls[0].request.url.params["search_query"]
            await adapter.close()

    async def test_search_normalizes_pdf_ligatures(self):
        ref = Reference(title="Certiﬁed adversarial robustness via randomized smoothing", year=2019)
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/1902.02918</id>
          <title>Certified Adversarial Robustness via Randomized Smoothing</title>
          <published>2019-02-08T00:00:00Z</published>
          <author><name>Jeremy M Cohen</name></author>
        </entry></feed>"""
        async with respx.mock:
            route = respx.get("https://export.arxiv.org/api/query").mock(
                return_value=httpx.Response(200, text=xml)
            )
            adapter = ArXivAdapter()
            results = await adapter.search(ref)
            assert results and route.calls.last.request.url.params["search_query"] == 'ti:"Certified adversarial robustness via randomized smoothing"'
            await adapter.close()

    async def test_search_falls_back_to_title_tokens_when_exact_query_is_empty(self):
        ref = Reference(title="A Robust Method for Certified Model Verification", year=2024)
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/2401.00001</id>
          <title>A Robust Method for Certified Model Verification</title>
          <published>2024-01-01T00:00:00Z</published>
        </entry></feed>"""
        def response(request):
            query = request.url.params["search_query"]
            if query.startswith("ti:("):
                return httpx.Response(200, text=xml)
            return httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>')

        async with respx.mock:
            route = respx.get("https://export.arxiv.org/api/query").mock(side_effect=response)
            respx.get("https://arxiv.org/api/query").mock(side_effect=response)
            adapter = ArXivAdapter()
            results = await adapter.search(ref)
            assert len(results) == 1
            assert results[0].arxiv_id == "2401.00001"
            assert route.calls[0].request.url.params["search_query"].startswith('ti:"')
            assert any(call.request.url.params["search_query"].startswith("ti:(") for call in route.calls)
            await adapter.close()

    async def test_search_uses_author_and_keyword_route_after_title_miss(self):
        """A changed/truncated title must still trigger an arXiv preprint search."""
        ref = Reference(
            title="Defense against adversarial attacks using spectral regularization",
            authors=["Dinghuai Zhang", "Yang Song"],
            year=2020,
        )
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>http://arxiv.org/abs/2001.00001</id>
          <title>Defending Models with Spectral Regularization</title>
          <published>2020-01-01T00:00:00Z</published>
          <author><name>Dinghuai Zhang</name></author>
        </entry></feed>"""

        def response(request):
            query = request.url.params["search_query"]
            if query.startswith('au:"Dinghuai Zhang"'):
                return httpx.Response(200, text=xml)
            return httpx.Response(
                200,
                text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>',
            )

        async with respx.mock:
            route = respx.get("https://export.arxiv.org/api/query").mock(side_effect=response)
            respx.get("https://arxiv.org/api/query").mock(side_effect=response)
            adapter = ArXivAdapter()
            results = await adapter.search(ref)
            assert results and results[0].arxiv_id == "2001.00001"
            queries = [call.request.url.params["search_query"] for call in route.calls]
            assert any(q.startswith('au:"Dinghuai Zhang"') for q in queries)
            await adapter.close()


def test_security_venues_enable_dblp_in_auto_mode():
    assert is_cs(Reference(title="Stealing machine learning models via prediction APIs", venue="USENIX Security Symposium"))


def test_abbreviated_neurips_venue_enables_dblp_in_auto_mode():
    assert is_cs(Reference(
        title="Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection",
        raw="in Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 21002-21012.",
    ))


def test_abbreviated_iclr_venue_enables_dblp_in_auto_mode():
    assert is_cs(Reference(
        title="DAB-DETR: Dynamic anchor boxes are better queries for DETR",
        raw="in Proc. Int. Conf. Learn. Representations, 2022.",
    ))


def test_compact_neurips_venue_enables_dblp_in_auto_mode():
    assert is_cs(Reference(
        title="Understanding the effective receptive field in deep convolutional neural networks",
        raw="in Proc.Adv.Neural Inform. Process. Syst., 2016, pp. 4898-4906.",
    ))


def test_dblp_semaphore_is_safe_across_python39_event_loops():
    """Repeated CLI-style asyncio.run calls must not reuse a loop-bound limiter."""
    payload = {
        "result": {
            "hits": {
                "hit": [{
                    "info": {
                        "title": "Attention is all you need",
                        "authors": {"author": [{"text": "Ashish Vaswani"}]},
                        "year": "2017",
                        "venue": "NeurIPS",
                    }
                }]
            }
        }
    }

    async def run_once():
        adapter = DBLPAdapter()
        try:
            results = await adapter.search(Reference(title="Attention is all you need"))
            assert results and results[0].year == 2017
        finally:
            await adapter.close()

    with respx.mock:
        respx.get("http://dblp.org/search/publ/api").mock(
            return_value=httpx.Response(200, json=payload)
        )
        asyncio.run(run_once())
        asyncio.run(run_once())


@pytest.mark.asyncio
class TestZenodoAdapter:
    async def test_lookup_doi(self):
        doi = "10.5281/zenodo.15753192"
        payload = {
            "doi": "10.5281/zenodo.15753193",
            "conceptdoi": doi,
            "metadata": {
                "title": "deepseek-ai/DeepSeek-R1: v1.0.0",
                "publication_date": "2025-06-27",
                "creators": [{"name": "Nano"}],
            },
        }
        async with respx.mock:
            respx.get("https://zenodo.org/api/records/15753192").mock(
                return_value=httpx.Response(200, json=payload)
            )
            adapter = ZenodoAdapter()
            cand = await adapter.lookup_doi(doi)
            assert cand is not None
            assert cand.title == "deepseek-ai/DeepSeek-R1: v1.0.0"
            assert cand.year == 2025
            await adapter.close()
