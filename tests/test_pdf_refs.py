from __future__ import annotations

from pathlib import Path

import pytest

from paperaudit.config import reset_settings
from paperaudit.parse.pdf_refs import (
    _MD_HEADING_RE,
    _extract_text,
    _find_ref_section,
    _remove_pdf_bibliography_noise,
    parse_pdf_refs,
)

MARKDOWN_DOC = """\
# A Paper About Things

## 1. Introduction

Some introductory text that mentions references in passing, padded out
across a few extra lines so the heading below sits well past the
first 20% of the document, which is what _find_ref_section requires
before it will treat a heading match as the real bibliography.

## 7. References

[1] A. Vaswani, N. Shazeer, N. Parmar,
"Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

[2] J. Devlin, M.-W. Chang, K. Lee,
"BERT: Pre-training of deep bidirectional transformers," in NAACL-HLT, 2019, pp. 4171-4186.

[3] Y. LeCun, Y. Bengio, G. Hinton, "Deep learning," Nature, vol. 521, pp. 436-444, 2015.

[4] I. Goodfellow, Y. Bengio, A. Courville,
"Generative adversarial networks," in NeurIPS, 2014, pp. 2672-2680.

## Appendix

Extra material that should not be treated as a reference.
"""


def _make_pdf(tmp_path: Path, text: str) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    pdf_path = tmp_path / "paper.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestMarkdownHeadings:
    def test_strips_atx_heading_markers(self):
        assert _MD_HEADING_RE.sub("", "## 7. References\n") == "7. References\n"
        assert _MD_HEADING_RE.sub("", "# Introduction\n") == "Introduction\n"

    def test_finds_section_under_markdown_heading(self):
        stripped = _MD_HEADING_RE.sub("", MARKDOWN_DOC)
        section = _find_ref_section(stripped)
        assert section is not None
        assert "Attention is all you need" in section
        assert "BERT" in section
        assert "Extra material" not in section

    def test_stops_at_lettered_appendix_with_title_on_next_line(self):
        text = """\
Paper text padded so the references heading is well past the first fifth
of the document and can be distinguished from incidental mentions.

References
[1] A. One, "A Great Paper," in Conf, 2020, pp. 1-2.
[2] B. Two, "Another Paper," in Journal, 2021, pp. 3-4.
A
Details of Exploratory Analysis
This appendix text must not be parsed as a reference.
B
Open Science
More appendix text.
"""
        section = _find_ref_section(text)
        assert section is not None
        assert "Another Paper" in section
        assert "Details of Exploratory Analysis" not in section
        assert "Open Science" not in section

    def test_stops_at_common_post_bibliography_headings(self):
        text = """\
Introductory text padded so the References heading is well past the first
fifth of the document and is treated as the bibliography.

References
[1] A. One, "A Great Paper," in Conf, 2020, pp. 1-2.

Supporting Information
Supplementary experiments.
"""
        section = _find_ref_section(text)
        assert section is not None
        assert "A Great Paper" in section
        assert "Supplementary experiments" not in section

    @pytest.mark.parametrize(
        "heading",
        [
            "Appendix A: Additional Results",
            "A. Additional Results",
            "A. Additional results",
            "A - Additional Results",
        ],
    )
    def test_stops_at_inline_appendix_heading(self, heading):
        text = f"""\
Introductory text padded so the References heading is well past the first
fifth of the document and is treated as the bibliography.

References
[1] A. One, "A Great Paper," in Conf, 2020, pp. 1-2.
{heading}
This appendix text must not be parsed as a reference.
"""
        section = _find_ref_section(text)
        assert section is not None
        assert "A Great Paper" in section
        assert "This appendix text" not in section

    def test_does_not_treat_author_initial_as_appendix(self):
        from paperaudit.parse.pdf_refs import _is_inline_appendix_start

        assert not _is_inline_appendix_start('A. One, "A Great Paper," in Conf, 2020.')

    def test_uses_conservative_numbered_tail_when_heading_is_missing(self):
        text = """\
Body text with no recognizable bibliography heading.
More body text.
\n[1] A. One, "A Great Paper," in Conf, 2020, pp. 1-2.
[2] B. Two, "Another Paper," in Journal, 2021, pp. 3-4.
[3] C. Three, "Third Paper," in Workshop, 2022, pp. 5-6.
Appendix A: Extra material
This must not be included.
"""
        section = _find_ref_section(text)
        assert section is not None
        assert "Third Paper" in section
        assert "This must not be included" not in section

    def test_returns_none_for_document_without_reference_cluster(self):
        text = """\
Body text with a single numbered list item.
1. This is not a bibliography.
"""
        assert _find_ref_section(text) is None

    def test_removes_repeated_running_review_header_inside_bibliography(self):
        lines = [
            "Alice Smith. First paper. Journal, 2020.",
            "Under review as a conference paper at ICLR 2026",
            "Bob Jones. Second paper. Journal, 2021.",
        ]
        cleaned = _remove_pdf_bibliography_noise(lines)
        assert all("Under review" not in line for line in cleaned)


class TestExtractTextMarkerToggle:
    def setup_method(self):
        reset_settings()

    def teardown_method(self):
        reset_settings()

    def test_marker_disabled_by_default_uses_pymupdf(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PAPERAUDIT_USE_MARKER", raising=False)
        reset_settings()
        pdf_path = _make_pdf(tmp_path, "Hello from PyMuPDF")
        text = _extract_text(pdf_path)
        assert "Hello from PyMuPDF" in text

    def test_marker_enabled_but_not_installed_falls_back(self, monkeypatch, tmp_path):
        # marker-pdf is an optional extra and isn't installed in this environment;
        # enabling the flag should still degrade gracefully to PyMuPDF.
        monkeypatch.setenv("PAPERAUDIT_USE_MARKER", "1")
        reset_settings()
        pdf_path = _make_pdf(tmp_path, "Hello from fallback")
        text = _extract_text(pdf_path)
        assert "Hello from fallback" in text

    def test_marker_enabled_and_used_when_available(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PAPERAUDIT_USE_MARKER", "1")
        reset_settings()
        monkeypatch.setattr(
            "paperaudit.parse.marker_pdf.convert_pdf_to_markdown",
            lambda path: "## References\n[1] Marker output.",
        )
        pdf_path = _make_pdf(tmp_path, "ignored")
        text = _extract_text(pdf_path)
        assert text == "## References\n[1] Marker output."


class TestParsePdfRefsEndToEnd:
    def test_parses_references_from_generated_pdf(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PAPERAUDIT_USE_MARKER", raising=False)
        reset_settings()
        body = (
            "Paper Title\n\n"
            "1. Introduction\n"
            "Some introductory text padded across a couple of lines so the\n"
            "references heading further down sits past the first 20% mark.\n\n"
            "References\n"
            '[1] A. One, "A Great Paper," in Conf, 2020, pp. 1-2.\n'
            '[2] B. Two, "Another Paper," in Journal, 2021, pp. 3-4.\n'
        )
        pdf_path = _make_pdf(tmp_path, body)
        refs = parse_pdf_refs(pdf_path, use_grobid=False)
        assert len(refs) == 2
        assert any("Great Paper" in (r.title or "") for r in refs)
        assert any("Another Paper" in (r.title or "") for r in refs)

    def test_stops_nature_bibliography_before_publisher_matter(self):
        text = """\
Introductory text padded so the References heading is well past the first
fifth of the document and is treated as the bibliography.

References
1.\tOpenAI et al. GPT4 technical report. Preprint at https://doi.org/10.48550/arXiv.2303.08774 (2024).
2.\tOpenAI. Introducing SimpleQA; https://openai.com/index/introducing-simpleqa/ (2024).
Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims.
© The Author(s) 2025
Methods
GRPO is described here.
"""
        section = _find_ref_section(text)
        assert section is not None
        assert "Introducing SimpleQA" in section
        assert "Publisher’s note" not in section
        assert "Methods" not in section
