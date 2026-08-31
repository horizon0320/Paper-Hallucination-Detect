from __future__ import annotations

from pathlib import Path

import pytest

from paperaudit.models import Reference
from paperaudit.parse.bibtex import parse_bibtex
from paperaudit.parse.ris import parse_ris
from paperaudit.parse.csl_json import parse_csl_json
from paperaudit.parse.nbib import parse_nbib
from paperaudit.parse.doi_list import parse_doi_list
from paperaudit.parse.plaintext import split, detect_style, extract_fields, Style, parse_plaintext
from paperaudit.parse.intext import canonical_author_key, extract_intext_citations
from paperaudit.parse.detect import auto_parse

FIXTURES = Path(__file__).parent / "fixtures"


class TestBibtex:
    def test_parses_known_real(self):
        refs = parse_bibtex((FIXTURES / "known_real.bib").read_text())
        assert len(refs) == 5
        titles = {r.title for r in refs}
        assert "Attention is all you need" in titles

    def test_extracts_doi(self):
        refs = parse_bibtex((FIXTURES / "known_real.bib").read_text())
        attention = next(r for r in refs if r.title and "Attention" in r.title)
        assert attention.doi is not None
        assert "1706.03762" in attention.doi

    def test_extracts_authors(self):
        refs = parse_bibtex((FIXTURES / "known_real.bib").read_text())
        attention = next(r for r in refs if r.title and "Attention" in r.title)
        assert len(attention.authors) >= 2
        assert any("Vaswani" in a for a in attention.authors)

    def test_extracts_year(self):
        refs = parse_bibtex((FIXTURES / "known_real.bib").read_text())
        attention = next(r for r in refs if r.title and "Attention" in r.title)
        assert attention.year == 2017


class TestRis:
    def test_parses_sample(self):
        refs = parse_ris((FIXTURES / "sample.ris").read_text())
        assert len(refs) == 2

    def test_roundtrip_title(self):
        refs = parse_ris((FIXTURES / "sample.ris").read_text())
        titles = {r.title for r in refs}
        assert "Attention is all you need" in titles

    def test_roundtrip_doi(self):
        refs = parse_ris((FIXTURES / "sample.ris").read_text())
        attention = next(r for r in refs if r.title and "Attention" in r.title)
        assert attention.doi is not None


class TestCslJson:
    def test_parses_sample(self):
        refs = parse_csl_json((FIXTURES / "sample.json").read_text())
        assert len(refs) == 2

    def test_roundtrip_title(self):
        refs = parse_csl_json((FIXTURES / "sample.json").read_text())
        titles = {r.title for r in refs}
        assert "Attention is all you need" in titles

    def test_roundtrip_authors(self):
        refs = parse_csl_json((FIXTURES / "sample.json").read_text())
        attention = next(r for r in refs if r.title and "Attention" in r.title)
        assert any("Vaswani" in a for a in attention.authors)


class TestNbib:
    def test_parses_sample(self):
        refs = parse_nbib((FIXTURES / "sample.nbib").read_text())
        assert len(refs) >= 1

    def test_extracts_pmid(self):
        refs = parse_nbib((FIXTURES / "sample.nbib").read_text())
        pmids = [r.pmid for r in refs if r.pmid]
        assert "34519632" in pmids

    def test_extracts_title(self):
        refs = parse_nbib((FIXTURES / "sample.nbib").read_text())
        titles = [r.title for r in refs if r.title]
        assert any("deep learning" in t.lower() for t in titles)

    def test_extracts_doi(self):
        refs = parse_nbib((FIXTURES / "sample.nbib").read_text())
        dois = [r.doi for r in refs if r.doi]
        assert any("10." in d for d in dois)


class TestDoiList:
    def test_parses_dois(self):
        text = "10.1234/test.001\n10.5678/test.002\n"
        refs = parse_doi_list(text)
        assert len(refs) == 2
        assert refs[0].doi == "10.1234/test.001"


class TestPlaintext:
    def test_canonical_author_key_handles_pdf_glued_citations(self):
        assert canonical_author_key("Su, H") == "su"
        assert canonical_author_key("Carvalho, O. L. F. d") == "carvalho"
        assert canonical_author_key("KCobbe et al.") == "cobbe"
        assert canonical_author_key("GPQA DiamondRein et al.") == "rein"
        assert canonical_author_key("MultiPL-ECassano et al.") == "cassano"
        assert canonical_author_key("MMLU- ReduxGema et al.") == "gema"
        assert canonical_author_key("BYang et al.") == "yang"

    def test_split_numbered_ieee(self):
        text = (FIXTURES / "ieee_style.txt").read_text()
        chunks = split(text)
        assert len(chunks) == 5

    def test_split_blank_lines_apa(self):
        text = (FIXTURES / "apa_style.txt").read_text()
        chunks = split(text)
        assert len(chunks) == 5

    def test_split_author_year_with_wrapped_publication_year(self):
        text = """\
Alon, U.; and Yahav, E. 2021. On the Bottleneck of Graph Neural Networks.
Appling, D. S.; Briscoe, E. J.; and Hutto, C. J.
2015. Discriminative models for predicting deception strategies.
Carson, T. L. 2010. Lying and deception: Theory and practice.
"""
        chunks = split(text)
        assert len(chunks) == 3
        assert "Appling" in chunks[1]

    def test_split_unnumbered_wrapped_refs_keeps_venue_lines_with_reference(self):
        text = """\
Alice Smith and Bob Jones. First example paper.
Journal of Testing, 2020.
Carol Brown. Second example paper. In Proceedings of Testing, 2021.
"""
        chunks = split(text)
        assert len(chunks) == 2
        assert chunks[0].startswith("Alice Smith")
        assert "Journal of Testing" in chunks[0]
        assert chunks[1].startswith("Carol Brown")

    def test_acl_emnlp_block_boundaries_and_author_year_fields(self):
        # ACL/EMNLP PDFs commonly expose one text block per reference, with
        # long entries split across blocks.  Blank-block boundaries must win
        # over the natural-name heuristic so wrapped author lines do not
        # become separate phantom references.
        text = """\
I Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou,
Junxian He, Graham Neubig, Pengfei Liu, et al. 2023.
Factool: Factuality detection in generative ai–a tool augmented framework for multi-task and multi-domain scenarios.
arXiv preprint arXiv:2307.13528.

Cheng-Han Chiang and Hung-Yi Lee. 2023.
Can large language models be an alternative to human evaluations?
In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi,
Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, et al. 2023.
Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288.
"""
        chunks = split(text)
        refs = [extract_fields(chunk, Style.AUTHOR_YEAR) for chunk in chunks]
        assert len(refs) == 3
        assert refs[0].title and refs[0].title.startswith("Factool:")
        assert refs[0].authors[-1] == "Pengfei Liu"
        assert refs[1].title == "Can large language models be an alternative to human evaluations?"
        assert refs[1].authors == ["Cheng-Han Chiang", "Hung-Yi Lee"]
        assert refs[2].title == "Llama 2: Open foundation and fine-tuned chat models"
        assert refs[2].arxiv_id == "2307.09288"

    def test_author_year_strips_year_from_last_author(self):
        ref = extract_fields(
            "Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, "
            "Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, "
            "Avinatan Hassidim, and Yossi Matias. 2022. True: Re-evaluating "
            "factual consistency evaluation.",
            Style.AUTHOR_YEAR,
        )
        assert ref.authors[-1] == "Yossi Matias"

    def test_aaai_semicolon_author_blocks_are_rejoined(self):
        # A two-column AAAI PDF may cut an author list at the column boundary;
        # the continuation starts with an initial (``G.;``), not a new record.
        text = """\
Cheng, B.; Misra, I.; Schwing, A.

G.; Kirillov, A.; and Girdhar, R. 2022. Masked-attention Mask Transformer
for Universal Image Segmentation. In CVPR.

Zhang, R.; Hu, X.; Li, B.; Huang, S.; Deng, H.; Li, H.; Qiao, Y.; and
Gao, P. 2023. Prompt, Generate, then Cache: Cascade of Foundation Models
makes Strong Few-shot Learners. arXiv preprint arXiv:2303.02151.
"""
        refs = parse_plaintext(text)
        assert len(refs) == 2
        assert refs[0].title == "Masked-attention Mask Transformer for Universal Image Segmentation"
        assert refs[0].authors[-1] == "Girdhar, R"
        assert refs[1].title == "Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners"
        assert refs[1].year == 2023

    def test_extract_single_author_title_after_period(self):
        ref = extract_fields(
            "Lei Su. Can we get rid of handcrafted feature extractors? "
            "sparsevit: Nonsemantics-centered, parameter-efficient method. "
            "AAAI, 2025.",
            Style.AUTHOR_YEAR,
        )
        assert ref.authors == ["Lei Su"]
        assert ref.title and ref.title.startswith("Can we get rid of handcrafted")

    def test_author_year_keeps_wrapped_initials_with_following_authors(self):
        samples = [
            (
                "Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., "
                "and Wong, E. Jailbreaking black box large language models in twenty queries. "
                "arXiv preprint arXiv:2310.08419, 2023.",
                "Jailbreaking black box large language models in twenty queries",
                ["Pappas, G. J", "Wong, E"],
            ),
            (
                "Chen, S. F. and Goodman, J. An empirical study of smoothing techniques "
                "for language modeling. Computer Speech & Language, 13(4):359–394, 1999.",
                "An empirical study of smoothing techniques for language modeling",
                ["Chen, S. F", "Goodman, J"],
            ),
            (
                "Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., "
                "de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., "
                "Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L., Lavril, T., Wang, T., "
                "Lacroix, T., and Sayed, W. E. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825.",
                "Mistral 7b",
                ["Chaplot, D. S", "de las Casas, D"],
            ),
        ]
        for raw, title, authors in samples:
            ref = extract_fields(raw, Style.AUTHOR_YEAR)
            assert ref.title == title
            for author in authors:
                assert author in ref.authors

    def test_author_year_does_not_truncate_long_author_lists(self):
        raw = (
            "Alpha, A., Beta, B., Gamma, C., Delta, D., Epsilon, E., Foxtrot, F., "
            "Golf, G., Hotel, H., India, I., Juliett, J., Kilo, K., Lima, L. "
            "A long paper title. arXiv preprint arXiv:2401.00001, 2024."
        )
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert len(ref.authors) == 12

    def test_author_year_drops_et_al_from_author_list(self):
        ref = extract_fields(
            "Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, et al. "
            "MultiPL-E: A scalable approach. arXiv preprint arXiv:2208.08227, 2022.",
            Style.AUTHOR_YEAR,
        )
        assert all("et al" not in author.casefold() for author in ref.authors)

    def test_split_merges_author_only_line_with_wrapped_title(self):
        text = """\
First Author, Second Author, and Third Author.
Wrapped paper title: A study of robust parsing. In 2024 IEEE Symposium on Security and Privacy (SP), 2024.
Next Author and Another Author. A second paper. In Conference, 2023.
"""
        chunks = split(text)
        assert len(chunks) == 2
        assert chunks[0].startswith("First Author, Second Author")
        assert "Wrapped paper title" in chunks[0]
        ref = extract_fields(chunks[0], Style.AUTHOR_YEAR)
        assert ref.title == "Wrapped paper title: A study of robust parsing"
        assert ref.authors == ["First Author", "Second Author", "Third Author"]

    def test_author_year_stops_title_at_ordinal_conference_marker(self):
        raw = (
            "Guangdong Bai, Jike Lei, Guozhu Meng, and Jin Song Dong. "
            "AUTHSCAN: automatic extraction of web authentication protocols from implementations. "
            "In 20th Annual Network and Distributed System Security Symposium, 2013."
        )
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert ref.title == "AUTHSCAN: automatic extraction of web authentication protocols from implementations"

    def test_author_year_does_not_use_url_marker_as_title(self):
        raw = "Anthropic, 2025. URL https://www.anthropic.com/."
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert ref.title is None
        assert ref.authors == ["Anthropic"]
        assert ref.year == 2025

    def test_author_year_stops_at_journal_and_report_venues(self):
        cases = [
            (
                "Guy Hacohen and Daphna Weinshall. Principal components bias in over-parameterized linear models, and its manifestation in deep neural networks. Journal of Machine Learning Research, 23 (155):1-46, 2022.",
                "Principal components bias in over-parameterized linear models, and its manifestation in deep neural networks",
            ),
            (
                "David Tan. Generative ai and copyright infringement. Center for Technology, Robotics, Artificial Intelligence & the Law (TRAIL), January 2024.",
                "Generative ai and copyright infringement",
            ),
        ]
        for raw, expected in cases:
            ref = extract_fields(raw, Style.AUTHOR_YEAR)
            assert ref.title == expected

    def test_author_year_stops_ieee_transactions_volume_pages_before_arxiv(self):
        raw = (
            "Xiaohong Liu. Pscc-net: Progressive spatio-channel correlation network "
            "for image manipulation detection and localization. IEEE Transactions on "
            "Circuits and Systems for Video Technology, 32 (11):7505–7517, 2022. "
            "arXiv:2103.10596."
        )
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert ref.title == "Pscc-net: Progressive spatio-channel correlation network for image manipulation detection and localization"
        assert ref.authors == ["Xiaohong Liu"]
        assert ref.arxiv_id == "2103.10596"

    def test_author_year_stops_generic_carrier_volume_pages(self):
        cases = [
            (
                "A. Researcher. A useful remote sensing method. Remote Sensing, 15 (18):4588, 2023.",
                "A useful remote sensing method",
                "Remote Sensing, 15 (18):4588, 2023",
            ),
            (
                "A. Researcher. A useful vision method. Pattern Recognition 123 (2022) 108456.",
                "A useful vision method",
                "Pattern Recognition 123 (2022) 108456",
            ),
        ]
        for raw, expected_title, expected_venue in cases:
            ref = extract_fields(raw, Style.AUTHOR_YEAR)
            assert ref.title == expected_title
            assert ref.venue == expected_venue

    def test_split_adjacent_web_organization_references_with_period_in_title(self):
        text = (
            "The Hollywood Reporter. Artists Score Major Win in Copyright Case "
            "Against AI Art Generators. https://www.hollywoodreporter.com/story, 2025. "
            "The White House. America’s AI Action Plan. "
            "https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/, "
            "July 2025. Released July 23, 2025 as mandated by EO 14179."
        )
        chunks = split(text)
        assert len(chunks) == 2
        assert chunks[0].startswith("The Hollywood Reporter.")
        assert chunks[1].startswith("The White House.")
        assert "Artists Score Major Win" in chunks[0]
        assert "America’s AI Action Plan" in chunks[1]

    def test_split_adjacent_dataset_records_and_arxiv_record(self):
        text = (
            "ModelScope Community. AI-ModelScope/CodeAlpaca-20k. "
            "https://huggingface.co/datasets/AI-ModelScope/CodeAlpaca-20k, 2023. "
            "Open-R1 Community. Verifiable Coding Problems Python 10k. "
            "https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python-10k, 2024."
        )
        chunks = split(text)
        assert len(chunks) == 2
        refs = parse_plaintext(text, Style.AUTHOR_YEAR)
        assert [r.title for r in refs] == [
            "AI-ModelScope/CodeAlpaca-20k",
            "Verifiable Coding Problems Python 10k",
        ]

    def test_split_adjacent_full_author_records_without_url(self):
        text = (
            "Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, et al. "
            "Mmlu-pro: A robust benchmark. Advances in Neural Information Processing Systems, 2024. "
            "An Yang, Anfeng Li, Baosong Yang, and Chenxu Lv. Qwen3 technical report. "
            "arXiv preprint arXiv:2505.09388, 2025."
        )
        chunks = split(text)
        assert len(chunks) == 2
        assert chunks[1].startswith("An Yang")

    def test_author_year_repairs_ocr_initial_chain_before_title(self):
        raw = (
            "David Rein, Stas Gaskin, Lajanugen Logeswaran, Adva Wolf, "
            "Yoav Goldberg, C. J. H. un, Swaroop Mishra, and Daniel Khashabi. "
            "Gpqa: A graduate-level google-proof q&a benchmark, 2023."
        )
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert ref.title == "Gpqa: A graduate-level google-proof q&a benchmark"
        assert "C. J. H. un" in ref.authors

    def test_author_year_stops_at_named_international_conference(self):
        raw = (
            "Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, "
            "Siva Reddy, Aaron Courville, and Nicolas Le Roux. VINE-PPO: Unlocking RL potential "
            "for LLM reasoning through refined credit assignment. In The Twelfth International "
            "Conference on Learning Representations, ICLR 2024, 2024."
        )
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert ref.title == "VINE-PPO: Unlocking RL potential for LLM reasoning through refined credit assignment"

    def test_extracts_sentence_style_web_organization_record(self):
        ref = extract_fields(
            "The White House. America’s AI Action Plan. "
            "https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/, "
            "July 2025. Released July 23, 2025 as mandated by EO 14179.",
            Style.AUTHOR_YEAR,
        )
        assert ref.title == "America’s AI Action Plan"
        assert ref.authors == ["The White House"]
        assert ref.year == 2025
        assert ref.urls == [
            "https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/"
        ]

    def test_split_doi_followed_by_author_and_compact_org_records(self):
        text = (
            "Xinguo Feng, Zhongkui Ma, Zihan Wang. Uncovering gradient inversion risks. "
            "doi: 10.1145/3658644.3690292. Four Flynn, Helen King, Anca Dragan. "
            "Strengthening our Frontier Safety Framework. "
            "https://deepmind.google/discover/blog/strengthening-our-frontier-safety-framework/, 2025. "
            "Nagabu, 2024. URL https://huggingface.co/datasets/Nagabu/HAM10000/."
        )
        chunks = split(text)
        assert any(c.startswith("Four Flynn, Helen King, Anca Dragan.") for c in chunks)
        assert any(c.startswith("Nagabu, 2024.") for c in chunks)

    def test_author_year_stops_at_year_prefixed_conference_venue(self):
        raw = (
            "Alex Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. "
            "2010 20th International Conference on Pattern Recognition, pp. 2366-2369, 2010."
        )
        ref = extract_fields(raw, Style.AUTHOR_YEAR)
        assert ref.title == "Image quality metrics: Psnr vs. ssim"

    def test_detect_style_ieee(self):
        text = (FIXTURES / "ieee_style.txt").read_text()
        chunks = split(text)
        style = detect_style(chunks)
        assert style == Style.IEEE

    def test_detect_style_apa(self):
        text = (FIXTURES / "apa_style.txt").read_text()
        chunks = split(text)
        style = detect_style(chunks)
        assert style == Style.APA

    def test_pdf_copypaste_hyphen_cleanup(self):
        text = (FIXTURES / "pdf_copypaste.txt").read_text()
        chunks = split(text)
        assert len(chunks) >= 3
        full_text = " ".join(chunks)
        assert "Neural" in full_text
        assert "Neu-" not in full_text

    def test_spacing_accents_in_pdf_names_are_composed(self):
        ref = extract_fields(
            "Herv´e J´egou. Training data-efﬁcient image transformers & distillation through attention. PMLR, 2021.",
            Style.UNKNOWN,
        )
        assert "Hervé Jégou" in ref.raw
        assert "efficient" in (ref.title or "")

    def test_web_citation_title_stops_before_center_for_metadata(self):
        ref = extract_fields(
            "David Tan. Generative ai and copyright infringement. Center for Technology, Robotics, Artificial Intelligence & the Law (TRAIL), January 2024. URL https://law.nus.edu.sg/trail/generative-ai-copyright-infringement/.",
            Style.UNKNOWN,
        )
        assert ref.title == "Generative ai and copyright infringement"
        assert ref.authors == ["David Tan"]
        assert ref.venue == "Center for Technology, Robotics, Artificial Intelligence & the Law (TRAIL)"
        assert ref.urls

    def test_extract_year(self):
        chunk = '[1] A. Author, "Some Paper," Journal, 2021.'
        ref = extract_fields(chunk, Style.IEEE)
        assert ref.year == 2021

    def test_extract_year_ignores_page_range(self):
        chunk = (
            "Davide Cozzolino, Justus Thies, Marius A Riegler, Matthias Niessner, "
            "and Luisa Verdoliva. Coverage: A novel database for copy-move forgery "
            "detection. In IEEE International Conference on Image Processing (ICIP), "
            "pp. 2082–2086, 2016."
        )
        ref = extract_fields(chunk)
        assert ref.year == 2016

    def test_extract_doi(self):
        chunk = 'Some paper. doi: 10.1234/test.001'
        ref = extract_fields(chunk)
        assert ref.doi == "10.1234/test.001"

    def test_extract_url(self):
        chunk = 'Some paper. Available at https://example.com/paper'
        ref = extract_fields(chunk)
        assert "https://example.com/paper" in ref.urls

    def test_extract_arxiv_id(self):
        chunk = 'Attention is all you need. arXiv:1706.03762, 2017.'
        ref = extract_fields(chunk)
        assert ref.arxiv_id == "1706.03762"

    def test_nature_style_organization_and_initial_authors(self):
        refs = parse_plaintext(
            "1.\tOpenAI et al. GPT4 technical report. Preprint at "
            "https://doi.org/10.48550/arXiv.2303.08774 (2024).\n"
            "2.\tKojima, T., Gu, S. S., Reid, M., Matsuo, Y. & Iwasawa, Y. "
            "Large language models are zero-shot reasoners. In Advances in "
            "Neural Information Processing Systems 35 (ACM, 2022).\n"
        )
        assert len(refs) == 2
        assert refs[0].title == "GPT4 technical report"
        assert refs[0].arxiv_id == "2303.08774"
        assert refs[1].title == "Large language models are zero-shot reasoners"

    def test_keeps_publisher_year_suffix_with_security_reference(self):
        refs = parse_plaintext(
            "Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. "
            "Membership inference attacks against machine learning models. In 2017 IEEE "
            "symposium on security and privacy (SP), pp. 3–18. IEEE, 2017.\n"
            "Zihan Wang, Zhongkui Ma, Xinguo Feng, Ruoxi Sun, Hu Wang, Minhui Xue, and Guangdong Bai. "
            "Corelocker: Neuron-level usage control. In 2024 IEEE Symposium on Security and Privacy (SP), "
            "pp. 2497–2514. IEEE Computer Society, 2024."
        )
        assert len(refs) == 2
        assert refs[0].title == "Membership inference attacks against machine learning models"
        assert refs[0].year == 2017
        assert refs[1].title == "Corelocker: Neuron-level usage control"
        assert refs[1].year == 2024

    def test_repairs_glued_initial_authors_from_pdf_text(self):
        ref = extract_fields(
            'S. Ren, K. R. HeGirshick, and J. Sun, "Faster R-CNN: Towards real-time '
            'object detection with region proposal networks," in Proc. Adv. Neural '
            'Inf. Process. Syst., 2015, pp. 91-99.',
            Style.IEEE,
        )
        assert ref.authors == ["S. Ren", "K. He", "R. Girshick", "J. Sun"]

    def test_repairs_compact_initials_with_glued_authors(self):
        ref = extract_fields(
            'Z. Tian, C.H. ShenChen, and T. He, "FCOS: Fully convolutional one-stage '
            'object detection," in Proc. Int. Conf. Comput. Vis., 2019, pp. 9626-9635.',
            Style.IEEE,
        )
        assert ref.authors == ["Z. Tian", "C.H. Shen", "H. Chen", "T. He"]


class TestIntextCitations:
    def test_extracts_multiple_citations_in_one_parenthesis(self):
        refs = extract_intext_citations(
            "Prior work (Zou et al., 2023; Jia et al., 2024; Chen & Goodman, 1999) "
            "and OpenAI (2025) are cited here."
        )
        assert [ref.raw for ref in refs] == [
            "Zou et al., 2023",
            "Jia et al., 2024",
            "Chen & Goodman, 1999",
            "OpenAI, 2025",
        ]
        assert all(ref.citation_only for ref in refs)

    def test_ignores_prose_years(self):
        refs = extract_intext_citations("In 2024, the method improved. As 2023 shows, results vary.")
        assert refs == []


class TestAutoDetect:
    def test_detects_bibtex_by_extension(self):
        refs = auto_parse(FIXTURES / "known_real.bib")
        assert len(refs) == 5

    def test_detects_ris_by_extension(self):
        refs = auto_parse(FIXTURES / "sample.ris")
        assert len(refs) == 2

    def test_detects_json_by_extension(self):
        refs = auto_parse(FIXTURES / "sample.json")
        assert len(refs) == 2

    def test_detects_nbib_by_extension(self):
        refs = auto_parse(FIXTURES / "sample.nbib")
        assert len(refs) >= 1

    def test_detects_plaintext_by_content(self):
        refs = auto_parse(FIXTURES / "ieee_style.txt")
        assert len(refs) >= 3
