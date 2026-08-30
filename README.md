# Paper-audit

[![PyPI](https://img.shields.io/pypi/v/paperaudit)](https://pypi.org/project/paperaudit/)
[![Python](https://img.shields.io/pypi/pyversions/paperaudit)](https://pypi.org/project/paperaudit/)
[![CI](https://github.com/horizon0320/Paper-audit/actions/workflows/publish.yml/badge.svg)](https://github.com/horizon0320/Paper-audit/actions/workflows/publish.yml)

Citation verification tool: check whether references actually exist, whether their URLs are live, and whether the content is relevant to the citation context.

## What it does

Three checks per reference:

1. **Existence** — resolves against OpenAlex, Crossref, Semantic Scholar, arXiv, DBLP (CS), PubMed (biomedical), and Google Books (textbooks). Domain records are queried as part of the main evidence pass, then linked across preprint, conference, and journal versions. For computer-science citations, arXiv is queried by exact title first and, when that misses, by author plus discriminative title keywords so title changes between preprint and publication do not hide a real record.
2. **URL liveness** — HTTP HEAD/GET check; classifies 2xx/4xx/timeout/bot-protection
3. **Content relevance** — LLM-backed check comparing fetched content to the cited title/topic (requires `DEEPSEEK_API_KEY` or Ollama)

## Verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `VERIFIED` | Paper found in a scholarly database with matching title, authors, year | None — citation is good |
| `METADATA_MISMATCH` | Paper found, but a field in your citation differs from the database record | Correct the mismatched field; the paper itself is real |
| `DEAD_URL` | Paper exists but one or more cited URLs return 4xx/5xx or time out | Update or remove the URL |
| `CONTENT_DRIFT` | Paper exists and URL is live, but fetched content doesn't match what the citation claims | Review whether you are citing the right paper |
| `NOT_FOUND` | Could not verify in the configured databases | Manual verification recommended; see note below |
| `UNRESOLVABLE` | Could not attempt verification — citation is missing enough fields (no title, no DOI, no authors) | Add missing fields (year, DOI, venue) and re-run |

### NOT_FOUND is not "fake"

`NOT_FOUND` means the tool could not confirm the paper in the databases it queries. Common legitimate reasons:

- **Recent publications** — papers from the past 6–12 months are often not yet indexed
- **Preprints** — papers only on institutional repositories or not yet on arXiv
- **Truncated or missing DOI** — without a DOI, title search may miss the paper
- **Obscure venues** — proceedings from smaller conferences may not be in major databases

In the HTML report, a strong author-list conflict is displayed separately as
`疑似幻觉` because it is more concerning than an ordinary version or URL
warning. This is a risk label, not proof that the source document was fabricated.

### Expected verification rates by publication year

| Publication year | Typical verification rate |
|---|---|
| ≤ 2023 | 85–100% |
| 2024 | 60–85% |
| 2025 | 30–60% |
| 2026 | 10–30% |

## Install

```bash
pip install paperaudit                 # basic install
pip install "paperaudit[cli-llm]"      # + DeepSeek for relevance checks
```

For development:

```bash
git clone https://github.com/horizon0320/Paper-audit
cd Paper-audit
pip install -e ".[dev]"
```

## CLI usage

```bash
# Check a PDF — extracts references automatically (PyMuPDF, multi-column aware)
paperaudit check paper.pdf
paperaudit check paper.pdf                    # basic existence check
paperaudit check paper.pdf --format md > report.md

# Check a BibTeX / RIS / CSL-JSON / NBIB / plaintext file
paperaudit check refs.bib
paperaudit check refs.ris
paperaudit check refs.json

# Read from stdin
cat refs.txt | paperaudit check -

# Single ad-hoc reference
paperaudit check-one "Vaswani et al. (2017). Attention is all you need. NeurIPS."

# Output formats: table (default), json, md
paperaudit check refs.bib --format json
paperaudit check refs.bib --format md > report.md

# Optional checks/cache (disabled by default)
paperaudit check refs.bib --enable-llm --llm ollama  # enable relevance (requires a configured LLM)
paperaudit check refs.bib --url              # enable URL liveness
paperaudit check refs.bib --cache            # enable SQLite cache

# Domain adapters (auto by default)
paperaudit check refs.bib --domain pubmed   # force PubMed only
paperaudit check refs.bib --domain dblp    # force DBLP only
paperaudit check refs.bib --domain none    # disable domain adapters

默认 `--domain auto` 按本工具的计算机论文范围处理：每条有标题的参考文献都会
查询 DBLP 和 arXiv。两者只提供独立证据，最终结果仍由标题、作者、年份以及其他
信源的交叉匹配决定，不会因为单一信源命中就直接判定。
```

Exit code is non-zero if any reference is `NOT_FOUND` or `DEAD_URL` (useful in CI).

## PDF support

Paper-audit uses **PyMuPDF** for PDF text extraction, which handles multi-column layouts (IEEE/ACM conference papers) correctly. References are automatically located, split, and parsed.

Supported citation styles auto-detected from the reference section:
- LNCS / Springer (`Lastname, I.: Title. Venue (Year)`)
- IEEE (`[N] Authors, "Title," Venue, Year`)
- APA, Vancouver, MLA, Chicago

### LLM fallback for garbled references

If some references can't be parsed (garbled PDF text, unusual formatting), Paper-audit can use the configured LLM to recover the fields.

To skip LLM entirely, keep the default mode (or pass `--no-llm`).

### GROBID (optional, best quality)

For the highest-quality reference extraction, run a local GROBID server:

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.1
export PAPERAUDIT_GROBID_URL=http://localhost:8070/api
paperaudit check paper.pdf
```

When GROBID is available, it is used as the primary extractor. PyMuPDF is the fallback when GROBID is not running.

### marker (optional, better text quality for hard PDFs)

[marker](https://github.com/datalab-to/marker) converts a PDF to Markdown using layout/OCR models instead of raw text extraction. It handles scanned pages, complex layouts, and tables better than PyMuPDF, and its `#`-style Markdown headings make the references section easier to locate reliably.

It's opt-in: `marker-pdf` pulls in PyTorch and several GB of layout/OCR models, and conversion is much slower than PyMuPDF (seconds to minutes per PDF, especially on CPU), so it's never enabled just because the package happens to be installed.

```bash
pip install "paperaudit[marker]"
export PAPERAUDIT_USE_MARKER=1
paperaudit check paper.pdf
```

When enabled, marker is tried first within the text-extraction fallback path (i.e. after GROBID, before PyMuPDF). If marker isn't installed or conversion fails, Paper-audit silently falls back to PyMuPDF → pypdf → pdfminer as usual.

## Environment variables

All API keys are **optional** — Paper-audit works without any keys but will hit anonymous rate limits faster when checking large reference lists.

| Variable | Default | Description |
|---|---|---|
| `PAPERAUDIT_MAILTO` | `paperaudit@example.com` | Polite email for OpenAlex/Crossref API (strongly recommended) |
| `SEMANTIC_SCHOLAR_API_KEY` | *(optional)* | Raises Semantic Scholar rate limit from ~1 req/s to 100 req/5s — see below |
| `GOOGLE_BOOKS_API_KEY` | *(optional)* | Raises Google Books limit from ~1k req/day to 100k/day; used for textbook lookup |
| `PAPERAUDIT_GROBID_URL` | *(optional)* | GROBID REST endpoint for high-quality PDF parsing; use `http://localhost:8070/api` for a local Docker instance |
| `PAPERAUDIT_USE_MARKER` | *(optional)* | Set to `1`/`true` to use [marker](https://github.com/datalab-to/marker) (PDF→Markdown via layout/OCR models) for PDF text extraction; requires `pip install "paperaudit[marker]"` |
| `DEEPSEEK_API_KEY` | *(optional)* | Enables relevance checks via DeepSeek; takes priority over Ollama if both are set |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint for DeepSeek |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name |
| `OLLAMA_MODEL` | *(optional)* | Enables relevance checks via local Ollama (e.g. `llama3.2`, `mistral`); used when `DEEPSEEK_API_KEY` is not set |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |

### Getting free API keys

**Semantic Scholar** (recommended — significantly improves reliability for large reference lists):
1. Go to **[semanticscholar.org/product/api#api-key](https://www.semanticscholar.org/product/api#api-key)**
2. Fill in the form — free, approved within minutes
3. Add to your shell profile: `export SEMANTIC_SCHOLAR_API_KEY=your_key`

**Ollama** (free, local, no internet required):
1. Install Ollama from [ollama.com](https://ollama.com) and pull a model: `ollama pull llama3.2`
2. Set `export OLLAMA_MODEL=llama3.2` — Paper-audit will use it automatically when `DEEPSEEK_API_KEY` is not set
3. Works with any model Ollama supports; `llama3.2` or `mistral` are good choices for relevance checking

**Google Books** (recommended when references include textbooks):
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the "Books API" and create an API key
3. Add to your shell profile: `export GOOGLE_BOOKS_API_KEY=your_key`

## Supported input formats

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | PyMuPDF extraction; multi-column aware; GROBID optional; marker optional |
| BibTeX | `.bib` | via bibtexparser |
| RIS | `.ris` | Zotero, Mendeley, EndNote, Web of Science |
| CSL JSON | `.json` | Zotero exports |
| PubMed NBIB | `.nbib` | PubMed direct export |
| DOI list | `.txt` | One DOI per line |
| Plaintext | `.txt` | IEEE, APA, LNCS/Springer, Vancouver, MLA, Chicago; auto-detected |

## Reference enrichment

When a citation is incomplete (missing year, DOI, or venue) but the tool finds a matching paper in a database, the result includes an `enriched` field with the complete metadata. Visible in JSON output:

```json
{
  "overall_verdict": "VERIFIED",
  "reference": { "title": "SOEN-101: ...", "year": null, "doi": null },
  "enriched":  { "title": "SOEN-101: ...", "year": 2025, "doi": "10.1109/ICSE55347.2025.00638", "venue": "ICSE" }
}
```

## Caching

Results are cached in SQLite (`~/.cache/paperaudit/cache.db`):
- **PASS / VERIFIED** results: cached for 30 days
- **FAIL / NOT_FOUND** results: cached for 1 day (so recent publications get re-checked as databases update)

To force a fresh lookup: `--no-cache`, or delete `~/.cache/paperaudit/cache.db`.
