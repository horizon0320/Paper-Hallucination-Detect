from __future__ import annotations

import html
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
import re


DBLP_DISAMBIGUATION_SUFFIX = re.compile(r'\s+\d{4}$')


ROOT = Path(__file__).resolve().parent
JSON_NAME = sys.argv[1] if len(sys.argv) > 1 else '00783_AAAI26_paperaudit_basic.json'
JSON_PATH = ROOT / 'outputs' / JSON_NAME
OUT_PATH = JSON_PATH.with_name(JSON_PATH.stem + '_report.html')


def report_subject_name() -> str:
    """Derive the user-facing input filename from the generated JSON name."""
    name = JSON_PATH.stem
    for suffix in ('_paperaudit_basic', '_paperaudit', '_report'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name or JSON_PATH.stem


def esc(value: object) -> str:
    # Older cached evidence may already contain HTML entities (for example
    # ``&amp;`` in a venue name). Decode once before escaping so the report does
    # not render double-escaped text such as ``&amp;amp;``.
    return html.escape(html.unescape('' if value is None else str(value)))


def normalize_pdf_artifacts(value: object) -> str:
    """Normalize ligatures and spacing accents in legacy extracted citations."""
    text = unicodedata.normalize('NFKC', '' if value is None else str(value))
    for mark, combining in (("´", "\u0301"), ("`", "\u0300"), ("¨", "\u0308"), ("ˆ", "\u0302")):
        text = re.sub(rf"\s*{re.escape(combining)}([A-Za-z])", lambda m, c=combining: unicodedata.normalize('NFC', m.group(1) + c), text)
        text = re.sub(rf"{re.escape(mark)}([A-Za-z])", lambda m, c=combining: unicodedata.normalize('NFC', m.group(1) + c), text)
    return unicodedata.normalize('NFC', text)


def normalize_reference_artifacts(item: dict) -> None:
    ref = item.get('reference') or {}
    for key in ('raw', 'title', 'venue'):
        if ref.get(key):
            ref[key] = normalize_pdf_artifacts(ref[key])
    ref['authors'] = [normalize_pdf_artifacts(a) for a in (ref.get('authors') or [])]


def clean_dblp_author(value: object) -> str:
    return DBLP_DISAMBIGUATION_SUFFIX.sub('', str(value or '')).strip()


def clean_dblp_evidence(item: dict) -> None:
    """Remove DBLP's author disambiguation IDs from legacy JSON artifacts."""
    evidence = ((item.get('checks') or [{}])[0]).get('evidence', {})
    if evidence.get('best_candidate_source') == 'dblp':
        evidence['best_candidate_authors'] = [
            clean_dblp_author(author) for author in (evidence.get('best_candidate_authors') or [])
        ]
    for record in evidence.get('candidate_sources') or []:
        if record.get('source') == 'dblp':
            record['authors'] = [clean_dblp_author(author) for author in (record.get('authors') or [])]
    enriched = item.get('enriched')
    if enriched and evidence.get('best_candidate_source') == 'dblp':
        enriched['authors'] = [clean_dblp_author(author) for author in (enriched.get('authors') or [])]


def has_author_mismatch(item: dict) -> bool:
    # A NOT_FOUND result may still contain a low-ranked, unrelated candidate.
    # Its author list is not evidence of a hallucinated citation. However, if
    # the matcher explicitly recorded an author conflict and the title is a
    # strong match, this is precisely the "疑似幻觉" case even when an older
    # JSON artifact carried a NOT_FOUND verdict.
    if item.get('overall_verdict') == 'UNRESOLVABLE':
        return False
    evidence = ((item.get('checks') or [{}])[0]).get('evidence', {})
    # “疑似幻觉” is a cross-source conclusion. If a primary CS registry was
    # unavailable for this item, a single remaining index with an incomplete
    # or version-specific author list is only a review signal, not enough to
    # accuse the citation of fabricating authors.
    primary_registry_unavailable = any(
        result.get('source') in {'dblp', 'arxiv'} and result.get('status') == 'error'
        for result in evidence.get('source_results', [])
    )
    if primary_registry_unavailable:
        return False
    author_mismatch_evidence = any(
        str(m).lower().startswith('authors:') for m in evidence.get('mismatches', [])
    )
    if item.get('overall_verdict') == 'NOT_FOUND' and not author_mismatch_evidence:
        return False
    if not (evidence.get('exact_doi_match') or evidence.get('exact_arxiv_match')
            or (evidence.get('title_score') or 0) >= 0.80):
        return False
    ref_authors = (item.get('reference') or {}).get('authors') or []
    found_authors = evidence.get('best_candidate_authors') or []
    # Initials such as ``Y. He`` and full names such as ``Yiran He`` are the
    # same bibliographic author, even if the old evidence string pre-dates
    # abbreviation-aware matching.
    if not ref_authors or not found_authors:
        return False
    # A short citation may intentionally list only the first author(s).
    if len(author_signatures(ref_authors)) < 3:
        return False
    _different, author_status = author_comparison(ref_authors, found_authors)
    return author_status not in ('一致', '缩写一致')


def normalized_surname(name: str) -> str:
    if ',' in name:
        name = name.split(',', 1)[0]
    else:
        name = name.split()[-1] if name.split() else ''
    # Normalize surname particles consistently across citation and index
    # formats (``van den Berg, R.`` vs ``Rianne van den Berg``).
    name = name.split()[-1] if name.split() else name
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(ch for ch in name if not unicodedata.combining(ch))
    return ''.join(ch for ch in name.casefold() if ch.isalnum())


def encoding_only_author_conflict(item: dict) -> bool:
    evidence = ((item.get('checks') or [{}])[0]).get('evidence', {})
    mismatches = evidence.get('mismatches', [])
    if not mismatches or not all(str(m).lower().startswith('authors:') for m in mismatches):
        return False
    ref_authors = (item.get('reference') or {}).get('authors') or []
    found_authors = evidence.get('best_candidate_authors') or []
    return bool(ref_authors and found_authors and author_comparison(ref_authors, found_authors)[1] == '缩写一致')


def abbreviation_only_author_conflict(item: dict) -> bool:
    evidence = ((item.get('checks') or [{}])[0]).get('evidence', {})
    if not any(str(m).lower().startswith('authors:') for m in evidence.get('mismatches', [])):
        return False
    ref_authors = (item.get('reference') or {}).get('authors') or []
    found_authors = evidence.get('best_candidate_authors') or []
    return bool(ref_authors and found_authors and author_comparison(ref_authors, found_authors)[1] == '缩写一致')


def normalized_title(value: object) -> str:
    text = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return ''.join(ch for ch in text if ch.isalnum())


def same_title(left: object, right: object) -> bool:
    a, b = normalized_title(left), normalized_title(right)
    if not a or not b:
        return False
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.94


def same_venue(left: object, right: object) -> bool:
    """Treat bibliographic carrier abbreviations and expanded forms alike.

    Indexes commonly shorten a venue (``NeurIPS``) while the citation uses its
    expanded container title (``Advances in Neural Information Processing
    Systems``), or omit proceedings numbers, pages, publisher and location.
    These are presentation/version differences, not evidence that the cited
    work is a different paper.  Keep the comparison conservative: arXiv and a
    conference proceedings record remain different carriers.
    """
    if not left or not right:
        return not left and not right
    def norm(value: object) -> str:
        text = unicodedata.normalize('NFKD', str(value)).casefold()
        text = re.sub(r"\b(?:in|the|proceedings|advances|of|for|and|on|an|a)\b", " ", text)
        text = re.sub(r"\b(?:19|20)\d{2}[a-z]?\b", " ", text)
        # Volume/page tails such as ``34:22745–22757`` are citation details,
        # not a different carrier.
        text = re.sub(r"\b\d{1,3}\s*:\s*\d+(?:\s*[–—\-]\s*\d+)?", " ", text)
        text = re.sub(r"\b(?:pp?|pages?)\.?\s*[\d–—\-]+(?:\s*[–—\-]\s*[\d–—\-]+)?", " ", text)
        text = re.sub(r"\b(?:new york|ny|usa|association for computing machinery|acm|ieee)\b", " ", text)
        text = text.replace('neural information processing systems', ' neurips ')
        text = text.replace('nips', ' neurips ')
        text = re.sub(r'international conference on (?:learning representations|representation learning)', ' iclr ', text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())
    a, b = norm(left), norm(right)
    if not a or not b:
        return False
    if a == b or (a == 'neurips' and b == 'neurips'):
        return True
    # A longer citation often appends volume/edition details to the same
    # proceedings title.  Require a substantial token overlap to avoid
    # collapsing unrelated venues that share only ``conference``.
    at, bt = set(a.split()), set(b.split())
    return bool(at and bt and (at <= bt or bt <= at) and len(at & bt) >= 2)


def authoritative_record(ref: dict, evidence: dict) -> dict:
    """Choose a report-facing record from cross-source evidence.

    Candidates with the same DOI/title are treated as versions of one work.
    A group supported by more independent sources wins over a single high-score
    candidate, while the cited year is used only as a tie-breaker.
    """
    records = [r for r in (evidence.get('candidate_sources') or [])
               if (r.get('title_score') or 0) >= 0.85]
    if not records and evidence.get('best_candidate_title') and (evidence.get('title_score') or 0) >= 0.70:
        records = [{
            'source': evidence.get('best_candidate_source'),
            'title': evidence.get('best_candidate_title'),
            'year': evidence.get('best_candidate_year'),
            'authors': evidence.get('best_candidate_authors') or [],
            'doi': evidence.get('best_candidate_doi'),
            'composite': evidence.get('composite', 0),
        }]

    groups = []
    for record in records:
        doi = str(record.get('doi') or '').casefold().rstrip('/')
        placed = None
        for group in groups:
            if doi and group['doi'] and doi == group['doi']:
                placed = group
                break
            if not doi and not group['doi'] and same_title(record.get('title'), group['records'][0].get('title')):
                placed = group
                break
            if doi and not group['doi'] and same_title(record.get('title'), group['records'][0].get('title')):
                placed = group
                break
        if placed is None:
            placed = {'doi': doi, 'records': []}
            groups.append(placed)
        placed['records'].append(record)

    if not groups:
        return {
            'title': None, 'year': None, 'authors': [], 'doi': None,
            'venue': None, 'sources': [], 'support_count': 0,
            'consensus': False,
        }
    cited_year = ref.get('year')
    groups.sort(key=lambda group: (
        len({r.get('source') for r in group['records'] if r.get('source')}),
        sum(1 for r in group['records'] if cited_year and r.get('year') == cited_year),
        max((r.get('composite') or 0) for r in group['records']),
    ), reverse=True)
    chosen = groups[0]
    chosen_records = chosen['records']
    best_source = evidence.get('best_candidate_source')
    best_title = evidence.get('best_candidate_title')
    representative = max(
        chosen_records,
        key=lambda r: (
            # The verifier has already selected the record with the most
            # compatible carrier/authors. Keep that source as the display
            # representative instead of replacing it with an arXiv version
            # solely because its year equals the cited (possibly mixed) year.
            1 if r.get('source') == best_source and same_title(r.get('title'), best_title) else 0,
            1 if cited_year and r.get('year') == cited_year else 0,
            r.get('composite') or 0,
        ),
    )
    # The current JSON format stores venue only for the selected candidate.
    # Reuse it when that candidate belongs to the selected consensus group.
    venue = evidence.get('best_candidate_venue') if any(
        same_title(r.get('title'), best_title) for r in chosen_records
    ) else None
    return {
        'title': representative.get('title'),
        'year': representative.get('year'),
        'authors': representative.get('authors') or [],
        'doi': representative.get('doi'),
        'venue': venue,
        'sources': sorted({r.get('source') for r in chosen_records if r.get('source')}),
        'support_count': len({r.get('source') for r in chosen_records if r.get('source')}),
        'consensus': len({r.get('source') for r in chosen_records if r.get('source')}) >= 2,
    }


def author_display(authors: list[str]) -> str:
    return ', '.join(str(author) for author in authors) if authors else '未提供'


def author_signatures(authors: list[str]) -> list[tuple[str, str]]:
    signatures = []
    for author in authors or []:
        value = ' '.join(str(author or '').split()).strip()
        pieces = [piece.strip() for piece in value.split(',')]
        if len(pieces) > 1 and all(re.match(r'^[A-ZÀ-ÖØ-Ý](?:\.\s*|\s+)[A-Za-zÀ-ÖØ-öø-ÿ\'’-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ\'’-]+)*$', piece) for piece in pieces):
            names = pieces
        else:
            names = [value]
        for name in names:
            if ',' in name:
                surname, given = (part.strip() for part in name.split(',', 1))
            else:
                tokens = name.split()
                surname, given = (tokens[-1], ' '.join(tokens[:-1])) if tokens else ('', '')
            surname = normalized_surname(surname)
            initial = normalized_title(given)[:1]
            if surname:
                signatures.append((surname, initial))
    return signatures


def author_comparison(cited: list[str], found: list[str]) -> tuple[bool, str]:
    if cited == found:
        return False, '一致'
    cited_sigs = author_signatures(cited)
    found_sigs = author_signatures(found)
    if cited_sigs and found_sigs:
        if len(cited_sigs) == len(found_sigs) and all(
            cited_sig[0] == found_sig[0]
            and (not cited_sig[1] or not found_sig[1] or cited_sig[1] == found_sig[1])
            for cited_sig, found_sig in zip(cited_sigs, found_sigs)
        ):
            return True, '缩写一致'
        available = list(found_sigs)
        matched = 0
        for surname, initial in cited_sigs:
            match_index = next((i for i, (other_surname, other_initial) in enumerate(available)
                                if surname == other_surname and (not initial or not other_initial or initial == other_initial)), None)
            if match_index is not None:
                matched += 1
                available.pop(match_index)
        if matched == len(cited_sigs):
            return True, '缩写一致'
        if matched / max(len(cited_sigs), len(found_sigs)) >= 0.5:
            return True, '部分一致'
    return True, '有差异'


def comparison_row(label: str, cited: object, found: object, different: bool, status: str | None = None) -> str:
    status = status or ('有差异' if different else '一致')
    cls = 'diff' if different else 'same'
    return (f'<div class="comparison-row {cls}"><div class="comparison-label">{esc(label)}</div>'
            f'<div class="comparison-cited">{esc(cited or "未提供")}</div>'
            f'<div class="comparison-found">{esc(found or "未提供")}</div>'
            f'<div class="comparison-status">{status}</div></div>')


def generate_report(json_path: str | Path | None = None) -> Path:
    """Generate an HTML report from a Paper-audit JSON artifact.

    The function is intentionally importable so the CLI can provide a true
    one-command ``--format html`` workflow.  ``json_path`` may be absolute or
    relative to the project's ``outputs`` directory.
    """
    global JSON_PATH, OUT_PATH
    if json_path is not None:
        candidate = Path(json_path)
        JSON_PATH = candidate if candidate.is_absolute() else ROOT / 'outputs' / candidate
        OUT_PATH = JSON_PATH.with_name(JSON_PATH.stem + '_report.html')
    raw = JSON_PATH.read_text()
    data = json.loads(raw[raw.find('['):])
    for item in data:
        normalize_reference_artifacts(item)
        clean_dblp_evidence(item)
        checks = item.get('checks') or []
        if abbreviation_only_author_conflict(item) or encoding_only_author_conflict(item):
            item['overall_verdict'] = 'VERIFIED'
            if checks:
                checks[0]['status'] = 'PASS'
                checks[0].get('evidence', {}).pop('mismatches', None)
            item['notes'] = []
        elif item.get('overall_verdict') == 'VERIFIED' and has_author_mismatch(item):
            item['overall_verdict'] = 'METADATA_MISMATCH'
            if checks:
                checks[0]['status'] = 'WARN'
                checks[0].setdefault('evidence', {}).setdefault(
                    'mismatches', ['authors: author list differs after abbreviation normalization']
                )
            item['notes'] = ['作者名单在缩写归一化后仍不一致，疑似引用幻觉；建议人工核对原始文献。']
        if checks and checks[0].get('evidence', {}).get('year_conflict_unconfirmed'):
            item['enriched'] = None
            checks[0]['confidence'] = min(float(checks[0].get('confidence', 0.0)), 0.85)
            conflict_sources = ', '.join(checks[0]['evidence'].get('year_conflict_unconfirmed', {}).get('conflicting_sources', []))
            item['notes'] = [f'年份差异仅出现在一个独立信源（{conflict_sources}）；系统未将其判定为元数据冲突。']
    # The Typer CLI writes a one-line progress message before JSON; normalize
    # the saved machine-readable artifact so it can be consumed directly.
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    verdicts = Counter(item.get('overall_verdict') for item in data)
    not_found_count = verdicts.get('NOT_FOUND', 0)
    suspected_count = sum(1 for item in data if has_author_mismatch(item))
    citation_only_count = sum(
        1 for item in data if (item.get('reference') or {}).get('citation_only')
    )
    verified_count = sum(
        1 for item in data
        if item.get('overall_verdict') == 'VERIFIED'
        and not has_author_mismatch(item)
        and not (item.get('reference') or {}).get('citation_only')
    )
    review_count = max(
        0, len(data) - verified_count - not_found_count - suspected_count - citation_only_count
    )
    total_count = len(data)
    def pct(count: int) -> float:
        return (count / total_count * 100.0) if total_count else 0.0
    cards = []
    for index, item in enumerate(data, 1):
        ref = item.get('reference', {})
        check = (item.get('checks') or [{}])[0]
        evidence = check.get('evidence', {})
        title = ref.get('title') or ref.get('raw') or '(unknown)'
        candidate = evidence.get('best_candidate_title') or '(no candidate)'
        source = evidence.get('best_candidate_source') or 'unknown'
        confidence = check.get('confidence', 0)
        status = item.get('overall_verdict', 'UNRESOLVABLE')
        # Backward compatibility for reports generated before web-root URLs
        # were preserved by the parser: any cited URL with no strong academic
        # match is a review item, never a fabricated/not-found claim.
        web_source_unresolved = bool(
            evidence.get('web_source_unresolved')
            or (
                (ref.get('urls') or [])
                and status == 'NOT_FOUND'
                and (evidence.get('title_score') or 0.0) < 0.85
            )
        )
        citation_only = bool(ref.get('citation_only'))
        suspected = has_author_mismatch(item)
        category = 'citation-only' if citation_only else ('verified' if status == 'VERIFIED' else ('not-found' if status == 'NOT_FOUND' and not web_source_unresolved else ('suspected' if suspected else 'review')))
        badge_class = 'citation-only' if citation_only else ('suspect' if suspected else ('ok' if status == 'VERIFIED' else ('warn' if web_source_unresolved or status in ('METADATA_MISMATCH', 'CONTENT_DRIFT') else 'danger')))
        display_status = '正文引用缺少文末记录' if citation_only else ('疑似幻觉' if suspected else ('网页来源待人工复核' if web_source_unresolved else status))
        authority = authoritative_record(ref, evidence)
        # A low-score generic search hit (for example ``Universal Code`` for
        # a Sourcegraph homepage citation) is not an authority record.
        authority_title = authority.get('title') or (
            candidate if (evidence.get('title_score') or 0) >= 0.70 else None
        )
        authority_sources = authority.get('sources') or ([source] if source != 'unknown' else [])
        support_count = authority.get('support_count') or (1 if authority_title and authority_title != '(no candidate)' else 0)
        authority_label = '多源共识权威记录' if authority.get('consensus') else '最佳匹配记录'
        if citation_only:
            authority_title = '未提供完整文末记录'
            authority_sources = []
            support_count = 0
            authority_label = '未执行唯一性检索'
        if category == 'not-found':
            authority_title = '未找到可确认的学术记录'
            authority_sources = []
            support_count = 0
            authority_label = '未形成权威记录'
        if web_source_unresolved:
            authority_title = '未形成可交叉验证的学术记录'
            authority_sources = []
            support_count = 0
            authority_label = '网页来源待人工复核'
        display_source = '未形成权威记录' if category == 'not-found' else source
        notes = '; '.join(item.get('notes') or [])
        if citation_only:
            notes = '正文中检测到该作者-年份引用，但文末 References 没有对应完整条目；需要补充书目后才能验证。'
        # Keep the report wording neutral even when an older JSON artifact
        # still contains the pre-consensus "may be fabricated" message.
        if status == 'NOT_FOUND':
            notes = '当前信源未能验证该参考文献，可能是未收录、信息不完整或检索受限；建议人工复核。'
        elif web_source_unresolved:
            notes = '该条目提供的是网页来源；当前未形成可交叉验证的学术记录，不将其判定为不存在，建议人工核对原始链接。'
        elif suspected:
            notes = '作者列表与权威学术记录存在明显冲突，疑似引用幻觉；建议人工核对原始文献。'
        cited_authors = ref.get('authors') or []
        found_authors = authority.get('authors') or (evidence.get('best_candidate_authors') or [])
        cited_year = ref.get('year')
        found_year = authority.get('year') or evidence.get('best_candidate_year')
        cited_venue = ref.get('venue')
        found_venue = authority.get('venue') or evidence.get('best_candidate_venue')
        cited_doi = ref.get('doi')
        found_doi = authority.get('doi') or evidence.get('best_candidate_doi')
        # NOT_FOUND may contain low-scoring search candidates. They are not an
        # authority and must never be displayed as the comparison record (a
        # common source of false author/DOI mismatches in the report).
        if category == 'not-found' or web_source_unresolved:
            found_authors = []
            found_year = None
            found_venue = None
            found_doi = None
        evidence_doi = '无可确认 DOI' if category == 'not-found' else (found_doi or '未提供')
        author_diff, author_status = author_comparison(cited_authors, found_authors)
        if category == 'not-found':
            not_found_label = '未找到可确认的学术记录'
            comparison_rows = [
                comparison_row(label, cited, not_found_label, True, '未找到')
                for label, cited in (
                    ('标题', title), ('作者', author_display(cited_authors)),
                    ('年份', cited_year), ('载体', cited_venue), ('DOI', cited_doi)
                )
            ]
        elif web_source_unresolved:
            web_review_label = '未形成可交叉验证的学术记录'
            comparison_rows = [
                comparison_row(label, cited, web_review_label, True, '待人工复核')
                for label, cited in (
                    ('标题', title), ('作者', author_display(cited_authors)),
                    ('年份', cited_year), ('载体', cited_venue), ('DOI', cited_doi)
                )
            ]
        elif category == 'review':
            # Verdict UNRESOLVABLE with no authority record: the check could
            # not run to completion because scholarly registries were throttled
            # or blocked.  Comparing the citation against an empty record made
            # every field read as a difference, which looked like a metadata
            # error rather than a missing lookup.
            unavailable_label = '信源未能应答，无可比对记录'
            comparison_rows = [
                comparison_row(label, cited, unavailable_label, True, '未能核验')
                for label, cited in (
                    ('标题', title), ('作者', author_display(cited_authors)),
                    ('年份', cited_year), ('载体', cited_venue), ('DOI', cited_doi)
                )
            ]
        else:
            comparison_rows = [
                comparison_row('标题', title, authority_title, not same_title(title, authority_title)),
                comparison_row('作者', author_display(cited_authors), author_display(found_authors),
                               author_diff and author_status != '缩写一致', author_status),
                comparison_row('年份', cited_year, found_year,
                               bool(cited_year != found_year and (cited_year or found_year))),
                comparison_row('载体', cited_venue, found_venue,
                not same_venue(cited_venue, found_venue)),
                comparison_row('DOI', cited_doi, found_doi,
                               bool(str(cited_doi or '').casefold().rstrip('/') != str(found_doi or '').casefold().rstrip('/')
                                    and (cited_doi or found_doi))),
            ]
        comparison_html = ''.join(comparison_rows)
        if authority_sources:
            authority_source_text = f'{authority_label} · {support_count} 个独立信源：{", ".join(authority_sources)}'
        else:
            authority_source_text = '未形成可确认的权威记录'
        comparison_block = (
            f'<div class="comparison"><div class="comparison-title">原始引用与搜索记录的字段对照</div>'
            f'<div class="comparison-head"><span>字段</span><span>原始引用</span><span>权威记录</span><span>结论</span></div>'
            f'{comparison_html}<div class="comparison-source">{esc(authority_source_text)}</div></div>'
        )
        match_block = (
            '<div class="match"><b>未执行唯一性检索：</b>正文只有作者-年份键，缺少题名、DOI 等字段，无法可靠定位具体论文。'
            ' <span class="muted">请补充对应的完整文末参考文献</span></div>'
            if citation_only else
            '<div class="match"><b>搜索结果：</b>未找到足够匹配的学术记录 '
            '<span class="muted">当前候选不足以确认该引用</span></div>'
            if category == 'not-found' else
            '<div class="match"><b>搜索结果：</b>该条目为网页来源，尚未形成可交叉验证的学术记录 '
            '<span class="muted">保留原始链接供人工核对</span></div>'
            if web_source_unresolved else
            f'<div class="match"><b>搜索出的权威记录：</b>{esc(authority_title)} '
            f'<span class="muted">{esc(authority.get("year") or evidence.get("best_candidate_year") or "")} · '
            f'{esc(", ".join(authority_sources) or source)} · {esc(authority_label)}</span></div>'
        )
        meta = []
        if ref.get('year'):
            meta.append(f"引用年份：{ref['year']}")
        if ref.get('arxiv_id'):
            meta.append(f"arXiv：{ref['arxiv_id']}")
        if ref.get('doi'):
            meta.append(f"DOI：{ref['doi']}")
        parallel_records = evidence.get('candidate_sources') or []
        record_lines = []
        for record in parallel_records[:8]:
            record_title = record.get('title') or '(无标题)'
            record_source = record.get('source') or 'unknown'
            record_year = record.get('year') or '?'
            record_lines.append(f"{esc(record_source)} · {esc(record_year)} · {esc(record_title)}")
        parallel_html = '<br>'.join(record_lines) if record_lines else '无其他候选记录'
        source_result_lines = []
        source_results = evidence.get('source_results') or []
        if not source_results:
            # Backward-compatible reconstruction for older JSON artifacts.
            counts = Counter(record.get('source', 'unknown') for record in parallel_records)
            queried = evidence.get('queried_sources') or sorted(counts)
            source_results = [
                {'source': name, 'status': 'matched' if counts.get(name) else 'no_match',
                 'candidate_count': counts.get(name, 0), 'error': None}
                for name in queried
            ]
        for source_result in source_results:
            source_name = source_result.get('source') or 'unknown'
            source_status = source_result.get('status')
            source_count = source_result.get('candidate_count', 0)
            if source_status == 'error':
                source_label = f'请求错误：{source_result.get("error") or "未知错误"}'
            elif source_status == 'matched':
                source_label = f'返回 {source_count} 条候选'
            else:
                source_label = '未返回候选'
            source_result_lines.append(f'{esc(source_name)} · {esc(source_label)}')
        source_results_html = '<br>'.join(source_result_lines) if source_result_lines else '未记录逐信源状态'
        cards.append(f'''<article class="ref-card" data-category="{category}" data-status="{esc(status)}">
  <div class="ref-head"><div class="ref-num">[{index}]</div><div class="ref-title">{esc(title)}</div><span class="badge {badge_class}">{esc(display_status)}</span></div>
  <div class="subline">{' · '.join(esc(x) for x in meta) if meta else '未提取 DOI/arXiv 标识'} · 置信度 {float(confidence):.3f} · 代表信源 {esc(display_source)} · 独立支持 {support_count} 个信源</div>
  {match_block}
  {comparison_block}
  {f'<div class="manual-note">{esc(notes)}</div>' if notes else ''}
  <details><summary>查看原始引用与证据</summary>
    <div class="raw">{esc(ref.get('raw') or '')}</div>
    <div class="evidence"><b>权威记录来源：</b>{esc(', '.join(authority_sources) or '未形成共识')}<br><b>候选 DOI：</b>{esc(evidence_doi)}<br><b>其他候选记录：</b>{parallel_html}<br><b>逐信源查询结果：</b>{source_results_html}<br><b>API 调用：</b>{esc(check.get('cost', {}).get('api_calls', 0))} · <b>总候选数：</b>{esc(evidence.get('total_candidates', 0))}</div>
  </details>
</article>''')

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    subject_name = report_subject_name()
    report_heading = 'Tele cite detection 验证报告'
    report_title = f'{report_heading} · {subject_name}'
    html_doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(report_title)}</title>
<style>
:root{{--bg:#f4f7fb;--ink:#172033;--muted:#5d6878;--line:#d9e1ec;--green:#166534;--green-bg:#dcfce7;--amber:#92400e;--amber-bg:#fef3c7;--red:#991b1b;--red-bg:#fee2e2;--not-found:#64748b;--not-found-bg:#e2e8f0;--blue:#1e40af;--citation:#7c3aed;--citation-bg:#f3e8ff}}
.stats{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0}}.citation-stat strong{{color:var(--citation)}}.swatch.citation-only{{background:var(--citation)}}.badge.citation-only{{background:var(--citation-bg);color:var(--citation)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}main{{max-width:1120px;margin:0 auto;padding:28px 18px 60px}}.hero{{background:linear-gradient(135deg,#17324d,#245a72);color:#fff;border-radius:18px;padding:28px 30px;margin-bottom:18px}}h1{{margin:0 0 8px;font-size:clamp(1.6rem,3.4vw,2.5rem);overflow-wrap:anywhere}}h2{{margin:0 0 10px}}.hero p{{margin:0;color:#dbeafe;max-width:900px}}.panel{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0;box-shadow:0 3px 12px #1720330b}}.stat{{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px 15px}}.stat-btn{{display:block;width:100%;text-align:left;cursor:pointer;font:inherit;color:var(--ink)}}.stat-btn:hover{{background:#eef8f5;border-color:#86bdb7;transform:translateY(-1px)}}.stat strong{{display:block;font-size:1.7rem;line-height:1.1}}.stat span{{display:block;color:var(--muted);margin-top:4px;font-size:.9rem}}.ok-stat strong{{color:var(--green)}}.suspect-stat strong{{color:#c2410c}}.warn-stat strong{{color:var(--amber)}}.danger-stat strong{{color:var(--not-found)}}.summary-visual{{display:grid;grid-template-columns:220px 1fr;gap:24px;align-items:center;margin:12px 0 4px}}.donut{{width:190px;height:190px;border-radius:50%;display:grid;place-items:center;margin:auto}}.donut::after{{content:"";width:126px;height:126px;border-radius:50%;background:#fff;grid-area:1/1}}.donut-label{{grid-area:1/1;z-index:1;text-align:center;font-weight:800;font-size:1.8rem;line-height:1.1}}.donut-label small{{display:block;font-size:.78rem;color:var(--muted);font-weight:500;margin-top:5px}}.legend{{display:grid;gap:9px}}.legend-row{{display:flex;align-items:center;gap:9px;color:var(--muted)}}.legend-btn{{border:0;background:transparent;padding:0;cursor:pointer;font:inherit;text-align:left;width:fit-content}}.legend-btn:hover{{color:var(--ink);text-decoration:underline}}.legend-row b{{color:var(--ink);min-width:42px}}.swatch{{width:11px;height:11px;border-radius:3px;display:inline-block}}.swatch.verified{{background:var(--green)}}.swatch.suspected{{background:#ea580c}}.swatch.review{{background:#d97706}}.swatch.not-found{{background:var(--not-found)}}.muted{{color:var(--muted)}}.filters{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}}.filter-btn{{border:1px solid #b9c8d8;background:#fff;border-radius:999px;padding:7px 12px;color:var(--ink);cursor:pointer;font:inherit}}.filter-btn:hover,.filter-btn.active{{background:#e6f4f1;border-color:#86bdb7;color:#115e59}}.filter-count{{color:var(--muted);font-size:.9rem;margin-left:auto}}.ref-card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 17px;margin:11px 0}}.ref-card.hidden{{display:none}}.ref-head{{display:grid;grid-template-columns:42px 1fr auto;gap:10px;align-items:start}}.ref-num{{font-weight:800;color:#486581}}.ref-title{{font-weight:700}}.subline{{font-size:.86rem;color:var(--muted);margin:5px 0 8px 52px;overflow-wrap:anywhere}}.match{{background:#f8fafc;border-radius:8px;padding:9px 11px;margin-left:52px;overflow-wrap:anywhere}}.comparison{{margin:12px 0 0 52px;border:1px solid #dbe7ef;border-radius:10px;overflow:hidden;background:#fbfdff}}.comparison-title{{padding:9px 11px;font-weight:700;color:#164e63;background:#f0f9ff}}.comparison-head,.comparison-row{{display:grid;grid-template-columns:80px minmax(0,1fr) minmax(0,1fr) 54px;gap:10px;padding:8px 11px;align-items:start}}.comparison-head{{font-size:.78rem;color:var(--muted);background:#f8fafc;border-top:1px solid #e5edf3;border-bottom:1px solid #e5edf3}}.comparison-row{{font-size:.88rem;border-bottom:1px solid #edf2f7}}.comparison-label{{font-weight:700;color:#486581}}.comparison-cited,.comparison-found{{overflow-wrap:anywhere}}.comparison-found{{color:#0f766e}}.comparison-row.diff .comparison-found{{color:#b45309;font-weight:600}}.comparison-status{{font-size:.76rem;color:#0f766e;white-space:nowrap}}.comparison-row.diff .comparison-status{{color:#b45309}}.comparison-source{{padding:8px 11px;color:var(--muted);font-size:.8rem;background:#f8fafc;border-top:1px solid #e5edf3}}.badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-size:.78rem;font-weight:800;white-space:nowrap}}.badge.ok{{background:var(--green-bg);color:var(--green)}}.badge.warn{{background:var(--amber-bg);color:var(--amber)}}.badge.suspect{{background:var(--suspect-bg);color:var(--suspect)}}.badge.danger{{background:var(--not-found-bg);color:var(--not-found)}}.manual-note{{margin:10px 0 0 52px;background:#fffbeb;color:#78350f;border-left:3px solid #d97706;padding:8px 10px;border-radius:4px}}details{{margin:10px 0 0 52px}}summary{{cursor:pointer;color:#075985}}.raw{{margin-top:8px;background:#f9fafb;border-left:3px solid #9ca3af;padding:10px;white-space:pre-wrap;overflow-wrap:anywhere}}.evidence{{margin-top:8px;color:var(--muted);font-size:.9rem}}footer{{color:var(--muted);font-size:.85rem;margin-top:20px}}@media(max-width:700px){{.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}}.summary-visual{{grid-template-columns:1fr;gap:14px}}.donut{{width:160px;height:160px}}.ref-head{{grid-template-columns:34px 1fr}}.ref-head>span{{grid-column:2;justify-self:start}}.subline,.match,.manual-note,details,.comparison{{margin-left:0}}.comparison-head,.comparison-row{{grid-template-columns:58px minmax(0,1fr) minmax(0,1fr)}}.comparison-status{{grid-column:3}}.filter-count{{width:100%;margin-left:0}}}}
:root{{--green:#087f5b;--green-bright:#25c98a;--green-bg:#d9fbe9;--suspect:#c2410c;--suspect-bg:#ffedd5}}.donut{{background:conic-gradient(var(--green-bright) 0 {pct(verified_count):.2f}%,#ea580c {pct(verified_count):.2f}% {pct(verified_count + suspected_count):.2f}%,#d97706 {pct(verified_count + suspected_count):.2f}% {pct(verified_count + suspected_count + review_count):.2f}%,var(--citation) {pct(verified_count + suspected_count + review_count):.2f}% {pct(verified_count + suspected_count + review_count + citation_only_count):.2f}%,var(--not-found) {pct(verified_count + suspected_count + review_count + citation_only_count):.2f}% 100%)}}.swatch.verified{{background:var(--green-bright)}}.swatch.suspected{{background:#ea580c}}.swatch.review{{background:#d97706}}.swatch.citation-only{{background:var(--citation)}}.badge.suspect{{background:var(--suspect-bg);color:var(--suspect)}}.badge.citation-only{{background:var(--citation-bg);color:var(--citation)}}.filter-btn:hover,.filter-btn.active{{background:#e5faf3;border-color:#7bd8b5;color:#087f5b}}
</style><style>.hero .subject-title{{font-size:1.15rem;font-weight:600;color:#fff;margin:0 0 6px;overflow-wrap:anywhere}}</style></head><body><main>
<section class="hero"><h1>{esc(report_heading)}</h1><p class="subject-title">{esc(subject_name)}</p><p>生成时间：{generated}</p></section>
<section class="panel"><h2>结果摘要</h2><div class="stats"><button type="button" class="stat stat-btn" data-jump-filter="all"><strong>{total_count}</strong><span>全部参考文献</span></button><button type="button" class="stat stat-btn ok-stat" data-jump-filter="verified"><strong>{verified_count}</strong><span>VERIFIED</span></button><button type="button" class="stat stat-btn suspect-stat" data-jump-filter="suspected"><strong>{suspected_count}</strong><span>疑似幻觉</span></button><button type="button" class="stat stat-btn warn-stat" data-jump-filter="review"><strong>{review_count}</strong><span>需要复核</span></button><button type="button" class="stat stat-btn citation-stat" data-jump-filter="citation-only"><strong>{citation_only_count}</strong><span>缺少文末条目</span></button><button type="button" class="stat stat-btn danger-stat" data-jump-filter="not-found"><strong>{not_found_count}</strong><span>NOT_FOUND</span></button></div>
<div class="summary-visual" role="img" aria-label="参考文献状态分布：全部 {total_count} 条，VERIFIED {verified_count} 条，疑似幻觉 {suspected_count} 条，需要复核 {review_count} 条，缺少文末条目 {citation_only_count} 条，NOT_FOUND {not_found_count} 条">
  <div class="donut"><div class="donut-label">{total_count}<small>条参考文献</small></div></div>
  <div class="legend"><button type="button" class="legend-row legend-btn" data-jump-filter="verified"><span class="swatch verified"></span><b>{verified_count}</b> VERIFIED（{pct(verified_count):.1f}%）</button><button type="button" class="legend-row legend-btn" data-jump-filter="suspected"><span class="swatch suspected"></span><b>{suspected_count}</b> 疑似幻觉（{pct(suspected_count):.1f}%）</button><button type="button" class="legend-row legend-btn" data-jump-filter="review"><span class="swatch review"></span><b>{review_count}</b> 需要复核（{pct(review_count):.1f}%）</button><button type="button" class="legend-row legend-btn" data-jump-filter="citation-only"><span class="swatch citation-only"></span><b>{citation_only_count}</b> 缺少文末条目（{pct(citation_only_count):.1f}%）</button><button type="button" class="legend-row legend-btn" data-jump-filter="not-found"><span class="swatch not-found"></span><b>{not_found_count}</b> NOT_FOUND（{pct(not_found_count):.1f}%）</button></div>
</div>
</section>
<section id="results" class="panel"><h2>逐条结果（共 {total_count} 条）</h2><div class="filters" role="toolbar" aria-label="按结果分类筛选"><span class="muted">查看：</span><button type="button" class="filter-btn active" data-filter="all" aria-pressed="true">全部 <span>({total_count})</span></button><button type="button" class="filter-btn" data-filter="verified" aria-pressed="false">VERIFIED <span>({verified_count})</span></button><button type="button" class="filter-btn" data-filter="suspected" aria-pressed="false">疑似幻觉 <span>({suspected_count})</span></button><button type="button" class="filter-btn" data-filter="review" aria-pressed="false">需要复核 <span>({review_count})</span></button><button type="button" class="filter-btn" data-filter="citation-only" aria-pressed="false">缺少文末条目 <span>({citation_only_count})</span></button><button type="button" class="filter-btn" data-filter="not-found" aria-pressed="false">NOT_FOUND <span>({not_found_count})</span></button><span id="filter-count" class="filter-count">显示 {total_count} 条</span></div><div id="refs">{''.join(cards)}</div></section>
<script>(function(){{const buttons=[...document.querySelectorAll('.filter-btn')];const cards=[...document.querySelectorAll('.ref-card')];const jumpers=[...document.querySelectorAll('[data-jump-filter]')];const count=document.getElementById('filter-count');const results=document.getElementById('results');function apply(filter){{let visible=0;cards.forEach(card=>{{const show=filter==='all'||card.dataset.category===filter;card.classList.toggle('hidden',!show);if(show)visible++;}});buttons.forEach(btn=>{{const active=btn.dataset.filter===filter;btn.classList.toggle('active',active);btn.setAttribute('aria-pressed',String(active));}});count.textContent='显示 '+visible+' 条';}}buttons.forEach(btn=>btn.addEventListener('click',()=>apply(btn.dataset.filter)));jumpers.forEach(btn=>btn.addEventListener('click',()=>{{apply(btn.dataset.jumpFilter);results.scrollIntoView({{behavior:'smooth',block:'start'}});}}));}})();</script>
<footer>验证结果文件：{esc(JSON_PATH.name)} · Paper-audit 本地修改版 · 仅基础存在性验证</footer>
</main></body></html>'''
    OUT_PATH.write_text(html_doc)
    return OUT_PATH


def main() -> None:
    out_path = generate_report()
    print(out_path)


if __name__ == '__main__':
    main()
