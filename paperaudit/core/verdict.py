from __future__ import annotations

from paperaudit.models import CheckResult, CheckStatus, Reference, Verdict


def compute_verdict(ref: Reference, checks: list[CheckResult]) -> tuple[Verdict, list[str]]:
    """Derive overall verdict from check results. Returns (verdict, notes)."""
    notes: list[str] = []

    if ref.citation_only:
        return Verdict.UNRESOLVABLE, [
            "正文中检测到作者-年份引用，但文末缺少完整书目字段；"
            "需要补充完整参考文献后才能验证。"
        ]

    by_name = {c.name: c for c in checks}

    existence = by_name.get("existence")
    url_liveness = by_name.get("url_liveness")
    relevance = by_name.get("relevance")

    if not ref.title and not ref.doi and not ref.authors and not ref.arxiv_id:
        return Verdict.UNRESOLVABLE, ["Reference could not be parsed into checkable fields."]

    if existence is None or existence.status == CheckStatus.ERROR:
        return Verdict.UNRESOLVABLE, ["Existence check could not run."]

    if existence.status == CheckStatus.FAIL:
        notes.append(
            "Could not verify this reference in any scholarly database. "
            "It may be obscure, not yet indexed, or affected by a search/parser limitation. "
            "Manual review recommended."
        )
        return Verdict.NOT_FOUND, notes

    if existence.status == CheckStatus.WARN:
        mismatches = existence.evidence.get("mismatches", [])
        if mismatches:
            notes.append(f"Metadata mismatches found: {'; '.join(mismatches)}")
            if url_liveness and url_liveness.status == CheckStatus.FAIL:
                notes.append("Additionally, one or more cited URLs are not reachable.")
            return Verdict.METADATA_MISMATCH, notes

    if existence.evidence.get("year_conflict_unconfirmed"):
        conflict = existence.evidence["year_conflict_unconfirmed"]
        notes.append(
            "年份差异仅出现在一个独立信源 "
            f"({', '.join(conflict.get('conflicting_sources', []))})；"
            "系统未将其判定为元数据冲突。"
        )

    if url_liveness and url_liveness.status == CheckStatus.FAIL:
        notes.append("One or more cited URLs returned an error status (4xx/5xx).")
        return Verdict.DEAD_URL, notes

    if relevance and relevance.status == CheckStatus.FAIL:
        label = relevance.evidence.get("label", "UNRELATED")
        rationale = relevance.evidence.get("rationale", "")
        notes.append(f"Content relevance check: {label}. {rationale}")
        return Verdict.CONTENT_DRIFT, notes

    if url_liveness and url_liveness.status == CheckStatus.WARN:
        notes.append("Some URLs timed out or had inconclusive responses.")

    if relevance and relevance.status == CheckStatus.WARN:
        label = relevance.evidence.get("label", "")
        rationale = relevance.evidence.get("rationale", "")
        if label:
            notes.append(f"Relevance check: {label}. {rationale}")

    return Verdict.VERIFIED, notes
