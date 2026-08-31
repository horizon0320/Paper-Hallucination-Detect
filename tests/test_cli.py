from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from paperaudit.cli import app
from paperaudit.models import CheckResult, CheckStatus, Reference, VerificationReport, Verdict


def test_check_html_is_one_command(monkeypatch, tmp_path: Path) -> None:
    """HTML mode persists JSON and renders its report without a second command."""
    monkeypatch.chdir(tmp_path)
    reference = Reference(raw="[1] A test paper. 2024.", title="A test paper", year=2024)
    report = VerificationReport(
        reference=reference,
        overall_verdict=Verdict.VERIFIED,
        checks=[
            CheckResult(
                name="existence",
                status=CheckStatus.PASS,
                confidence=0.99,
                evidence={"source_results": [], "candidate_sources": [], "total_candidates": 0},
            )
        ],
    )

    monkeypatch.setattr("paperaudit.parse.detect.auto_parse", lambda _path: [reference])

    async def fake_verify_many(_refs, _opts):
        return [report]

    monkeypatch.setattr("paperaudit.core.engine.verify_many", fake_verify_many)

    result = CliRunner().invoke(app, ["check", "paper.pdf", "--format", "html"])

    assert result.exit_code == 0, result.stdout
    json_path = tmp_path / "outputs" / "paper_paperaudit_basic.json"
    html_path = tmp_path / "outputs" / "paper_paperaudit_basic_report.html"
    assert json_path.exists()
    assert html_path.exists()
    assert json.loads(json_path.read_text())[0]["overall_verdict"] == "VERIFIED"
    assert "Tele cite detection 验证报告" in html_path.read_text()
    assert str(html_path) in result.stdout
