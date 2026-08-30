from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from paperaudit.models import Verdict, VerificationReport

app = typer.Typer(name="paperaudit", help="Citation verification: existence, URL liveness, relevance.")
console = Console()
err_console = Console(stderr=True)

_VERDICT_STYLES = {
    Verdict.VERIFIED: "green",
    Verdict.METADATA_MISMATCH: "yellow",
    Verdict.DEAD_URL: "red",
    Verdict.CONTENT_DRIFT: "magenta",
    Verdict.NOT_FOUND: "bold red",
    Verdict.UNRESOLVABLE: "dim",
}


def _build_sources():
    from paperaudit.sources.crossref import CrossrefAdapter
    from paperaudit.sources.openalex import OpenAlexAdapter
    from paperaudit.sources.semantic_scholar import SemanticScholarAdapter
    from paperaudit.sources.arxiv import ArXivAdapter
    from paperaudit.sources.zenodo import ZenodoAdapter
    from paperaudit.sources.webpage import WebPageAdapter

    return [CrossrefAdapter(), OpenAlexAdapter(), SemanticScholarAdapter(), ArXivAdapter(), ZenodoAdapter(), WebPageAdapter()]


def _resolve_llm_client(
    no_llm: bool,
    llm: str | None,
    llm_model: str | None,
    llm_url: str | None,
):
    """
    Build the LLM client from explicit flags, falling back to env-var auto-detection.

    --llm none        → disabled
    --llm deepseek    → force DeepSeek (needs DEEPSEEK_API_KEY)
    --llm ollama      → force Ollama (needs OLLAMA_MODEL or --llm-model)
    (default)         → auto: DeepSeek if key set, else Ollama if model set, else None
    """
    if no_llm or llm == "none":
        return None

    provider = (llm or "auto").lower()

    if provider in ("deepseek", "auto"):
        try:
            from paperaudit.llm.deepseek import make_deepseek_client
            from paperaudit.config import get_settings
            s = get_settings()
            # Allow --llm-model to override the model name for DeepSeek
            if llm_model and provider == "deepseek":
                from paperaudit.llm.deepseek import DeepSeekClient
                if s.deepseek_api_key:
                    client = DeepSeekClient(s.deepseek_api_key, llm_url or s.deepseek_base_url, llm_model)
                    return client
            client = make_deepseek_client()
            if client:
                return client
        except ImportError:
            pass
        if provider == "deepseek":
            err_console.print(
                "[yellow]--llm deepseek: DEEPSEEK_API_KEY not set or openai package missing.[/yellow]"
            )
            return None

    if provider in ("ollama", "auto"):
        try:
            from paperaudit.llm.ollama import OllamaClient, make_ollama_client
            from paperaudit.config import get_settings
            s = get_settings()
            model = llm_model or s.ollama_model
            if model:
                url = llm_url or s.ollama_base_url
                return OllamaClient(base_url=url, model=model)
            if provider == "ollama":
                err_console.print(
                    "[yellow]--llm ollama: set --llm-model or OLLAMA_MODEL env var.[/yellow]"
                )
        except ImportError:
            pass

    return None


def _make_options(
    check_relevance: bool,
    check_url: bool,
    mailto: str | None,
    concurrency: int,
    use_cache: bool,
    domain: str,
    model: str | None,
    llm: str | None = None,
    llm_model: str | None = None,
    llm_url: str | None = None,
):
    from paperaudit.core.cascade import VerifyOptions
    from paperaudit.config import get_settings

    if mailto:
        settings = get_settings()
        settings.mailto = mailto

    llm_client = _resolve_llm_client(not check_relevance, llm, llm_model, llm_url)

    return VerifyOptions(
        check_url=check_url,
        check_relevance=check_relevance and llm_client is not None,
        use_cache=use_cache,
        sources=_build_sources(),
        llm_client=llm_client,
        concurrency=concurrency,
        domain_mode=domain,
    )


def _render_table(reports: list[VerificationReport]) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Title / Raw", max_width=50, overflow="fold")
    table.add_column("Verdict", min_width=20)
    table.add_column("Notes", max_width=50, overflow="fold")

    for r in reports:
        title = r.reference.title or r.reference.raw[:60] or "(unknown)"
        verdict_label = r.verdict_display()
        style = _VERDICT_STYLES.get(r.overall_verdict, "")
        notes = "; ".join(r.notes[:2]) if r.notes else ""
        table.add_row(title, f"[{style}]{verdict_label}[/{style}]", notes)

    console.print(table)

    total = len(reports)
    bad = sum(
        1 for r in reports
        if r.overall_verdict in (Verdict.NOT_FOUND, Verdict.DEAD_URL, Verdict.METADATA_MISMATCH, Verdict.CONTENT_DRIFT)
    )
    verified = sum(1 for r in reports if r.overall_verdict == Verdict.VERIFIED)
    console.print(
        f"\n[bold]Total:[/bold] {total}  "
        f"[green]Verified: {verified}[/green]  "
        f"[red]Issues: {bad}[/red]  "
        f"[dim]Other: {total - verified - bad}[/dim]"
    )


def _render_md(reports: list[VerificationReport]) -> str:
    lines = ["# Reference Verification Report", ""]
    for r in reports:
        title = r.reference.title or r.reference.raw[:60] or "(unknown)"
        lines.append(f"## {title}")
        lines.append(f"**Verdict:** {r.verdict_display()}")
        if r.reference.doi:
            lines.append(f"**DOI:** {r.reference.doi}")
        if r.notes:
            lines.append("**Notes:**")
            for n in r.notes:
                lines.append(f"- {n}")
        for c in r.checks:
            lines.append(f"- `{c.name}`: {c.status.value} (confidence={c.confidence:.2f})")
        lines.append("")
    return "\n".join(lines)


@app.command("check")
def check_cmd(
    file: str = typer.Argument(..., help="Reference file or '-' for stdin"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, md"),
    check_relevance: bool = typer.Option(False, "--enable-llm/--no-llm", help="启用 LLM 内容相关性检查（默认关闭）"),
    check_url: bool = typer.Option(False, "--url/--no-url", help="启用 URL 存活检查（默认关闭）"),
    llm_provider: Optional[str] = typer.Option(None, "--llm", "--llm-provider", "--provider", help="LLM provider: auto, deepseek, ollama, none"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="Override model name (e.g. llama3.2, deepseek-chat)"),
    llm_url: Optional[str] = typer.Option(None, "--llm-url", help="Override LLM base URL (e.g. http://localhost:11434/v1)"),
    model: Optional[str] = typer.Option(None, "--model", help="Alias for --llm-model (deprecated)", hidden=True),
    mailto: Optional[str] = typer.Option(None, "--mailto", help="Email for API politeness"),
    concurrency: int = typer.Option(8, "--concurrency", "-c", help="Max concurrent checks"),
    use_cache: bool = typer.Option(False, "--cache/--no-cache", help="启用 SQLite 缓存（默认关闭）"),
    domain: str = typer.Option("auto", "--domain", help="Domain adapters: auto (CS: DBLP+arXiv), pubmed, dblp, all, none"),
    style: Optional[str] = typer.Option(None, "--style", help="Override plaintext style detection"),
):
    """Verify references from a file or stdin."""
    if file == "-":
        text = sys.stdin.read()
        from paperaudit.parse.detect import _sniff_and_parse as parse
        refs = parse(text)
    else:
        from paperaudit.parse.detect import auto_parse
        refs = auto_parse(Path(file))

    if not refs:
        err_console.print("[yellow]No references found in input.[/yellow]")
        raise typer.Exit(0)

    opts = _make_options(check_relevance, check_url, mailto, concurrency, use_cache, domain, model or llm_model, llm_provider, llm_model or model, llm_url)

    async def _run():
        from paperaudit.core.engine import verify_many
        enriched = refs
        if opts.llm_client is not None:
            unresolvable = sum(1 for r in refs if not r.title and not r.doi and not r.arxiv_id)
            if unresolvable:
                err_console.print(
                    f"[dim]{unresolvable} reference(s) unresolvable by regex — "
                    f"using LLM to recover fields...[/dim]"
                )
                from paperaudit.parse.plaintext import enrich_with_llm
                enriched = await enrich_with_llm(refs, opts.llm_client)
        return await verify_many(enriched, opts)

    # Keep machine-readable JSON/MD output clean: progress belongs on stderr
    # so callers can redirect stdout directly to a valid artifact.
    err_console.print(f"[dim]Parsed {len(refs)} reference(s). Checking...[/dim]")
    reports = asyncio.run(_run())

    if format == "json":
        print(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))
    elif format == "md":
        print(_render_md(reports))
    else:
        _render_table(reports)

    bad_verdicts = {Verdict.NOT_FOUND, Verdict.DEAD_URL}
    if any(r.overall_verdict in bad_verdicts for r in reports):
        raise typer.Exit(1)


@app.command("check-one")
def check_one_cmd(
    reference: str = typer.Argument(..., help="Raw reference string"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, md"),
    check_relevance: bool = typer.Option(False, "--enable-llm/--no-llm", help="启用 LLM 内容相关性检查（默认关闭）"),
    check_url: bool = typer.Option(False, "--url/--no-url", help="启用 URL 存活检查（默认关闭）"),
    llm_provider: Optional[str] = typer.Option(None, "--llm", "--llm-provider", "--provider", help="LLM provider: auto, deepseek, ollama, none"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="Override model name"),
    llm_url: Optional[str] = typer.Option(None, "--llm-url", help="Override LLM base URL"),
    mailto: Optional[str] = typer.Option(None, "--mailto"),
    use_cache: bool = typer.Option(False, "--cache/--no-cache", help="启用 SQLite 缓存（默认关闭）"),
    domain: str = typer.Option("auto", "--domain"),
):
    """Verify a single raw reference string."""
    from paperaudit.parse.plaintext import extract_fields, Style

    ref = extract_fields(reference, Style.UNKNOWN)

    opts = _make_options(check_relevance, check_url, mailto, 1, use_cache, domain, None, llm_provider, llm_model, llm_url)
    from paperaudit.core.engine import verify_one
    report = asyncio.run(verify_one(ref, opts))

    if format == "json":
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    elif format == "md":
        print(_render_md([report]))
    else:
        _render_table([report])

    if report.overall_verdict in {Verdict.NOT_FOUND, Verdict.DEAD_URL}:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
