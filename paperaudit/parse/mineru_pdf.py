"""MinerU PDF-to-Markdown adapter used by the PDF reference parser.

MinerU is intentionally kept behind a small subprocess adapter rather than
imported into Paper-audit.  This lets the application use the user's pinned
Python 3.12 environment while Paper-audit itself remains installable on
Python 3.9 and still has a deterministic fallback when MinerU is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_MINERU_PROJECT = Path("/Users/acc/Documents/ChatGPT/paper-ethical-audit")
DEFAULT_MINERU_VENV = DEFAULT_MINERU_PROJECT / ".venv-mineru"
DEFAULT_MINERU_CLI = DEFAULT_MINERU_VENV / "bin" / "mineru"


def _repo_root() -> Path:
    # parse/mineru_pdf.py -> parse -> paperaudit -> project root
    return Path(__file__).resolve().parents[2]


def _mineru_cli() -> Path | str | None:
    configured = os.getenv("PAPERAUDIT_MINERU_CLI")
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_file() else None
    if DEFAULT_MINERU_CLI.is_file():
        return DEFAULT_MINERU_CLI
    return shutil.which("mineru")


def _cache_dir(path: Path) -> Path:
    """Return a stable extraction directory for this exact file revision."""
    stat = path.stat()
    key = hashlib.sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:20]
    return _repo_root() / "tmp" / "mineru" / f"{path.stem[:80]}-{key}"


def _markdown_from_output(output_dir: Path, stem: str) -> Path | None:
    candidates = list(output_dir.rglob("*.md")) if output_dir.exists() else []
    if not candidates:
        return None
    # Prefer the file MinerU names after the input PDF; otherwise choose the
    # largest Markdown file (the auxiliary files are not Markdown documents).
    exact = [p for p in candidates if p.stem == stem]
    return max(exact or candidates, key=lambda p: p.stat().st_size)


def _content_list_text(output_dir: Path, stem: str) -> str | None:
    """Build a readable Markdown-like stream from MinerU's JSON evidence.

    ``content_list_v2.json`` is useful when a run produced structured blocks
    but its Markdown renderer was omitted or empty.  We keep titles as ATX
    headings so bibliography section boundaries remain available to the
    generic parser.
    """
    candidates = list(output_dir.rglob("*content_list_v2.json")) if output_dir.exists() else []
    if not candidates:
        return None
    source = next((p for p in candidates if p.stem.startswith(stem)), candidates[0])
    try:
        pages = json.loads(source.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    lines: list[str] = []
    for page in pages if isinstance(pages, list) else []:
        blocks = page if isinstance(page, list) else [page]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type", "")).casefold()
            content = block.get("content")
            if not isinstance(content, dict):
                continue
            key = "title_content" if kind == "title" else "paragraph_content"
            pieces = content.get(key)
            if not isinstance(pieces, list):
                continue
            value = "".join(
                str(piece.get("content", ""))
                for piece in pieces
                if isinstance(piece, dict) and piece.get("content") is not None
            ).strip()
            if value:
                lines.append(("# " if kind == "title" else "") + value)
    text = "\n\n".join(lines)
    return text if text.strip() else None


def _with_text_layer(path: Path, markdown: str, output_dir: Path) -> str:
    """Return MinerU Markdown repaired with the PDF embedded text layer.

    MinerU layout is kept untouched: only the characters inside individual
    blocks are re-read from the PDF text layer, so bibliography boundaries
    and section headings stay exactly as MinerU detected them.
    """
    from paperaudit.parse.text_layer import recover_pdf_text_layer

    repaired, _replaced = recover_pdf_text_layer(path, markdown, output_dir)
    return repaired


def convert_pdf_to_markdown(path: Path, timeout: float = 900.0) -> str | None:
    """Run the pinned MinerU CLI and return its Markdown output.

    The output is cached by input path, size, and mtime so repeated web checks
    do not pay the model startup cost again.  A failed/empty invocation returns
    ``None`` and leaves the caller free to use its safe legacy fallback.  Every
    successful result is passed through the PDF text-layer repair, which restores
    characters that MinerU drops for kerned/ligature fonts (see text_layer).
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return None
    cli = _mineru_cli()
    if cli is None:
        return None

    output_dir = _cache_dir(path)
    markdown = _markdown_from_output(output_dir, path.stem)
    if markdown is not None:
        try:
            value = markdown.read_text(encoding="utf-8", errors="replace")
            if value.strip():
                return _with_text_layer(path, value, output_dir)
        except OSError:
            pass
    structured = _content_list_text(output_dir, path.stem)
    if structured:
        return _with_text_layer(path, structured, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    project = Path(os.getenv("PAPERAUDIT_MINERU_PROJECT", str(DEFAULT_MINERU_PROJECT))).expanduser()
    env = os.environ.copy()
    # Keep the production route deterministic and aligned with the pinned
    # environment supplied for Paper-audit. Callers can select a different
    # CLI/project explicitly, but inherited device settings must not silently
    # change extraction behaviour.
    env["MINERU_MODEL_SOURCE"] = "modelscope"
    env["MINERU_DEVICE_MODE"] = "cpu"
    # MinerU's CPU pipeline imports ONNX Runtime.  On macOS, ORT 1.29 can
    # abort during interpreter teardown while its optional telemetry HTTP
    # worker is shutting down (a native SIGABRT outside Python's control).
    # Disable that non-essential telemetry path for this isolated subprocess.
    env["ORT_DISABLE_TELEMETRY"] = "1"
    command = [
        str(cli),
        "--path",
        str(path),
        "--output",
        str(output_dir),
        "--method",
        "txt",
        "--backend",
        "pipeline",
        "--formula",
        "false",
        "--table",
        "false",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(project) if project.is_dir() else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # A native library can abort only during process teardown after MinerU has
    # already written a complete Markdown/JSON result.  Prefer that produced
    # evidence over silently falling back to the legacy PDF parser solely
    # because the subprocess returned a non-zero exit code.
    markdown = _markdown_from_output(output_dir, path.stem)
    if markdown is not None:
        try:
            value = markdown.read_text(encoding="utf-8", errors="replace")
        except OSError:
            value = ""
        if value.strip():
            return _with_text_layer(path, value, output_dir)
    structured = _content_list_text(output_dir, path.stem)
    return _with_text_layer(path, structured, output_dir) if structured else None
