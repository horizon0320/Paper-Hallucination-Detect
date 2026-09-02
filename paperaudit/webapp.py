"""Local browser UI for running a Paper-audit PDF check.

Run with ``python -m paperaudit.webapp`` and open the printed localhost URL.
The browser only talks to this local process; PDF contents are not uploaded to
an external service by this UI.
"""

from __future__ import annotations

import asyncio
import cgi
import json
import mimetypes
import shutil
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from paperaudit.cli import _build_sources
from paperaudit.core.cascade import VerifyOptions
from paperaudit.core.engine import verify_many
from paperaudit.parse.detect import auto_parse


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs"
EXTRA_REPORT_DIRS = (REPORT_DIR, ROOT.parent / "outputs")
UPLOAD_DIR = REPORT_DIR / ".web_uploads"
INDEX_PATH = Path(__file__).resolve().parent / "web" / "index.html"
DEMO_PATH = Path(__file__).resolve().parent / "web" / "demo.html"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(updates)


def _get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def _run_job(job_id: str, input_path: Path, stem: str) -> None:
    json_path = REPORT_DIR / f"{stem}_paperaudit_basic.json"
    try:
        _set_job(job_id, phase="parse", progress=5, message="正在读取 PDF…")
        refs = auto_parse(input_path)
        total = len(refs)
        if not refs:
            raise ValueError("没有识别到文末参考文献")
        _set_job(
            job_id,
            total=total,
            completed=0,
            phase="verify",
            progress=15,
            message=f"已识别 {total} 条参考文献，开始多源验证…",
        )

        sources = _build_sources()
        opts = VerifyOptions(
            check_url=False,
            check_relevance=False,
            use_cache=False,
            sources=sources,
            concurrency=8,
            domain_mode="auto",
        )

        def on_progress(completed: int, count: int, report) -> None:
            # Verification work is concurrent, so progress is based on
            # completed records rather than a guessed time estimate.
            percent = 15 + round((completed / max(count, 1)) * 80)
            _set_job(
                job_id,
                completed=completed,
                total=count,
                progress=min(percent, 95),
                phase="verify",
                message=f"正在验证第 {completed}/{count} 条…",
            )

        async def run() -> list:
            try:
                return await verify_many(refs, opts, progress_callback=on_progress)
            finally:
                # Generic adapters are shared by all records and must be
                # closed once the batch is complete to release HTTP sockets.
                for source in sources:
                    close = getattr(source, "close", None)
                    if close is not None:
                        await close()

        reports = asyncio.run(run())
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        from tools_generate_report import generate_report

        html_path = generate_report(json_path)
        # The HTML renderer normalizes abbreviation-only author differences and
        # promotes genuine author conflicts to the visible “疑似幻觉” bucket.
        # Read its normalized JSON back so the progress API reports the same
        # totals users see in the generated page.
        normalized = json.loads(json_path.read_text(encoding="utf-8"))
        from tools_generate_report import has_author_mismatch
        counts = {"VERIFIED": 0, "SUSPECTED": 0, "REVIEW": 0, "NOT_FOUND": 0}
        for item in normalized:
            status = item.get("overall_verdict")
            if status == "NOT_FOUND":
                counts["NOT_FOUND"] += 1
            elif has_author_mismatch(item):
                counts["SUSPECTED"] += 1
            elif status == "VERIFIED":
                counts["VERIFIED"] += 1
            else:
                counts["REVIEW"] += 1
        _set_job(
            job_id,
            phase="done",
            progress=100,
            completed=total,
            total=total,
            message="验证完成",
            report_url="/reports/" + html_path.name,
            json_url="/reports/" + json_path.name,
            counts=counts,
        )
    except Exception as exc:
        _set_job(job_id, phase="error", progress=100, message=str(exc), error=str(exc))
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperAuditWeb/1.0"

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/demo":
            self._send_bytes(DEMO_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/api/jobs/"):
            job = _get_job(unquote(parsed.path.rsplit("/", 1)[-1]))
            self._send_json(job or {"phase": "unknown", "message": "任务不存在"}, 200 if job else 404)
            return
        if parsed.path.startswith("/reports/"):
            name = Path(unquote(parsed.path.rsplit("/", 1)[-1])).name
            for report_dir in EXTRA_REPORT_DIRS:
                target = report_dir / name
                if target.is_file() and target.suffix in {".html", ".json"}:
                    content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "application/json"
                    self._send_bytes(target.read_bytes(), content_type)
                    return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/api/check":
            self._send_json({"error": "not found"}, 404)
            return
        content_type, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
        if content_type != "multipart/form-data":
            self._send_json({"error": "请使用 multipart/form-data 上传 PDF"}, 400)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
        )
        item = form["file"] if "file" in form else None
        filename = Path(getattr(item, "filename", "") or "paper.pdf").name
        if item is None or not filename.lower().endswith(".pdf"):
            self._send_json({"error": "请选择 PDF 文件"}, 400)
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        input_path = UPLOAD_DIR / f"{job_id}.pdf"
        with input_path.open("wb") as handle:
            shutil.copyfileobj(item.file, handle)
        stem = Path(filename).stem or "paper"
        _set_job(job_id, phase="queued", progress=0, completed=0, total=0, message="任务已加入队列")
        threading.Thread(target=_run_job, args=(job_id, input_path, stem), daemon=True).start()
        self._send_json({"job_id": job_id}, HTTPStatus.ACCEPTED)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Paper-audit 本地网页已启动： http://127.0.0.1:8765")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
