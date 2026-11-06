"""
nanowaka-fin-ocr-agent
~~~~~~~~~~~~~~~~~~~~~~
Converts PDF files to Markdown using the fine-tuned OCR model published on
Hugging Face (default: zheungyikgain/nanowaka.fin.ocr, a finetune of
ATH-MaaS/OvisOCR2). Accepts ONE pdf, MANY pdfs, or a DIRECTORY of pdfs.

Two ways in:

1. ADK routes (what the agent platform will call) -- POST /run and
   POST /run_async, JSON + HMAC like every other BYOA agent. PDFs come in
   as base64 (`files`) or as server-side paths (`paths` / `directory`).
   The ADK's HMAC middleware only understands JSON bodies, so raw file
   upload cannot go through /run.

2. POST /ocr -- multipart/form-data, for Postman / curl: one or more
   `file` parts (each a PDF). Auth is the shared secret in the
   `X-Api-Key` header (compared in constant time) unless OCR_REQUIRE_KEY=0 (open). ASYNC BY DEFAULT: returns
   a job_id immediately; poll GET /ocr_status/<job_id>. A request taking >100s
   dies at the Cloudflare edge. Pass `?async=false` for a blocking call
   (short PDFs only).

Markdown for every request is also written to OUTPUT_DIR/<task_id>/.
Server-side paths are restricted to OCR_ALLOWED_INPUT_ROOTS (path traversal
and arbitrary file read are otherwise a real risk with a `paths` input).
"""
import base64
import binascii
import hmac
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from flask import jsonify, request
from waitress import serve

from byoa_adk import Agent, attach_async_routes
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

import ocr_engine

log = logging.getLogger("nanowaka_ocr")

BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR", HERE / "output")).expanduser().resolve()
MAX_UPLOAD_MB = int(os.environ.get("OCR_MAX_UPLOAD_MB", "100"))
# /ocr and /ocr_status are open when OCR_REQUIRE_KEY=0 (browser frontends cannot keep a secret).
OCR_REQUIRE_KEY = os.environ.get("OCR_REQUIRE_KEY", "1") != "0"
CORS_ORIGINS = [o.strip() for o in os.environ.get("OCR_CORS_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ROOTS = [
    Path(p).expanduser().resolve()
    for p in os.environ.get("OCR_ALLOWED_INPUT_ROOTS", "~/Documents:~/Downloads").split(":") if p.strip()
]


# ---------------------------------------------------------------------------
# Core: a list of (name, pdf_bytes) -> per-file results
# ---------------------------------------------------------------------------

def _safe_rel(name: str) -> Path:
    """Relative, traversal-free output path (<name>.md) for any caller-supplied name."""
    parts = [re.sub(r"[^\w.\- ]", "_", p) for p in Path(name.replace("\\", "/")).parts
             if p not in ("", ".", "..", "/")]
    rel = Path(*parts) if parts else Path("document")
    return rel.with_suffix(".md")


def _resolve_allowed(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    if not any(p == r or r in p.parents for r in ALLOWED_ROOTS):
        raise ValueError(f"path not allowed (outside OCR_ALLOWED_INPUT_ROOTS): {raw}")
    if not p.exists():
        raise ValueError(f"path not found: {raw}")
    return p


def _collect_paths(raw_paths: list) -> list:
    """Files and/or directories -> [(display_name, Path)]; directories are searched recursively."""
    items = []
    for raw in raw_paths:
        p = _resolve_allowed(str(raw))
        if p.is_dir():
            items += [(str(f.relative_to(p.parent)), f) for f in sorted(p.rglob("*.pdf"))]
        else:
            items.append((p.name, p))
    return items


def combined_markdown(results: list) -> str:
    """One flat string for frontends: a single PDF's Markdown as-is; several PDFs joined, each under a
    '<!-- file: name -->' marker so the parts can be split again."""
    done = [r for r in results if r["markdown"]]
    if len(done) == 1 and len(results) == 1:
        return done[0]["markdown"]
    return "\n\n".join(f"<!-- file: {r['filename']} -->\n\n{r['markdown']}" for r in done)


def public_view(res: dict, detail: bool = False, keep_paths: bool = False) -> dict:
    """Response body without repetition: the Markdown appears ONCE (top-level `markdown`).
    detail=True also puts each file's Markdown in results[].markdown (for splitting by file).
    Server paths (output_path / output_dir) are only kept for the platform routes."""
    drop = set() if detail else {"markdown"}
    if not keep_paths:
        drop.add("output_path")
    summary = {k: v for k, v in res["summary"].items() if keep_paths or k != "output_dir"}
    return {"model": res["model"], "markdown": res["markdown"], "summary": summary,
            "results": [{k: v for k, v in r.items() if k not in drop} for r in res["results"]]}


def run_ocr(items: list, task_id: str, dpi: int, max_new: int) -> dict:
    """items: [(name, bytes | Path)]. One bad file never stops the batch."""
    out_root = OUTPUT_DIR / re.sub(r"[^\w\-]", "_", task_id)
    results, seen = [], set()
    for name, src in items:
        t0 = time.time()
        rel = _safe_rel(name)
        n = 1
        while rel in seen:                                  # two uploads with the same name
            n += 1
            rel = _safe_rel(name).with_name(f"{_safe_rel(name).stem}_{n}.md")
        seen.add(rel)
        entry = {"filename": name, "status": "ok", "pages": 0, "markdown": "", "flags": [], "output_path": None}
        try:
            data = src.read_bytes() if isinstance(src, Path) else src
            if not data[:5] == b"%PDF-":
                raise ValueError("not a PDF file")
            md, pages, flags = ocr_engine.ocr_pdf_bytes(data, dpi=dpi, max_new=max_new)
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(md, encoding="utf-8")
            entry.update(pages=pages, markdown=md, flags=flags, output_path=str(dest),
                         status="flagged" if flags else "ok")
        except Exception as exc:
            log.exception("OCR failed for %s", name)
            entry.update(status="error", flags=[f"{type(exc).__name__}: {exc}"])
        entry["seconds"] = round(time.time() - t0, 1)
        results.append(entry)
    count = lambda s: sum(r["status"] == s for r in results)
    return {
        "model": ocr_engine.MODEL_ID,
        "markdown": combined_markdown(results),
        "summary": {"files": len(results), "ok": count("ok"), "flagged": count("flagged"),
                    "errors": count("error"), "pages": sum(r["pages"] for r in results),
                    "output_dir": str(out_root)},
        "results": results,
    }


# ---------------------------------------------------------------------------
# ADK agent (JSON + HMAC)
# ---------------------------------------------------------------------------

class NanowakaOcrAgent(Agent):
    name = "nanowaka-fin-ocr-agent"
    version = "1.0.0"
    description = ("Converts PDF files (one, several, or a whole directory) to Markdown with the "
                   "nanowaka.fin.ocr model, an OvisOCR2 finetune trained on financial documents "
                   "(invoices, purchase orders, delivery orders). Input: base64 PDFs in `files`, or "
                   "server-side `paths` / `directory`. Output: one Markdown document per PDF, pages "
                   "separated by '---', plus quality flags (runaway output, malformed tables).")
    capabilities = ["text-extract"]

    def run(self, payload: TaskPayload) -> TaskResult:
        inp = payload.inputs
        items = []

        files = inp.get("files") or []
        if not isinstance(files, list):
            return _fail(payload, "inputs.files must be a list of {filename, content_base64}")
        for i, f in enumerate(files):
            try:
                items.append((str(f.get("filename") or f"document_{i + 1}.pdf"),
                              base64.b64decode(f["content_base64"], validate=True)))
            except (KeyError, AttributeError, binascii.Error, TypeError):
                return _fail(payload, f"inputs.files[{i}] needs filename and valid content_base64")

        raw_paths = list(inp.get("paths") or [])
        if inp.get("directory"):
            raw_paths.append(inp["directory"])
        try:
            items += _collect_paths(raw_paths)
        except ValueError as exc:
            return _fail(payload, str(exc))

        if not items:
            return _fail(payload, "provide at least one PDF via inputs.files, inputs.paths or inputs.directory "
                                  "(a directory with no .pdf files also counts as empty)")
        try:
            dpi = int(inp.get("dpi") or ocr_engine.DEFAULT_DPI)
            max_new = int(inp.get("max_new_tokens") or ocr_engine.DEFAULT_MAX_NEW)
        except (TypeError, ValueError):
            return _fail(payload, "dpi and max_new_tokens must be integers")

        outputs = public_view(run_ocr(items, payload.task_id, dpi, max_new),
                              detail=bool(inp.get("per_file_markdown", False)), keep_paths=True)
        if not inp.get("include_markdown", True):
            outputs["markdown"] = ""
            for r in outputs["results"]:
                r.pop("markdown", None)
        ok = outputs["summary"]["errors"] < outputs["summary"]["files"]
        return TaskResult(task_id=payload.task_id,
                          status=TaskStatus.SUCCESS if ok else TaskStatus.FAILURE,
                          outputs=outputs,
                          error_msg=None if ok else "every file failed; see outputs.results[].flags")


def _fail(payload, msg):
    return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE, error_msg=msg)


INPUT_SCHEMA = {
    "files": {"type": "array", "required": False,
              "description": "[{filename, content_base64}] -- PDFs sent inline as base64."},
    "paths": {"type": "array", "required": False,
              "description": "Server-side PDF file or directory paths (inside OCR_ALLOWED_INPUT_ROOTS)."},
    "directory": {"type": "string", "required": False,
                  "description": "Server-side directory; searched recursively for *.pdf."},
    "dpi": {"type": "integer", "required": False, "description": "Render DPI. Default 200 (matches training)."},
    "max_new_tokens": {"type": "integer", "required": False, "description": "Per-page token limit. Default 4096."},
    "per_file_markdown": {"type": "boolean", "required": False,
                          "description": "true = also return each PDF's Markdown in results[].markdown "
                                         "(default false: the Markdown is returned once, in `markdown`)."},
    "include_markdown": {"type": "boolean", "required": False,
                         "description": "false = return only output_path/flags, not the Markdown text."},
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "markdown": {"type": "string",
                     "description": "All Markdown in one string (several PDFs are joined, each under an "
                                    "'<!-- file: name -->' marker). Same text as results[].markdown."},
        "summary": {"type": "object", "description": "files/ok/flagged/errors/pages/output_dir"},
        "results": {"type": "array",
                    "description": "per PDF: filename, status (ok|flagged|error), pages, markdown, flags, "
                                   "output_path, seconds"},
    },
}


# ---------------------------------------------------------------------------
# Flask app: ADK routes + multipart /ocr
# ---------------------------------------------------------------------------

def build_app(agent):
    app = build_flask_app(agent, BYOA_SECRET, extra_protected_paths={"/run_async"})
    attach_async_routes(app, agent)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    jobs, jobs_lock = {}, threading.Lock()

    def _authed():
        if not OCR_REQUIRE_KEY:
            return True
        return hmac.compare_digest(request.headers.get("X-Api-Key", ""), BYOA_SECRET)

    def _detail():
        return request.args.get("detail", "").lower() in ("1", "true", "yes")

    @app.after_request
    def _cors(resp):
        """CORS for the browser-facing /ocr routes only ('*' or a comma-separated allow-list)."""
        if CORS_ORIGINS and (request.path == "/ocr" or request.path.startswith("/ocr_status/")):
            origin = request.headers.get("Origin", "")
            if "*" in CORS_ORIGINS:
                resp.headers["Access-Control-Allow-Origin"] = "*"
            elif origin in CORS_ORIGINS:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Headers"] = "X-Api-Key, Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Max-Age"] = "600"
        return resp

    def _parse_upload():
        uploads = [f for f in request.files.getlist("file") if f.filename]
        if not uploads:
            return None, (jsonify({"error": "send one or more PDFs as multipart form-data field 'file'"}), 400)
        items = [(Path(f.filename).name, f.read()) for f in uploads]
        try:
            dpi = int(request.form.get("dpi") or ocr_engine.DEFAULT_DPI)
            max_new = int(request.form.get("max_new_tokens") or ocr_engine.DEFAULT_MAX_NEW)
        except ValueError:
            return None, (jsonify({"error": "dpi and max_new_tokens must be integers"}), 400)
        return (items, dpi, max_new), None

    @app.post("/ocr")
    def ocr_upload():
        if not _authed():
            return jsonify({"error": "missing or wrong X-Api-Key"}), 401
        parsed, err = _parse_upload()
        if err:
            return err
        items, dpi, max_new = parsed
        task_id = "upload-" + uuid.uuid4().hex[:12]
        if request.args.get("async", "true").lower() not in ("0", "false", "no"):   # async by default
            with jobs_lock:
                jobs[task_id] = {"status": "running", "result": None, "error": None}

            def worker():
                try:
                    res, st = run_ocr(items, task_id, dpi, max_new), "done"
                except Exception as exc:
                    res, st = None, "error"
                    with jobs_lock:
                        jobs[task_id]["error"] = str(exc)
                with jobs_lock:
                    jobs[task_id].update(status=st, result=res)

            threading.Thread(target=worker, daemon=True).start()
            return jsonify({"job_id": task_id, "poll": f"/ocr_status/{task_id}"}), 202
        res = run_ocr(items, task_id, dpi, max_new)
        return jsonify({"task_id": task_id, "status": "done", "error": None,
                        **public_view(res, detail=_detail())}), 200

    @app.get("/ocr_status/<job_id>")
    def ocr_status(job_id):
        if not _authed():
            return jsonify({"error": "missing or wrong X-Api-Key"}), 401
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "unknown job_id"}), 404
        view = {"status": job["status"], "error": job["error"]}
        if job["result"]:
            view.update(public_view(job["result"], detail=_detail()))
        return jsonify(view), 200

    @app.get("/model_status")
    def model_status():
        return jsonify({"model": ocr_engine.MODEL_ID, "loaded": ocr_engine.is_loaded()}), 200

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("AGENT_PORT", 5008))
    agent = NanowakaOcrAgent()
    try:
        agent.generate_manifest(
            endpoint=os.environ.get("AGENT_ENDPOINT", "https://nanowaka-ocr.example.com"),
            output_path=HERE / "Agent.md",
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
        )
    except Exception as exc:                       # e.g. AGENT_ENDPOINT not https
        raise SystemExit(f"Could not generate Agent.md: {exc}")
    if os.environ.get("OCR_PRELOAD", "1") != "0":   # load now so the first request isn't a 30s+ wait
        ocr_engine._load()
    app = build_app(agent)
    serve(app, host=os.environ.get("AGENT_HOST", "0.0.0.0"), port=port, threads=4)
