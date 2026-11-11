"""
BYOA Knowledge Ingestion Service -- standalone GX10 service.

Ingests company documents into a local ChromaDB collection for later RAG
retrieval (retrieval agent is a separate, LATER piece -- this service only
covers ingestion/management of the vector store itself).

Design notes:
  - This is a plain Flask service (like registration_server.py), NOT a
    byoa_adk Agent wrapped with build_flask_app -- the ADK agents in this
    platform (fx/summarize/webcrawl/websearch) only demonstrate pure-JSON
    TaskPayload.inputs, with no established pattern for binary file upload
    through that path. registration_server.py's /register endpoint is the
    one place in this codebase that already does multipart file upload via
    request.files, so this service follows that precedent instead of
    guessing at ADK's multipart handling for large binary content.
  - The retrieval half (query -> relevant chunks, with fallback-to-websearch
    when confidence is low) is intentionally NOT built here. That will be a
    normal capability-tagged ADK agent ("knowledge-retrieve") registered
    through registration_server.py, since retrieval is pure JSON in/out and
    fits the existing Agent pattern cleanly. This service only owns the
    vector store's contents.
  - Incremental ingestion: every document gets a stable doc_id (sha256 of
    filename, or an explicit one the caller supplies). Re-ingesting a
    doc_id first deletes *only that doc_id's* existing chunks from Chroma,
    then re-adds the freshly chunked+embedded content -- other documents'
    chunks are never touched.
  - Embeddings come from vLLM (OpenAI-compatible /v1/embeddings), same
    backend/model as retrieve_agent.py's document-side embedding -- MUST
    stay in sync with that agent, or vectors won't be comparable in the
    same collection. Ollama is being phased out platform-wide; no Ollama
    calls remain in this service.
  - No pyngrok. Public exposure (if this is ever exposed beyond localhost --
    by default treat this as an admin-only, LAN/localhost-bound service
    since it can delete/overwrite the knowledge base) would be handled the
    same way as everywhere else: systemd-managed cloudflared unit.
  - Config comes from a .env file, same convention as every other service.

Supported formats: .txt, .md, .html/.htm, .docx, .pdf, .pptx, .xlsx/.xls, .csv
Explicitly rejected: video and image files (no OCR/transcription attempted here).
"""
import csv
import hashlib
import io
import os
import re
import threading
import time
import unicodedata
import uuid
from datetime import datetime, timezone

import requests as _req
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from waitress import serve

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "5405"))
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://knowledge.zackloh.dev")

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "/opt/byoa-knowledge/chroma_data")
# The ChromaDB store is now owned by exactly one process: the standalone
# `chroma run` server (byoa-chromadb6.service). This service must NOT open
# its own direct PersistentClient onto CHROMA_DB_PATH's sqlite file --
# retrieve_agent.py needs access to the same data, and two processes each
# independently holding a PersistentClient on the same on-disk store was
# the dual-writer bug that made ingested documents invisible to retrieval.
CHROMADB_SERVER_HOST = os.environ.get("CHROMADB_SERVER_HOST", "localhost")
CHROMADB_SERVER_PORT = int(os.environ.get("CHROMADB_SERVER_PORT", "4420"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "company_docs")

# Feedback memory lives in its OWN collection on the same standalone chroma
# server -- deliberately never mixed into COLLECTION_NAME. KB docs and
# feedback rows have different embedding subjects (chunk content vs. a
# user's goal text), different lifecycles (KB docs get purged per-session/
# TTL; feedback should accumulate forever), and different write patterns
# (chunked bulk writes vs. single small inserts) -- see chat history for
# the full rationale. Same server, same process, separate namespace.
FEEDBACK_COLLECTION_NAME = os.environ.get("CHROMA_FEEDBACK_COLLECTION", "byoa_feedback_memory")

# Embeddings: vLLM only -- Ollama is being phased out platform-wide. This
# MUST stay the same backend + model as retrieve_agent.py's document-side
# embedding, or vectors from the two won't be comparable in the same
# collection. Documents get NO instruction prefix (only queries do -- see
# retrieve_agent.py's embed_query() for why).
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11501")
VLLM_EMBED_MODEL = os.environ.get("VLLM_EMBED_MODEL", "qwen3-embedding-8b")

# Chunking tunables
TARGET_CHUNK_CHARS = int(os.environ.get("TARGET_CHUNK_CHARS", "1200"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1800"))
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "120"))
# Semantic chunker: how many sentences per embedding "window" before comparing
# adjacent windows for a breakpoint, and how aggressive the breakpoint
# threshold is (higher percentile = fewer, larger chunks).
SEMANTIC_WINDOW_SENTENCES = int(os.environ.get("SEMANTIC_WINDOW_SENTENCES", "3"))
SEMANTIC_BREAKPOINT_PERCENTILE = float(os.environ.get("SEMANTIC_BREAKPOINT_PERCENTILE", "85"))

REJECTED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv",           # video
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg",  # image
}

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".html", ".htm",
    ".docx", ".pdf", ".pptx", ".xlsx", ".xls", ".csv",
}

# ── Concurrency safety (added after the 2026-08-05 stress-test incident) ────
# A 100-file concurrent upload burst (many /goal jobs uploading at once, each
# hitting this service) caused chromadb.errors.InternalError under write
# contention, and once that started, waitress's 4 request threads backed up
# and never drained -- even /health stopped responding for hours. Root cause
# was two-fold:
#   1. ChromaDB's SQLite-backed store doesn't handle multiple concurrent
#      writers to the same collection well -- collection.add()/delete()
#      calls racing across waitress's 4 threads is what actually threw
#      InternalError.
#   2. There was no retry on that error and no bound on how many /ingest
#      requests could be doing real work (chunking + embedding + writing) at
#      once -- so a burst just piled every waitress thread onto the same
#      contended resource with no fallback, leaving nothing free to answer
#      even a trivial /health check.
#
# Fix, in two parts:
#   - _CHROMA_WRITE_LOCK serializes the actual collection.add()/delete()
#     calls within this process, so writes queue safely instead of racing.
#     _chroma_write_with_retry() additionally retries with backoff, since
#     the standalone `chroma run` server can still throw a transient error
#     under its OWN internal contention (WAL checkpoints etc.) even with
#     this process's writes serialized.
#   - _INGEST_SEMAPHORE caps how many requests can be doing the CPU/GPU-bound
#     work (chunking, embedding) at once, independent of collection writes.
#     Acquiring it has a generous timeout (INGEST_ACQUIRE_TIMEOUT_SECS,
#     10 minutes by default): a request that can't get a slot waits in
#     line rather than being rejected outright, so a burst of legitimate
#     concurrent users all eventually get served instead of some of them
#     getting a 503 just for bad timing. This is a deliberate choice to
#     favor "everyone queues and gets in" over "reject anyone who didn't
#     arrive first" -- but the wait is NOT unbounded: if the backend is
#     genuinely broken (not just busy -- e.g. ChromaDB itself crashed,
#     like the 2026-08-05 incident this whole mechanism exists because
#     of), an unbounded wait would just reproduce that exact incident's
#     symptom -- a request hanging forever with zero feedback, now "by
#     design" instead of by bug. After INGEST_ACQUIRE_TIMEOUT_SECS, it
#     gives up and returns a real 503 so the failure is visible rather
#     than silent.
MAX_CONCURRENT_INGESTS = int(os.environ.get("MAX_CONCURRENT_INGESTS", "6"))
INGEST_ACQUIRE_TIMEOUT_SECS = int(os.environ.get("INGEST_ACQUIRE_TIMEOUT_SECS", str(10 * 600)))
CHROMA_WRITE_MAX_RETRIES = int(os.environ.get("CHROMA_WRITE_MAX_RETRIES", "4"))
CHROMA_WRITE_RETRY_BASE_SECS = float(os.environ.get("CHROMA_WRITE_RETRY_BASE_SECS", "0.4"))

_CHROMA_WRITE_LOCK = threading.Lock()
_INGEST_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INGESTS)


def _chroma_write_with_retry(fn, *args, **kwargs):
    """
    Runs one ChromaDB write (collection.add or collection.delete) under the
    process-wide write lock, retrying with exponential backoff on failure.
    The lock alone stops THIS process's threads from racing each other, but
    the standalone Chroma server can still throw a transient error under its
    own internal contention -- a short retry loop absorbs that instead of
    surfacing a 500 for something that would have succeeded moments later.
    Raises the last exception if every attempt fails.
    """
    last_exc = None
    for attempt in range(CHROMA_WRITE_MAX_RETRIES):
        try:
            with _CHROMA_WRITE_LOCK:
                return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < CHROMA_WRITE_MAX_RETRIES - 1:
                time.sleep(CHROMA_WRITE_RETRY_BASE_SECS * (2 ** attempt))
    raise last_exc


# ── vLLM embedding readiness ──────────────────────────────────────────────────
def verify_vllm_embed_ready():
    try:
        r = _req.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"vLLM not reachable at {VLLM_BASE_URL} -- is the embedding server running? ({e})"
        )
    model_ids = [m.get("id") for m in r.json().get("data", [])]
    if VLLM_EMBED_MODEL not in model_ids:
        raise RuntimeError(
            f"Embedding model {VLLM_EMBED_MODEL!r} not found on vLLM server at {VLLM_BASE_URL}.\n"
            f"Currently served: {model_ids}"
        )
    print(f"OK vLLM reachable, embedding model {VLLM_EMBED_MODEL!r} present.")


VLLM_EMBED_TIMEOUT_SECS = int(os.environ.get("VLLM_EMBED_TIMEOUT_SECS", "120"))
# A long document can produce thousands of chunks. Sending them all to
# vLLM in ONE /v1/embeddings call means the whole request lives or dies on
# a single timeout, no matter how large VLLM_EMBED_TIMEOUT_SECS is set --
# and a big enough document will eventually outrun any fixed timeout.
# Sub-batching keeps each individual HTTP call bounded and fast regardless
# of overall document size, and a slow/failed batch only has to retry that
# batch, not re-embed the entire document from scratch.
VLLM_EMBED_BATCH_SIZE = int(os.environ.get("VLLM_EMBED_BATCH_SIZE", "64"))


# qwen3-embedding is asymmetric (see retrieve_agent.py's embed_query for
# the same pattern against the KB collection): the QUERY side needs an
# instruction prefix, the stored/document side does not. Goal text is
# embedded as a plain document (embed_texts, no prefix) when WRITTEN into
# feedback memory, and with this instruction prefix when used to SEARCH
# feedback memory for similar past goals -- same asymmetry, different task
# description since this is matching goal-to-goal, not query-to-passage.
FEEDBACK_QUERY_INSTRUCT_TASK = os.environ.get(
    "FEEDBACK_QUERY_INSTRUCT_TASK",
    "Given a user's goal for an AI agent platform, retrieve similar past "
    "goals along with their outcome and user rating",
)


def embed_feedback_query(text: str) -> list:
    instructed_query = f"Instruct: {FEEDBACK_QUERY_INSTRUCT_TASK}\nQuery:{text}"
    r = _req.post(
        f"{VLLM_BASE_URL}/v1/embeddings",
        json={"model": VLLM_EMBED_MODEL, "input": instructed_query},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def embed_texts(texts: list) -> list:
    """Embed a list of strings via vLLM's OpenAI-compatible /v1/embeddings,
    sub-batched at VLLM_EMBED_BATCH_SIZE per call so one huge document's
    worth of chunks can't turn into a single all-or-nothing request that
    outruns VLLM_EMBED_TIMEOUT_SECS. Order is preserved across batches; the
    OpenAI-compatible endpoint returns `index`-tagged items PER-BATCH (0-
    based within that batch, not global), so each batch is sorted on its
    own indices before being appended, not sorted globally afterward."""
    all_embeddings = []
    for start in range(0, len(texts), VLLM_EMBED_BATCH_SIZE):
        batch = texts[start:start + VLLM_EMBED_BATCH_SIZE]
        r = _req.post(
            f"{VLLM_BASE_URL}/v1/embeddings",
            json={"model": VLLM_EMBED_MODEL, "input": batch},
            timeout=VLLM_EMBED_TIMEOUT_SECS,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        all_embeddings.extend(d["embedding"] for d in data)
    return all_embeddings


def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Text normalization ────────────────────────────────────────────────────
# OCR/document-conversion tools upstream of this service (not ours -- we
# only receive their output) occasionally emit non-standard characters in
# place of real whitespace: e.g. one OCR v2 pipeline was observed emitting
# U+2337 (APL FUNCTIONAL SYMBOL SQUISH QUAD) for every detected space
# instead of U+0020. Unicode's own "is this whitespace" category (Zs)
# doesn't cover such symbol-category artifacts, so str.isspace()/regex \s
# silently treat them as ordinary word characters. That collapses an
# entire OCR'd table into one giant "word" with no space for our sentence
# splitter to find a boundary on, which starves semantic_chunk() of any
# split points -- see _split_sentences() and the single-sentence branch in
# semantic_chunk() below.
#
# This is a defensive normalization layer, not a fix for that specific
# OCR tool (we don't own it and it may change what it emits). It runs on
# every block, from every extractor, right before chunking:
#   1. Known artifact characters seen in the wild -> regular space.
#   2. Any Unicode space-separator (category Zs, e.g. NBSP, figure space)
#      -> regular space, via NFKC-style category normalization.
#   3. Zero-width/format characters (category Cf, e.g. zero-width space,
#      word joiner) -> stripped entirely (they carry no visible content).
# Collapses runs of resulting whitespace down to single spaces afterward.
_OCR_ARTIFACT_CHARS = "\u2337"  # extend this set if new tools surface new artifacts
_OCR_ARTIFACT_TABLE = {ord(c): " " for c in _OCR_ARTIFACT_CHARS}


def normalize_ocr_whitespace(text: str) -> str:
    if not text:
        return text
    text = text.translate(_OCR_ARTIFACT_TABLE)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Zs":
            out.append(" ")
        elif cat == "Cf":
            continue  # zero-width/format chars carry no visible content
        else:
            out.append(ch)
    text = "".join(out)
    return re.sub(r"[ \t]+", " ", text)


# ── Block model ────────────────────────────────────────────────────────────
# Every extractor below normalizes its source format down to a flat list of
# blocks with the same shape, so chunking logic only needs to be written once:
#   {"type": "header" | "paragraph" | "list_item" | "table_row",
#    "level": int | None,   # header level (1-6), else None
#    "text": str}

def _md_like_blocks(text: str) -> list:
    """Shared block parser for plain text / markdown (and used as a
    fallback for anything we can only get raw text out of)."""
    blocks = []
    lines = text.splitlines()
    para_buf = []

    def flush_para():
        if para_buf:
            joined = " ".join(l.strip() for l in para_buf if l.strip())
            if joined:
                blocks.append({"type": "paragraph", "level": None, "text": joined})
            para_buf.clear()

    header_re = re.compile(r"^(#{1,6})\s+(.*)")
    bullet_re = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)")
    table_row_re = re.compile(r"^\s*\|.*\|\s*$")

    for line in lines:
        if not line.strip():
            flush_para()
            continue
        m = header_re.match(line)
        if m:
            flush_para()
            blocks.append({"type": "header", "level": len(m.group(1)), "text": m.group(2).strip()})
            continue
        if table_row_re.match(line):
            flush_para()
            # strip leading/trailing pipe, split on |, drop markdown separator rows like |---|---|
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{1,}:?", c) for c in cells if c):
                continue
            blocks.append({"type": "table_row", "level": None, "text": " | ".join(cells)})
            continue
        m = bullet_re.match(line)
        if m:
            flush_para()
            blocks.append({"type": "list_item", "level": None, "text": m.group(2).strip()})
            continue
        para_buf.append(line)
    flush_para()
    return blocks


def extract_txt_md(raw: bytes) -> list:
    return _md_like_blocks(raw.decode("utf-8", errors="replace"))


def extract_html(raw: bytes) -> list:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe"]):
        tag.decompose()

    blocks = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        text = " ".join(el.get_text(separator=" ", strip=True).split())
        if not text:
            continue
        if el.name.startswith("h"):
            blocks.append({"type": "header", "level": int(el.name[1]), "text": text})
        elif el.name == "li":
            blocks.append({"type": "list_item", "level": None, "text": text})
        elif el.name == "tr":
            cells = [" ".join(c.get_text(separator=" ", strip=True).split())
                     for c in el.find_all(["td", "th"])]
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                blocks.append({"type": "table_row", "level": None, "text": row_text})
        else:  # p
            blocks.append({"type": "paragraph", "level": None, "text": text})
    return blocks


def extract_docx(raw: bytes) -> list:
    import docx as _docx
    doc = _docx.Document(io.BytesIO(raw))
    blocks = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "") if para.style else ""
        if style.startswith("Heading"):
            m = re.search(r"\d+", style)
            level = int(m.group()) if m else 1
            blocks.append({"type": "header", "level": min(level, 6), "text": text})
        elif "List" in style:
            blocks.append({"type": "list_item", "level": None, "text": text})
        else:
            blocks.append({"type": "paragraph", "level": None, "text": text})
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                blocks.append({"type": "table_row", "level": None, "text": row_text})
    return blocks


def extract_pdf(raw: bytes) -> list:
    import pdfplumber
    blocks = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            # Tables first, since extract_text() will otherwise mix table
            # cell text into the prose stream.
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table in tables:
                for row in table:
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    row_text = " | ".join(c for c in cells if c)
                    if row_text:
                        blocks.append({"type": "table_row", "level": None, "text": row_text})

            text = page.extract_text() or ""
            # PDFs don't expose reliable heading semantics through plain text
            # extraction. Heuristic: a short, standalone line (no trailing
            # punctuation) is treated as a header. This is imperfect by
            # nature of the format -- prose that doesn't match just falls
            # through as paragraph blocks, which still get semantic-chunked
            # correctly.
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                looks_like_header = (
                    len(line) <= 80
                    and not line.endswith((".", ",", ";", ":"))
                    and (line.isupper() or line.istitle())
                )
                if looks_like_header:
                    blocks.append({"type": "header", "level": 1, "text": line})
                else:
                    blocks.append({"type": "paragraph", "level": None, "text": line})

    # Chronology/timeline sections often render as one flowing block of text
    # with no line break before each dated entry (pdfplumber's extract_text()
    # merges wrapped visual lines), so line-position heuristics can't catch
    # them -- the year boundary has to be found *within* the text itself.
    # Scoped to paragraphs following a header that names a chronology/
    # timeline, so this doesn't misfire on ordinary prose that happens to
    # mention a year in passing (e.g. "In 1841-43 Galt's most important
    # publisher...").
    blocks = _split_chronology_paragraphs(blocks)
    return blocks


_CHRONOLOGY_HEADER_RE = re.compile(r"chronology|timeline", re.IGNORECASE)
# Splits immediately before a 4-digit year (optionally a range like
# "1795-1804"), as long as: it's not itself preceded by a digit (so it
# doesn't fire mid-number), and it's not the second half of a year range
# it was just preceded by (so "1795-1804" stays one entry, not two).
_YEAR_ENTRY_SPLIT_RE = re.compile(
    r"(?<!\d)(?<!\d{4}-)(?<!\d{4}\u2013)(?=\d{4}(?:[\u2013\-]\d{2,4})?\s)"
)


def _split_chronology_paragraphs(blocks: list) -> list:
    """
    pdfplumber's extract_text() yields many short physical lines, each
    becoming its own paragraph block -- so a chronology entry that wraps
    across lines has its year boundary split across two *different*
    blocks, not sitting inside one. Splitting each block in isolation (the
    first attempt at this) missed nearly everything, since most individual
    lines don't contain a year boundary themselves. Fix: buffer all
    consecutive paragraph blocks under a chronology header into one
    continuous text first, THEN split that merged text on year boundaries.
    """
    result = []
    in_chronology_section = False
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        merged = " ".join(buffer)
        pieces = [p.strip() for p in _YEAR_ENTRY_SPLIT_RE.split(merged) if p.strip()]
        if len(pieces) > 1:
            for p in pieces:
                result.append({"type": "list_item", "level": None, "text": p})
        else:
            result.append({"type": "paragraph", "level": None, "text": merged})
        buffer.clear()

    for b in blocks:
        if b["type"] == "header":
            flush_buffer()
            in_chronology_section = bool(_CHRONOLOGY_HEADER_RE.search(b["text"]))
            result.append(b)
        elif b["type"] == "paragraph" and in_chronology_section:
            buffer.append(b["text"])
        else:
            flush_buffer()
            result.append(b)
    flush_buffer()
    return result
    return blocks


def extract_pptx(raw: bytes) -> list:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(raw))
    blocks = []
    for slide in prs.slides:
        title_text = None
        if slide.shapes.title and slide.shapes.title.text.strip():
            title_text = slide.shapes.title.text.strip()
            blocks.append({"type": "header", "level": 1, "text": title_text})
        for shape in slide.shapes:
            if shape == slide.shapes.title or not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if not text:
                    continue
                # Slide body text is bullet-oriented by convention.
                blocks.append({"type": "list_item", "level": None, "text": text})
    return blocks


def extract_tabular(raw: bytes, ext: str) -> list:
    import pandas as pd
    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(raw))
    else:
        df = pd.read_excel(io.BytesIO(raw))
    cols = [str(c) for c in df.columns]
    blocks = []
    for _, row in df.iterrows():
        cells = [f"{col}: {row[col]}" for col in cols if str(row[col]).strip() not in ("", "nan")]
        row_text = " | ".join(cells)
        if row_text:
            blocks.append({"type": "table_row", "level": None, "text": row_text})
    return blocks


EXTRACTORS = {
    ".txt": extract_txt_md,
    ".md": extract_txt_md,
    ".markdown": extract_txt_md,
    ".html": extract_html,
    ".htm": extract_html,
    ".docx": extract_docx,
    ".pdf": extract_pdf,
    ".pptx": extract_pptx,
    ".xlsx": lambda raw: extract_tabular(raw, ".xlsx"),
    ".xls": lambda raw: extract_tabular(raw, ".xls"),
    ".csv": lambda raw: extract_tabular(raw, ".csv"),
}


# ── Sectioning ────────────────────────────────────────────────────────────────
def split_into_sections(blocks: list) -> list:
    """
    Groups a flat block list into sections at header boundaries. Returns a
    list of {"path": "Header > Subheader", "blocks": [...]}. If there are
    no headers at all, returns a single section with path "" containing
    every block.
    """
    if not any(b["type"] == "header" for b in blocks):
        return [{"path": "", "blocks": blocks}]

    sections = []
    current_path_stack = []  # list of (level, text)
    current_blocks = []

    def flush():
        if current_blocks:
            path = " > ".join(t for _, t in current_path_stack)
            sections.append({"path": path, "blocks": list(current_blocks)})
            current_blocks.clear()

    for b in blocks:
        if b["type"] == "header":
            flush()
            level = b["level"] or 1
            # Pop any breadcrumb entries at this level or deeper, then push.
            current_path_stack[:] = [(lv, t) for lv, t in current_path_stack if lv < level]
            current_path_stack.append((level, b["text"]))
            current_blocks.append(b)
        else:
            current_blocks.append(b)
    flush()
    return sections


# ── Chunking strategies ───────────────────────────────────────────────────────
def _batch_atomic_units(items: list, max_chars: int) -> list:
    """
    Each atomic unit (a table row, a bullet/list item, a chronology entry)
    is kept as its own chunk by default. A complete row or list entry is
    already a coherent, independently meaningful unit -- gluing several
    together just to fill up toward max_chars dilutes each individual
    item's contribution to the resulting embedding, the same problem we
    fixed for prose (never cut mid-sentence; don't blend unrelated facts
    into one semantic-chunked blob). Merging here only happens to rescue
    a genuinely undersized fragment (below MIN_CHUNK_CHARS) by attaching
    it to its neighbor -- not to hit a size target. If merging would push
    past max_chars, it's skipped even for an undersized item, since the
    hard cap still matters.
    """
    if not items:
        return []
    merged = [items[0]]
    for text in items[1:]:
        if (len(merged[-1]) < MIN_CHUNK_CHARS
                and len(merged[-1]) + len(text) + 1 <= max_chars):
            merged[-1] = merged[-1] + "\n" + text
        else:
            merged.append(text)
    return merged


def _split_sentences(text: str) -> list:
    # Lightweight sentence splitter -- good enough for chunk boundaries,
    # not meant to be linguistically perfect.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunk(text: str) -> list:
    """
    Splits continuous prose using embedding-similarity breakpoints:
    group sentences into small windows, embed each window, and cut wherever
    similarity to the next window drops below the threshold percentile
    (i.e. topic shift), then enforce min/max chunk size by merging
    undersized pieces and hard-splitting oversized ones.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= SEMANTIC_WINDOW_SENTENCES:
        # Even with too few sentences to run breakpoint detection, this
        # text still needs to respect MAX_CHUNK_CHARS -- a text with no
        # detectable sentence boundaries (e.g. a wall of text with no
        # whitespace at all) must not bypass the size cap and go straight
        # to embedding unsplit, since that can exceed the embedding
        # model's max sequence length and fail the whole ingest.
        return _enforce_size_bounds([text]) if text.strip() else []

    windows = []
    for i in range(0, len(sentences), SEMANTIC_WINDOW_SENTENCES):
        windows.append(" ".join(sentences[i:i + SEMANTIC_WINDOW_SENTENCES]))

    if len(windows) == 1:
        return [windows[0]]

    try:
        vectors = embed_texts(windows)
    except Exception:
        # Embedding service hiccup shouldn't lose the document -- fall back
        # to fixed-size chunking rather than failing the whole ingest.
        return _fixed_size_chunk(text)

    sims = [_cosine_sim(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    if not sims:
        return [text]

    sorted_sims = sorted(sims)
    idx = int(len(sorted_sims) * (1 - SEMANTIC_BREAKPOINT_PERCENTILE / 100.0))
    idx = max(0, min(idx, len(sorted_sims) - 1))
    threshold = sorted_sims[idx]

    chunks = []
    current = [windows[0]]
    for i, sim in enumerate(sims):
        if sim <= threshold:
            chunks.append(" ".join(current))
            current = [windows[i + 1]]
        else:
            current.append(windows[i + 1])
    chunks.append(" ".join(current))

    return _enforce_size_bounds(chunks)


def _fixed_size_chunk(text: str, max_chars: int = None) -> list:
    """
    Packs whole sentences into chunks up to max_chars, never splitting a
    sentence. This is the fallback path (embedding service unreachable) and
    also the oversized-chunk splitter in _enforce_size_bounds -- in both
    cases, staying within the size cap is secondary to never breaking a
    sentence in half.

    If a single sentence is itself longer than max_chars, it becomes its
    own oversized chunk rather than being cut -- preserving the sentence's
    logical integrity takes priority over the size cap in that edge case.
    """
    limit = max_chars if max_chars is not None else TARGET_CHUNK_CHARS
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks, buf, buf_len = [], [], 0
    for s in sentences:
        s_len = len(s) + 1
        if buf and buf_len + s_len > limit:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += s_len
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _enforce_size_bounds(chunks: list) -> list:
    # Merge undersized neighbors so tiny fragments don't stand alone.
    # For oversized chunks, repack by whole sentences up to MAX_CHUNK_CHARS
    # -- never a mid-sentence cut. If packing still leaves a chunk over the
    # cap (one long sentence on its own), that's left as-is on purpose:
    # a slightly oversized but logically intact chunk beats a broken one.
    merged = []
    for c in chunks:
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    final = []
    for c in merged:
        if len(c) > MAX_CHUNK_CHARS:
            final.extend(_fixed_size_chunk(c, max_chars=MAX_CHUNK_CHARS))
        else:
            final.append(c)
    return final


def chunk_section(section: dict) -> list:
    """
    Returns a list of {"text": str, "chunk_type": str} for one section,
    applying the decision tree:
      - mostly table rows  -> row batching
      - mostly list items  -> point-form batching
      - else (prose)       -> single chunk if short, else semantic chunking
    """
    blocks = [b for b in section["blocks"] if b["type"] != "header"]
    if not blocks:
        return []

    total_chars = sum(len(b["text"]) for b in blocks)
    table_chars = sum(len(b["text"]) for b in blocks if b["type"] == "table_row")
    list_chars = sum(len(b["text"]) for b in blocks if b["type"] == "list_item")

    def _chunk_stray_prose(paragraph_blocks: list) -> list:
        # Shared prose handling used both as the main branch (prose-dominant
        # sections) and as leftover-rescue for table-/list-dominant sections
        # (e.g. an invoice's Subtotal/GST/Total-Due/Payment-Terms footer
        # sitting in the SAME section as the line-items table, with no
        # header in between) -- without this, that trailing prose was
        # silently dropped whenever a table or list happened to make up
        # over half the section's characters, since the table/list branches
        # only ever read their own dominant block type.
        prose = " ".join(b["text"] for b in paragraph_blocks)
        if not prose.strip():
            return []
        if len(prose) <= int(TARGET_CHUNK_CHARS * 1.3):
            return [{"text": prose.strip(), "chunk_type": "section_prose"}]
        return [{"text": piece, "chunk_type": "semantic"} for piece in semantic_chunk(prose)]

    results = []
    if total_chars and table_chars / total_chars > 0.5:
        rows = [b["text"] for b in blocks if b["type"] == "table_row"]
        for batch in _batch_atomic_units(rows, MAX_CHUNK_CHARS):
            results.append({"text": batch, "chunk_type": "row_batch"})
        # Rescue any stray list items and any stray prose (e.g. a totals/
        # footer paragraph trailing the table) so they aren't dropped.
        stray_items = [b["text"] for b in blocks if b["type"] == "list_item"]
        for batch in _batch_atomic_units(stray_items, MAX_CHUNK_CHARS):
            results.append({"text": batch, "chunk_type": "point_batch"})
        results.extend(_chunk_stray_prose([b for b in blocks if b["type"] == "paragraph"]))
    elif total_chars and list_chars / total_chars > 0.5:
        items = [b["text"] for b in blocks if b["type"] == "list_item"]
        for batch in _batch_atomic_units(items, MAX_CHUNK_CHARS):
            results.append({"text": batch, "chunk_type": "point_batch"})
        # Rescue any stray table rows and any stray prose, same rationale
        # as the table branch above.
        stray_rows = [b["text"] for b in blocks if b["type"] == "table_row"]
        for batch in _batch_atomic_units(stray_rows, MAX_CHUNK_CHARS):
            results.append({"text": batch, "chunk_type": "row_batch"})
        results.extend(_chunk_stray_prose([b for b in blocks if b["type"] == "paragraph"]))
    else:
        results.extend(_chunk_stray_prose([b for b in blocks if b["type"] in ("paragraph", "list_item")]))
        # Any stray table rows in a prose-dominant section still need a home.
        stray_rows = [b["text"] for b in blocks if b["type"] == "table_row"]
        for batch in _batch_atomic_units(stray_rows, MAX_CHUNK_CHARS):
            results.append({"text": batch, "chunk_type": "row_batch"})

    return results


def chunk_document(blocks: list) -> list:
    """Top-level entry point: sectioning + per-section chunking.
    Returns list of {"text", "chunk_type", "section_path"}."""
    # Normalize whitespace artifacts here, once, regardless of which
    # extractor produced these blocks -- see normalize_ocr_whitespace().
    for b in blocks:
        b["text"] = normalize_ocr_whitespace(b["text"])
    sections = split_into_sections(blocks)
    all_chunks = []
    for section in sections:
        for c in chunk_section(section):
            c["section_path"] = section["path"]
            all_chunks.append(c)
    return all_chunks


# ── ChromaDB ─────────────────────────────────────────────────────────────────
# Fix (2026-08-11): get_collection() used to create a brand-new
# chromadb.HttpClient -- with its own underlying HTTP connection pool --
# on every single call (once per /ingest, once per DELETE, etc.). Nothing
# ever closed those, so after a few hundred requests the process leaked
# enough sockets/file descriptors (on both this process and the standalone
# chroma-server process it talks to) to start throwing opaque 500s.
# Restarting the service freed the leaked FDs, which is why that "fixed"
# it -- it wasn't actually fixed, just reset. Now: build the HttpClient
# once at import time and reuse it for the life of the process.
import chromadb  # noqa: E402

_chroma_client = chromadb.HttpClient(host=CHROMADB_SERVER_HOST, port=CHROMADB_SERVER_PORT)
_collection_cache = None
_collection_cache_lock = threading.Lock()


def get_collection():
    # Fix (2026-08-12): get_or_create_collection() is a network round-trip
    # to the standalone chroma server, and calling it concurrently from
    # multiple waitress worker threads on the one shared _chroma_client
    # turned out to be able to hang/deadlock a thread rather than fail
    # fast -- one stuck thread never returns, permanently shrinking the
    # worker pool until every thread is dead and the service stops
    # responding to ANY request, even ones unrelated to ingestion.
    # Fix: do this network call at most once per process lifetime under a
    # lock, then cache and reuse the resulting collection handle. The
    # collection's identity/metadata don't change between requests, so
    # there's no reason to re-fetch it every time anyway.
    global _collection_cache
    if _collection_cache is None:
        with _collection_cache_lock:
            if _collection_cache is None:  # re-check: another thread may have won the race
                # Explicitly cosine, not Chroma's default (squared L2) -- cosine
                # similarity is the standard, interpretable metric for text
                # embeddings, and the retrieval agent's confidence threshold
                # depends on a clean 1-distance similarity score. NOTE: metadata
                # is only applied at *creation* time -- changing this after the
                # collection already exists has no effect on existing data; see
                # deployment notes for the one-time migration.
                _collection_cache = _chroma_client.get_or_create_collection(
                    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
    return _collection_cache


_feedback_collection_cache = None
_feedback_collection_cache_lock = threading.Lock()


def get_feedback_collection():
    # Same lazy-cached-handle pattern as get_collection() above, and for
    # the same reason: get_or_create_collection() is a network round-trip
    # to the standalone chroma server, and doing it more than once per
    # process (or concurrently from multiple threads) risks the same
    # hang/deadlock get_collection() was fixed for on 2026-08-12.
    global _feedback_collection_cache
    if _feedback_collection_cache is None:
        with _feedback_collection_cache_lock:
            if _feedback_collection_cache is None:
                _feedback_collection_cache = _chroma_client.get_or_create_collection(
                    FEEDBACK_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
    return _feedback_collection_cache


def store_feedback(goal_text: str, rating: int, comment: str, dag_summary: str,
                    capabilities_used: list, response_excerpt: str, plan_id: str,
                    thread_id: str, controller_version: str) -> dict:
    """
    Embeds goal_text as a plain document (no instruction prefix -- see
    embed_feedback_query above) and writes one row into feedback memory.
    Reuses _chroma_write_with_retry, the same write lock/backoff already
    proven under the KB collection's concurrent-write stress test -- both
    collections live on the same standalone chroma server, so serializing
    writes process-wide (not per-collection) is the correct scope; the
    server-side contention this guards against doesn't care which
    collection a write targets.
    """
    collection = get_feedback_collection()
    embedding = embed_texts([goal_text])[0]
    fb_id = f"fb_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "goal_text": goal_text,
        "rating": int(rating),
        "comment": comment or "",
        "dag_summary": dag_summary or "",
        # Chroma metadata values must be scalar -- lists aren't supported,
        # so capabilities are flattened to a comma-joined string and split
        # back out again on the read side (see query_feedback below).
        "capabilities_used": ",".join(capabilities_used or []),
        "response_excerpt": (response_excerpt or "")[:500],
        "plan_id": plan_id or "",
        "thread_id": thread_id or "",
        "controller_version": controller_version or "",
        "submitted_at": now,
    }
    _chroma_write_with_retry(
        collection.add, ids=[fb_id], documents=[goal_text],
        embeddings=[embedding], metadatas=[metadata],
    )
    return {"feedback_id": fb_id, "submitted_at": now}


def query_feedback(goal_text: str, top_k: int = 10) -> list:
    """
    Returns up to top_k past feedback rows whose goal_text is nearest to
    goal_text, each with a cosine similarity score (1 - distance, same
    convention as retrieve_agent.py's KB query). Returns raw matches --
    bucketing by rating, recency weighting, and capping how many actually
    get injected into a prompt are the CALLER's job (platform_controller),
    not this service's; this just answers "what's similar", nothing more.
    """
    collection = get_feedback_collection()
    query_vec = embed_feedback_query(goal_text)
    raw = collection.query(
        query_embeddings=[query_vec], n_results=top_k,
        include=["metadatas", "distances"],
    )
    if not raw.get("ids") or not raw["ids"][0]:
        return []

    matches = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]
        matches.append({
            "goal_text": meta.get("goal_text", ""),
            "rating": meta.get("rating"),
            "comment": meta.get("comment", ""),
            "dag_summary": meta.get("dag_summary", ""),
            "capabilities_used": [c for c in meta.get("capabilities_used", "").split(",") if c],
            "submitted_at": meta.get("submitted_at", ""),
            "similarity": round(1 - distance, 4),
        })
    return matches


def ingest_document(doc_id: str, source_filename: str, chunks: list, session_id: str = None) -> dict:
    collection = get_collection()

    # Incremental ingestion: wipe only this doc_id's previous chunks, so
    # re-ingesting (or updating) one document never touches any other
    # document already sitting in the collection.
    #
    # This used to be a bare `except Exception: pass` on the assumption
    # that any failure here just meant "first-time ingest, nothing to
    # delete yet." That's not actually distinguishable from a genuine
    # write failure (e.g. the InternalError seen under concurrent-write
    # contention -- see _chroma_write_with_retry above), so silently
    # swallowing it risked proceeding to add() on top of a doc_id whose
    # old chunks may or may not have actually been removed. Now: retry
    # like any other write, and if it still fails after retries, surface
    # it as a real ingestion failure instead of guessing.
    try:
        _chroma_write_with_retry(collection.delete, where={"doc_id": doc_id})
    except Exception as e:
        raise RuntimeError(f"Failed to clear existing chunks for doc_id={doc_id}: {e}")

    if not chunks:
        return {"doc_id": doc_id, "chunks_written": 0}

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    now = datetime.now(timezone.utc).isoformat()
    metadatas = [
        {
            "doc_id": doc_id,
            "source": source_filename,
            "chunk_index": i,
            "chunk_type": c["chunk_type"],
            "section_path": c["section_path"],
            "ingested_at": now,
        }
        for i, c in enumerate(chunks)
    ]
    # Only session_id itself is conditional -- Chroma metadata values can't
    # be None, and goal-attached uploads are the only caller expected to
    # pass this. is_goal_upload, below, is DELIBERATELY unconditional
    # (always written, True or False, never omitted) -- retrieve_agent.py's
    # unscoped-query isolation filter needs a field that's guaranteed
    # present on every single chunk to filter reliably. A conditional field
    # (present only for uploads, absent otherwise -- the ORIGINAL design
    # here) forces that filter to test for a possibly-missing key, and
    # ChromaDB's behavior for a `where` clause against a key some documents
    # simply don't have is version-dependent (see the "documented quirk"
    # this service and retrieve_agent.py both already work around
    # elsewhere) -- not something safe to build a data-isolation guarantee
    # on. An always-present boolean sidesteps that ambiguity completely:
    # every chunk, upload or not, has this key, so equality-filtering on it
    # behaves the same on every ChromaDB version.
    if session_id:
        for m in metadatas:
            m["session_id"] = session_id
    for m in metadatas:
        m["is_goal_upload"] = bool(session_id)

    _chroma_write_with_retry(collection.add, ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    breakdown = {}
    for c in chunks:
        breakdown[c["chunk_type"]] = breakdown.get(c["chunk_type"], 0) + 1

    return {"doc_id": doc_id, "chunks_written": len(chunks), "chunk_type_breakdown": breakdown}


# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "Missing file field: file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "file is empty"}), 400

    ext = os.path.splitext(f.filename)[1].lower()

    if ext in REJECTED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type: {ext}. Video and image files are not ingested "
                     f"(no OCR/transcription in this service)."
        }), 415

    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": f"Unrecognized extension: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        }), 415

    raw = f.read()
    doc_id = request.form.get("doc_id") or hashlib.sha256(f.filename.encode("utf-8")).hexdigest()[:24]
    session_id = request.form.get("session_id") or None

    # Everything from here on is the CPU/GPU-bound half (chunking +
    # embedding + the write-locked Chroma calls) -- gated behind
    # _INGEST_SEMAPHORE so at most MAX_CONCURRENT_INGESTS requests are
    # doing this work at once, regardless of how many waitress threads
    # exist. This BLOCKS (queues) rather than rejecting outright -- a
    # burst of legitimate concurrent users should all eventually get
    # served, not have some of them fail just for arriving mid-burst.
    # Bounded at INGEST_ACQUIRE_TIMEOUT_SECS (10 min default) so a
    # genuinely broken backend (not just a busy one) still surfaces a
    # real 503 instead of hanging forever with no feedback -- see the
    # comment above MAX_CONCURRENT_INGESTS for why that distinction
    # matters.
    acquired = _INGEST_SEMAPHORE.acquire(timeout=INGEST_ACQUIRE_TIMEOUT_SECS)
    if not acquired:
        return jsonify({
            "error": f"Ingestion service is at capacity (max {MAX_CONCURRENT_INGESTS} concurrent "
                     f"ingests) and this request waited {INGEST_ACQUIRE_TIMEOUT_SECS}s without "
                     f"getting a slot. Please retry shortly.",
        }), 503

    try:
        try:
            blocks = EXTRACTORS[ext](raw)
        except Exception as e:
            return jsonify({"error": f"Failed to parse {f.filename}: {e}"}), 422

        if not blocks:
            return jsonify({"error": f"No extractable text content found in {f.filename}"}), 422

        try:
            chunks = chunk_document(blocks)
            result = ingest_document(doc_id, f.filename, chunks, session_id=session_id)
        except Exception as e:
            return jsonify({"error": f"Ingestion failed: {e}"}), 500
    finally:
        _INGEST_SEMAPHORE.release()

    return jsonify({
        "status": "ingested",
        "filename": f.filename,
        **result,
    }), 200


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """
    Writes one rated goal->outcome row into feedback memory. Called by
    platform_controller.py's POST /feedback right after a user rates a
    completed goal -- the controller resolves plan_id -> goal/answer/
    capabilities_used itself and sends the fully-formed record here; this
    endpoint only owns embedding + storage, same division of labor as
    /ingest owns storage for the KB collection.
    """
    body = request.get_json(silent=True) or {}
    goal_text = (body.get("goal_text") or "").strip()
    rating = body.get("rating")

    if not goal_text:
        return jsonify({"error": "goal_text is required"}), 400
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be an integer 1-5"}), 400
    if rating < 1 or rating > 5:
        return jsonify({"error": "rating must be between 1 and 5"}), 400

    try:
        result = store_feedback(
            goal_text=goal_text,
            rating=rating,
            comment=body.get("comment", ""),
            dag_summary=body.get("dag_summary", ""),
            capabilities_used=body.get("capabilities_used", []),
            response_excerpt=body.get("response_excerpt", ""),
            plan_id=body.get("plan_id", ""),
            thread_id=body.get("thread_id", ""),
            controller_version=body.get("controller_version", ""),
        )
    except Exception as e:
        return jsonify({"error": f"Failed to store feedback: {e}"}), 500

    return jsonify({"status": "stored", **result}), 200


@app.route("/feedback/query", methods=["POST"])
def feedback_query_endpoint():
    """
    Retrieves past feedback rows similar to goal_text. Called by
    platform_controller.py right before it builds the Goal Expander prompt
    for a NEW goal, so it can inject "here's what worked / didn't work on
    a similar past goal" as context.
    """
    body = request.get_json(silent=True) or {}
    goal_text = (body.get("goal_text") or "").strip()
    top_k = body.get("top_k", 10)

    if not goal_text:
        return jsonify({"error": "goal_text is required"}), 400
    try:
        top_k = max(1, min(int(top_k), 50))
    except (TypeError, ValueError):
        top_k = 10

    try:
        matches = query_feedback(goal_text, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Feedback query failed: {e}"}), 500

    return jsonify({"goal_text": goal_text, "matches": matches}), 200


@app.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """
    Diagnostics: rating distribution per capability, so a human can see
    which agent/capability tends to draw low ratings -- separate from the
    automatic in-context loop above, this is for a person to go fix
    something. Reads the whole feedback collection; fine at feedback-memory
    scale (small rows, no chunking), unlike /docs' equivalent over the much
    larger KB collection.
    """
    collection = get_feedback_collection()
    try:
        data = collection.get(include=["metadatas"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    per_capability = {}
    overall = {"count": 0, "rating_sum": 0}
    for meta in data.get("metadatas", []):
        rating = meta.get("rating", 0)
        overall["count"] += 1
        overall["rating_sum"] += rating
        caps = [c for c in meta.get("capabilities_used", "").split(",") if c]
        for cap in caps:
            entry = per_capability.setdefault(cap, {"count": 0, "rating_sum": 0})
            entry["count"] += 1
            entry["rating_sum"] += rating

    def _finalize(entry):
        return {
            "count": entry["count"],
            "avg_rating": round(entry["rating_sum"] / entry["count"], 2) if entry["count"] else None,
        }

    return jsonify({
        "overall": _finalize(overall),
        "by_capability": {cap: _finalize(e) for cap, e in per_capability.items()},
    }), 200


@app.route("/feedback/list", methods=["GET"])
def list_feedback():
    """
    Returns every stored feedback row, raw -- unlike /feedback/query
    (which requires a goal_text to search against) this is a blanket
    listing, for auditing/clearing workflows rather than the Goal
    Expander's similarity lookup.
    """
    collection = get_feedback_collection()
    try:
        data = collection.get(include=["metadatas"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    rows = []
    for fb_id, meta in zip(data.get("ids", []), data.get("metadatas", [])):
        rows.append({
            "feedback_id": fb_id,
            "goal_text": meta.get("goal_text", ""),
            "rating": meta.get("rating"),
            "comment": meta.get("comment", ""),
            "dag_summary": meta.get("dag_summary", ""),
            "capabilities_used": [c for c in meta.get("capabilities_used", "").split(",") if c],
            "plan_id": meta.get("plan_id", ""),
            "thread_id": meta.get("thread_id", ""),
            "submitted_at": meta.get("submitted_at", ""),
        })
    rows.sort(key=lambda r: r["submitted_at"], reverse=True)
    return jsonify({"count": len(rows), "feedback": rows}), 200


@app.route("/feedback/<feedback_id>", methods=["DELETE"])
def delete_one_feedback(feedback_id):
    """Deletes a single feedback row by its feedback_id (from /feedback/list or the original POST /feedback response)."""
    collection = get_feedback_collection()
    try:
        existing = collection.get(ids=[feedback_id])
        if not existing.get("ids"):
            return jsonify({"error": f"No feedback found with id: {feedback_id}"}), 404
        _chroma_write_with_retry(collection.delete, ids=[feedback_id])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "deleted", "feedback_id": feedback_id}), 200


@app.route("/feedback", methods=["DELETE"])
def delete_all_feedback():
    """
    Wipes every row in feedback memory. Separate collection from the KB
    (company_docs) on the same chroma server -- this does NOT touch any
    ingested document, only accumulated ratings/comments. Requires
    ?confirm=true to avoid an accidental blanket wipe from a stray call.
    """
    if request.args.get("confirm") != "true":
        return jsonify({"error": "Pass ?confirm=true to actually wipe all feedback"}), 400

    collection = get_feedback_collection()
    try:
        existing = collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            _chroma_write_with_retry(collection.delete, ids=ids)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "deleted", "count": len(ids)}), 200


@app.route("/docs/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    collection = get_collection()
    try:
        _chroma_write_with_retry(collection.delete, where={"doc_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "deleted", "doc_id": doc_id}), 200


@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """
    Removes every chunk tagged with this session_id -- used to clean up
    goal-attached uploads once the platform controller no longer needs
    them (plan executed, rejected, or expired unapproved). Docs ingested
    any other way (no session_id at ingest time) are untouched, since they
    were never tagged with this key in the first place.
    """
    collection = get_collection()
    try:
        _chroma_write_with_retry(collection.delete, where={"session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "deleted", "session_id": session_id}), 200


@app.route("/docs/<doc_id>/chunks", methods=["GET"])
def get_doc_chunks(doc_id):
    collection = get_collection()
    try:
        data = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not data.get("ids"):
        return jsonify({"error": f"No chunks found for doc_id: {doc_id}"}), 404

    chunks = []
    for i, chunk_id in enumerate(data["ids"]):
        meta = data["metadatas"][i]
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_index": meta.get("chunk_index"),
            "chunk_type": meta.get("chunk_type"),
            "section_path": meta.get("section_path"),
            "text": data["documents"][i],
        })
    chunks.sort(key=lambda c: c["chunk_index"])
    return jsonify({"doc_id": doc_id, "chunk_count": len(chunks), "chunks": chunks}), 200


@app.route("/docs", methods=["GET"])
def list_docs():
    collection = get_collection()
    try:
        data = collection.get(include=["metadatas"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    docs = {}
    for meta in data.get("metadatas", []):
        doc_id = meta["doc_id"]
        entry = docs.setdefault(doc_id, {"doc_id": doc_id, "source": meta["source"],
                                          "chunk_count": 0, "ingested_at": meta["ingested_at"]})
        entry["chunk_count"] += 1
    return jsonify(list(docs.values())), 200


if __name__ == "__main__":
    verify_vllm_embed_ready()
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    print("=" * 60)
    print("BYOA Knowledge Ingestion Service is live")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  Chroma path  : {CHROMA_DB_PATH}  (collections: {COLLECTION_NAME}, {FEEDBACK_COLLECTION_NAME})")
    print(f"  Embed model  : {VLLM_EMBED_MODEL} via {VLLM_BASE_URL}")
    print(f"  Concurrency  : max {MAX_CONCURRENT_INGESTS} concurrent ingests, "
          f"{CHROMA_WRITE_MAX_RETRIES}x write retry")
    print(f"  POST   {AGENT_ENDPOINT}/ingest              (multipart: file=<doc>, doc_id=<optional>, session_id=<optional>)")
    print(f"  GET    {AGENT_ENDPOINT}/docs                (list ingested docs)")
    print(f"  DELETE {AGENT_ENDPOINT}/docs/<doc_id>       (remove one doc's chunks)")
    print(f"  DELETE {AGENT_ENDPOINT}/session/<session_id> (remove all chunks tagged with a session_id)")
    print(f"  POST   {AGENT_ENDPOINT}/feedback             (write one rated goal outcome)")
    print(f"  POST   {AGENT_ENDPOINT}/feedback/query       (find similar past rated goals)")
    print(f"  GET    {AGENT_ENDPOINT}/feedback/stats       (rating diagnostics by capability)")
    print(f"  GET    {AGENT_ENDPOINT}/health")
    print("=" * 60)
    # threads=8, not 4: MAX_CONCURRENT_INGESTS caps actual ingest processing
    # at 3 concurrent by default, so with 8 total waitress threads there
    # are always threads free to answer /health, /docs, and other
    # lightweight routes even when every ingest slot is busy -- this is
    # what actually keeps the service visibly alive under load, rather
    # than relying on the semaphore alone (see MAX_CONCURRENT_INGESTS
    # comment above for the incident this addresses).
    serve(app, host="0.0.0.0", port=PORT, threads=8)
