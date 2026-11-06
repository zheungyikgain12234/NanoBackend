"""
BYOA Platform Controller -- standxlone GX10 service.

Ported from 208-byoa-platform-controller.ipynb. All orchestration logic is
unchanged: Goal Expander prompt, $task_N.field resolution (incl. array
indexing), the two-step approval gate (/goal -> /goal/<id>/approve/reject),
NDJSON streaming execution, and all /dev/* controls.

What's different from the Kaggle version:
  - Does NOT install/start Ollama. GX10 runs `ollama` as its own systemd
    service, shared with the summarize agent. This process only verifies
    it's reachable and the configured model is pulled.
  - No pyngrok. Public exposure is handled by the systemd-managed
    `cloudflared` unit on this box, proxying
    https://controller.zackloh.dev -> http://localhost:PORT.
  - No `while True: sleep(60)` keepalive cell -- systemd keeps this alive.
  - Config comes from a .env file instead of hardcoded notebook cell values.
  - /dev/restartOllama now shells out to `sudo systemctl restart ollama`
    instead of managing its own `ollama serve` subprocess, since Ollama is
    a shared system service here, not something this process spawned.
    Requires a passwordless sudo rule for this one command -- see README.
"""
import hashlib
import hmac as _hmac
import io
import json
import os
import re as _re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, TimeoutError as FuturesTimeout
from datetime import date, datetime, timezone

import psycopg2
import requests as req
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agent_credentials import get_agent_secret

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
PLATFORM_DB_URL = os.environ["PLATFORM_DB_URL"]
PORT = int(os.environ.get("PORT", "10301"))
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://controller.zackloh.dev")

# ── Goal plan audit log ───────────────────────────────────────────────────────
# Plain-file audit trail: every goal's input text + its expanded task plan
# gets written here as one JSON file per goal, independent of the in-memory
# _goal_jobs/_pending_plans/_completed_goals structures (all ephemeral --
# pending plans expire via PLAN_TTL_SECS, completed goals get pruned by
# _prune_stale_completed_goals, and _goal_jobs itself doesn't survive a
# restart). See _save_goal_plan().
GOALS_LOG_DIR = os.environ.get("GOALS_LOG_DIR", "/opt/v6/byoa-controller")
os.makedirs(GOALS_LOG_DIR, exist_ok=True)

# ── Knowledge base ingestion (byoa-rag-ingest3 / knowledge.zackloh.dev) ──────
# NOTE: this is a *pre-processing* call the controller makes directly, over
# localhost, before planning -- unlike the knowledge-retrieve AGENT (behind
# retrieve5.zackloh.dev / port 10406), which is dispatched the normal way
# through the `registered_agents` table + call_agent(), not hardcoded here.
KNOWLEDGE_BASE_URL = os.environ.get("KNOWLEDGE_BASE_URL", "http://localhost:10405")
KNOWLEDGE_INGEST_PATH = os.environ.get("KNOWLEDGE_INGEST_PATH", "/ingest")
# ASSUMPTION -- confirm/adjust once you add this endpoint to the ingest agent:
# DELETE {KNOWLEDGE_BASE_URL}/session/<session_id> removes every doc ingested
# under that session_id. Change the template (and the requests verb in
# purge_session_docs() below) if you land on a different path/method.
KNOWLEDGE_SESSION_PURGE_PATH_TEMPLATE = os.environ.get(
    "KNOWLEDGE_SESSION_PURGE_PATH_TEMPLATE", "/session/{session_id}"
)
# DELETE {KNOWLEDGE_BASE_URL}/docs/<doc_id> removes just ONE doc's chunks --
# used for partial-failure cleanup within a single turn (see ingest_uploaded_files)
# so a failed upload doesn't nuke every OTHER doc already attached earlier in
# the same conversation, the way purge_session_docs(conversation_id) would.
KNOWLEDGE_DOC_PURGE_PATH_TEMPLATE = os.environ.get(
    "KNOWLEDGE_DOC_PURGE_PATH_TEMPLATE", "/docs/{doc_id}"
)

# ── Feedback memory (same chromadb_ingest_service.py process, separate
# ChromaDB collection) ────────────────────────────────────────────────────
# Same direct-localhost-call pattern as KNOWLEDGE_BASE_URL above -- this is
# a pre/post-processing call the controller makes itself, not a registered
# agent dispatched through call_agent(). Reuses KNOWLEDGE_BASE_URL's host:
# port since /feedback and /feedback/query live in the same service
# process as /ingest, just against a different collection.
FEEDBACK_BASE_URL = os.environ.get("FEEDBACK_BASE_URL", KNOWLEDGE_BASE_URL)
FEEDBACK_SUBMIT_PATH = os.environ.get("FEEDBACK_SUBMIT_PATH", "/feedback")
FEEDBACK_QUERY_PATH = os.environ.get("FEEDBACK_QUERY_PATH", "/feedback/query")
# Best-effort timeouts: feedback retrieval sits on the Goal Expander's
# critical path (see _build_experience_block below), so it must fail fast
# and fall back to "" rather than stall every single goal if the feedback
# service is slow or down. Submission (after the user already has their
# answer) can afford to be a little more patient.
FEEDBACK_QUERY_TIMEOUT_SECS = float(os.environ.get("FEEDBACK_QUERY_TIMEOUT_SECS", "3"))
FEEDBACK_SUBMIT_TIMEOUT_SECS = float(os.environ.get("FEEDBACK_SUBMIT_TIMEOUT_SECS", "10"))
# How many good/bad exemplars actually get injected into the Goal Expander
# prompt -- capped independently of how many top_k matches come back from
# /feedback/query, so prompt size stays bounded even once feedback memory
# has thousands of rows (see chat history: unbounded growth here just
# drowns the signal and bloats the prompt).
FEEDBACK_GOOD_EXEMPLARS = int(os.environ.get("FEEDBACK_GOOD_EXEMPLARS", "2"))
FEEDBACK_BAD_EXEMPLARS = int(os.environ.get("FEEDBACK_BAD_EXEMPLARS", "2"))
FEEDBACK_GOOD_RATING_MIN = int(os.environ.get("FEEDBACK_GOOD_RATING_MIN", "4"))
FEEDBACK_BAD_RATING_MAX = int(os.environ.get("FEEDBACK_BAD_RATING_MAX", "2"))
# Ratings strictly between the good/bad thresholds (i.e. 3, given the
# defaults above) previously fell into neither bucket and were never
# surfaced at all, even when the comment contained a genuinely actionable
# note (e.g. "this was fine but should be more concise"). These aren't
# "this approach was wrong" (bad) or "reuse this exact pattern" (good) --
# they're a middle note worth surfacing as its own tier, without implying
# either the approach failed or that it's a pattern to copy.
FEEDBACK_NEUTRAL_EXEMPLARS = int(os.environ.get("FEEDBACK_NEUTRAL_EXEMPLARS", "2"))
# Unlike the KB's own retrieve agent (which has DEFAULT_MIN_SIMILARITY and a
# fallback_recommended flag specifically so a weak match doesn't get treated
# as relevant), /feedback/query has no floor at all -- it returns whatever's
# nearest up to top_k, even if that "nearest" match is barely related. With
# only a handful of total feedback rows, an unrelated entry (e.g. a currency
# conversion rating showing up for a biography goal) can still rank in the
# top few purely because nothing better exists yet to outrank it. This
# threshold filters those out before bucketing by rating.
FEEDBACK_MIN_SIMILARITY = float(os.environ.get("FEEDBACK_MIN_SIMILARITY", "0.55"))
# A SINGLE low-rated similar goal turned out to be unreliable signal on
# its own -- even with dag_summary stripped, one "rated 1/5" entry sitting
# next to a near-identical goal still correlated with the model building a
# LESS structured plan (skipping a needed step), the wrong direction. The
# fix isn't to stop using bad feedback automatically -- it's to stop
# acting on any SINGLE anecdote. Bad feedback only becomes an active
# planning signal once at least this many similar low-rated goals agree;
# below that, it's still stored and visible via /feedback/list, just not
# yet trusted enough to influence planning on its own.
FEEDBACK_BAD_CORROBORATION_MIN = int(os.environ.get("FEEDBACK_BAD_CORROBORATION_MIN", "2"))
# Neutral (middling-rating) feedback isn't "this failed" in the first
# place, so it doesn't carry the same single-anecdote risk -- one comment
# is still a real, actionable note (e.g. "keep this concise") rather than
# an unreliable failure signal, so no corroboration threshold is needed
# here; it's capped by count instead (below).
FEEDBACK_NEUTRAL_MIN_CORROBORATION = int(os.environ.get("FEEDBACK_NEUTRAL_MIN_CORROBORATION", "1"))
# How long a completed goal's context (answer, capabilities used, dag
# summary) stays available for /feedback to rate after the fact. Plans
# themselves are already popped from _pending_plans on approve/reject
# (see pop_pending_plan) so they can't be reused for this -- this is a
# SEPARATE, longer-lived record purely for post-hoc rating.
COMPLETED_GOAL_TTL_SECS = int(os.environ.get("COMPLETED_GOAL_TTL_SECS", str(24 * 3600)))
# This must stay ABOVE chromadb_ingest_service.py's own INGEST_ACQUIRE_TIMEOUT_SECS
# (10 min default) plus real processing time (embedding calls there have their
# own 120s timeout, writes retry for a few more seconds) -- that service now
# QUEUES a request under load rather than rejecting it outright (see its
# MAX_CONCURRENT_INGESTS comment), so this timeout needs enough headroom for a
# request to legitimately sit in that queue and still complete, or the
# controller would give up and report a timeout for something the ingest
# service was still honestly working through. 13 minutes = 10 min queue wait +
# ~3 min buffer for actual processing once a slot opens.
UPLOAD_INGEST_TIMEOUT = int(os.environ.get("UPLOAD_INGEST_TIMEOUT", str(13 * 60)))
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", "10"))
# This is the cap on a single FILE's total size (after chunked uploads are
# reassembled -- see CHUNKED UPLOADS below), not the per-chunk transport
# size. Raised from the old 25MB default now that files can arrive via
# /upload/chunk instead of only inline in one /goal multipart body -- the
# real ceiling worth caring about now is "how big a doc do we want in the
# knowledge base," not "what fits through Cloudflare in one shot."
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200MB/file

# ── Chunked uploads (added 2026-08-12) ───────────────────────────────────────
# Cloudflare's edge enforces a hard ~100MB cap on a single request body,
# independent of anything this app does -- a large file sent inline as part
# of one /goal multipart POST gets a 413 from Cloudflare itself, before this
# process ever sees the request. This can't be fixed by raising any timeout
# or Flask/waitress setting; the request body is rejected at the edge.
#
# Workaround: the CLIENT splits any file over CHUNK_UPLOAD_TRIGGER_BYTES
# into CHUNK_UPLOAD_CHUNK_BYTES-sized pieces (default 50MB -- comfortably
# under Cloudflare's ~100MB cap even with multipart overhead) and POSTs each
# piece separately to /upload/chunk. Once every piece has arrived, POST
# /upload/finalize/<upload_id> to have this process concatenate them back
# into the exact original bytes on disk. The resulting upload_id is then
# passed to /goal (as a "file_upload_ids" field, alongside/instead of
# "files") instead of re-sending the raw content -- /goal reads the
# reassembled file straight off disk into a normal file_payloads entry, so
# everything downstream of that point (ingest_uploaded_files, MAX_UPLOAD_
# BYTES, ALLOWED_UPLOAD_EXTS) behaves exactly as it does for a small,
# directly-attached file.
#
# Each raw chunk is written to its own temp file rather than held in memory
# and reassembly streams chunk-by-chunk into the final file (see
# _finalize_chunk_upload) -- so a 200MB file never needs to sit fully in
# RAM twice over (once as chunks, once reassembled) at any point.
CHUNK_UPLOAD_DIR = os.environ.get("CHUNK_UPLOAD_DIR", "/tmp/byoa_chunk_uploads")
CHUNK_UPLOAD_CHUNK_BYTES = int(os.environ.get("CHUNK_UPLOAD_CHUNK_BYTES", str(50 * 1024 * 1024)))  # 50MB
# Generous slack over the nominal chunk size -- clients shouldn't send
# chunks bigger than CHUNK_UPLOAD_CHUNK_BYTES, but this is a sanity ceiling
# against a misbehaving/malicious client, not a strict enforcement of exact
# chunk sizing (the last chunk of a file is almost always smaller anyway).
CHUNK_UPLOAD_MAX_CHUNK_BYTES = int(os.environ.get("CHUNK_UPLOAD_MAX_CHUNK_BYTES", str(60 * 1024 * 1024)))  # 60MB
CHUNK_UPLOAD_TTL_SECS = int(os.environ.get("CHUNK_UPLOAD_TTL_SECS", str(3600)))  # abandon incomplete uploads after 1hr
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

_chunk_uploads = {}  # upload_id -> {filename, total_chunks, received:set(), status, created_at, ...}
_chunk_uploads_lock = threading.Lock()


def _prune_stale_chunk_uploads():
    """Mirrors _prune_stale_goal_jobs below -- drop abandoned uploads (client
    started sending chunks, then vanished) so /tmp/byoa_chunk_uploads doesn't
    grow unbounded. Only prunes uploads that are NOT yet consumed by a /goal
    call -- a finalized-but-not-yet-consumed upload still counts as "stale"
    after the TTL too, since a client that finalizes and never calls /goal
    with that upload_id has equally abandoned it."""
    cutoff = time.time() - CHUNK_UPLOAD_TTL_SECS
    with _chunk_uploads_lock:
        stale = [uid for uid, u in _chunk_uploads.items() if u.get("created_at", cutoff) < cutoff]
        for uid in stale:
            del _chunk_uploads[uid]
    for uid in stale:
        shutil.rmtree(os.path.join(CHUNK_UPLOAD_DIR, uid), ignore_errors=True)

# Each /goal job ingests its OWN files sequentially (see the `for fp in
# file_payloads` loop in ingest_uploaded_files below) -- one job alone was
# never the problem. The 2026-08-05 incident came from MANY concurrent jobs
# (a burst of /goal calls, each spawning its own background _worker thread
# immediately) each sequentially calling /ingest at the same time as every
# other job -- so the aggregate concurrent load on the shared ingest service
# (byoa-rag-ingest, 4 waitress threads + one shared ChromaDB writer at the
# time) could spike far past what it could actually handle, regardless of
# any one job's own file count being under MAX_UPLOAD_FILES.
#
# This semaphore caps concurrent /ingest calls PLATFORM-WIDE, across every
# job's worker thread, not just within one job. Keep this at or below
# chromadb_ingest_service.py's own MAX_CONCURRENT_INGESTS -- there's no harm
# in throttling slightly earlier here (a job's worker thread just waits its
# turn, cheap) versus discovering the limit only after already making an
# HTTP call that comes back a 503. Raised from 3 to 6 alongside that
# service's own limit (2026-08-05) to shorten queue wait under concurrent-
# user load -- keep these two values in sync if either is tuned further.
#
# Acquiring this has NO timeout of its own (see the `with` below) -- it just
# blocks the job's background worker thread until a turn opens up, which is
# fine since that thread isn't tied to any client connection (the person
# already got their 202 response). The actual bound on total wait time comes
# from UPLOAD_INGEST_TIMEOUT once the HTTP call itself is made, matching
# chromadb_ingest_service.py's own INGEST_ACQUIRE_TIMEOUT_SECS -- see that
# constant's definition there for why an eventual bound still matters even
# though the goal here is "queue, don't reject."
GLOBAL_INGEST_CONCURRENCY_LIMIT = int(os.environ.get("GLOBAL_INGEST_CONCURRENCY_LIMIT", "6"))
_INGEST_CONCURRENCY_SEMAPHORE = threading.Semaphore(GLOBAL_INGEST_CONCURRENCY_LIMIT)
ALLOWED_UPLOAD_EXTS = {".txt", ".md", ".pdf", ".docx", ".csv", ".json"}
# Formats where the raw uploaded bytes, decoded as text, ARE the complete
# and correct plain-text representation -- no extraction step needed or
# wanted. See ingest_uploaded_files() below for why this matters: these
# are exactly the formats where an upstream OCR/MD pipeline (like the
# document-processing pipeline's own tesseract+paddleocr output) encodes
# real structural information directly in whitespace -- a placeholder
# character's run-length signaling column gaps, or a blank line marking a
# table row boundary -- that a downstream task may need to read directly.
# ChromaDB ingestion's own chunking pipeline runs normalize_ocr_whitespace()
# on every block unconditionally (correct for THAT pipeline's purpose:
# clean, compact text for embedding/semantic search), which collapses any
# run of whitespace down to a single space -- deliberately destroying
# exactly the fine-grained spacing signal a structured-extraction task
# over the SAME document might depend on. Reconstructing "full_text" from
# post-chunking text for these formats silently hands a downstream task a
# lossy, restructured version of the document it never asked for and has
# no way to detect it received.
PLAIN_TEXT_INLINE_EXTS = {".txt", ".md", ".markdown"}
# Historical / no-longer load-bearing: JSON eligibility used to be decided
# by checking a file's extension against this set. That was fragile -- a
# caller could send genuinely-JSON content under a non-.json filename (see
# _try_parse_as_json's docstring for the real incident this caused) -- so
# ingest_uploaded_files() now decides JSON eligibility by attempting to
# parse the content itself, regardless of filename. Left defined here only
# because ALLOWED_UPLOAD_EXTS below still needs ".json" listed as an
# accepted extension at all (an upload named "*.json" must still pass the
# initial file-type gate); this set itself is no longer read anywhere.
JSON_UPLOAD_EXTS = {".json"}
# Per-file cap for inlining an uploaded doc's full text directly into the
# plan (see preseed_uploaded_docs() below). Under this, the file's content
# is handed straight to whatever task needs it -- no embedding, no
# similarity search, no `where` filter -- which sidesteps the ChromaDB
# small-collection quirks entirely for the common "ask about the file I
# just attached" case. Over this, the doc still gets session-scoped into
# the knowledge base as before, and the note tells the Goal Expander to
# use knowledge-retrieve (with session_id) for it instead, since a doc this
# size genuinely benefits from chunked retrieval rather than being dumped
# whole into a single task's input.
MAX_INLINE_UPLOAD_CHARS = int(os.environ.get("MAX_INLINE_UPLOAD_CHARS", "20000"))
# $task_N references only match digits (see _TASK_REF_TOKEN_RE), so
# preseeded uploads are assigned reserved integer "task_ids" starting here
# rather than a string key like "upload_1" -- that wouldn't match the
# regex at all and would get silently left as literal unresolved text (or
# flagged as a malformed reference by validate_task_references). Chosen
# high enough that it won't collide with a real plan's own task numbering
# (goal plans typically run 1..~15).
UPLOAD_TASK_ID_BASE = 9001

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:35b-a3b-q4_K_M")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")
# Bounds expand_goal()'s own call to vLLM. Was hardcoded to 12000s (3.3
# hours) -- effectively no timeout at all, so a busy/queued vLLM left a
# /goal job stuck in "planning" indefinitely with zero log output, rather
# than failing fast with a catchable error like every other vLLM call in
# this codebase (see summarize_agent.py's VLLM_QUEUE_WAIT_SECS for the
# equivalent bounded pattern there). Not set in .env -- this default of
# 120s applies unless a GOAL_EXPANDER_TIMEOUT_SECS entry is added later.
GOAL_EXPANDER_TIMEOUT_SECS = int(os.environ.get("GOAL_EXPANDER_TIMEOUT_SECS", "120"))
MAX_PARALLEL_AGENTS = int(os.environ.get("MAX_PARALLEL_AGENTS", "5"))
AGENT_CALL_TIMEOUT = int(os.environ.get("AGENT_CALL_TIMEOUT", "6000"))
PLAN_TTL_SECS = int(os.environ.get("PLAN_TTL_SECS", "600"))

BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")  # no longer used for per-agent signing -- see agent_credentials.get_agent_secret()
# "data-matching" added here after a confirmed production failure:
# byoa-match6 legitimately started taking 100-160+ seconds under
# concurrent load once its own retry/concurrency-limiting fix went in
# (correctly absorbing transient vLLM contention instead of crashing --
# see that agent's own history), which pushed it past Cloudflare
# Tunnel's ~100s edge timeout on the synchronous /run path -- the exact
# same ceiling text-summarize was moved off of below, for the exact
# same reason. Confirmed via byoa-match6's own logs: it reported
# TaskStatus.SUCCESS and a 200 on /run every time, but the response
# never reached the controller, which recorded an empty task_outputs
# and "(no answer was produced for this turn)" instead -- the request
# succeeded server-side and was lost in transit, not actually failed.
ASYNC_CAPABILITIES = {"text-summarize", "image-generate", "data-matching"}

# Capabilities whose plans skip the human approval gate (/goal ->
# /goal/<id>/approve) and execute immediately. This is a DELIBERATE,
# narrow exception to the platform's non-negotiable "AI recommends, a
# human who clicks POST is accountable" design constraint -- it is safe
# ONLY because every capability listed here is read-only/diagnostic and
# never itself writes anywhere (a data-matching comparison produces a
# report; it does not submit to the AP Document Store or any other
# system of record). A capability belongs in this set ONLY if running it
# unattended can never itself constitute the accountable "click POST"
# action the stakeholder requirement is protecting -- do NOT add a
# write/submit-capable capability here.
AUTO_APPROVE_CAPABILITIES = {"data-matching"}

# Identifies which physical deployment answered a goal -- this codebase is
# shared verbatim across /opt/v5, /opt/v6, /opt/v7 (etc.), each its own
# systemd unit/port/database, so a response alone doesn't otherwise say
# which one produced it. Set explicitly per-deployment's .env (e.g.
# CONTROLLER_VERSION=v7 in /opt/v7/byoa-controller/.env) -- there's no
# reliable way to self-detect this from inside the process (the port and
# install path are configuration, not identity), so this must be set, not
# inferred. Appended to the final answer in approve_goal's generate() --
# see the "final_answer" line there -- specifically to cut down on the
# "which controller actually handled this" confusion (e.g. the
# controller7.zackloh.dev-was-actually-pointing-at-controller6 mixup).
CONTROLLER_VERSION = os.environ.get("CONTROLLER_VERSION", "6.1")

# ── Chat conversation history (new) ──────────────────────────────────────────
# The frontend is a chat interface: a `conversation_id` ties together every
# /goal call in the same thread, so a follow-up like "now translate that to
# French" can be resolved against what was actually asked/answered earlier in
# THIS conversation. This is intentionally separate from the upload-ingestion
# `session_id` (job_id) above, which only scopes a single goal's attached
# files in the knowledge base and has nothing to do with chat turn history.
CONVERSATION_TTL_SECS = int(os.environ.get("CONVERSATION_TTL_SECS", str(4 * 3600)))  # 4 hours idle
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "20"))
MAX_HISTORY_GOAL_CHARS = int(os.environ.get("MAX_HISTORY_GOAL_CHARS", "500"))
MAX_HISTORY_ANSWER_CHARS = int(os.environ.get("MAX_HISTORY_ANSWER_CHARS", "2000"))

# Cap on how many distinct attached files a single conversation_id keeps
# offering to the Goal Expander (oldest dropped first) -- see
# append_conversation_docs below. Same bounding principle as
# MAX_HISTORY_TURNS: every tracked doc adds a line (or more, for small
# ones -- their full inline text) to the "0. PRIORITY" upload rule in
# EVERY subsequent turn's prompt, so an unbounded list grows that prompt
# without bound over a long-running thread.
MAX_HISTORY_DOCS = int(os.environ.get("MAX_HISTORY_DOCS", "20"))


# ── Ollama readiness (does NOT install/pull/start -- ollama.service owns that) ─
# NON-FATAL BY DESIGN: goal-expansion and summarize_agent.py have both since
# migrated to vLLM (VLLM_BASE_URL, see expand_goal's actual chat-completions
# call) -- Ollama is confirmed idle during normal operation and is no longer
# on this controller's live request path at all. OLLAMA_BASE_URL/OLLAMA_MODEL
# are kept only for the admin-facing /dev/restartOllama endpoint and the
# /dev/status "ollama_responsive" field, in case Ollama is deliberately
# brought back for something later. This function used to raise and crash
# the whole controller on startup if the shared Ollama systemd service
# happened to be down -- confirmed as a real incident (controller6
# crash-looping on 2026-09-08 purely because Ollama was unreachable, despite
# Ollama being unused). It now only warns, so an unrelated Ollama outage
# can't take the controller down with it.
def verify_ollama_ready():
    try:
        r = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"WARN: Ollama not reachable at {OLLAMA_BASE_URL} -- "
              f"is `systemctl status ollama` running? ({e})\n"
              f"      Continuing startup anyway: Ollama is not on the live "
              f"request path (goal-expansion uses vLLM); this only affects "
              f"/dev/restartOllama and /dev/status's ollama_responsive field.")
        return
    tags = [m["name"] for m in r.json().get("models", [])]
    if OLLAMA_MODEL not in tags:
        print(f"WARN: Model {OLLAMA_MODEL!r} is not pulled. Run: ollama pull {OLLAMA_MODEL}\n"
              f"      Currently pulled: {tags}")
        return
    print(f"OK Ollama reachable, model {OLLAMA_MODEL!r} present.")


# ── Registry helpers (Cell 4, unchanged) ───────────────────────────────────────
def get_conn():
    return psycopg2.connect(PLATFORM_DB_URL)


def fetch_active_agents() -> list:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT agent_id, display_name, endpoint_url,
               invoke_path, invoke_method,
               capabilities, input_schema, manifest
        FROM registered_agents
        WHERE status = 'REGISTERED_ACTIVE'
        ORDER BY registered_at DESC
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()

    agents = []
    for r in rows:
        manifest = r["manifest"] if isinstance(r["manifest"], dict) else json.loads(r["manifest"])
        agents.append({
            "agent_id": r["agent_id"],
            "display_name": r["display_name"],
            "description": manifest.get("description", ""),
            "capabilities": r["capabilities"] if isinstance(r["capabilities"], list) else json.loads(r["capabilities"]),
            "input_schema": r["input_schema"] if isinstance(r["input_schema"], dict) else json.loads(r["input_schema"]),
            "output_schema": manifest.get("output_schema", {}),
            "endpoint_url": r["endpoint_url"],
            "invoke_path": r["invoke_path"],
            "invoke_method": r["invoke_method"],
            "invoke_content_type": manifest.get("invoke_content_type", "multipart/form-data"),
        })
    return agents


# ── TEMP toggle: persist original uploads to disk ─────────────────────────────
# When True, every uploaded file's raw original bytes (exactly as received,
# before any KB ingestion/chunking/JSON parsing) are additionally written to
# ORIGINAL_UPLOADS_DIR. Purely additive -- doesn't change what gets ingested,
# what's returned, or what lands in doc_refs; this is just a local copy for
# debugging/audit. Flip to False to stop writing (files already saved are
# left alone either way).
SAVE_ORIGINAL_UPLOADS = True
ORIGINAL_UPLOADS_DIR = "/opt/v6/byoa-controller"


def _save_original_upload(filename: str, content: bytes) -> None:
    """Best-effort write of one upload's raw bytes to ORIGINAL_UPLOADS_DIR,
    gated by SAVE_ORIGINAL_UPLOADS. Never raises -- a disk/permission
    problem here must not fail the actual upload/ingestion flow, so any
    error is logged and swallowed. Filenames are prefixed with a
    timestamp + short uuid so same-named re-uploads (or two different
    conversations uploading "report.pdf") don't collide/overwrite.
    """
    if not SAVE_ORIGINAL_UPLOADS:
        return
    try:
        os.makedirs(ORIGINAL_UPLOADS_DIR, exist_ok=True)
        unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
        dest = os.path.join(ORIGINAL_UPLOADS_DIR, unique_name)
        with open(dest, "wb") as f:
            f.write(content)
    except Exception as e:
        print(f"WARN: failed to save original upload {filename!r} to {ORIGINAL_UPLOADS_DIR}: {e}")


# ── Goal-attached file uploads → knowledge base ingestion ────────────────────
def _try_parse_as_json(content: bytes):
    """
    Attempts to treat `content` as JSON regardless of what the file was
    named -- replaces the old "ext in JSON_UPLOAD_EXTS" filename check in
    ingest_uploaded_files below, which trusted the caller's filename to
    correctly reflect the content. That trust turned out to be misplaced
    in practice (2026-09 auto-match investigation): documents freshly
    fetched from an external system were named "*.json" by their own
    fetch code, but a locally-OCR'd document kept its original,
    non-".json" filename even after its content became a JSON string --
    silently routing it down the plain-text/knowledge-base path instead
    of the matching path, with no error surfaced anywhere. Checking the
    actual bytes instead of the filename closes that entire class of bug.

    Returns (True, parsed_object, raw_text) if `content` decodes as
    UTF-8 and that text parses as valid JSON. Returns (False, None, None)
    for anything else, including genuinely binary files (.pdf, .docx) --
    the UTF-8 decode itself is expected to fail for those, which is
    caught here and treated the same as "not JSON" rather than raised.

    Cheap either way: both the decode and the parse bail out on the
    first invalid byte/token, so a real PDF/DOCX never gets past the
    decode step -- this costs virtually nothing extra over the old
    extension check, even at the MAX_UPLOAD_BYTES ceiling.
    """
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False, None, None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return False, None, None
    return True, parsed, raw_text


def ingest_uploaded_files(session_id: str, file_payloads: list, enforce_file_limit: bool = True) -> list:
    """
    Validates and ingests each uploaded file into the knowledge base
    (byoa-rag-ingest5, localhost:10405), tagged with `session_id` so the
    knowledge-retrieve agent can later be scoped to just THIS goal's uploads
    rather than the whole knowledge base.

    file_payloads: list of {"filename": str, "mimetype": str, "content": bytes}.
    Plain bytes, NOT Flask FileStorage objects -- FileStorage.stream is tied
    to the request context and gets closed by Werkzeug as soon as the
    request/response cycle ends, which happens before this function ever
    runs now that it's called from the background worker rather than the
    request thread. The request handler reads each upload into memory
    (cheap, no network involved) and hands over plain bytes instead.

    Returns a list of doc refs (whatever /ingest hands back per file, e.g.
    {"doc_id":..., "collection":..., "chunks": N}). Raises ValueError for
    validation failures (caller turns these into 400s) and RuntimeError for
    ingestion/network failures (caller turns these into 502s).
    """
    # enforce_file_limit=False is passed ONLY for an all-JSON upload that went
    # over MAX_UPLOAD_FILES at intake (see goal_endpoint): the cap is then
    # decided AFTER the plan exists, because only a plan containing a
    # data-matching task is allowed to exceed it. Every other caller keeps the
    # default (True), i.e. the original behaviour.
    if enforce_file_limit and len(file_payloads) > MAX_UPLOAD_FILES:
        raise ValueError(f"Too many files (max {MAX_UPLOAD_FILES})")

    doc_refs = []
    for fp in file_payloads:
        filename = fp.get("filename") or ""
        content = fp["content"]
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            raise ValueError(f"Unsupported file type: {filename}")

        size = len(content)
        if size == 0:
            raise ValueError(f"{filename} is empty")
        if size > MAX_UPLOAD_BYTES:
            raise ValueError(f"{filename} exceeds {MAX_UPLOAD_BYTES} byte limit")

        safe_name = secure_filename(filename) or "upload"
        _save_original_upload(safe_name, content)

        # Content-based JSON detection (see _try_parse_as_json) -- NOT
        # gated on the file's extension/name. JSON uploads skip the KB
        # /ingest call entirely and are handled standalone -- chromadb_
        # ingest_service.py's SUPPORTED_EXTENSIONS doesn't include .json
        # anyway (it 415s), and JSON documents don't need knowledge-
        # retrieve/semantic search: the data-matching rule only ever
        # consumes the inline $task_N.json reference built here, never
        # anything retrieve-based. doc_id is synthesized locally (same
        # sha256-of-filename pattern the KB service itself uses) since
        # there's no KB response to pull one from.
        is_json, parsed_json, raw_text = _try_parse_as_json(content)
        if is_json:
            doc_ref = {
                "filename": safe_name,
                "status": "ingested",
                "doc_id": hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24],
                "full_text": raw_text,
                "parsed_json": parsed_json,
                "is_json": True,
            }
            doc_refs.append(doc_ref)
            continue

        try:
            # Acquired for the duration of this one file's /ingest call --
            # released as soon as it returns (success or failure), so a
            # slow file doesn't starve other jobs' turns any longer than
            # necessary. See GLOBAL_INGEST_CONCURRENCY_LIMIT above for why
            # this exists at all: without it, N concurrent /goal jobs each
            # running this same sequential loop could still add up to N
            # simultaneous /ingest calls platform-wide.
            with _INGEST_CONCURRENCY_SEMAPHORE:
                resp = req.post(
                    f"{KNOWLEDGE_BASE_URL}{KNOWLEDGE_INGEST_PATH}",
                    files={"file": (safe_name, io.BytesIO(content),
                                     fp.get("mimetype") or "application/octet-stream")},
                    data={"session_id": session_id, "source": "goal_upload", "original_filename": filename},
                    timeout=UPLOAD_INGEST_TIMEOUT,
                )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ingestion failed for {filename}: {e}")

        try:
            doc_ref = resp.json()
        except Exception:
            doc_ref = {"filename": safe_name, "status": "ingested"}
        doc_ref.setdefault("filename", safe_name)

        # Pull the full text back out immediately -- the ingest service
        # already chunked+embedded it into the KB, and we also want text
        # on hand so small uploads can be inlined directly into a task's
        # input (see preseed_uploaded_docs below) instead of requiring a
        # similarity search to get their own content back out.
        #
        # PLAIN_TEXT_INLINE_EXTS get their `full_text` from the ORIGINAL
        # uploaded bytes directly, NOT from the post-chunking reconstruction
        # below. Root-caused via a long debugging session (see chat
        # history): chromadb_ingest_service.py's normalize_ocr_whitespace()
        # runs unconditionally on every chunked block and collapses any
        # run of whitespace -- including a placeholder character an
        # upstream OCR/MD pipeline uses to encode real column-gap and row-
        # boundary structure -- down to a single space. That's the CORRECT
        # behavior for what that function exists to do (clean, compact
        # text for embedding quality), but reconstructing full_text FROM
        # that already-collapsed, already-rechunked text for a
        # downstream extraction task silently hands it a lossy,
        # restructured version of the document -- e.g. an OCR'd table
        # whose column spacing and per-row blank-line structure a
        # downstream agent (byoa-summarize-agent) specifically depends on
        # to correctly segment rows, arriving already destroyed, with no
        # signal anywhere that this happened (confirmed: a 4565-character
        # source MD was silently reduced to ~1498 characters this way,
        # and re-chunked/rejoined block boundaries no longer lined up
        # with the original document's actual row boundaries at all).
        # For a plain-text format, there is no extraction step to
        # meaningfully bypass in the first place -- the raw bytes ARE the
        # complete, correct text -- so there's no reason to accept a lossy
        # reconstruction when the original is sitting right there.
        doc_id = doc_ref.get("doc_id")
        if ext in PLAIN_TEXT_INLINE_EXTS:
            try:
                doc_ref["full_text"] = content.decode("utf-8")
            except UnicodeDecodeError:
                doc_ref["full_text"] = content.decode("utf-8", errors="replace")
        elif doc_id:
            try:
                chunks_resp = req.get(
                    f"{KNOWLEDGE_BASE_URL}/docs/{doc_id}/chunks",
                    timeout=UPLOAD_INGEST_TIMEOUT,
                )
                chunks_resp.raise_for_status()
                chunks = chunks_resp.json().get("chunks", [])
                doc_ref["full_text"] = "\n\n".join(c.get("text", "") for c in chunks)
            except Exception as e:
                # Not fatal -- ingestion itself already succeeded, this
                # doc just won't be eligible for inlining and will fall
                # back to the session-scoped knowledge-retrieve path below.
                doc_ref["full_text"] = ""
                print(f"WARN: could not fetch full text for doc_id {doc_id!r}: {e}")

        doc_refs.append(doc_ref)

    return doc_refs


def purge_docs(doc_ids: list) -> None:
    """
    Deletes a specific set of docs by doc_id (not a whole session/conversation).
    Best-effort, per-doc -- used when only THIS turn's newly-ingested files
    need to be rolled back (e.g. goal expansion errored after ingestion) but
    earlier turns' files in the same conversation must be left alone.
    """
    for doc_id in doc_ids:
        if not doc_id:
            continue
        url = f"{KNOWLEDGE_BASE_URL}{KNOWLEDGE_DOC_PURGE_PATH_TEMPLATE.format(doc_id=doc_id)}"
        try:
            resp = req.delete(url, timeout=UPLOAD_INGEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            print(f"WARN: failed to purge doc_id {doc_id!r} from knowledge base: {e}")


def purge_session_docs(session_id: str) -> bool:
    """
    Deletes every document ingested under this session_id from the knowledge
    base, so uploaded files don't linger and accumulate there once they're no
    longer needed. Called from every path a session's uploaded docs can stop
    being needed:
      - goal expansion errored or found no tasks (plan never created)
      - the plan was rejected
      - the plan expired unapproved (PLAN_TTL_SECS, via _cleanup_expired_plans)
      - all pending plans were flushed by a /dev/* prompt change
      - the plan was approved and execute_tasks() has finished running

    Returns True if the purge call succeeded, False otherwise (logged, not
    raised -- a failed purge shouldn't break the goal/plan flow itself, it
    just means that session's docs linger in the KB until manually cleaned up).
    """
    url = f"{KNOWLEDGE_BASE_URL}{KNOWLEDGE_SESSION_PURGE_PATH_TEMPLATE.format(session_id=session_id)}"
    try:
        resp = req.delete(url, timeout=UPLOAD_INGEST_TIMEOUT)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"WARN: failed to purge session {session_id!r} docs from knowledge base: {e}")
        return False


def preseed_uploaded_docs(doc_refs: list) -> dict:
    """
    Builds pre-completed pseudo-task entries for uploaded docs small enough
    to inline directly (see MAX_INLINE_UPLOAD_CHARS) -- keyed with reserved
    integer "task_ids" starting at UPLOAD_TASK_ID_BASE (9001, 9002, ...),
    NOT string keys like "upload_1" -- $task_N references only match
    digits (_TASK_REF_TOKEN_RE), so a non-numeric key would never even be
    recognized as a reference. Seeded into execute_tasks()'s `completed`
    dict BEFORE the DAG runs, so the Goal Expander can reference
    "$task_9001.text" exactly like any other task output -- no agent call
    happens for these, the value is just already there.

    Each entry matches the same shape a real task result has
    (response.outputs.text) so the existing resolve_inputs()/_resolve_path()
    machinery needs no special-casing to handle it.
    """
    preseeded = {}
    n = 0
    for d in doc_refs:
        text = d.get("full_text", "")
        if not text or len(text) > MAX_INLINE_UPLOAD_CHARS:
            continue
        task_id = UPLOAD_TASK_ID_BASE + n
        n += 1
        outputs = {
            "text": text,
            "filename": d.get("filename"),
            "doc_id": d.get("doc_id"),
        }
        # JSON uploads additionally get their parsed object available as
        # $task_N.json -- see JSON_UPLOAD_EXTS/ingest_uploaded_files(). This
        # is purely additive: "text" above is still the same raw string it
        # always was, so anything already relying on $task_N.text for
        # content questions is completely unaffected.
        if d.get("is_json"):
            outputs["json"] = d.get("parsed_json")
        preseeded[task_id] = {
            "task_id": task_id,
            "task_description": f"Uploaded file: {d.get('filename', 'unknown')}",
            "agent_id": "goal-upload",
            "success": True,
            "status_code": 200,
            "response": {
                "status": "success",
                "error_msg": None,
                "outputs": outputs,
            },
            "error": None,
        }
    return preseeded


def _build_upload_rule(doc_refs: list, session_id: str) -> str:
    """
    A dedicated "0. PRIORITY" rule describing THIS request's uploaded
    file(s) -- spliced into extra_rules (build_expander_prompt) BEFORE the
    RAG rule, NOT appended to the goal text.

    Two things had to change to get this to actually stick, in order:

    1. A paragraph tacked onto the end of {goal} was empirically
       unreliable -- it competes against rule 10's own extensive,
       example-backed instructions (which the model has much more to
       latch onto) and consistently lost, with the model defaulting
       straight back to building a knowledge-retrieve task even when a
       small, directly-relevant upload was available and the note
       explicitly said not to. Giving uploads the same structural weight
       -- a proper rule with an explicit override clause, plus a worked
       example (see _build_upload_example) -- was the first fix.

    2. Even with that structural weight, numbering this "11." STILL lost
       reliably -- because GOAL_EXPANDER_RAG_RULE already contains its OWN
       internal "11. Coordinate multi-entity goals..." sub-rule, so the
       prompt ended up with two different, unrelated "rule 11"s in it,
       and placed AFTER rule 10 to boot (the model had already "locked
       in" the knowledge-retrieve pattern by the time it got here). Fixed
       by (a) using a non-colliding "0. PRIORITY" label instead of a
       number that clashes with an existing rule, and (b) splicing this
       BEFORE rag_rules in build_expander_prompt, so the override is read
       first, not as a contradiction arriving after the fact.
    """
    inline_lines = []
    json_lines = []
    retrieve_lines = []
    n = 0
    for d in doc_refs:
        filename = d.get("filename") or d.get("original_filename") or "a document"
        text = d.get("full_text", "")
        if text and len(text) <= MAX_INLINE_UPLOAD_CHARS:
            task_id = UPLOAD_TASK_ID_BASE + n
            n += 1
            inline_lines.append(f'      - "{filename}" -- full text already available as $task_{task_id}.text')
            if d.get("is_json"):
                json_lines.append(f'      - "{filename}" -- parsed JSON object already available as $task_{task_id}.json')
        else:
            doc_id = d.get("doc_id")
            retrieve_lines.append(
                f'      - "{filename}" (too large to inline) -- doc_id="{doc_id}"'
            )

    if not inline_lines and not retrieve_lines:
        return ""

    parts = ["\n0. PRIORITY -- files attached to THIS specific goal (read this FIRST, "
             "before rule 10 below or any other rule that would otherwise apply to the "
             "same content):\n"]

    if inline_lines:
        parts.append(
            "    The following file(s) were just uploaded with this goal. Their full text is\n"
            "    ALREADY AVAILABLE directly -- no agent call needed to retrieve it:\n"
            + "\n".join(inline_lines) + "\n"
            "    For any part of the goal asking about the CONTENT of one of these files (e.g.\n"
            "    \"what does this file say about X\", \"summarize the attached document\", \"what\n"
            "    is [term] according to the attached file\"), you MUST:\n"
            "      a. Reference the file's content directly as a task input, e.g.:\n"
            "           \"text\": \"$task_" + str(UPLOAD_TASK_ID_BASE) + ".text\"\n"
            "         (use the EXACT reserved task_id number listed above for that specific file --\n"
            "         never invent a different number, never use one of the real plan task_ids for it.)\n"
            "      b. Give this task an EMPTY depends_on: [] -- the content is already available,\n"
            "         there is nothing to wait on.\n"
            "      c. Do NOT build a knowledge-retrieve task, and do NOT build a web-search task,\n"
            "         for this part of the goal. THIS RULE OVERRIDES RULE 10 for content covered\n"
            "         by an uploaded file -- even when a knowledge-retrieve agent is registered, do\n"
            "         not search the knowledge base for something an attached file already answers\n"
            "         directly. Searching for content that was just handed to you directly is\n"
            "         redundant, slower, and (for a very small/freshly-scoped search) can return\n"
            "         zero or unreliable matches anyway.\n"
            "      d. If the goal has ANOTHER part unrelated to the uploaded file's content (e.g.\n"
            "         \"...and also tell me about Y\"), handle that part normally per whichever other\n"
            "         rule fits (web-search, knowledge-retrieve, etc) -- this override applies ONLY\n"
            "         to the part of the goal the uploaded file's content actually answers.\n"
        )

    if json_lines:
        parts.append(
            "\n    The following of the above file(s) are JSON files, and ALSO have their parsed\n"
            "    JSON object (not just raw text) already available:\n"
            + "\n".join(json_lines) + "\n"
            "    If (and ONLY if) a data-matching agent is registered and the MATCH rule below\n"
            "    applies to this goal, use \"$task_N.json\" (the parsed object) -- NOT\n"
            "    \"$task_N.text\" (the raw string) -- as that document's element in the\n"
            "    data-matching task's \"documents\" array. \"$task_N.text\" remains the right\n"
            "    reference for ordinary content questions about the file per (a)-(d) above --\n"
            "    this JSON-object form exists purely to additionally support matching, not to\n"
            "    replace the text reference for anything else.\n"
        )

    if retrieve_lines:
        parts.append(
            "    The following file(s) were also uploaded but were too large to inline, and were\n"
            f"    instead ingested into the internal knowledge base scoped to session_id=\"{session_id}\":\n"
            + "\n".join(retrieve_lines) + "\n"
            "    Two different cases for these, depending on what the goal actually asks:\n"
            "      a. SPECIFIC QUESTION the file might answer (e.g. \"what does the file say about\n"
            "         X\", \"find the section on Y\"): build a knowledge-retrieve task per rule 10,\n"
            f"         but additionally include \"session_id\": \"{session_id}\" in its inputs\n"
            "         alongside \"query\", so retrieval is scoped to just this session's uploads\n"
            "         rather than the full knowledge base.\n"
            "      b. WHOLE-DOCUMENT summary/description (e.g. \"summarize this document\", \"what\n"
            "         does the attached file say\", \"describe the uploaded file\", or any goal with\n"
            "         no specific question narrower than the whole file): do NOT use a normal\n"
            "         similarity query for this -- a query like \"summarize the document X.pdf\" has\n"
            "         almost no semantic overlap with the document's actual content and will score\n"
            "         low or wrongly report no confident match, even though the document IS there.\n"
            "         Instead, build a knowledge-retrieve task with:\n"
            f"           \"session_id\": \"{session_id}\", \"fetch_all\": true\n"
            "         (\"query\" is still required by the schema but is ignored in this mode --\n"
            "         any short placeholder value is fine). This bypasses similarity ranking\n"
            "         entirely and returns the file's ENTIRE content, in original order, as a\n"
            "         SINGLE field: $task_N.full_text (N = that task's own task_id). Reference\n"
            "         $task_N.full_text directly as the summarize task's \"text\" input -- do NOT\n"
            "         use $task_N.results[0].text/results[1].text for this case, since fetch_all\n"
            "         mode returns the whole document as one field, not a short ranked list.\n"
            + ("         CRITICAL -- more than one large file is attached to this session: "
               "fetch_all\n"
               "         scoped by session_id ALONE returns every large file in this session "
               "MERGED\n"
               "         together as a single full_text blob -- there is no way to tell which\n"
               "         content came from which file, and a goal like \"compare the two\" would\n"
               "         silently end up comparing the same merged blob against itself. You MUST\n"
               "         additionally include \"doc_id\" in each retrieve task's inputs, using the\n"
               "         EXACT doc_id value listed above for that specific file, so each retrieve\n"
               "         task fetches only its own document:\n"
               f"           \"session_id\": \"{session_id}\", \"fetch_all\": true, \"doc_id\": \"<that file's doc_id>\"\n"
               "         Build ONE retrieve task PER large file this way, each with its own doc_id.\n"
               if len(retrieve_lines) > 1 else "")
        )

    return "".join(parts)


def _build_upload_example(doc_refs: list) -> str:
    """
    Concrete worked example(s) matching the style/format of
    GOAL_EXPANDER_RAG_EXAMPLES (including its double-brace JSON literal
    convention), showing the task shapes the PRIORITY rule expects.
    Two independent examples, emitted based on which case(s) actually
    apply to this request's uploads:
      - inline example: small file, content already available directly.
      - fetch_all example: large file, whole-document summary needed --
        similarity search is the wrong tool for this (see
        _build_upload_rule), so this shows the fetch_all + full_text
        pattern instead of a normal knowledge-retrieve query.
    """
    parts = []

    if any(d.get("full_text") and len(d["full_text"]) <= MAX_INLINE_UPLOAD_CHARS for d in doc_refs):
        parts.append(f"""
Example for a goal about the CONTENT of an uploaded file, per the PRIORITY
rule above -- note there is NO knowledge-retrieve task and NO web-search
task anywhere; the summarise-agent task references the uploaded file's
content directly and has an EMPTY depends_on, since nothing needs to be
fetched first (again, agent_id below is PURELY FICTIONAL, use the real
registered one):
[
  {{
    "task_id": 1,
    "task_description": "Answer the question using the uploaded file's content",
    "agent_id": "summarise-agent",
    "inputs": {{
      "text": "$task_{UPLOAD_TASK_ID_BASE}.text",
      "focus": "Using the excerpt below, answer the original question directly and naturally."
    }},
    "depends_on": []
  }}
]
""")

    if any(d.get("full_text") and len(d["full_text"]) > MAX_INLINE_UPLOAD_CHARS for d in doc_refs):
        parts.append("""
Example for a WHOLE-DOCUMENT summary of a large uploaded file (too big to
inline), per the PRIORITY rule above -- note "fetch_all": true replaces a
normal similarity "query", and the summarise task references
$task_1.full_text (a single field), NOT $task_1.results[0].text -- fetch_all
mode returns the whole document as one joined field, not a short ranked
list (again, agent_ids are PURELY FICTIONAL, use the real registered ones):
[
  {
    "task_id": 1,
    "task_description": "Retrieve the full content of the uploaded document",
    "agent_id": "knowledge-retrieve-agent",
    "inputs": {
      "query": "document content",
      "session_id": "<use the exact session_id given above>",
      "fetch_all": true
    },
    "depends_on": []
  },
  {
    "task_id": 2,
    "task_description": "Summarize the uploaded document",
    "agent_id": "summarise-agent",
    "inputs": {
      "text": "$task_1.full_text",
      "focus": "Using the excerpt below, provide a clear summary of the document's main points."
    },
    "depends_on": [1]
  }
]
""")

    return "".join(parts)


# ── LLM Goal Expander (Cell 5, unchanged) ─────────────────────────────────────
GOAL_EXPANDER_PROMPT_TEMPLATE = """You are a task planner for an AI agent platform.

A user has submitted the following goal:
\"\"\"{goal}\"\"\"
{history_block}
{experience_block}
The following agents are currently registered and available:
{agents_block}

Your job:
1. Analyse the goal.
2. Decide which agents (from the list above) are relevant to fulfil it.
3. Break the goal into an ordered Kanban task list where each task maps to one agent.
4. Only use agents from the list above. Do not invent agents.
5. If no agent is relevant, return an empty task list.
6. If a task needs output from a previous task, reference it using $task_N.path
   where N is the task_id and path navigates the upstream agent's real output_schema.
   Supported path syntax:
     $task_1.field              — plain field (e.g. $task_1.text)
     $task_1.field[0]           — first item of an array field (e.g. $task_1.results[0])
     $task_1.field[0].subkey    — a key inside an array item (e.g. $task_1.results[0].url)
   RULES:
   - field names MUST be exact keys from that agent's output_schema keys listed above.
   - NEVER invent a field name.
   - NEVER copy field names from the example below — it is illustrative syntax only.
   - For array fields (type: array), use [0] to access the first item, then .subkey
     to access a specific property of that item (e.g. $task_1.results[0].url to get
     the URL of the first search result).
   - If you need the whole array, reference just the field: $task_1.results
   - A $task_N.path reference can also be EMBEDDED inside a larger literal string,
     not just used as the entire input value. Everything outside the $task_N.path
     token is kept as literal text; only the token itself is substituted with the
     real upstream value at execution time. This lets you compose a new input out
     of an upstream result plus your own wording, e.g.:
       "query": "$task_3.summary reviews and attractions"
     or concatenate more than one upstream text into a single field, e.g.:
       "text": "$task_2.text\n\n$task_5.text"
     Use this whenever a later task's input needs to combine an upstream value
     with additional literal wording, rather than being only the raw value.
     CRITICAL: that blank-line separator MUST be written as the two-character
     ESCAPED sequence backslash-n-backslash-n (\n\n as literal characters in
     your output), never as an actual line break inside the string. Your
     entire response is parsed as JSON, and JSON strings cannot contain a
     literal raw newline character -- if you press what would be "enter"
     inside a quoted string instead of writing the escaped \n\n sequence,
     the ENTIRE response fails to parse and the whole plan is lost, even
     though every other task was correct. Keep every "text" field's value
     as one single unbroken line in your raw output, using \n\n only as an
     escaped two-character token wherever a separator is needed.

7. Multi-part / dependent-entity goals:
   Some goals have two (or more) parts where a LATER part asks about, evaluates,
   or reviews a SPECIFIC entity that can only be known once an EARLIER part of
   the chain has resolved it — e.g. "find the winning entry in a competition,
   then tell me about the team behind it", "find the top-rated item in category Y, then
   tell me about its manufacturer". In this situation you MUST NOT build two
   independent parallel branches that never reference each other — the later
   part cannot search or reason about "the team behind it" / "its manufacturer"
   etc. generically, because the entity is unknown until the earlier chain runs.
   Instead:
     a. Build the earlier chain as normal (search → page-fetch as needed) to
        gather the raw content the entity will be extracted from. The
        page-fetch is NOT optional whenever the extraction task in step (b)
        needs to compare specific figures (prices, ratings, ranks, scores)
        across sources -- raw search-result objects (titles, snippets,
        short text_previews) are truncated, inconsistent in units/currency,
        and frequently include noise the extraction task can't filter (e.g.
        a "flights from X" search surfacing a "flights TO X" listing from a
        different origin entirely). NEVER feed a search task's raw
        "results" array directly into an extraction task's "text" input
        (e.g. "text": "$task_1.results") as a shortcut past the crawl step
        -- always crawl the top result(s) first and feed the extraction
        task the actual fetched page text ("$task_2.text"), never the
        un-crawled search metadata.
     b. Add a dedicated extraction task using a summarise-capable agent whose
        task_description explicitly asks it to identify ONLY the specific
        entity needed for the NEXT step's search -- not the final fact the
        goal ultimately wants. For "find the winning entry in a competition,
        then tell me about the team behind it", the next step needs the
        winning entry's identity, so extract that: "Identify the winning
        entry in the competition." For "find the top-rated item in category
        Y, then tell me about its manufacturer", the next step needs to
        search for the item's manufacturer, which means it needs the ITEM's
        name first -- NOT the manufacturer itself (that hasn't been searched
        for yet at this point in the chain, so asking this task to identify
        it is asking for something it cannot actually know): "Identify the
        name of the top-rated item found" -- never "...its manufacturer" or
        any other phrasing that names the LATER fact instead of the entity
        needed to go search for that fact. With that upstream task's fetched
        text as its input, this task's output becomes the resolved entity for
        later steps. IMPORTANT when the goal involves a superlative
        ("cheapest", "highest-rated", "best", "fastest", "lowest", etc.): the
        upstream text this task reads is often a COMBINED crawl of several
        sources, and more than one of those sources will independently frame
        its OWN content as "the cheapest" / "the best" relative only to
        itself -- e.g. a page entirely about flights to one specific
        destination will describe its own fare as "the cheapest one-way
        price found," even if a completely different, cheaper destination is
        named on another page later in the same combined text. Trusting
        whichever source's own editorial framing claims the superlative is
        not reliable. The extraction task's focus MUST explicitly instruct
        comparing the actual figures (prices, ratings, scores, etc.)
        mentioned anywhere in the text and identifying the entity tied to the
        objectively best one -- e.g. "Compare every price mentioned in the
        text, then respond with only the name of the destination tied to the
        single lowest price found -- not whichever destination a single
        source calls cheapest" -- rather than a bare "identify the
        destination in the cheapest offer" phrasing that lets the model
        anchor on the first or most prominent self-described "cheapest"
        claim it encounters.
        Also watch for two related traps in the same combined text: (1) if
        the goal names a specific origin or starting point (e.g. "cheapest
        flights FROM Singapore"), crawled sources sometimes include
        reversed-direction or unrelated content (e.g. a page about flights
        TO Singapore from elsewhere) -- the extraction focus must
        explicitly exclude the named origin itself from being a valid
        candidate answer, since it can never correctly be its own
        destination. (2) if the goal states an explicit numeric constraint
        (e.g. "under $1000", "within a $500 budget"), some crawled sources
        will rank/recommend items OUTSIDE that constraint under a similar-
        sounding but different superlative (e.g. a general "best laptops of
        2026" list recommending a $1,300 laptop as its overall top pick,
        crawled alongside a page specifically about laptops "under $1000").
        The extraction focus must explicitly require verifying the
        candidate actually satisfies the stated constraint before accepting
        it, not merely that some source calls it "best" or "top-rated" in
        an unrelated or unconstrained context.
        CRITICAL WORDING CONSTRAINT: whatever comparison instruction you use
        MUST still read as a single-entity extraction ask to the summarise
        agent, which only responds with a bare name (rather than a full
        paragraph) when the focus contains NO disqualifying words -- avoid
        "and", "describe", "explain", "whether", "how", "why", or "should"
        anywhere in this task's focus. A focus like "compare every price...
        AND identify..." disqualifies bare-entity mode purely because of
        that "and", forcing a full comparative paragraph back out instead of
        a clean name -- and that whole paragraph then gets embedded verbatim
        into the next task's query (per step c), diluting the actual target
        entity among several other names, numbers, and irrelevant context
        mentioned along the way. Use commas or "then" to join clauses
        instead of "and", and phrase the request as "identify" / "extract"
        the winning entity rather than "describe" or "explain" why it wins.
     c. The later part's tasks MUST set depends_on to include this extraction
        task (directly or transitively), and MUST embed the extraction task's
        output field inside their own inputs (per rule 6's embedded-token
        syntax) rather than writing a generic, self-contained query that does
        not reference the resolved entity at all.
     d. The final task in this chain is a single summarise-agent call over the
        concatenated raw text from both parts (per rule 6's concatenation
        syntax) -- NOT another extraction task. Word its task_description as
        a full, evaluative request covering everything the original goal
        asked, e.g. "Summarize the winning entry and give a full
        assessment of the team's background, covering
        both parts of the request" -- NOT a short "identify"/"name" style
        phrase like the extraction task in step (b) used. The wording matters:
        an extraction-style description will be (correctly) answered with
        just a bare name by the summarise agent, which would silently discard
        the entire evaluative half of the goal since this is the plan's final
        answer with nothing downstream to catch the loss. Its focus must
        require the summary to EXPLICITLY state each fact the original goal
        asked for as its own clause, not merely include details that imply
        it -- e.g. for "...and who makes it", the summary must contain a
        direct statement such as "It is made by [company]", not just mention
        the company's name in passing inside a product name or spec line
        where the reader has to infer the connection themselves.
        Its focus MUST also embed the extraction task's output token (e.g.
        "$task_3.summary") as the named subject to write about -- e.g.
        "Summarize what the text says about $task_3.summary, covering its
        price and a full assessment of its quality" -- rather than asking
        the model to independently re-derive "the cheapest X" or "the
        top-rated Y" from the raw combined text a second time. The entity
        was already resolved and committed to in step (b); this task's job
        is to report ON that specific, already-named entity, not to
        re-litigate whether it was the right choice. Re-deriving from raw
        text a second time risks a different (and possibly hedging or
        contradictory) answer than step (b) already gave, since the same
        messy multi-source text that step (b) had to resolve down to one
        entity will still look ambiguous on a second, independent pass --
        embedding the token keeps both tasks consistent with each other.
   Build this full chain (steps a-d) even in cases where the later part's
   answer might plausibly already be sitting in the same upstream text (e.g.
   a product's manufacturer is often named right there in the same review
   page that named the product) -- do NOT skip the fresh search-and-fetch in
   step (a)/(c) on the assumption that it's unnecessary. Deciding "is this
   fact already in the text or not" is not a judgment call the plan should
   make: it has produced two different broken plans in testing (once folding
   both parts into a single task, once building a second task that silently
   re-derives the same answer instead of consuming the first task's output).
   Always build the complete, uniform chain instead: extract the entity in
   its own task (step b), then run a NEW search/fetch specifically about
   that entity (step a's pattern, applied to the later part), then a single
   final summarise task (step d) over the CONCATENATION of both parts' raw
   fetched text -- exactly as in the destination-quality example above, with
   no shortcut for cases that look like the fact might already be available.
   Only skip this chaining if the goal's parts are genuinely independent of one
   another (e.g. "what's the current value of Metric A and the current status of Item B" —
   neither part depends on an entity resolved by the other).

8. Finalizing answers built from multiple summarise-agent outputs:
   The platform automatically combines results for you — you do NOT need to,
   and must NOT, add a task that feeds multiple summarise-agent outputs into
   ANOTHER summarise-agent task to "combine", "synthesize", or "finalize"
   them. After your plan runs, the platform detects every summarise-agent
   task whose output nothing else in your plan depends on (i.e. each
   branch's actual final piece of the answer) and automatically appends one
   more step that joins them together verbatim — with no LLM call, so
   nothing gets reworded, reordered, or dropped. This means:
     a. If a goal has two or more INDEPENDENT parts (e.g. "how's Item X
        doing, and what similar items exist" — see the third example
        below), just let EACH branch end in its own summarise-agent task.
        Do not add any task after them to join their outputs — leave both
        as leaves. The platform appends the combine step automatically, and
        it will already be visible in the plan shown to the user, since
        this happens before the plan is returned for approval.
     b. If a goal has only ONE summarise-agent output feeding the final
        answer (including a dependent-entity chain per rule 7, where the
        LAST task is a single summarise-agent call over concatenated raw
        text), that output IS the final answer already — do not add
        anything after it. The platform only appends a combine step when it
        finds MORE than one independent final summary; a single one is left
        exactly as you built it.
     c. Never add your own "synthesis", "final answer", or "combine" task
        that takes other summarise-agent outputs as input — that duplicates
        (and risks undoing) what the platform now does for you automatically.
     d. The same auto-combine step also picks up leaves from non-text
        agents -- most commonly an image-generate task. This is where an
        easy mistake creeps in: if a goal asks for BOTH information AND an
        image (e.g. "where did X live, and how good is it -- generate an
        illustration"), do NOT make the image task depend_on the
        summarise-agent task just because the image's subject matter comes
        from that summary. Making the image task's inputs reference
        $task_N.summary creates a depends_on edge, and ANY task something
        else depends on is excluded from the leaf set -- so the summarise
        task's actual text answer would silently disappear from the final
        output, leaving only the image behind. Instead, write the image
        task's prompt input as a short, self-contained literal description
        of the subject (using what you already know from the goal itself
        or from an earlier task's OUTPUT VALUES copied in as plain text --
        not a $task_N reference), and give it depends_on: [] (or only the
        upstream task that actually gathered the raw facts, if the image
        genuinely cannot be described without them -- prefer describing it
        without them whenever possible). This keeps the summarise task and
        the image task as two independent leaves, so the platform's
        auto-combine step surfaces both the text answer and the image
        together, instead of the image silently swallowing the text.

9. Content depth rule:
   - If the goal contains a URL (e.g. "summarize https://...", "crawl https://..."):
     STEP 1: use a page-fetch agent (web-crawl capability) to retrieve that URL.
     STEP 2: pass the fetched text field (e.g. $task_1.text) to the summarise agent.
     NEVER pass a URL string directly as the text input to a summarise agent —
     summarise agents only accept plain text content, not URL strings. A URL is
     not text. Always crawl it first.
   - If the goal does NOT contain a URL and requires understanding, explaining,
     analysing, or summarising a topic in any depth:
     STEP 1: use a search agent to find relevant URLs.
     STEP 2: use a page-fetch agent on $task_N.results[0].url to get full article text.
     STEP 3: pass the fetched text to the summarise agent.
     Never summarise a short snippet — always fetch the full page first.
   - Only skip the page-fetch step if the goal explicitly asks for a brief overview
     or snippet-level answer, or if no page-fetch agent is registered.
   - Default to CONCISE final answers unless the goal explicitly asks for
     detail, depth, or thoroughness (e.g. "explain in detail", "give me a
     full breakdown", "comprehensive overview"): phrase the focus of any
     summarise-agent task whose output will end up in the FINAL answer (a
     leaf task per rule 8, or a task feeding directly into the combine
     step) to request a concise result -- e.g. include wording like "in
     4-5 sentences" or "briefly" in that task's focus. This is a DEFAULT
     added ALONGSIDE whatever else that task's focus already needs to say
     (e.g. rule 7's single-entity extraction phrasing, rule 10's KB-vs-
     web-search leaf framing) -- not a replacement for those requirements.
     When a goal produces MULTIPLE leaves that the platform will
     auto-combine (per rule 8), keep each individual leaf's target shorter
     than you would for a single-leaf goal (e.g. 2-3 sentences each rather
     than 4-5) -- the combined final answer is the sum of every leaf, so
     several leaves each independently aiming for "concise" can still add
     up to a long combined answer if each one isn't scaled down
     individually.

{extra_rules}
Respond ONLY with a valid JSON array. No explanation, no markdown, no preamble.
Each item in the array must have exactly these fields:
  - task_id: integer starting from 1
  - task_description: string — what this task does
  - agent_id: string — must match one of the agent_ids above
  - inputs: object — key-value pairs matching the agent's input_schema keys
  - depends_on: array of task_ids that must complete before this task (empty if none)

Example for a three-step search → fetch → summarise chain (ALL agent_ids and field
names below are PURELY FICTIONAL — use only real agent_ids and real output_schema keys
from the registered agents listed above. Do not copy any of these literally):
[
  {{
    "task_id": 1,
    "task_description": "Retrieve records relevant to the request",
    "agent_id": "collect-agent",
    "inputs": {{"dataset": "the dataset or subject the user asked about"}},
    "depends_on": []
  }},
  {{
    "task_id": 2,
    "task_description": "Transform the first retrieved record into the required format",
    "agent_id": "transform-agent",
    "inputs": {{"item_id": "$task_1.records[0].id"}},
    "depends_on": [1]
  }},
  {{
    "task_id": 3,
    "task_description": "Deliver the transformed result",
    "agent_id": "notify-agent",
    "inputs": {{"message": "$task_2.formatted_output"}},
    "depends_on": [2]
  }}
]

Example for a dependent-entity goal like "find the cheapest item in category X,
then tell me if its maker has a good reputation" (again, all agent_ids and field
names are PURELY FICTIONAL — do not copy them literally):
[
  {{
    "task_id": 1,
    "task_description": "Search for the cheapest item in category X",
    "agent_id": "search-agent",
    "inputs": {{"query": "cheapest item in category X"}},
    "depends_on": []
  }},
  {{
    "task_id": 2,
    "task_description": "Fetch the full page for the top search result to see maker details",
    "agent_id": "fetch-agent",
    "inputs": {{"url": "$task_1.results[0].url"}},
    "depends_on": [1]
  }},
  {{
    "task_id": 3,
    "task_description": "Identify only the name of the maker of the cheapest item found",
    "agent_id": "summarise-agent",
    "inputs": {{"text": "$task_2.text"}},
    "depends_on": [2]
  }},
  {{
    "task_id": 4,
    "task_description": "Search for reputation and reviews of that specific maker",
    "agent_id": "search-agent",
    "inputs": {{"query": "$task_3.summary reputation reviews"}},
    "depends_on": [3]
  }},
  {{
    "task_id": 5,
    "task_description": "Fetch the full page for the top reputation search result",
    "agent_id": "fetch-agent",
    "inputs": {{"url": "$task_4.results[0].url"}},
    "depends_on": [4]
  }},
  {{
    "task_id": 6,
    "task_description": "Summarise the cheapest item's price and its maker's reputation together",
    "agent_id": "summarise-agent",
    "inputs": {{"text": "$task_2.text\n\n$task_5.text"}},
    "depends_on": [2, 5]
  }}
]

Example for a goal with two INDEPENDENT parts, each needing its own
search → fetch → summarise chain. Note the plan ends at task 6 — per rule 8,
you do NOT add a further task to join tasks 3 and 6's outputs; the platform
appends that step automatically (again, all agent_ids and field names below
are PURELY FICTIONAL):
[
  {{
    "task_id": 1,
    "task_description": "Search for information on topic A",
    "agent_id": "search-agent",
    "inputs": {{"query": "topic A"}},
    "depends_on": []
  }},
  {{
    "task_id": 2,
    "task_description": "Fetch the full page for the top result on topic A",
    "agent_id": "fetch-agent",
    "inputs": {{"url": "$task_1.results[0].url"}},
    "depends_on": [1]
  }},
  {{
    "task_id": 3,
    "task_description": "Summarise topic A",
    "agent_id": "summarise-agent",
    "inputs": {{"text": "$task_2.text"}},
    "depends_on": [2]
  }},
  {{
    "task_id": 4,
    "task_description": "Search for information on topic B",
    "agent_id": "search-agent",
    "inputs": {{"query": "topic B"}},
    "depends_on": []
  }},
  {{
    "task_id": 5,
    "task_description": "Fetch the full page for the top result on topic B",
    "agent_id": "fetch-agent",
    "inputs": {{"url": "$task_4.results[0].url"}},
    "depends_on": [4]
  }},
  {{
    "task_id": 6,
    "task_description": "Summarise topic B",
    "agent_id": "summarise-agent",
    "inputs": {{"text": "$task_5.text"}},
    "depends_on": [5]
  }}
]

{extra_examples}
"""


GOAL_EXPANDER_RAG_RULE = """
10. Internal knowledge base (knowledge-retrieve capability):
    If an agent with the knowledge-retrieve capability is registered, and the
    goal is a factual/informational question about something that could
    plausibly sit in a STATIC, previously-written document (policies,
    historical/biographical facts, reports, reference material — anything
    that doesn't change from minute to minute), build it as its OWN
    INDEPENDENT branch, exactly like the independent-parts pattern in rule 8:
      a. knowledge-retrieve task, "query" = the user's core question,
         depends_on: [] (runs immediately, in parallel with everything else).
      b. A summarise-agent task depending on it, with:
           "text": "$task_N.results[0].text\n$task_N.results[1].text"
           (top TWO results' clean text fields only — NEVER embed the whole
           results array directly; it mixes chunk_id/similarity/doc_id
           metadata in with the actual text, which confuses a plain-text
           summarizer.)
           "focus": "Using the excerpt below, answer the original question directly and naturally (e.g. state who/what/where as relevant, in a brief natural sentence) -- unless fallback_recommended is true (current value: $task_N.fallback_recommended), in which case skip that and instead state plainly that the internal knowledge base did not have a confident match for this question, giving the reason: $task_N.fallback_reason."
      c. ALSO build the normal web-search -> page-fetch -> summarise chain per
         rule 9, completely independently (no depends_on linking it to the
         knowledge-retrieve branch) — this branch runs unconditionally, every
         time, regardless of the knowledge-retrieve branch's confidence. Skip
         it only if no web-search agent is registered.
      d. Per rule 8, leave BOTH branches' summarise-agent tasks as leaves —
         do NOT add your own task to combine them. The platform's existing
         auto-combine step picks up both automatically, the same way it
         already handles any other multi-part goal.

    DO skip building the knowledge-retrieve branch entirely for anything
    inherently LIVE, real-time, or continuously changing, since a static
    document store can never contain the current value of something that
    updates by the second/minute/hour. This includes (reason by the same
    principle, not an exhaustive list): live prices or valuations, named
    market or ranking indices, live rate-conversion values between two
    things, continuously-updated readings of any kind, live event
    results, or today's breaking developments on any topic. For
    these, go straight to the normal web-search/live-data-agent chain with no
    knowledge-retrieve task at all.

    You do NOT know what's actually inside the knowledge base at planning
    time, and must NOT try to guess by SUBJECT MATTER (e.g. "sounds like
    company data" vs "sounds like public knowledge") whether it's likely to
    have relevant content for anything else — that judgment belongs to the
    agent itself, reported back via fallback_recommended after it actually
    searches.

    Dependent compound questions (structurally similar to rule 7, but with
    ONE CRITICAL DIFFERENCE -- read this before applying rule 7's pattern
    here, since rule 7's own worked examples are ALL pure web-search chains
    and contain no knowledge-retrieve step anywhere; pattern-matching
    directly to rule 7's wording for step (a) below is WRONG in this case):
    the parallel branch pattern above (a-d) assumes the web-search half
    doesn't need to know the knowledge-retrieve half's answer first. That's
    true for a goal like "where did X live" alone, but NOT for a goal like
    "where did X live, and how good are those places" — the second half
    can only be searched precisely once the actual place name(s) are
    known; searching blind in parallel just searches for the person's name
    again and returns irrelevant results (e.g. towns/places merely NAMED
    after the person, not the actual places they lived). When a goal's
    second part genuinely depends on the first part's answer like this:
      a. THE FIRST PART OF THE CHAIN IS A SINGLE knowledge-retrieve TASK,
         depends_on: [] -- NOT a web-search+web-crawl pair. This is the one
         part of this whole pattern that does NOT follow rule 7's step (a)
         literally: rule 7's step (a) says "search -> page-fetch", but that
         wording describes a PURE web-search chain with no knowledge base
         involved at all. Here, one knowledge-retrieve task fully replaces
         that entire search+fetch pair for the FIRST part only -- there is
         no page-fetch step for the knowledge-retrieve branch (its results
         already ARE clean text, unlike a raw web search result). Build
         NO web-search task and NO web-crawl task for the first part of the
         goal under this rule -- only the second part (step c/d below) uses
         web-search+crawl.
      b. Insert an EXTRACTION-focused summarise-agent task depending on the
         knowledge-retrieve task from (a) — same EXTRACTION PRINCIPLE as
         rule 7's entity-extraction step (identify one specific entity
         needed for the next search), but its input is the knowledge-retrieve
         task's own result field, NOT a crawled page (there is no crawled
         page for this branch). Feed it ONLY the single top result
         ($task_N.results[0].text, NOT results[0] AND results[1]) —
         extraction needs the most precise, narrowest source available, not
         broad coverage; a second, more general chunk gives the model
         material to hedge and generalize with instead of naming one
         specific thing. Phrase the focus as asking for exactly ONE entity,
         in strictly singular language — NEVER phrase it with a
         parenthetical plural like "place(s)" or "name(s)", since that reads
         as a request for a LIST of several items (a separate mode that
         explicitly does NOT trigger clean single-entity extraction) rather
         than a single clean answer. Use wording like: "Identify the single
         specific city or place where the subject was born or lived.
         Respond with only that place name, nothing else." — this returns a
         short clean value like "[City], [Region]" rather than a paragraph
         listing every place mentioned anywhere in the source.
      c. Use that extracted value as the actual web-search query
         ($task_N.summary) for the SECOND part of the goal only, so the
         search is properly targeted at the real place(s) instead of
         guessing blind in parallel. This step, and step (d) below, ARE a
         normal web-search+crawl pair per rule 7/9 -- the "no web-search"
         instruction in step (a) applies ONLY to the first part of the
         chain, not this second part.
      d. Continue with fetch + summarise as normal per rule 9, depends_on
         chained through from the search step. Give this final summarise task
         an EXPLICIT focus, not just its task_description -- something like
         "The text below is about $task_N.summary, a place. Assess its
         quality, reputation, and characteristics as a place. This place is
         significant only as the historical location of a separate
         individual -- do not expect or remark on that individual's presence
         in this text; just assess the place itself." Without this, the task
         auto-inherits a focus built from its own task_description (which
         likely still names the person), and the summarizer may treat "the
         person isn't mentioned in this content" as something noteworthy to
         flag, even though that's expected and not actually a gap.
      e. MANDATORY, NOT OPTIONAL: ALSO build a second, independent
         summarise-agent task depending ONLY on the knowledge-retrieve task
         from step (a) (depends_on: the knowledge-retrieve task's id ONLY —
         NOT the second part's chain), reporting what the knowledge base
         itself says in answer to the first part of the goal. This is a
         SEPARATE leaf from step (d)'s final task, left uncombined per rule
         8 — the platform's auto-combine step joins both leaves
         automatically. Skipping this step means the knowledge base's own
         answer never surfaces in the final response at all, even though a
         knowledge-retrieve task was built and ran — do not build the
         knowledge-retrieve task in step (a) without ALSO building this leaf
         to report its findings.
      f. If the goal has only ONE part with no such downstream dependency,
         the simpler independent-parallel-branch pattern earlier in this rule
         still applies — don't build the extraction step unnecessarily.

11. Coordinate multi-entity goals with a cross-entity question:
    Rule 7 covers goals where a LATER part depends on an entity only
    resolved by an EARLIER part (sequential dependency). This rule covers a
    DIFFERENT shape: the goal names two or more entities UP FRONT that are
    each independently searchable (neither needs the other resolved first),
    e.g. "how are Index A and Index B doing", "compare Company A and
    Company B's latest earnings". Distinguish two cases:

    CRITICAL PRECONDITION before applying this rule at all: it only applies
    when the goal itself NAMES the specific entities (real proper nouns the
    user already knows, e.g. "Index A and Index B"). It does NOT apply to
    "N best/top-rated UNNAMED items" goals (e.g. "the 5 best noodle stalls
    in a hawker centre, and what makes them stand out") -- there, the N
    items' identities are completely unknown upfront and must be DISCOVERED
    from a single ranking source, not independently searched for one at a
    time. Building N separate branches with ordinal-labeled queries like
    "second best noodle stall", "third best noodle stall", "fourth best
    noodle stall" is WRONG and will return irrelevant noise for every
    ordinal beyond the first or second, because "the Nth best X" is not a
    real, independently searchable concept -- no website is indexed under
    that exact ordinal phrasing, so the search engine falls back to
    unrelated results. For this shape instead: run ONE search with a broad
    query (e.g. "best noodle stalls hawker centre", no ordinal words), ONE
    page-fetch of the top result(s) (list-style "top N" articles naturally
    name several ranked items in a single source), then ONE summarise task
    whose focus asks it to name each of the N items AND what makes each
    stand out, using the list_instruction mechanics (name actual items, not
    just aggregate stats) -- exactly like the single-chain pattern used for
    a plain "top 10 X" ranking request, just with a per-item elaboration
    clause added to the focus. Never spend more than one search/fetch pair
    discovering an unranked top-N set; only escalate to multiple
    search/fetch pairs (per rule 7) if a LATER part of the goal needs fresh
    information about ONE specific item only after it has been identified.

    a. NO cross-entity question attached (e.g. "how are Index A and
       Index B doing today" — just wants each reported on):
       Build a fully separate search -> fetch -> summarise branch per
       entity, each with its OWN focused query naming only that one
       entity — never combine multiple entity names into a single search
       query, since this dilutes results and biases the crawl toward
       whichever entity happens to dominate the top hits. Leave each
       branch's summarise-agent task as an independent leaf per rule 8 —
       do NOT add a task to combine them; the platform's auto-combine
       mechanism (rule 8) joins the leaves automatically.

    b. A cross-entity question IS attached on top (e.g. "...shall I
       go with option A?", "...which is the better choice?", "...and
       is that a good sign overall?"): this question can only be
       answered by reasoning over ALL entities together, so it must NOT
       be answered inside any single per-entity summarise task, and per
       rule 8 you also must NOT chain multiple summarise-agent outputs
       into another summarise-agent task. Instead:
         i.   Build a separate search -> fetch branch per entity, each
              with its own focused single-entity query, exactly as in (a) —
              but do NOT end each branch in its own summarise-agent task.
         ii.  Add exactly ONE final summarise-agent task depending on
              ALL the fetch tasks, with "text" set to the CONCATENATION of
              every branch's raw fetched text (per rule 6's concatenation
              syntax, e.g. "text": "$task_2.text\n\n$task_5.text") — never
              the branches' summaries, since none exist yet in this
              pattern, and never just one branch's text alone.
         iii. Word this final task's task_description and an EXPLICIT
              focus as the FULL evaluative question from the original
              goal, covering every entity by name AND the cross-cutting
              question itself (e.g. "Summarize the current standing of
              Index A and Index B, and provide actionable advice
              on which option to go with based on current
              conditions.") — not a bare per-entity report.
         iv.  This produces a single summarise-agent leaf for the whole
              goal (not one per entity), so the platform's auto-combine
              step has nothing to join — that is correct and expected
              here, since the cross-entity reasoning already happened in
              one call, unlike case (a) where the leaves stay genuinely
              separate.

    The deciding question when a goal names multiple entities: does
    anything in the goal require weighing the entities AGAINST each other
    or reaching ONE decision that spans all of them? If yes, use pattern
    (b). If the goal is just asking to report on each entity with no such
    combined judgement, use pattern (a) and let each stay independent.

"""

GOAL_EXPANDER_RAG_EXAMPLES = """
Example for a goal answerable from internal company knowledge, per rule 10 --
note the knowledge-retrieve branch (tasks 1-2) and the web-search branch
(tasks 3-5) are completely independent of each other (neither depends_on the
other), each ends in its own summarise-agent leaf, and NOTHING combines them
-- the platform's existing auto-combine mechanism (rule 8) picks up both
leaves automatically, exactly as it would for any other multi-part goal
(again, all agent_ids and field names below are PURELY FICTIONAL):
[
  {{
    "task_id": 1,
    "task_description": "Search internal knowledge base for the vacation policy",
    "agent_id": "knowledge-retrieve-agent",
    "inputs": {{"query": "what is the vacation policy"}},
    "depends_on": []
  }},
  {{
    "task_id": 2,
    "task_description": "Report the internal knowledge base's answer on the vacation policy",
    "agent_id": "summarise-agent",
    "inputs": {{
      "text": "$task_1.results[0].text\n$task_1.results[1].text",
      "focus": "Using the excerpt below, answer the original question directly and naturally (e.g. state who/what/where as relevant, in a brief natural sentence) -- unless fallback_recommended is true (current value: $task_1.fallback_recommended), in which case skip that and instead state plainly that the internal knowledge base did not have a confident match for this question, giving the reason: $task_1.fallback_reason."
    }},
    "depends_on": [1]
  }},
  {{
    "task_id": 3,
    "task_description": "Search the web for the vacation policy",
    "agent_id": "search-agent",
    "inputs": {{"query": "company vacation policy"}},
    "depends_on": []
  }},
  {{
    "task_id": 4,
    "task_description": "Fetch the full page for the top web search result",
    "agent_id": "fetch-agent",
    "inputs": {{"url": "$task_3.results[0].url"}},
    "depends_on": [3]
  }},
  {{
    "task_id": 5,
    "task_description": "Summarise the vacation policy from the web search result",
    "agent_id": "summarise-agent",
    "inputs": {{"text": "$task_4.text"}},
    "depends_on": [4]
  }}
]

Example for a DEPENDENT compound question, per rule 10's dependent-chain
sub-pattern -- "where did X work, and how good is that office" needs the
actual office name/location before the web search can be targeted, so task 2
extracts just that name, and task 3's query uses it (not a blind guess at the
original goal text). Task 6 is the independent leaf reporting the knowledge
base's own findings, same as the simple pattern -- tasks 5 and 6 both stay
uncombined leaves, auto-combined by the platform per rule 8 (again, all
agent_ids and field names below are PURELY FICTIONAL):
[
  {{
    "task_id": 1,
    "task_description": "Search internal knowledge base for where the employee worked",
    "agent_id": "knowledge-retrieve-agent",
    "inputs": {{"query": "where did the employee work"}},
    "depends_on": []
  }},
  {{
    "task_id": 2,
    "task_description": "Extract just the office name/location from the knowledge base result",
    "agent_id": "summarise-agent",
    "inputs": {{
      "text": "$task_1.results[0].text",
      "focus": "Identify the single specific office or location mentioned in the text above where the person worked. Respond with only that place name, nothing else."
    }},
    "depends_on": [1]
  }},
  {{
    "task_id": 3,
    "task_description": "Search the web for reviews/quality of the extracted office location",
    "agent_id": "search-agent",
    "inputs": {{"query": "$task_2.summary reviews quality"}},
    "depends_on": [2]
  }},
  {{
    "task_id": 4,
    "task_description": "Fetch the full page for the top result on the office location",
    "agent_id": "fetch-agent",
    "inputs": {{"url": "$task_3.results[0].url"}},
    "depends_on": [3]
  }},
  {{
    "task_id": 5,
    "task_description": "Summarise the quality assessment of the office location",
    "agent_id": "summarise-agent",
    "inputs": {{
      "text": "$task_4.text",
      "focus": "The text below is about $task_2.summary, a place. Assess its quality, reputation, and characteristics as a place. This place is significant only as the historical location of a separate individual -- do not expect or remark on that individual's presence in this text; just assess the place itself."
    }},
    "depends_on": [4]
  }},
  {{
    "task_id": 6,
    "task_description": "Report the internal knowledge base's own findings on where the employee worked",
    "agent_id": "summarise-agent",
    "inputs": {{
      "text": "$task_1.results[0].text\n$task_1.results[1].text",
      "focus": "Using the excerpt below, answer the original question directly and naturally (e.g. state who/what/where as relevant, in a brief natural sentence) -- unless fallback_recommended is true (current value: $task_1.fallback_recommended), in which case skip that and instead state plainly that the internal knowledge base did not have a confident match for this question, giving the reason: $task_1.fallback_reason."
    }},
    "depends_on": [1]
  }}
]
"""

_DEFAULT_GOAL_EXPANDER_PROMPT_TEMPLATE = GOAL_EXPANDER_PROMPT_TEMPLATE


def _build_history_block(history: list) -> str:
    """
    Formats prior turns of THIS chat conversation into a labeled block
    spliced into the Goal Expander prompt right after the current goal (see
    build_expander_prompt), so the model can resolve references like "that",
    "it", "the previous one", "again", "do the same for Y" against what was
    actually asked and answered earlier in the SAME conversation.

    Deliberately framed as read-only context, not as additional work: the
    current goal (quoted just above where this block is spliced in) remains
    the ONLY thing this plan must build tasks for. Each turn is already
    truncated (MAX_HISTORY_GOAL_CHARS / MAX_HISTORY_ANSWER_CHARS) and the
    list is already capped to the last MAX_HISTORY_TURNS entries by
    append_conversation_turn() before it ever reaches here, so no further
    capping happens in this function -- it's purely formatting.
    """
    if not history:
        return ""

    parts = [
        "\nCONVERSATION CONTEXT -- this goal was submitted inside an ongoing "
        "chat thread. The turns below are what the user already asked and "
        "what was already answered earlier in THIS SAME conversation, oldest "
        "first. Use them ONLY to resolve what an ambiguous word or phrase in "
        "the CURRENT goal above refers to (e.g. \"that\", \"it\", \"the previous "
        "one\", \"do the same for Y\", \"summarize it in French instead\", "
        "\"what about tomorrow\"). Do NOT build any task for these earlier "
        "turns themselves, and do NOT treat them as additional instructions "
        "to fulfil now -- build tasks ONLY for the current goal quoted above, "
        "reinterpreted with this context in mind where relevant:\n"
    ]
    for i, turn in enumerate(history, start=1):
        parts.append(f'  Turn {i} -- user asked: "{turn.get("goal", "")}"')
        parts.append(f'  Turn {i} -- answer given: "{turn.get("answer", "")}"\n')

    return "\n".join(parts) + "\n"


GOAL_EXPANDER_MATCH_RULE = """
MATCH. Cross-document matching / reconciliation (data-matching capability):
    If an agent with the data-matching capability is registered, and the
    goal calls for comparing or reconciling qty/amount data ACROSS two or
    more already-available JSON documents, build a single data-matching
    task instead of treating each document as an independent
    summarise/report target.

    JSON-ONLY -- this is a hard precondition, not just a preference:
    matching NEVER triggers for non-JSON documents. A document only
    qualifies as JSON for this rule if it is EITHER:
      - a prior task's extracted-JSON output (e.g. an OCR/extraction task's
        "$task_N.outputs"), which is inherently structured JSON, OR
      - an uploaded file the "0. PRIORITY" rule above explicitly lists as
        having a parsed JSON object available via "$task_N.json".
    Every other uploaded file type (.txt, .md, .pdf, .docx, .csv that
    was NOT run through extraction, or any file the PRIORITY rule only
    lists a "$task_N.text" reference for, with no accompanying
    "$task_N.json" line) does NOT qualify, no matter how directly the goal
    asks to "compare" or "match" them -- there is no structured field-level
    data in a plain-text/prose file for this agent to reconcile. If the
    goal asks to compare documents but fewer than two of them qualify as
    JSON by the above test, do NOT build a data-matching task; fall back to
    handling each document individually per the upload/knowledge-retrieve
    rules (e.g. separate summarise tasks), or per rule 8's combine
    mechanism if the goal also wants them read together narratively.

    a. Detecting the intent -- two cases (still requires the JSON-only
       precondition above to actually be met before either case applies):
       DIRECT: the goal explicitly says "match", "compare", "reconcile",
       "cross-check", "verify ... against ...", "3-way match", or asks
       whether documents "line up" / "agree" / "tally".
       INDIRECT: the goal doesn't use any of those words, but names or
       implies two-or-more documents being checked AS A SET rather than
       described individually -- e.g. "check this invoice and PO", "does
       the delivery note match what was billed", "confirm the amounts are
       right across these files". The signal is whether the goal treats
       the documents relationally (checking one against another) rather
       than asking about each document's own content in isolation.

       CRITICAL DISAMBIGUATOR: do NOT build a data-matching task just
       because two or more documents happen to be attached/available. If
       the goal only asks about ONE document's own content (e.g.
       "summarize this invoice", "what's the total on the PO") while a
       second, unrelated file is merely also present, handle that document
       normally per the upload/knowledge-retrieve rules above and do NOT
       invoke matching. Matching only applies when the goal itself asks a
       relational question spanning the documents.

    b. Building the task:
       Set "documents" to a JSON array with ONE ELEMENT PER DOCUMENT, each
       element a WHOLE-STRING $task_N reference to that document's parsed
       JSON per the JSON-ONLY test above -- "$task_N.outputs" for a prior
       extraction task, or "$task_N.json" (NEVER "$task_N.text") for a
       directly-uploaded JSON file -- never a string built by concatenating
       documents into one field, and never the entire completed-tasks
       object. depends_on must list every task_id referenced in
       "documents".

    c. Reference-document ordering -- this determines which document the
       agent treats as ground truth for the comparison; getting the order
       wrong silently produces a comparison from the wrong direction
       rather than an error, so this needs a deliberate choice, not the
       order documents happen to appear in the goal text:
         - If the goal names or implies which document is being verified
           AGAINST another (e.g. "check the invoice against the PO" --
           the PO is the commitment being verified against, so the PO
           goes first), put that document's task_id first in the array.
         - In an Invoice / PO / Delivery Note three-way match with no
           explicit direction stated, default to PO first (the order
           commitment), then Delivery Note, then Invoice.
         - If the goal gives no basis to decide direction at all, default
           to the order the documents were extracted/uploaded (i.e. the
           lowest task_id first) and note this default doesn't change
           whether mismatches are DETECTED -- only which side of the
           comparison output each field is labeled on.

    d. Leave the data-matching task as a leaf per rule 8 -- do not chain
       its output into a further summarise-agent task unless the goal
       explicitly also asks for a written narrative explanation of the
       comparison result, not just the comparison itself.
"""

GOAL_EXPANDER_MATCH_EXAMPLES = """
Example for "Check if this invoice matches the PO and delivery note" with
three prior extraction tasks already producing JSON output (task_ids 1, 2, 3
for PO, Delivery Note, and Invoice respectively -- PO first per the 3-way
match default in rule MATCH, since no explicit direction was stated) (again,
agent_ids are PURELY FICTIONAL, use the real registered one):
[
  {
    "task_id": 4,
    "task_description": "Match PO, delivery note, and invoice line items and totals",
    "agent_id": "match-agent",
    "inputs": {
      "documents": ["$task_1.outputs", "$task_2.outputs", "$task_3.outputs"]
    },
    "depends_on": [1, 2, 3]
  }
]

Example for a goal that only asks about ONE of two attached files, per the
CRITICAL DISAMBIGUATOR in rule MATCH -- note there is NO data-matching task
here even though a second file is available, because the goal never asks a
relational question:
Goal: "What's the total on the PO?" (PO and Invoice both uploaded)
[
  {
    "task_id": 1,
    "task_description": "Answer the question using the uploaded PO's content",
    "agent_id": "summarise-agent",
    "inputs": {
      "text": "$task_101.text",
      "focus": "State the total amount on the PO."
    },
    "depends_on": []
  }
]

Example for a goal that directly asks to "compare" two attached files, per
the JSON-ONLY precondition in rule MATCH -- note there is STILL NO
data-matching task here, because neither uploaded file has a "$task_N.json"
line in the "0. PRIORITY" rule above (both are .docx, so only
"$task_N.text" -- raw prose -- is available, never "$task_N.json"). Handle
each individually instead:
Goal: "Compare these two contracts and tell me what's different"
(contract_v1.docx and contract_v2.docx both uploaded, neither is JSON)
[
  {
    "task_id": 1,
    "task_description": "Summarize contract_v1.docx",
    "agent_id": "summarise-agent",
    "inputs": {
      "text": "$task_101.text",
      "focus": "Summarize the key terms of this contract."
    },
    "depends_on": []
  },
  {
    "task_id": 2,
    "task_description": "Summarize contract_v2.docx",
    "agent_id": "summarise-agent",
    "inputs": {
      "text": "$task_102.text",
      "focus": "Summarize the key terms of this contract."
    },
    "depends_on": []
  }
]
"""


def build_expander_prompt(goal: str, agents: list, doc_refs: list = None, session_id: str = None,
                          history: list = None) -> str:
    agent_descriptions = []
    for a in agents:
        agent_descriptions.append(
            f'- agent_id: {a["agent_id"]}\n'
            f'  display_name: {a["display_name"]}\n'
            f'  description: {a["description"]}\n'
            f'  capabilities: {a["capabilities"]}\n'
            f'  input_schema keys: {[k for k in a["input_schema"].keys() if k != "max_attempts"]}\n'
            f'  output_schema keys: {list(a.get("output_schema", {}).get("properties", a.get("output_schema", {})).keys())}'
        )
    agents_block = "\n".join(agent_descriptions)

    # Only include the RAG rule/examples when a knowledge-retrieve agent is
    # actually registered -- otherwise every goal (including plain conversion
    # tasks, simple summaries, etc.) would carry that extra ~150 lines
    # of unrelated instructions for no reason, competing for the model's
    # attention on goals it can never apply to. Cheap to check: `agents` is
    # already the exact list about to be shown in agents_block.
    has_rag_agent = any("knowledge-retrieve" in a.get("capabilities", []) for a in agents)
    rag_rules = GOAL_EXPANDER_RAG_RULE if has_rag_agent else ""
    rag_examples = GOAL_EXPANDER_RAG_EXAMPLES if has_rag_agent else ""

    # Same "only include when relevant" principle as has_rag_agent above --
    # the data-matching rule only applies when a data-matching-capable
    # agent is actually registered.
    has_match_agent = any("data-matching" in a.get("capabilities", []) for a in agents)
    match_rules = GOAL_EXPANDER_MATCH_RULE if has_match_agent else ""
    match_examples = GOAL_EXPANDER_MATCH_EXAMPLES if has_match_agent else ""

    # Same "only include when relevant" principle -- the upload PRIORITY
    # rule only applies when files were actually attached to THIS goal.
    # Placed BEFORE the RAG rule/examples (not after, as originally tried)
    # -- appending it after rule 10 meant the model had already "locked
    # in" the knowledge-retrieve pattern by the time it reached the
    # override, and empirically kept building a knowledge-retrieve task
    # anyway despite the override text. Reading the override FIRST, before
    # the ~150-line RAG block it's meant to pre-empt, held up better.
    upload_rules = _build_upload_rule(doc_refs, session_id) if doc_refs else ""
    upload_examples = _build_upload_example(doc_refs) if doc_refs else ""

    extra_rules = upload_rules + rag_rules + match_rules
    extra_examples = upload_examples + rag_examples + match_examples

    history_block = _build_history_block(history) if history else ""
    experience_block = _build_experience_block(goal)
    print(f"[FEEDBACK] goal={goal!r} experience_block_chars={len(experience_block)} has_experience={bool(experience_block)}")

    return GOAL_EXPANDER_PROMPT_TEMPLATE.format(
        goal=goal,
        agents_block=agents_block,
        extra_rules=extra_rules,
        extra_examples=extra_examples,
        history_block=history_block,
        experience_block=experience_block,
    )


# Matches a $task_N.path reference that may appear ANYWHERE inside a string --
# not just as the entire value. This is what lets the Goal Expander compose
# inputs like "reviews of $task_2.summary attractions" (a literal query with
# an upstream value embedded in it) or "$task_2.text\n\n$task_5.text" (two
# upstream texts concatenated into one field), instead of being restricted to
# a task's input being ONE bare reference and nothing else. The path group is
# optional so a bare "$task_1" (whole outputs object, no field) still matches.
_TASK_REF_TOKEN_RE = _re.compile(r"\$task_(\d+)((?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)*)")


def _validate_ref_string(val: str, task_id, by_id: dict, agent_index: dict, errors: list) -> None:
    """Validates every $task_N.path token found inside a single string value."""
    if not isinstance(val, str) or "$task_" not in val:
        return

    matches = list(_TASK_REF_TOKEN_RE.finditer(val))
    if not matches:
        errors.append(f"Task {task_id}: malformed reference in {val!r}")
        return

    for m in matches:
        ref_task_id = int(m.group(1))
        field = m.group(2)[1:] if m.group(2) else ""

        if ref_task_id >= UPLOAD_TASK_ID_BASE:
            # Preseeded upload reference (see preseed_uploaded_docs) --
            # not a real plan task, so there's no upstream agent/schema to
            # check it against. Shape is fixed and controlled by us
            # (text/filename/doc_id), so always valid.
            continue

        upstream_task = by_id.get(ref_task_id)
        if not upstream_task:
            errors.append(f"Task {task_id}: references unknown task_id {ref_task_id}")
            continue

        upstream_agent = agent_index.get(upstream_task["agent_id"])
        if not upstream_agent:
            errors.append(f"Task {task_id}: upstream agent "
                          f"{upstream_task['agent_id']!r} not in registry")
            continue

        if not field:
            # Bare $task_N reference (no .path) -- refers to the
            # whole outputs object, nothing further to validate.
            continue

        out_schema = upstream_agent.get("output_schema", {})
        valid_keys = set(out_schema.get("properties", out_schema).keys())
        base_key = _re.split(r"[\[\.]", field)[0]
        if base_key not in valid_keys:
            errors.append(
                f"Task {task_id}: '${ref_task_id}.{field}' is not a valid "
                f"output field of agent {upstream_task['agent_id']!r}. "
                f"Valid fields: {sorted(valid_keys)}"
            )


def validate_task_references(tasks: list, agent_index: dict) -> list:
    """
    Post-generation guard: reject any $task_N.field reference (whether it's
    the whole input value, embedded inside a larger literal string, or one
    element of a literal JSON array of references -- e.g. a "summaries"
    input built as ["$task_3.summary", "$task_6.summary"]) whose field is
    not an actual key in the upstream agent's real output_schema.
    """
    by_id = {t["task_id"]: t for t in tasks}
    errors = []

    for task in tasks:
        for key, val in task.get("inputs", {}).items():
            if isinstance(val, list):
                for item in val:
                    _validate_ref_string(item, task["task_id"], by_id, agent_index, errors)
            else:
                _validate_ref_string(val, task["task_id"], by_id, agent_index, errors)

    return errors


_JSON_CONTROL_CHAR_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _repair_json_control_chars(raw: str) -> str:
    """
    One-pass repair for the single most common way the Goal Expander's JSON
    output fails to parse: a literal, unescaped control character (almost
    always a raw newline) sitting inside a JSON string value -- e.g. rule
    6's blank-line concatenation token written as an actual line break
    instead of the escaped "\\n\\n" sequence. Standard JSON forbids literal
    control characters inside strings, so json.loads() raises "Invalid
    control character" and the ENTIRE plan is discarded even though every
    other task was fine -- this is a purely mechanical formatting slip, not
    a content problem, and is fully recoverable by escaping the character
    in place.
    Walks the raw text tracking whether the cursor is inside a string
    literal (toggling on each unescaped double-quote) and escapes any raw
    control character found there; everything outside string literals
    (the actual JSON structure -- braces, commas, real newlines between
    tokens) is left completely untouched.
    """
    out = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch in _JSON_CONTROL_CHAR_ESCAPES:
                out.append(_JSON_CONTROL_CHAR_ESCAPES[ch])
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _drop_unreferenced_dependencies(tasks: list) -> list:
    """
    Deterministic safety net for cross-document (and generally, cross-branch)
    over-serialization: this system's whole task-DAG design assumes
    depends_on is a MIRROR of actual $task_N references in a task's own
    inputs (see the Goal Expander prompt's own framing -- "$task_N.summary
    creates a depends_on edge" -- and validate_task_references(), which
    checks those same tokens). Nothing enforces the reverse direction
    though: nothing stops the Goal Expander from adding a depends_on entry
    for a task_id that ISN'T referenced anywhere in this task's inputs.

    Observed failure shape: multiple independent document-extraction tasks
    (e.g. one JSON-extraction task per uploaded PO/DO/GRN, each reading its
    own $task_N.text with an empty depends_on) get chained by the Goal
    Expander into task_id order -- task 3 depends_on [2], task 4 depends_on
    [3], etc -- even though task 3's inputs never reference anything from
    task 2. Nothing about the goal required that order; it's the model
    defaulting to "process documents one after another" phrasing, or
    listing task_ids in ascending order out of habit. Once wave-based
    execute_tasks() sees that depends_on edge, though, it treats it as a
    real gate and runs the whole batch strictly one at a time no matter how
    many workers MAX_PARALLEL_AGENTS allows -- silently re-serializing work
    the plan didn't actually need serialized, and stretching an N-document
    batch's wall-clock time from ~max(single-doc time) toward N times that.

    Fix mirrors _ensure_upload_content_used's philosophy: don't re-tune the
    prompt (already unreliable per that function's docstring), check the
    LLM's ACTUAL output mechanically instead. For every task, keep only the
    depends_on entries whose task_id is actually mentioned via a $task_N
    token somewhere in that task's own inputs; drop the rest. A depends_on
    entry backed by a real reference (the normal case -- e.g. a combine-
    style task reading $task_2.summary) is left completely untouched, so
    genuinely dependent chains are unaffected.
    """
    def _referenced_ids(task) -> set:
        ids = set()
        for val in task.get("inputs", {}).values():
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, str) and "$task_" in item:
                    ids.update(int(m.group(1)) for m in _TASK_REF_TOKEN_RE.finditer(item))
        return ids

    for task in tasks:
        deps = task.get("depends_on", []) or []
        if not deps:
            continue
        referenced = _referenced_ids(task)
        kept = [d for d in deps if d in referenced]
        dropped = [d for d in deps if d not in referenced]
        if dropped:
            print(f"[PLAN] task={task['task_id']} dropped unreferenced depends_on={dropped} "
                  f"(no $task_N mention in its own inputs) -- now runs in parallel with those tasks",
                  flush=True)
            task["depends_on"] = kept

    return tasks


def _drop_web_search_for_financial_json_upload(tasks: list, doc_refs: list, goal: str, agents: list) -> list:
    """
    Deterministic safety net for ONE specific misfire: a financial
    document (invoice/PO/delivery order/GRN/etc) is uploaded for JSON
    extraction, and the Goal Expander ALSO builds an independent
    web-search -> webcrawl -> summarise chain alongside the upload-answer
    task -- e.g. goal="process" with an invoice attached produces a
    spurious "search the web for the meaning of the term 'process'"
    branch on top of the correct JSON-extraction task.

    Deliberately scoped to ONLY this case -- a financial-document upload
    where JSON output applies -- not short/ambiguous goals in general.
    Web-search/webcrawl/summarise chains are a normal, wanted part of
    plenty of other goals (including goals WITH an upload attached, per
    rule "0. PRIORITY".d: "if the goal has ANOTHER part unrelated to the
    uploaded file's content... handle that part normally"), so this must
    NOT touch those. The trigger mirrors the SAME two checks the
    dispatch-time auto-JSON logic already uses (see
    _looks_like_financial_document / _wants_json_output /
    _looks_like_narrow_factual_question and their use in
    _auto_json_from_upload above) so "does this goal+upload combination
    end up as a JSON-extraction answer" is judged identically in both
    places, just once at plan time and once at dispatch time:
      - at least one attached upload's own text looks like a financial
        document, AND
      - the goal either explicitly asks for JSON, or would hit the
        content-based auto-JSON trigger (i.e. is NOT a narrow single-fact
        question -- "what's the PO number" on an invoice should still be
        free to also web-search an unrelated second part, if any).

    When both hold: for a financial-document-JSON goal, the answer comes
    entirely from the document -- there is no legitimate reason for an
    independent web-search branch to exist at all, so every INDEPENDENT
    (depends_on: []) web-search task is dropped, cascading through
    anything (webcrawl/summarise) that exists only to serve it, same
    cascade logic as elsewhere in this module. Any web-search task that
    ISN'T independent (i.e. is itself downstream of something else,
    which shouldn't happen for a top-of-chain agent but is checked
    anyway) is left alone rather than guessed about.
    """
    if not doc_refs:
        return tasks

    financial_upload = any(
        d.get("full_text") and _looks_like_financial_document(d["full_text"])
        for d in doc_refs
    )
    if not financial_upload:
        return tasks

    json_intent = _wants_json_output(goal) or not _looks_like_narrow_factual_question(goal)
    if not json_intent:
        return tasks

    agent_caps = {a["agent_id"]: set(a.get("capabilities", [])) for a in agents}

    def _is_websearch(task) -> bool:
        return "web-search" in agent_caps.get(task.get("agent_id"), set())

    suspect_ids = {
        t["task_id"] for t in tasks
        if _is_websearch(t) and not (t.get("depends_on") or [])
    }
    if not suspect_ids:
        return tasks

    # Cascade: a task whose EVERY depends_on entry is already dropped has
    # no remaining purpose (it only existed to consume the branch we just
    # removed) -- drop it too. Anything with even one surviving dependency
    # is left alone, so a task that happens to ALSO draw on a real branch
    # is never touched.
    dropped = set(suspect_ids)
    changed = True
    while changed:
        changed = False
        for t in tasks:
            if t["task_id"] in dropped:
                continue
            deps = t.get("depends_on") or []
            if deps and all(d in dropped for d in deps):
                dropped.add(t["task_id"])
                changed = True

    kept = [t for t in tasks if t["task_id"] not in dropped]
    print(f"[PLAN] dropped web-search branch for financial-doc JSON goal {goal!r}: "
          f"task_ids={sorted(dropped)}", flush=True)
    return _drop_unreferenced_dependencies(kept)


def _ensure_upload_content_used(tasks: list, doc_refs: list, agents: list, goal: str, session_id: str = None,
                                 force_doc_ids: set = None) -> list:
    """
    Deterministic safety net, added after TWO rounds of prompt engineering
    both failed to reliably get the Goal Expander to use uploaded content
    instead of searching for it:
      1. A dedicated "0. PRIORITY" rule + worked example (equal structural
         weight to the RAG rule) -- didn't help.
      2. Reordering it BEFORE rule 10, and fixing a numbering collision
         with GOAL_EXPANDER_RAG_RULE's own internal "11." sub-rule --
         still didn't help. The model reliably rebuilds the standard
         knowledge-retrieve branch regardless of how the override is
         phrased or positioned (plausibly related to this model being
         called with enable_thinking=False -- suppressing a
         knowledge-retrieve pattern it's already deep into completing may
         need a deliberate reasoning step this call doesn't get).

    Rather than keep tuning the prompt, this checks the LLM's ACTUAL
    output mechanically: if inline-eligible uploaded content exists but
    NOTHING in the returned plan references it (no $task_N for N in the
    reserved UPLOAD_TASK_ID_BASE+ range, anywhere, in any task's inputs),
    deterministically APPEND a guaranteed leaf task that answers directly
    from the uploaded content -- rather than replacing or removing
    whatever the LLM already built, since that might legitimately also be
    needed (e.g. a goal genuinely requiring both the attachment AND a
    live web search for an unrelated part). The appended leaf is picked
    up by the normal auto-combine step automatically, since this runs
    BEFORE _inject_combine_task in expand_goal().

    If the LLM DID reference the upload correctly, this is a no-op --
    nothing is added, nothing is changed.

    LARGE-FILE EXTENSION: the above only ever covered inline-eligible
    (small, <=MAX_INLINE_UPLOAD_CHARS) uploads. Large uploads (ingested
    into the knowledge base, retrieved via session-scoped fetch_all) had
    no equivalent backstop -- confirmed missing when a bare, unadorned
    goal like "compare" (no "attached"/"the two" wording to anchor the
    model toward the RAG branch) produced a plan with ZERO
    knowledge-retrieve-agent tasks at all, even with two large PDFs
    attached to the request. The root cause isn't file size specifically
    -- it's that the plan doesn't reference the uploads AT ALL, which can
    happen for large files too, just via a different failure path (the
    model choosing an entirely different branch, e.g. web search on the
    goal's literal words) rather than the small-file failure path (model
    rebuilding a knowledge-retrieve task despite being told the content's
    already inline). Same principle, so same fix shape: check the actual
    output, and if none of it does a session-scoped fetch_all retrieval
    of these uploads, append a guaranteed retrieve+answer pair.

    SNOWBALLING FIX: `force_doc_ids`, when given (as a set of doc_id
    strings), restricts every check below to only the doc(s) in that
    set -- any doc in `doc_refs` whose doc_id ISN'T in `force_doc_ids` is
    excluded from both the inline and large-doc backstop entirely, as if
    it weren't in doc_refs at all for the purposes of this function. Left
    as None (all docs eligible, matching every prior behavior) for any
    caller that doesn't pass it.

    Why: `doc_refs` here is typically the FULL conversation-wide list of
    every file ever uploaded (see get_conversation_docs), kept around
    on purpose so a genuine follow-up ("summarize that PDF again") still
    has the file available. But this function's backstop doesn't
    distinguish "uploaded this turn" from "uploaded three goals ago" --
    unrestricted, it force-appends a task answering the CURRENT goal
    from an OLD upload's content the instant the model's own plan
    doesn't reference it, even when the current goal has nothing to do
    with that old file. Confirmed on production data: an unrelated
    follow-up goal, with no new upload of its own, got a task force-
    appended that answered it from an invoice attached to an earlier,
    unrelated goal in the same conversation. The /goal worker passes
    only this turn's freshly-ingested doc_id(s) here, so the forced
    "you WILL use this content" guarantee applies only to what was just
    handed to the model, not to everything it's ever seen in this
    thread. Older docs remain visible to the model via the normal
    prompt-level upload rule (build_expander_prompt) and can still be
    used naturally on a real follow-up -- they just aren't mechanically
    forced in when the model decides (rightly, on an unrelated goal)
    not to touch them.
    """
    if force_doc_ids is not None:
        doc_refs = [d for d in doc_refs if d.get("doc_id") in force_doc_ids]

    inline_task_ids = []
    inline_doc_by_tid = {}
    n = 0
    for d in doc_refs:
        text = d.get("full_text", "")
        if text and len(text) <= MAX_INLINE_UPLOAD_CHARS:
            tid = UPLOAD_TASK_ID_BASE + n
            inline_task_ids.append(tid)
            inline_doc_by_tid[tid] = d
            n += 1

    # Each large (non-inline) doc gets tracked individually by doc_id --
    # NOT just as a single session-wide "large_file_present" flag. That
    # session-wide check was the root of a real bug: with two large files
    # in one session, a fetch_all call scoped by session_id ALONE returns
    # BOTH files merged into a single full_text blob (confirmed via
    # retrieve_agent.py's fetch_all where-filter only ever including
    # session_id, never doc_id). So "compare the two" plans that build
    # one retrieve task per file end up feeding the identical merged
    # blob into both summarize tasks -- the "comparison" silently
    # compares the same content against itself. The fix has two halves:
    # (1) retrieve_agent.py now accepts an optional doc_id to scope
    # fetch_all to one specific document, and (2) this function checks
    # per-document usage so it can inject a properly-scoped retrieve
    # task for whichever specific large file(s) the LLM's plan missed,
    # rather than treating any one fetch_all call as covering all of them.
    large_docs = [
        d for d in doc_refs
        if d.get("full_text") and len(d["full_text"]) > MAX_INLINE_UPLOAD_CHARS
    ]

    if not inline_task_ids and not large_docs:
        return tasks

    def _inline_ids_in_text(t: dict) -> set:
        """
        Distinct inline upload task_ids referenced inside THIS task's
        own `text` input alone. Deliberately scoped to `text` only (not
        every input field, unlike the old whole-session scan this
        replaces) -- a data-matching task legitimately references
        several uploads at once via its own `documents` input, and that
        is correct, expected behavior, not the bug this function guards
        against. Only `text` (the field byoa-summarize-agent actually
        extracts from) can exhibit the failure mode below.
        """
        raw_val = (t.get("inputs", {}) or {}).get("text")
        if not isinstance(raw_val, str):
            return set()
        return {tid for tid in inline_task_ids if f"$task_{tid}" in raw_val}

    # A task whose `text` input contains TWO OR MORE distinct inline
    # upload ids means two (or more) separately-uploaded files' raw
    # content got concatenated into a SINGLE extraction call. Confirmed
    # in production as the root cause of "3 files uploaded, only 1 JSON
    # came back, every field doubled up as field/field_2/field_3" --
    # byoa-summarize-agent's label-value backfill dedups the resulting
    # cross-document field collisions by suffixing _2, _3, ... rather
    # than recognizing them as two different source documents, since it
    # has no way to know its single `text` input wasn't really one
    # document. There is no legitimate case where merging separate
    # uploaded files' text into one extraction call is correct -- every
    # properly-built per-file task references exactly one upload id --
    # so any task matching this is dropped outright rather than left to
    # run alongside the corrected per-file tasks injected below (leaving
    # it in would still hand back an extra, wrong, merged JSON on top of
    # the correct individual ones).
    if len(inline_task_ids) > 1:
        _merged_upload_tasks = [t for t in tasks if len(_inline_ids_in_text(t)) > 1]
        if _merged_upload_tasks:
            tasks = [t for t in tasks if t not in _merged_upload_tasks]

    def _inline_id_used_anywhere(tid: int) -> bool:
        """
        True if this inline upload id is referenced in ANY input field of
        ANY surviving task -- not just `text`. This is the "was this file
        actually used by the plan at all" question, and it must accept
        every legitimate way a task can reference an upload, including a
        data-matching task's `documents` array (which, BY DESIGN, lists
        several uploads in one field -- see AUTO_APPROVE_CAPABILITIES /
        _is_auto_approve_eligible: a pure-matching plan must be able to
        pass this check without also getting extra summarize tasks
        force-injected, or the plan stops being pure-matching and loses
        its auto-approve eligibility). Confirmed as a real regression:
        scoping this to `text` only (as the merge-detection check
        correctly does, see _inline_ids_in_text above) meant every
        inline doc used ONLY via a matching task's `documents` field
        read as "never used", forcing spurious "Answer the goal
        directly..." summarize tasks onto an otherwise-pure matching
        plan and silently disabling auto-approve for it.
        """
        for t in tasks:
            for v in (t.get("inputs", {}) or {}).values():
                for item in (v if isinstance(v, list) else [v]):
                    if isinstance(item, str) and f"$task_{tid}" in item:
                        return True
        return False

    def _doc_already_used(doc_id) -> bool:
        if not session_id:
            return False
        for t in tasks:
            inputs = t.get("inputs", {})
            if inputs.get("session_id") != session_id or inputs.get("fetch_all") is not True:
                continue
            task_doc_id = inputs.get("doc_id")
            # A single-large-file session never needed doc_id at all
            # (nothing else in the session to disambiguate from), so a
            # bare session_id+fetch_all call still counts as covering
            # that one document. Once there's more than one large file,
            # only a call whose doc_id matches THIS document counts.
            if len(large_docs) == 1 or task_doc_id == doc_id:
                return True
        return False

    # PER-DOCUMENT, not session-wide: the old `_references_inline()` asked
    # "is ANY inline upload referenced ANYWHERE", so as soon as the LLM's
    # plan handled even ONE of several uploaded files correctly, this
    # whole backstop went quiet for the OTHERS too -- a plan covering 1 of
    # 3 uploaded files (or covering all 3 in one merged task, now dropped
    # above) looked identical to one covering all 3 properly. Checking
    # each inline id individually means each file gets its own guaranteed
    # task regardless of what happened to the others -- while still
    # accepting ANY legitimate reference (text OR a matching task's
    # documents array), not just a lone `text` reference (see
    # _inline_id_used_anywhere's docstring for why that distinction
    # matters for matching goals specifically).
    unreferenced_inline_ids = [tid for tid in inline_task_ids if not _inline_id_used_anywhere(tid)]
    unreferenced_large_docs = [d for d in large_docs if not _doc_already_used(d.get("doc_id"))]

    if not unreferenced_inline_ids and not unreferenced_large_docs:
        return tasks

    # MATCH-TASK REPAIR: confirmed on production data -- with exactly 4
    # inline JSON documents attached to a matching goal, the Goal Expander
    # reliably writes a data-matching task whose "documents" array is
    # missing ONE of the 4 (never a fixed one -- position varies across
    # otherwise-identical requests, so this is a genuine LLM reliability
    # limit on this model, not something a prompt reword fixes for good).
    # Reproduced with a bare, direct call, no external caller involved.
    #
    # Left alone, the dropped document falls through to the generic
    # "answer the goal directly" backstop below, which hands it to
    # byoa-summarize-agent instead. That agent's normal pipeline (this is
    # the SAME agent that owns AP document intake/extraction/submission,
    # not a plain summarizer) can make a live outward call as part of its
    # ordinary processing -- which is never appropriate for what should be
    # a read-only matching request, and confirmed as the trigger for a
    # runaway resubmission loop on at least one caller's system.
    #
    # Since this is a JSON document dropped from an array the model
    # itself already tried to build, the correct repair is mechanical:
    # put it back in that same array, as $task_N.json, rather than routing
    # it to a different agent entirely. Only applies when (a) the plan
    # already contains exactly one data-matching task, and (b) the
    # dropped document is_json (has a parsed JSON object available) --
    # anything else falls through to the existing backstop below,
    # unchanged.
    match_agent_ids = {a["agent_id"] for a in agents if "data-matching" in a.get("capabilities", [])}
    match_tasks = [t for t in tasks if t.get("agent_id") in match_agent_ids]
    if len(match_tasks) == 1 and unreferenced_inline_ids:
        match_task = match_tasks[0]
        documents = match_task.get("inputs", {}).get("documents")
        if isinstance(documents, list):
            still_unreferenced = []
            for tid in unreferenced_inline_ids:
                d = inline_doc_by_tid.get(tid)
                if d is not None and d.get("is_json"):
                    documents.append(f"$task_{tid}.json")
                else:
                    still_unreferenced.append(tid)
            unreferenced_inline_ids = still_unreferenced

    if not unreferenced_inline_ids and not unreferenced_large_docs:
        return tasks

    summarize_agent = next((a for a in agents if "text-summarize" in a.get("capabilities", [])), None)
    if not summarize_agent:
        return tasks  # nothing sensible to inject without a summarizer registered

    next_id = max([t["task_id"] for t in tasks], default=0) + 1

    if unreferenced_inline_ids:
        for tid in unreferenced_inline_ids:
            tasks.append({
                "task_id": next_id,
                "task_description": "Answer the goal directly using the uploaded file's content",
                "agent_id": summarize_agent["agent_id"],
                "inputs": {
                    "text": f"$task_{tid}.text",
                    "focus": f"Using the excerpt below, answer this question directly and naturally: {goal}",
                },
                "depends_on": [],
            })
            next_id += 1

    if unreferenced_large_docs and session_id:
        retrieve_agent = next((a for a in agents if "knowledge-retrieve" in a.get("capabilities", [])), None)
        if retrieve_agent:
            for d in unreferenced_large_docs:
                retrieve_id = next_id
                filename = d.get("filename") or d.get("original_filename") or "the uploaded document"
                retrieve_inputs = {
                    "query": "document content",
                    "session_id": session_id,
                    "fetch_all": True,
                }
                # Only stamp doc_id when there's more than one large file --
                # matches the single-file exemption in _doc_already_used
                # above, and avoids sending a doc_id the retrieve agent
                # doesn't need (and can't hurt, but keep the two paths
                # symmetric rather than always including it).
                if len(large_docs) > 1:
                    retrieve_inputs["doc_id"] = d.get("doc_id")
                tasks.append({
                    "task_id": retrieve_id,
                    "task_description": f"Retrieve {filename}'s content (whole-document, backstop-injected)",
                    "agent_id": retrieve_agent["agent_id"],
                    "inputs": retrieve_inputs,
                    "depends_on": [],
                })
                next_id += 1

                tasks.append({
                    "task_id": next_id,
                    "task_description": f"Summarize {filename}'s content (backstop-injected)",
                    "agent_id": summarize_agent["agent_id"],
                    "inputs": {
                        "text": f"$task_{retrieve_id}.full_text",
                        "focus": (
                            f"Provide a detailed summary of {filename}'s main arguments, "
                            f"methodology, and findings, to help with the goal: {goal}"
                        ),
                    },
                    "depends_on": [retrieve_id],
                })
                next_id += 1

    return tasks


def _save_goal_plan(job_id: str, goal: str, tasks: list, conversation_id: str = None) -> None:
    """
    Writes this goal's input text and its expanded task plan to
    GOALS_LOG_DIR as one JSON file per goal -- a plain, human-readable
    audit trail on disk. Called once per goal, right after expand_goal()
    returns successfully (including an empty plan), from the /goal
    background worker.

    Best-effort only: a failure to write this log must never fail the
    goal itself, so any exception here is caught and printed, not raised.
    Filename is f"{unix_timestamp}_{job_id}.json" so files sort
    chronologically by name and never collide (job_id is a fresh uuid per
    /goal call).
    """
    try:
        record = {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "timestamp": time.time(),
            "goal": goal,
            "expanded_tasks": tasks,
        }
        filename = f"{int(record['timestamp'])}_{job_id}.json"
        path = os.path.join(GOALS_LOG_DIR, filename)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
    except Exception as e:
        print(f"WARN: failed to write goal plan log to {GOALS_LOG_DIR}: {e}")


def expand_goal(goal: str, agents: list, doc_refs: list = None, session_id: str = None,
                history: list = None, force_use_doc_ids: set = None) -> list:
    """
    `force_use_doc_ids`, when given, restricts _ensure_upload_content_used's
    deterministic "make the plan actually use the upload" backstop (see
    that function's docstring) to only the doc_id(s) in this set. Left as
    None (the old, unrestricted behavior -- every doc in doc_refs is
    force-checked) for any caller that doesn't pass it explicitly, so this
    is purely additive.

    Why this exists: doc_refs, as built by the /goal worker, is the FULL
    running list of every file ever uploaded anywhere in this
    conversation (see get_conversation_docs), not just this turn's. That
    breadth is intentional -- it's what lets a follow-up like "summarize
    that PDF again" work without re-attaching the file. But
    _ensure_upload_content_used's backstop doesn't know the difference
    between "uploaded this turn" and "uploaded three goals ago" -- it
    forces ANY doc in doc_refs into the plan's output the moment the
    plan's own tasks don't already reference it. Combined with the full
    conversation-wide doc_refs list, that meant an old, unrelated upload
    from an earlier goal was being force-fed into the answer for a
    completely unrelated LATER goal in the same conversation, just
    because it was still sitting in the running doc list ("snowballing"
    into goals it has nothing to do with). Confirmed on production data:
    a goal with no new upload of its own, in a conversation that had an
    invoice attached to an earlier goal, produced a plan force-appended
    with a task answering the new (unrelated) goal from that old
    invoice's text.

    Passing force_use_doc_ids={doc_ids of just this turn's fresh
    uploads} keeps the backstop's original guarantee for what it was
    actually built for (a FRESH upload getting ignored by the plan) while
    letting older, carried-over docs fall back to being naturally
    available -- via the "0. PRIORITY" upload rule / RAG rule in the
    prompt itself, same as before -- without being mechanically forced
    into every subsequent goal regardless of relevance.
    """
    prompt = build_expander_prompt(goal, agents, doc_refs=doc_refs, session_id=session_id, history=history)

    r = req.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=GOAL_EXPANDER_TIMEOUT_SECS,
    )
    r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        tasks = json.loads(raw)
    except json.JSONDecodeError as e:
        # Most common real-world cause: a literal raw newline (or other
        # control char) inside a JSON string value -- e.g. rule 6's
        # concatenation token written with an actual line break instead of
        # the escaped "\n\n" sequence. That's a mechanical formatting slip,
        # not a content error, and the plan is fully recoverable -- repair
        # it once before giving up and discarding an otherwise-valid plan.
        try:
            tasks = json.loads(_repair_json_control_chars(raw))
        except json.JSONDecodeError:
            raise ValueError(f"Goal Expander returned invalid JSON: {e}\nRaw:\n{raw}")

    if not isinstance(tasks, list):
        raise ValueError(f"Expected JSON array, got: {type(tasks)}")

    agent_index = {a["agent_id"]: a for a in agents}
    errors = validate_task_references(tasks, agent_index)
    if errors:
        raise ValueError(
            "Goal Expander produced invalid task references:\n  - " + "\n  - ".join(errors)
        )

    tasks = _drop_unreferenced_dependencies(tasks)
    tasks = _drop_web_search_for_financial_json_upload(tasks, doc_refs or [], goal, agents)
    tasks = _ensure_upload_content_used(tasks, doc_refs or [], agents, goal, session_id=session_id,
                                         force_doc_ids=force_use_doc_ids)
    tasks = _inject_combine_task(tasks, agent_index)

    return tasks


# The combine step is a CONTROLLER-NATIVE step, not a real registered agent --
# there is no HTTP call, no registry entry, no separate service to deploy.
# execute_tasks() special-cases this agent_id and runs _run_combine_task()
# in-process instead of dispatching over the network. It's still emitted in
# the exact same task_result JSON shape as every other task so the frontend
# needs zero special-casing to render it.
COMBINE_AGENT_ID = "byoa-combine-agent"


def _inject_combine_task(tasks: list, agent_index: dict) -> list:
    """
    Deterministic, controller-side post-processing of the Goal Expander's
    plan: find every task whose output nothing else in the plan depends on
    (a "leaf" -- i.e. one of the plan's actual final answer pieces, as
    opposed to an intermediate step like rule 7's dependent-entity
    extraction, whose output feeds a LATER task and is therefore excluded
    here). This is NOT limited to text-summarize agents -- any leaf counts,
    including structured/calculation agents like a conversion agent, a
    live-data agent, or anything else that returns a direct answer rather than prose.
    A goal like "1 Item A to Item B, and explain the gap" has TWO independent leaf
    branches (the live conversion, and the explanatory summary) and both
    need to reach the final answer, not just the one that happens to be a
    text-summarize task.

    If there's more than one such leaf, append one more task that combines
    all of their outputs -- executed entirely in-controller, deterministically,
    with no LLM call, so nothing gets compressed or dropped the way asking a
    model to re-summarise already-summarised text can. If there's zero or
    one, nothing is added: a single leaf output already IS the final answer.

    This runs as part of expand_goal(), so the combine task is already
    present in the plan returned to the frontend for approval -- not
    silently added later at execution time.
    """
    depended_on = set()
    for t in tasks:
        depended_on.update(t.get("depends_on", []) or [])

    tasks_by_id = {t["task_id"]: t for t in tasks}

    def _chain_depth(task_id, _memo={}):
        # Longest path (in task hops) from any root task down to this one.
        # A leaf fed by a short, direct chain (e.g. one knowledge-retrieve
        # lookup -> one summarise call) is typically reporting brief
        # context/identity -- who, what, where. A leaf fed by a long chain
        # (search -> crawl -> summarise -> search -> crawl -> summarise...)
        # is typically an elaborated investigation built ON TOP of that
        # context. Ordering leaves by chain depth (shortest first) means
        # the brief/contextual answer naturally comes first in the final
        # combined text, with the deeper investigation following it --
        # without needing the model to number tasks in any particular way,
        # so this doesn't depend on anything in the model-facing prompt.
        if task_id in _memo:
            return _memo[task_id]
        deps = tasks_by_id.get(task_id, {}).get("depends_on", []) or []
        depth = 1 if not deps else 1 + max(_chain_depth(d) for d in deps)
        _memo[task_id] = depth
        return depth

    leaves = []  # each: {"task_id", "ref", "label"}
    for t in tasks:
        if t["task_id"] in depended_on:
            continue  # something else consumes this task's output -- an intermediate step, not a final answer piece
        agent = agent_index.get(t.get("agent_id"))
        if not agent:
            continue

        # Always reference the WHOLE outputs object, never just
        # "$task_N.summary" directly. A text-summarize agent's `summary`
        # field is only a real prose summary in PROSE mode -- in JSON
        # mode (byoa-summarize-agent's _run_json_mode) it's deliberately
        # just a short placeholder sentence ("JSON extraction complete
        # -- see 'structured' field for the full object.") pointing at
        # the REAL data in `structured`. Whether a given leaf actually
        # ran in JSON mode isn't knowable here at plan-build time (it's
        # decided later, per-task, from the goal/document content at
        # dispatch time -- see _auto_json_from_upload) -- so this can't
        # be branched on here. Instead, the whole object is handed to
        # _run_combine_task, which inspects each entry's actual
        # `output_format`/`structured`/`summary` fields at combine time,
        # once the real values are known, and picks correctly per entry.
        ref = f"$task_{t['task_id']}"

        leaves.append({
            "task_id": t["task_id"],
            "ref": ref,
            "label": t.get("task_description", "") or t["agent_id"],
            "depth": _chain_depth(t["task_id"]),
        })

    # Shortest chain first (brief/contextual leaves before deep-investigation
    # leaves); task_id as a stable tiebreaker when depths match, preserving
    # the previous behavior for goals where this heuristic doesn't apply
    # (e.g. two same-depth independent branches like two named entities).
    leaves.sort(key=lambda leaf: (leaf["depth"], leaf["task_id"]))

    if len(leaves) <= 1:
        return tasks

    next_id = max(t["task_id"] for t in tasks) + 1
    combine_task = {
        "task_id": next_id,
        "task_description": f"Combine the {len(leaves)} results into the final answer",
        "agent_id": COMBINE_AGENT_ID,
        "inputs": {
            "summaries": [leaf["ref"] for leaf in leaves],
            "labels": [leaf["label"] for leaf in leaves],
        },
        "depends_on": [leaf["task_id"] for leaf in leaves],
    }
    return tasks + [combine_task]


# Leading imperative-verb phrasing that shows up in task_description text
# (since it's written as an instruction to an agent, e.g. "Fetch the live
# conversion rate for..."), stripped when a label is actually needed, so it
# reads as a short topic tag instead of an echoed command.
_LABEL_LEADING_VERB_RE = _re.compile(
    r"^(fetch|search for|search|summarize|summarise|identify|retrieve|get|"
    r"find|combine|analyze|analyse|convert|look up|check|gather)\b\s*(the\s+)?",
    _re.IGNORECASE,
)


def _clean_label(label: str) -> str:
    if not label:
        return label
    cleaned = _LABEL_LEADING_VERB_RE.sub("", label).strip()
    cleaned = cleaned.rstrip(".:")
    if not cleaned:
        return label
    return cleaned[0].upper() + cleaned[1:]


def _run_combine_task(resolved_inputs: dict) -> dict:
    """
    The combine step's actual logic -- no LLM call, no network call, purely
    deterministic text assembly:
      - exactly one non-empty entry in -> that entry out, verbatim.
      - more than one -> joined with a blank-line separator. Nothing
        reworded, reordered, or silently dropped.
      - a structured (non-string) entry, e.g. a conversion agent's
        {"input":.., "output":.., "rate":..} -- rendered as readable
        "key: value" pairs and prefixed with a cleaned-up label (e.g. "Live
        conversion result for 1 Item A to Item B: input: 1.0, ...") since raw numbers
        alone don't say what they represent.
      - a prose entry (already a coherent summary paragraph) is left
        UNLABELED -- prefixing it with its task's instruction-phrased
        description (e.g. "Summarize the key factors...: <paragraph>") turns
        an already-coherent paragraph into a labeled fragment and reads like
        a to-do list rather than an answer. Prose already establishes its
        own topic in its own words, so no header is needed.
    Raises ValueError on bad input; the caller wraps that into the same
    failure-shaped task_result every other task uses.
    """
    raw = resolved_inputs.get("summaries")
    if raw is None:
        raise ValueError("inputs.summaries is required")

    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"inputs.summaries must be an array of strings, got {type(raw).__name__}")

    labels = resolved_inputs.get("labels")
    if not isinstance(labels, list):
        labels = []
    if len(labels) < len(raw):
        labels = labels + [None] * (len(raw) - len(labels))

    # Keys that hold large/binary payloads (base64 images, files, etc.) --
    # these must NEVER be string-ified into the combined prose text. They
    # are pulled out separately and returned as their own structured
    # "media" list so the frontend can render them without regexing a
    # giant base64 blob out of a paragraph.
    _MEDIA_KEYS = {"image_base64", "audio_base64", "file_base64", "video_base64"}

    entries = []  # list of (text, needs_label, label)
    media = []    # list of dicts pulled out of structured entries
    for item, label in zip(raw, labels):
        if isinstance(item, str):
            text = item.strip()
            needs_label = False
        elif isinstance(item, dict):
            if item.get("output_format") == "json" and item.get("structured") is not None:
                # JSON-mode extraction result (see summarize_agent.py's
                # _run_json_mode) -- `structured` is the real extracted
                # object; `summary` in this mode is just a placeholder
                # sentence ("JSON extraction complete -- see 'structured'
                # field..."), never the data itself. Always labeled (even
                # if `label` is falsy this still identifies it as JSON)
                # so that when several of these appear side by side --
                # e.g. one extraction task per uploaded file -- each
                # block stays clearly attributed to its own source
                # instead of reading as one anonymous blob repeated N
                # times.
                text = json.dumps(item["structured"], indent=2, ensure_ascii=False)
                needs_label = True
                entries.append((text, needs_label, label))
                continue
            media_keys_present = _MEDIA_KEYS & item.keys()
            if media_keys_present:
                # Split off the media fields entirely; don't stringify them.
                media_entry = {k: v for k, v in item.items() if k in _MEDIA_KEYS or k in
                                ("width", "height", "format", "seed", "num_inference_steps",
                                 "generation_time_secs")}
                if label:
                    media_entry["label"] = _clean_label(label)
                media.append(media_entry)

                remaining = {k: v for k, v in item.items() if k not in media_entry or k == "label"}
                remaining.pop("label", None)
                if remaining:
                    text = ", ".join(f"{k}: {v}" for k, v in remaining.items())
                    needs_label = True
                else:
                    # Nothing left to say in prose -- skip this entry's text
                    # entirely; its content now lives only in `media`.
                    text = ""
                    needs_label = False
            else:
                text = ", ".join(f"{k}: {v}" for k, v in item.items())
                needs_label = True
        else:
            text = str(item).strip()
            needs_label = False
        entries.append((text, needs_label, label))

    entries = [e for e in entries if e[0]]

    if not entries and not media:
        raise ValueError("inputs.summaries contained no non-empty entries")

    if not entries:
        combined = ""
    elif len(entries) == 1:
        text, needs_label, label = entries[0]
        combined = f"{_clean_label(label)}: {text}" if (needs_label and label) else text
    else:
        pieces = []
        for text, needs_label, label in entries:
            if needs_label and label:
                pieces.append(f"{_clean_label(label)}: {text}")
            else:
                pieces.append(text)
        combined = "\n\n".join(pieces)

    result = {
        "combined": combined,
        "count": len(entries),
        "mode": "copy" if len(entries) == 1 else "combine",
    }
    if media:
        result["media"] = media
    return result


# ── Master Orchestrator (Cell 6, unchanged) ───────────────────────────────────
def build_agent_index(agents: list) -> dict:
    return {a["agent_id"]: a for a in agents}


def call_agent(agent: dict, inputs: dict) -> dict:
    agent_id = agent["agent_id"]
    url = agent["endpoint_url"].rstrip("/") + agent["invoke_path"]
    method = agent["invoke_method"].upper()
    ctype = agent["invoke_content_type"]

    try:
        headers = {"ngrok-skip-browser-warning": "true"}  # harmless no-op for non-ngrok agents
        if method == "POST":
            if "application/json" in ctype:
                resp = req.post(url, json=inputs, headers=headers, timeout=AGENT_CALL_TIMEOUT)
            else:
                schema = agent.get("input_schema", {})
                files, fields = {}, {}
                for k, v in inputs.items():
                    field_def = schema.get(k, {})
                    if field_def.get("type") == "file":
                        files[k] = (k, v)
                    else:
                        fields[k] = str(v)
                if not files:
                    files = {"_dummy": ("", b"")}
                resp = req.post(url, data=fields, files=files, headers=headers, timeout=AGENT_CALL_TIMEOUT)
        else:
            resp = req.get(url, params=inputs, headers=headers, timeout=AGENT_CALL_TIMEOUT)

        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}

        return {
            "agent_id": agent_id,
            "success": resp.status_code < 400,
            "status_code": resp.status_code,
            "response": body,
            "error": None,
        }
    except req.exceptions.Timeout:
        return {"agent_id": agent_id, "success": False, "status_code": None,
                "response": None, "error": f"Timeout after {AGENT_CALL_TIMEOUT}s"}
    except Exception as e:
        return {"agent_id": agent_id, "success": False, "status_code": None,
                "response": None, "error": str(e)}


def call_agent_async(agent: dict, capability: str, task_id, goal: str, resolved_inputs: dict,
                      poll_interval: float = 3.0, max_wait: int = None) -> dict:
    """
    Submit-then-poll dispatch for agents whose /run_async + /status/<job_id>
    routes exist (see ASYNC_CAPABILITIES for which capabilities use this --
    currently text-summarize, image-generate, and data-matching). Added
    because these are the agents slow enough to occasionally exceed
    Cloudflare Tunnel's ~100s edge timeout when called via the normal
    blocking /run path -- every submit and poll here completes in well
    under a second, so none of them can ever hit that ceiling, regardless
    of how long the underlying work actually takes. Every other capability
    (webcrawl, websearch, fx, knowledge-retrieve) is untouched and keeps
    using call_agent()/the synchronous /run path above.
    """
    agent_id = agent["agent_id"]
    base = agent["endpoint_url"].rstrip("/")
    max_wait = max_wait or AGENT_CALL_TIMEOUT

    ts = int(time.time())
    tid = str(task_id)
    msg = f"{tid}:{goal}:{capability}:{ts}".encode()
    sig = _hmac.new(get_agent_secret(agent_id).encode(), msg, hashlib.sha256).hexdigest()

    try:
        submit = req.post(
            f"{base}/run_async",
            json={
                "task_id": tid,
                "goal_id": goal,
                "capability": capability,
                "inputs": resolved_inputs,
                "timeout_secs": max_wait,
                "hmac_sig": sig,
                "timestamp": ts,
            },
            headers={"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"},
            timeout=10,
        )
        submit.raise_for_status()
        job_id = submit.json()["job_id"]
    except Exception as e:
        return {"agent_id": agent_id, "success": False, "status_code": None,
                "response": None, "error": f"run_async submit failed: {e}"}

    elapsed = 0.0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            r = req.get(f"{base}/status/{job_id}", timeout=10)
            body = r.json()
        except Exception:
            continue  # transient poll hiccup -- keep trying until max_wait

        status = body.get("status")
        if status in ("done", "error"):
            return {
                "agent_id": agent_id,
                "success": status == "done",
                "status_code": 200,
                "response": body.get("result") or {},
                "error": body.get("error"),
            }

    return {"agent_id": agent_id, "success": False, "status_code": None,
            "response": None, "error": f"Timed out after {max_wait}s waiting on job {job_id}"}


class InputResolutionError(ValueError):
    """Raised when a $task_N.field reference cannot be resolved."""


_NUMERIC_PATTERN = _re.compile(r"\d[\d,]*\.?\d*\s*%?")
_DATE_SIGNAL_PATTERN = _re.compile(
    r"\b(ago|20\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", _re.IGNORECASE
)
_RESULT_STOPWORDS = {
    "news", "market", "markets", "summary", "stock", "stocks", "today", "update",
    "updates", "recent", "the", "and", "for", "current",
    "guide", "overview", "review", "reviews", "report", "reports", "latest",
    "breaking", "official", "top", "best", "list", "results", "result",
    "info", "information", "details", "week", "month", "year", "season",
    "episode", "here", "this", "that", "your", "how", "why", "what", "live", "new",
}

# Detects when the QUERY ITSELF is asking for something time-sensitive (e.g.
# "top 10 AI models THIS WEEK", "TODAY's news", "LATEST releases"). Only when
# this fires does recency get weighed in scoring below -- for a query with no
# recency intent (e.g. "history of the Roman Empire"), an old page is not a
# worse result, so this must not penalize it.
_RECENCY_QUERY_SIGNAL_RE = _re.compile(
    r"\b(today|this week|this month|latest|recent(ly)?|current(ly)?|now|newest|"
    r"up[- ]to[- ]date|this year)\b",
    _re.IGNORECASE,
)

_MONTH_NAME_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# A handful of common date shapes seen in titles/snippets/previews. Deliberately
# best-effort -- this is a ranking signal, not a strict parser, so silently
# skipping a shape we don't recognize is fine.
_DATE_PATTERNS = [
    # "July 9, 2026" / "Jul 9 2026"
    _re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(20\d{2})\b", _re.IGNORECASE),
    # "9 July 2026"
    _re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(20\d{2})\b", _re.IGNORECASE),
    # ISO "2026-07-09"
    _re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b"),
    # "July 2026" (month + year, no day -- treated as the 1st of the month)
    _re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(20\d{2})\b", _re.IGNORECASE),
]

_RELATIVE_RECENCY_RE = _re.compile(r"\b(\d+)\s*(hour|hr|min|minute|day)s?\s+ago\b", _re.IGNORECASE)


def _extract_most_recent_date(text: str):
    """
    Best-effort extraction of the MOST RECENT date mentioned in a result's
    title/snippet/preview text. Returns a datetime.date, or None if nothing
    parseable was found. When several dates appear (e.g. an article's own
    publish/update date alongside older dates it references in passing),
    the most recent one is used -- a page's own dateline is usually the
    most recent date on it, while historical context dates it discusses
    are typically older.
    """
    candidates = []

    if _RELATIVE_RECENCY_RE.search(text):
        candidates.append(datetime.now(timezone.utc).date())

    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            try:
                if len(groups) == 3 and groups[0].isdigit() and groups[1].isdigit() and groups[2].isdigit():
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                elif len(groups) == 3 and groups[0].isdigit():
                    day, month, year = int(groups[0]), _MONTH_NAME_TO_NUM.get(groups[1][:3].lower()), int(groups[2])
                elif len(groups) == 3:
                    month, day, year = _MONTH_NAME_TO_NUM.get(groups[0][:3].lower()), int(groups[1]), int(groups[2])
                elif len(groups) == 2:
                    month, year, day = _MONTH_NAME_TO_NUM.get(groups[0][:3].lower()), int(groups[1]), 1
                else:
                    continue
                if not month:
                    continue
                candidates.append(date(year, month, day))
            except (ValueError, KeyError):
                continue

    return max(candidates) if candidates else None


def _recency_score(text: str, recency_query: bool) -> int:
    """
    Score contribution from how recently-dated a result appears to be --
    ONLY applied when the query itself signals recency intent (see
    _RECENCY_QUERY_SIGNAL_RE). Otherwise returns 0 so age never penalizes
    results for queries that aren't time-sensitive.
    """
    if not recency_query:
        return 0

    found = _extract_most_recent_date(text)
    if found is None:
        return 0  # no date found -- neutral, don't penalize undated pages

    age_days = (datetime.now(timezone.utc).date() - found).days
    if age_days < 0:
        age_days = 0
    if age_days <= 3:
        return 40
    if age_days <= 14:
        return 25
    if age_days <= 30:
        return 10
    if age_days <= 90:
        return -10
    return -30


def _looks_like_search_result(item) -> bool:
    """Duck-type check: does this look like a websearch-agent result item
    (as opposed to some other kind of array element)?"""
    return isinstance(item, dict) and "url" in item and ("title" in item or "snippet" in item)


def _query_keywords(query: str) -> list:
    """Meaningful terms from the search query, dropping generic filler words
    that appear in almost any web result regardless of actual
    topical relevance (e.g. 'guide', 'news', 'summary')."""
    words = _re.findall(r"[a-zA-Z]+", (query or "").lower())
    return [w for w in words if w not in _RESULT_STOPWORDS and len(w) > 2]


def _score_search_result(result: dict, query_keywords: list = None, recency_query: bool = False) -> int:
    """
    Heuristic score to prefer substantive, ON-TOPIC articles/data pages over
    generic marketing/landing pages OR generic homepages when picking which
    search result to crawl.

    The Goal Expander writes $task_N.results[0].url at expansion time,
    before any real search results exist, so it can never meaningfully
    choose an index itself -- it always ends up writing [0]. This scores
    the REAL results once they exist (at execution time) and substitutes
    in whichever one looks most substantive AND relevant, regardless of
    what literal index was requested.

    Signals, combined:
      - Relevance: does the result's title/snippet/text_preview actually
        contain the search query's own keywords? Weighted heavily (x10 per
        match) because it's the strongest signal available, and without it
        a generic homepage with many unrelated headlines (each carrying a
        timestamp and a number) can out-score a genuinely on-topic but
        shorter result purely by volume -- e.g. a general news homepage
        covering sports/celebrities/politics scores high on "looks
        data-rich" while being almost entirely off-topic.
      - Substantiveness: presence of concrete numbers/percentages/dates,
        as a secondary signal among comparably relevant candidates.
      - Recency: ONLY when the query itself signals recency intent (e.g.
        "top 10 X THIS WEEK", "LATEST releases") -- a keyword match alone
        can't tell a 6-month-stale "Top 10 X" listicle apart from one
        actually covering the requested week, since both use identical
        wording. When recency_query is True, a result's own extracted date
        shifts its score up (fresher) or down (stale) on top of the
        keyword-relevance score; when False, this contributes nothing, so
        results for non-time-sensitive queries are never penalized for age.
    """
    text = " ".join(str(result.get(k, "")) for k in ("title", "snippet", "text_preview"))
    text_lower = text.lower()

    numeric_hits = min(len(_NUMERIC_PATTERN.findall(text)), 6)
    has_percent = 1 if "%" in text else 0
    has_date_signal = 1 if _DATE_SIGNAL_PATTERN.search(text) else 0
    density_score = numeric_hits + (2 * has_percent) + has_date_signal

    relevance_score = 0
    if query_keywords:
        relevance_score = 10 * sum(
            1 for kw in query_keywords if _re.search(r"\b" + _re.escape(kw) + r"\b", text_lower)
        )

    recency_bonus = _recency_score(text, recency_query)

    # Hard penalty for thin content, using the websearch agent's OWN
    # already-verified text_length. A title can be keyword-stuffed (e.g. a
    # video link ending in "| [Index A] | [Index B]") while the actual
    # page has almost nothing crawlable -- text_length exposes that
    # regardless of how relevant the title/snippet looks.
    thin_content_penalty = -30 if result.get("text_length", 0) < 500 else 0

    return relevance_score + density_score + recency_bonus + thin_content_penalty


def _pick_best_result_index(results: list, query: str = None) -> int:
    """Index of the most substantive AND relevant-looking result; ties
    prefer the earlier (originally higher-ranked) index."""
    if not results:
        return 0
    keywords = _query_keywords(query) if query else []
    recency_query = bool(query and _RECENCY_QUERY_SIGNAL_RE.search(query))
    return max(
        range(len(results)),
        key=lambda i: (_score_search_result(results[i], keywords, recency_query), -i),
    )


def _resolve_path(value, path_str, full_ref, allow_result_ranking=False):
    segments = path_str.split(".")
    current = value

    for seg in segments:
        m = _re.match(r"^([^\[]+)(\[(\d+)\])?$", seg)
        if not m:
            raise InputResolutionError(f"Cannot resolve '{full_ref}' — malformed path segment: {seg!r}")
        key = m.group(1)
        index = m.group(3)

        parent = current  # keep a reference before descending, to read sibling fields (e.g. 'query')

        if isinstance(current, dict):
            if key not in current:
                # NOT a template/schema bug in the overwhelming majority of
                # cases -- validate_task_references() already confirmed
                # this exact field name is a real key in the upstream
                # agent's DECLARED output_schema at plan-creation time. If
                # it's still missing now, at execution time, that means
                # the upstream task genuinely failed and returned partial/
                # empty outputs (e.g. {"outputs": {}}) -- a normal runtime
                # condition, not a broken reference. Degrading to "" here
                # (rather than raising) matters most for the combine step:
                # without this, ONE failed branch out of several took down
                # the ENTIRE final answer, discarding other branches that
                # succeeded perfectly fine alongside it. The combine step's
                # own downstream logic can still note a summary came back
                # empty; that's a much better failure mode than losing
                # every other leaf's result too.
                return ""
            current = current[key]
        else:
            raise InputResolutionError(
                f"Cannot resolve '{full_ref}' — expected dict at '{key}', got {type(current).__name__}"
            )

        if index is not None:
            idx = int(index)
            if not isinstance(current, list):
                raise InputResolutionError(
                    f"Cannot resolve '{full_ref}' — '{key}' is not a list (got {type(current).__name__}), cannot index with [{idx}]"
                )
            if idx >= len(current):
                # NOT a template/schema bug -- array length legitimately
                # varies at runtime (top_k is a request, not a guarantee;
                # a small or session-scoped knowledge-base search often
                # returns fewer matches than requested). Degrade just this
                # ONE reference to an empty string rather than failing the
                # whole downstream task -- e.g.
                # "$task_1.results[0].text\n$task_1.results[1].text" still
                # resolves (to result 0's text plus a blank trailing line)
                # when only one match was found, instead of the summarize
                # task never running at all. Genuine template bugs (wrong
                # key name, indexing a non-list) still raise above/below --
                # this only softens the "fewer items than expected" case.
                return ""
            # Override the literal index ONLY when the upstream agent declared
            # the 'web-search' capability -- i.e. this behaviour is opted into
            # via the agent's own declared contract, not guessed from field
            # name/shape. Any agent that happens to have a field called
            # 'results' but doesn't declare 'web-search' is left untouched,
            # preserving agent-agnosticism: the controller still only acts on
            # what agents declare, never on assumptions about their internals.
            if allow_result_ranking and key == "results" and all(_looks_like_search_result(item) for item in current):
                # The search agent's own output already echoes back the
                # query it searched for (outputs.query) -- use it to weight
                # relevance, not just "does this look data-rich."
                original_query = parent.get("query") if isinstance(parent, dict) else None
                idx = _pick_best_result_index(current, original_query)
            current = current[idx]

    return current


def _resolve_task_reference(task_ref: str, path: str, full_ref: str,
                            completed: dict, agent_index: dict = None):
    """
    Resolves a single $task_N.path token to its real value. Shared by
    resolve_inputs whether the token is the entire input value or just one
    piece embedded inside a larger literal string.
    """
    prior = completed.get(task_ref)
    if prior is None:
        raise InputResolutionError(
            f"Cannot resolve '{full_ref}' — upstream task '{task_ref}' has not completed or is not in the completed-tasks map."
        )

    response = prior.get("response")
    if response is None:
        raise InputResolutionError(
            f"Cannot resolve '{full_ref}' — task '{task_ref}' has no response (error: {prior.get('error')!r})."
        )

    outputs = response.get("outputs") if isinstance(response, dict) else None

    if not path:
        return outputs if outputs is not None else response

    if outputs is None:
        raise InputResolutionError(
            f"Cannot resolve '{full_ref}' — task '{task_ref}' response has no 'outputs' object. "
            f"Raw response keys: {list(response.keys()) if isinstance(response, dict) else type(response)}"
        )

    # Only allow the result-ranking override when the upstream agent
    # DECLARED the 'web-search' capability -- looked up via agent_index,
    # never guessed from the shape of its output.
    allow_result_ranking = False
    if agent_index is not None:
        upstream_agent_id = prior.get("agent_id")
        upstream_agent = agent_index.get(upstream_agent_id) if upstream_agent_id else None
        if upstream_agent and "web-search" in upstream_agent.get("capabilities", []):
            allow_result_ranking = True

    return _resolve_path(outputs, path, full_ref, allow_result_ranking=allow_result_ranking)


def _resolve_value(v, completed: dict, agent_index: dict = None):
    """
    Resolves a single input value, which may be:
      - a plain literal (non-string, or a string with no $task_ reference)
        -- returned unchanged
      - a list -- each element resolved recursively via this same function
        (this is what lets an agent declare an "array" input like
        ["$task_3.summary", "$task_6.summary"] and receive the real,
        individually-resolved values rather than literal token strings)
      - a whole-string $task_N.path reference -- returns the raw resolved
        value untouched (preserves dict/list/str typing)
      - a string with one or more $task_N.path tokens embedded inside
        literal text -- returns a composed string with each token
        substituted in (stringified if the resolved value isn't already a
        string)
    """
    if isinstance(v, list):
        return [_resolve_value(item, completed, agent_index) for item in v]

    if not (isinstance(v, str) and "$task_" in v):
        return v

    matches = list(_TASK_REF_TOKEN_RE.finditer(v))
    if not matches:
        return v

    # Whole-string exact match (the historical/common case, e.g. the
    # entire "text" input IS "$task_2.text") -- preserve the original
    # behaviour of returning the raw resolved value untouched (could be
    # a dict, list, or string) rather than stringifying it.
    if len(matches) == 1 and matches[0].span() == (0, len(v)):
        m = matches[0]
        task_ref = f"task_{m.group(1)}"
        path = m.group(2)[1:] if m.group(2) else None
        return _resolve_task_reference(task_ref, path, v, completed, agent_index)

    # Otherwise: one or more tokens embedded inside a larger literal
    # string -- e.g. "reviews of $task_2.summary attractions" (composing
    # a downstream search query around an upstream-resolved entity), or
    # "$task_2.text\n\n$task_5.text" (concatenating two upstream texts
    # into a single field). Substitute each token with its resolved
    # value, stringified, and leave the surrounding literal text as-is.
    composed = v
    for m in reversed(matches):  # walk backwards so earlier spans/offsets stay valid
        task_ref = f"task_{m.group(1)}"
        path = m.group(2)[1:] if m.group(2) else None
        full_ref = m.group(0)
        value = _resolve_task_reference(task_ref, path, full_ref, completed, agent_index)
        value_str = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        composed = composed[:m.start()] + value_str + composed[m.end():]
    return composed


def resolve_inputs(task: dict, completed: dict, agent_index: dict = None) -> dict:
    return {
        k: _resolve_value(v, completed, agent_index)
        for k, v in task.get("inputs", {}).items()
    }


def run_orchestrator(tasks: list, agent_index: dict) -> list:
    completed = {}
    all_results = []
    pending = {t["task_id"]: t for t in tasks}

    while pending:
        ready = [t for t in pending.values() if all(dep in completed for dep in t.get("depends_on", []))]
        if not ready:
            for task_id in list(pending.keys()):
                all_results.append({
                    "task_id": task_id,
                    "task_description": pending[task_id].get("task_description", ""),
                    "agent_id": pending[task_id]["agent_id"],
                    "success": False,
                    "response": None,
                    "error": "Unresolvable dependency — possible cycle",
                })
            break

        with ThreadPoolExecutor(max_workers=min(len(ready), MAX_PARALLEL_AGENTS)) as pool:
            future_to_task = {}
            for task in ready:
                agent = agent_index.get(task["agent_id"])
                if not agent:
                    result = {
                        "task_id": task["task_id"],
                        "task_description": task.get("task_description", ""),
                        "agent_id": task["agent_id"],
                        "success": False,
                        "status_code": None,
                        "response": None,
                        "error": f"Agent not found in registry: {task['agent_id']}",
                    }
                    completed[task["task_id"]] = result
                    all_results.append(result)
                    del pending[task["task_id"]]
                    continue

                try:
                    inputs = resolve_inputs(task, {f"task_{k}": v for k, v in completed.items()}, agent_index)
                except InputResolutionError as e:
                    result = {
                        "task_id": task["task_id"],
                        "task_description": task.get("task_description", ""),
                        "agent_id": task["agent_id"],
                        "success": False,
                        "status_code": None,
                        "response": None,
                        "error": f"Input resolution failed: {e}",
                    }
                    completed[task["task_id"]] = result
                    all_results.append(result)
                    del pending[task["task_id"]]
                    continue

                future_to_task[pool.submit(call_agent, agent, inputs)] = task

            for future in as_completed(future_to_task, timeout=AGENT_CALL_TIMEOUT + 10):
                task = future_to_task[future]
                try:
                    result = future.result()
                except FuturesTimeout:
                    result = {"agent_id": task["agent_id"], "success": False,
                              "status_code": None, "response": None, "error": "Future timed out"}
                except Exception as e:
                    result = {"agent_id": task["agent_id"], "success": False,
                              "status_code": None, "response": None, "error": str(e)}

                result["task_id"] = task["task_id"]
                result["task_description"] = task.get("task_description", "")
                completed[task["task_id"]] = result
                all_results.append(result)
                del pending[task["task_id"]]

    return all_results


# ── Pending Plans Store (Cell 6.5, unchanged) ─────────────────────────────────
PENDING_PLANS = {}
PENDING_PLANS_LOCK = threading.Lock()


def _cleanup_expired_plans() -> None:
    """
    Removes expired plans from PENDING_PLANS. MUST be called with
    PENDING_PLANS_LOCK already held -- it only touches the dict.

    NOTE: this no longer purges any KB docs. `session_id` on a plan is now
    the conversation_id (files are tagged/scoped at the conversation level,
    not per-plan -- see the /goal worker), so purging it here on plan
    expiry would delete every file attached anywhere in that chat thread,
    including ones from earlier turns that are still perfectly valid.
    Conversation-scoped purging happens only in _prune_stale_conversations()
    and clear_conversation() now.
    """
    now = time.time()
    expired = [pid for pid, p in PENDING_PLANS.items() if now - p["created_at"] > PLAN_TTL_SECS]
    for pid in expired:
        PENDING_PLANS.pop(pid)


def create_pending_plan(goal: str, tasks: list, agents: list, session_id: str = None,
                        preseeded: dict = None, conversation_id: str = None) -> str:
    plan_id = uuid.uuid4().hex
    with PENDING_PLANS_LOCK:
        _cleanup_expired_plans()
        PENDING_PLANS[plan_id] = {
            "goal": goal,
            "tasks": tasks,
            "agents": agents,
            "created_at": time.time(),
            "status": "pending_approval",
            "session_id": session_id,
            "preseeded": preseeded or {},
            "conversation_id": conversation_id,
        }
    return plan_id


def get_pending_plan(plan_id: str):
    with PENDING_PLANS_LOCK:
        _cleanup_expired_plans()
        plan = PENDING_PLANS.get(plan_id)
    return plan


def pop_pending_plan(plan_id: str):
    with PENDING_PLANS_LOCK:
        _cleanup_expired_plans()
        plan = PENDING_PLANS.pop(plan_id, None)
    return plan


# ── TEMP kill-switch for conversation_id HISTORY linking ──────────────────────
# Flip to True to make passing a conversation_id a no-op for cross-turn
# GOAL/ANSWER history only (get_conversation_history / append_conversation_turn
# below), without touching any call site, route, or the API contract.
#
# Deliberately scoped to just those two accessors -- NOT get_conversation_docs
# / append_conversation_docs. Those two also carry THIS turn's own
# freshly-uploaded file within a single request (ingest -> append -> immediate
# read-back later in the same /goal worker call), so routing them through
# this switch breaks same-turn file processing, not just cross-turn doc
# continuity. Learned this the hard way -- see conversation history.
#
# This works because get_conversation_history/append_conversation_turn already
# have an `if not conversation_id: ...` no-op guard for the "unknown/fresh
# thread" case. Routing their calls through _resolve_conversation_id() just
# forces that existing guard to fire every time. Flip
# DISABLE_CONVERSATION_ID_LINKING back to False to restore normal behavior
# with zero other changes.
#
# Remove this whole block (and its two call sites) once the conversation_id
# history reference bug is actually fixed.
DISABLE_CONVERSATION_ID_LINKING = True


def _resolve_conversation_id(conversation_id: str):
    """Returns conversation_id unchanged, or None while the kill-switch above
    is on. Callers should assign the result back over their local
    conversation_id before using it, e.g.:
        conversation_id = _resolve_conversation_id(conversation_id)
    """
    if DISABLE_CONVERSATION_ID_LINKING:
        return None
    return conversation_id


# ── Conversation History Store (new) ──────────────────────────────────────────
# Keyed by `conversation_id` (frontend-generated on turn 1, echoed back on
# every subsequent /goal call in the same chat thread). Each entry holds a
# short, capped list of past (goal, answer) turns -- NOT the raw task graph
# or agent outputs, which can be large/binary and would blow up the Goal
# Expander prompt fast. See _build_history_block() for how this gets spliced
# into the prompt, and _extract_history_answer() for how the "answer" side
# of a turn gets produced once a plan finishes executing.
#
# In-memory + TTL, same pattern as PENDING_PLANS/_goal_jobs. A conversation
# naturally outlives a single plan's PLAN_TTL_SECS (a chat thread can sit
# idle between messages far longer than a pending approval should), so it
# gets its own, longer CONVERSATION_TTL_SECS. If the controller restarts,
# in-flight conversations lose their history same as pending plans/jobs
# already do today -- acceptable for now, but worth moving to Postgres
# (alongside registered_agents) if that becomes a problem in practice.
CONVERSATIONS = {}
CONVERSATIONS_LOCK = threading.Lock()


def _prune_stale_conversations() -> None:
    cutoff = time.time() - CONVERSATION_TTL_SECS
    with CONVERSATIONS_LOCK:
        stale = [cid for cid, c in CONVERSATIONS.items() if c["last_active"] < cutoff]
        for cid in stale:
            del CONVERSATIONS[cid]
    # Docs attached anywhere in a now-stale conversation are no longer
    # reachable by any future turn (the conversation_id itself is gone), so
    # this is the one place a whole-session KB purge is still correct --
    # mirrors the old per-job purge, just keyed to conversation_id instead
    # of job_id now that uploads are tagged/scoped by conversation_id.
    for cid in stale:
        purge_session_docs(cid)


def get_conversation_docs(conversation_id: str) -> list:
    """
    Returns a shallow copy of every doc_ref attached anywhere in THIS
    conversation so far (oldest first), or [] if none/unknown. Mirrors
    get_conversation_history() but for uploaded-file metadata instead of
    goal/answer turns -- this is what lets a follow-up turn with NO new
    upload still reference a file that was attached two turns ago.

    NOTE: deliberately NOT routed through the conversation_id kill-switch
    below -- this also carries THIS turn's own freshly-ingested upload
    within a single request (ingest -> append -> immediate read-back a few
    lines later in the /goal worker), not just cross-turn continuity.
    Disabling it here broke same-turn file processing entirely.
    """
    if not conversation_id:
        return []
    _prune_stale_conversations()
    with CONVERSATIONS_LOCK:
        conv = CONVERSATIONS.get(conversation_id)
        return list(conv.get("docs", [])) if conv else []


def append_conversation_docs(conversation_id: str, doc_refs: list) -> None:
    """
    Merges newly-ingested doc_refs into this conversation's running doc
    list, deduped by doc_id (a resend/re-ingest of the same doc_id -- e.g.
    the same filename re-uploaded -- replaces the earlier entry AND counts
    as freshly used again, rather than leaving it stuck at its original
    position). No-op if conversation_id is falsy or doc_refs is empty.

    Trims to the last MAX_HISTORY_DOCS entries (oldest dropped first) --
    same bounding principle as append_conversation_turn's MAX_HISTORY_TURNS
    trim, so a long-running thread with many attachments doesn't grow the
    Goal Expander's "0. PRIORITY" upload rule without bound. A dropped doc
    simply stops being offered to future turns; it is NOT individually
    deleted from the knowledge base here (there's no per-doc delete
    endpoint, only the session-level one -- see purge_session_docs) -- it
    still gets cleaned up eventually via the normal conversation-level
    purge (_prune_stale_conversations / clear_conversation).

    NOTE: deliberately NOT routed through the conversation_id kill-switch
    below -- see get_conversation_docs for why (same-turn round-trip, not
    just cross-turn continuity).
    """
    if not conversation_id or not doc_refs:
        return
    _prune_stale_conversations()
    with CONVERSATIONS_LOCK:
        conv = CONVERSATIONS.setdefault(conversation_id, {"turns": [], "docs": [], "last_active": time.time()})
        existing = conv.get("docs", [])
        by_id = {d["doc_id"]: d for d in existing if d.get("doc_id")}
        no_id = [d for d in existing if not d.get("doc_id")]
        order = [d["doc_id"] for d in existing if d.get("doc_id")]
        for d in doc_refs:
            doc_id = d.get("doc_id")
            if doc_id:
                if doc_id in by_id:
                    order.remove(doc_id)  # re-attached -- move to the end below
                by_id[doc_id] = d
                order.append(doc_id)
            else:
                no_id.append(d)
        merged = [by_id[doc_id] for doc_id in order] + no_id
        conv["docs"] = merged[-MAX_HISTORY_DOCS:]
        conv["last_active"] = time.time()


def get_conversation_history(conversation_id: str) -> list:
    """
    Returns a shallow copy of this conversation_id's turn list (oldest
    first), or [] if conversation_id is falsy or unknown/expired. Never
    raises -- an unrecognized conversation_id is just treated as the start
    of a fresh thread rather than an error, since the frontend may send one
    it generated client-side before the controller has seen it yet.
    """
    conversation_id = _resolve_conversation_id(conversation_id)
    if not conversation_id:
        return []
    _prune_stale_conversations()
    with CONVERSATIONS_LOCK:
        conv = CONVERSATIONS.get(conversation_id)
        return list(conv["turns"]) if conv else []


def append_conversation_turn(conversation_id: str, goal: str, answer: str) -> None:
    """
    Appends one (goal, answer) turn, creating the conversation entry if this
    is its first turn. Trims to the last MAX_HISTORY_TURNS turns (oldest
    dropped first) and caps each field's length -- same bounding principle
    as MAX_INLINE_UPLOAD_CHARS for uploads -- so a long-running chat thread
    doesn't grow the Goal Expander prompt without bound. Called once a
    plan's execution has actually finished (see approve_goal's generate()),
    never for a goal that was only planned, rejected, or left to expire
    unapproved -- there is no real "answer" to remember in those cases.
    """
    conversation_id = _resolve_conversation_id(conversation_id)
    if not conversation_id:
        return
    _prune_stale_conversations()
    with CONVERSATIONS_LOCK:
        conv = CONVERSATIONS.setdefault(conversation_id, {"turns": [], "last_active": time.time()})
        conv["turns"].append({
            "goal": (goal or "")[:MAX_HISTORY_GOAL_CHARS],
            "answer": (answer or "")[:MAX_HISTORY_ANSWER_CHARS],
        })
        conv["turns"] = conv["turns"][-MAX_HISTORY_TURNS:]
        conv["last_active"] = time.time()


# ── Final synthesis (new) ──────────────────────────────────────────────────
# Rationale: per-task fixes (result ranking, focus scoping, "prefer one
# combined chain") each only see a narrow slice of the problem and can still
# individually miss the mark -- e.g. a single-chain run whose one summarize
# call still wanders off-topic despite the right raw content being present.
# This adds ONE guaranteed final step, enforced by the controller itself
# (not dependent on the Goal Expander remembering to add it -- same pattern
# as the `focus` injection), that sees the ENTIRE picture at once: the
# original unscoped goal, the task plan, and the raw content every task
# actually produced -- then produces one final answer from that, explicitly
# flagging anything the goal asked for that the gathered material doesn't
# actually cover.

SYNTHESIS_MAX_CONTEXT_CHARS = 24000
SYNTHESIS_PER_TASK_MAX_CHARS = 20000
# These were originally 12000/3000, sized for a single normal crawl of one
# page. Since the fan-out crawl change, a single crawl TASK can itself be a
# combined blob of up to FANOUT_TOTAL_MAX_CHARS (20000) chars from up to 15
# sources, already sorted by relevance. The old 3000-char per-task cap was
# silently re-truncating that already-sorted content down to roughly its
# first source only -- discarding genuinely relevant later sources (e.g. a
# current data table that happened to sort second) before synthesis ever
# saw them, even though the same task's direct summarize call (which reads
# the untruncated text) found the right data. Raised to comfortably fit one
# full fan-out crawl's output without re-truncating it.


def _extract_raw_content_for_synthesis(task_result: dict) -> str:
    """Pull the most useful raw text out of a single task's result, for
    feeding into the final synthesis call. Prefers already-condensed content
    (a prior summary) over raw crawled text, and raw crawled text over a
    dump of search results, since crawled/summarized text is denser signal
    per character than a search-results list."""
    response = task_result.get("response")
    if not isinstance(response, dict):
        return ""
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        return ""

    if outputs.get("output_format") == "json" and outputs.get("structured") is not None:
        # JSON-mode outputs keep the parsed object in `structured` as the
        # single source of truth (see summarize_agent.py's _run_json_mode)
        # -- `summary` in this mode is deliberately just a short human
        # label, not a duplicate JSON string, so there's exactly one JSON
        # representation to read rather than the same content appearing
        # twice (once escaped as text, once as real nested JSON) in the
        # same task result. Re-serialize `structured` once here for
        # whatever consumes this return value (final_answer, synthesis
        # context) -- this is the only place that JSON text gets built.
        return json.dumps(outputs["structured"], indent=2, ensure_ascii=False)
    if outputs.get("summary"):
        return str(outputs["summary"])
    if outputs.get("text"):
        return str(outputs["text"])[:SYNTHESIS_PER_TASK_MAX_CHARS]
    if isinstance(outputs.get("results"), list):
        parts = []
        for r in outputs["results"][:5]:
            if isinstance(r, dict):
                parts.append(f"{r.get('title', '')}: {r.get('snippet', '')}")
        return " | ".join(parts)[:SYNTHESIS_PER_TASK_MAX_CHARS]

    return json.dumps(outputs)[:SYNTHESIS_PER_TASK_MAX_CHARS]


def _build_synthesis_context(tasks: list, completed: dict, agent_index: dict) -> str:
    """
    Assemble a labeled, length-capped block of every task's RAW content
    (search results, crawled pages) for the final synthesis prompt.

    Deliberately EXCLUDES text-summarize task outputs. Testing showed an
    intermediate summarize task can introduce its own scope drift (e.g.
    mentioning an entity that wasn't actually asked about) since it only
    sees a narrower slice of context than the final synthesis step does.
    Including that flawed summary alongside the good raw material adds
    noise and a contradiction risk for no benefit -- the synthesis step
    already has direct access to the same raw content the summarize task
    was working from, so nothing is lost by skipping the summary itself.
    """
    sections = []
    excluded_summaries = []
    for task in tasks:
        result = completed.get(task["task_id"])
        if not result or not result.get("success"):
            continue

        task_agent = agent_index.get(result.get("agent_id"))
        is_summarize_task = bool(task_agent and "text-summarize" in task_agent.get("capabilities", []))

        content = _extract_raw_content_for_synthesis(result)
        if not content:
            continue

        label = f"[{task.get('task_description', 'Task ' + str(task['task_id']))}]\n{content}"
        if is_summarize_task:
            excluded_summaries.append(label)
        else:
            sections.append(label)

    # Fallback: if there's no raw search/crawl content at all (e.g. a goal
    # that only ever ran a summarize task), fall back to including the
    # summaries rather than sending nothing to synthesize.
    if not sections and excluded_summaries:
        sections = excluded_summaries

    context = "\n\n".join(sections)
    if len(context) > SYNTHESIS_MAX_CONTEXT_CHARS:
        context = context[:SYNTHESIS_MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return context


def _find_summarize_agent(agent_index: dict):
    for agent in agent_index.values():
        if "text-summarize" in agent.get("capabilities", []):
            return agent
    return None


def _find_agent_by_capability(agent_index: dict, capability: str):
    for agent in agent_index.values():
        if capability in agent.get("capabilities", []):
            return agent
    return None


def _parse_missing_data_marker(summary_text: str):
    """
    Splits a summarize agent's response into (clean_answer, missing_entities).
    Looks for a final line "MISSING_DATA: <comma-separated list>" or
    "MISSING_DATA: none". If no marker is found, treats the whole text as
    the clean answer with no missing entities (backward compatible with
    calls that don't request the marker).
    """
    if not summary_text:
        return summary_text, []

    m = _re.search(r"(?im)^\s*MISSING_DATA:\s*(.*)\s*$", summary_text)
    if not m:
        return summary_text.strip(), []

    raw_list = m.group(1).strip()
    clean_text = (summary_text[:m.start()] + summary_text[m.end():]).strip()

    if not raw_list or raw_list.lower() in ("none", "n/a", "-"):
        return clean_text, []

    entities = [e.strip() for e in raw_list.split(",") if e.strip()]
    return clean_text, entities


def _extract_period_hint(goal: str) -> str:
    """
    Pulls a time-period phrase out of the goal to use in the supplementary
    search query, instead of always assuming "current". A goal asking about
    full-year 2025 performance needs a search for 2025 data, not "current"
    data -- searching for "current" when the goal asked about a past year
    actively steers the supplementary search away from the answer.

    The default (no year, no YTD phrasing in the goal) is deliberately
    topic-neutral: an earlier finance-flavored default ("current
    performance data") produced nonsense queries for non-finance goals
    (e.g. "Company A current performance data" for a goal that just asked
    to crawl a reference page).
    """
    m = _re.search(r"\b(19|20)\d{2}\b", goal)
    if m:
        return f"{m.group(0)} data"
    if _re.search(r"year.to.date|\bYTD\b|this year", goal, _re.IGNORECASE):
        return "year-to-date"
    return "latest information"


def _run_supplementary_search_and_crawl(missing_entities: list, agent_index: dict, goal: str) -> dict:
    """
    One targeted follow-up: search specifically for whatever entities the
    first synthesis pass flagged as missing, then crawl every result the
    same way the main fan-out crawl does.

    Rationale: the Goal Expander is instructed to prefer ONE combined
    search covering multiple named entities (for efficiency, and because
    combined pages are often genuinely better for closely-related topics).
    But that same combined query can under-represent a smaller/less
    globally-indexed entity paired with a much bigger one (e.g. a smaller
    regional index alongside two much larger global ones) -- the bigger name dominates the
    result set. Rather than switching search providers (which wouldn't fix
    this -- any popularity-ranked engine has the same bias), this runs one
    narrow, entity-specific search only when a gap is actually detected.

    Returns a dict so the caller can tell WHETHER real content was found,
    not just get back an opaque string -- this is what lets us verify a
    retry's answer is actually grounded in something real, rather than the
    model inventing a plausible-sounding number because it was told a
    supplementary search happened.
    """
    query = " ".join(missing_entities) + " " + _extract_period_hint(goal)

    search_agent = _find_agent_by_capability(agent_index, "web-search")
    crawl_agent = _find_agent_by_capability(agent_index, "web-crawl")
    if not search_agent or not crawl_agent:
        return {"query": query, "found": False, "text": "", "sources_succeeded": 0}

    url = search_agent["endpoint_url"].rstrip("/") + search_agent["invoke_path"]
    ts = int(time.time())
    tid = "gap_fill_search"
    msg = f"{tid}:{goal}:web-search:{ts}".encode()
    sig = _hmac.new(get_agent_secret(search_agent["agent_id"]).encode(), msg, hashlib.sha256).hexdigest()

    json_body = {
        "task_id": tid,
        "goal_id": goal,
        "capability": "web-search",
        "inputs": {"query": query, "min_results": 10},
        "timeout_secs": AGENT_CALL_TIMEOUT,
        "hmac_sig": sig,
        "timestamp": ts,
    }

    try:
        resp = req.post(
            url, json=json_body,
            headers={"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"},
            timeout=AGENT_CALL_TIMEOUT,
        )
        body = resp.json()
        results = body.get("outputs", {}).get("results")
        if resp.status_code >= 400 or not isinstance(results, list) or not results:
            return {"query": query, "found": False, "text": "", "sources_succeeded": 0}
    except Exception:
        return {"query": query, "found": False, "text": "", "sources_succeeded": 0}

    dummy_task = {"task_id": tid, "inputs": {}}
    crawl_outputs = crawl_all_search_results(crawl_agent, results, query, dummy_task, goal)
    text = crawl_outputs.get("text", "")
    return {
        "query": query,
        "found": bool(text.strip()),
        "text": text,
        "sources_succeeded": crawl_outputs.get("sources_succeeded", 0),
    }


# Detects when the ORIGINAL top-level goal is asking for structured/JSON
# output (e.g. "give me in json format", "return as json", "structured
# output"), as distinct from a task's own generated `focus`. The Goal
# Expander's task-decomposition step routinely drops this constraint when
# writing a task's `focus` field -- that field gets authored as a prose
# instruction template ("Summarize the key details... in 4-6 sentences")
# regardless of what the top-level goal actually asked for, since the
# expander LLM treats "summarize the document" and "summarize the document
# as json" as the same underlying task and only carries forward the parts
# it judges essential. Checking the ORIGINAL goal directly here, at
# dispatch time -- not the task's own focus/description -- means the
# format request survives even when the Goal Expander's JSON output never
# mentions it. Same "safety net regardless of what the LLM happened to
# generate" rationale as _build_enriched_focus below.
_JSON_FORMAT_REQUEST_RE = _re.compile(
    r"\bjson\b|\bstructured\s+(?:format|output|data)\b|\bmachine[\s-]readable\b",
    _re.IGNORECASE,
)


def _wants_json_output(goal: str) -> bool:
    return bool(goal) and bool(_JSON_FORMAT_REQUEST_RE.search(goal))


# Auto-detects a scanned business document (invoice, PO, delivery order,
# goods receipt note, put-away report, etc.) from its own CONTENT, so JSON
# extraction doesn't depend on the user's goal phrasing at all -- "process
# this", "summarize this", or no format-related wording whatsoever should
# still get structured JSON if what was actually uploaded is this kind of
# document. Two-tier on purpose, same reasoning as the earlier per-task
# regexes: a doc-TYPE cue alone (e.g. a novel that happens to mention "an
# invoice arrived") isn't enough by itself, so at least one doc-type cue
# AND at least two structural/financial support cues must BOTH be present.
# The gaps inside the multi-word cues are \s* (zero or more), not \s+: the
# OCR sometimes drops the space between two words of a title it read as one
# glued token -- confirmed on a real tax invoice whose ONLY "invoice" text
# anywhere in the OCR output was the annotation "WTCE NUMBER TAXINVOICE".
# With \s+ and \binvoice\b, "TAXINVOICE" matched nothing (no word boundary
# before the "I"), so the document-type cue was missing, the document was
# not recognised as financial, and the upload fell to the prose summary
# path instead of JSON extraction.
_FINANCIAL_DOC_TYPE_RE = _re.compile(
    r"\b(tax\s*invoice|invoice|purchase\s*order|delivery\s*order|goods?\s*"
    r"receipt(?:\s*note)?|packing\s*slip|quotation|credit\s*note|debit\s*"
    r"note|receipt|put[\s-]?away\s*report)\b",
    _re.IGNORECASE,
)
_FINANCIAL_SUPPORT_RE = _re.compile(
    r"\b(sub[\s-]?total|gst|vat|tax\s+amount|total\s+due|grand\s+total|"
    r"amount\s+payable|unit\s+price|qty|quantity|line\s+total|inv(?:oice)?"
    r"\s*(?:no|number|#)|p\.?o\.?\s*(?:no|number|#)|grn\s*(?:no|number|#)?|"
    r"do\s*(?:no|number|#)|vendor|warehouse|zone\s+code|bin\s+code)\b",
    _re.IGNORECASE,
)

# The OCR pipeline emits this placeholder character in place of most
# inter-word gaps whose exact whitespace couldn't be confidently
# transcribed -- e.g. "TAX\u2337INVOICE" instead of "TAX INVOICE" (see
# summarize_agent.py's own _OCR_WS_PLACEHOLDER, which names the same
# character for the same reason -- kept as a separate constant here
# rather than imported since the two files aren't otherwise coupled).
# None of Python's \s matches it, so every \s+/\s*-based multi-word
# cue above (tax invoice, unit price, grand total, zone code, ...)
# silently stopped matching real OCR text the day this platform's OCR
# started emitting this character -- only single-word cues (gst, qty,
# vendor, ...) kept matching, so detection went from reliable to
# "works if a lucky single-word cue happens to be present" rather than
# outright always-broken, which is what made this hard to notice.
# Normalizing runs of the placeholder to a single space before
# matching restores exact parity with plain-whitespace OCR text.
_OCR_WS_PLACEHOLDER = "\u2337"
_OCR_WS_PLACEHOLDER_RUN_RE = _re.compile(_OCR_WS_PLACEHOLDER + "+")


def _normalize_ocr_placeholder_whitespace(text: str) -> str:
    if not text or _OCR_WS_PLACEHOLDER not in text:
        return text
    return _OCR_WS_PLACEHOLDER_RUN_RE.sub(" ", text)


def _looks_like_financial_document(text: str) -> bool:
    if not text:
        return False
    text = _normalize_ocr_placeholder_whitespace(text)
    type_hit = bool(_FINANCIAL_DOC_TYPE_RE.search(text))
    support_hits = len(set(m.lower() for m in _FINANCIAL_SUPPORT_RE.findall(text)))
    return type_hit and support_hits >= 2


# Detects when the ORIGINAL goal is a narrow, single-fact lookup ("what is
# the purchaser name again?", "who signed this", "how much was the total")
# rather than a request to process/extract the WHOLE document ("process
# this", "what's in this file", "extract the details"). _auto_json_from_
# upload below exists to catch that SECOND case with no format wording at
# all -- it was never meant to also hijack a plain follow-up QUESTION about
# a financial document into a full structured-JSON dump just because the
# underlying file happens to look financial. Without this check, "what is
# the purchaser name again?" and "process this invoice" were
# indistinguishable to _auto_json_from_upload -- both got the identical
# full JSON object, burying the one fact actually asked for in an object
# with 20+ other fields.
#
# Deliberately goal-text-only and field-agnostic -- same "no hardcoding"
# principle as every other cue regex on this platform. This has no
# knowledge of "purchaser", "total", or any other specific field/document
# type; it only judges the QUESTION's shape (a bare fact lookup vs. a bulk
# extraction/processing request), so it works identically for any document
# type or any registered summarize-capable agent, not just
# byoa-summarize-agent by name.
_NARROW_QUESTION_CUE_RE = _re.compile(
    r"^\s*(?:what|who|when|where|which|how\s+(?:much|many))\b|\bagain\s*\??\s*$",
    _re.IGNORECASE,
)
_BULK_PROCESSING_CUE_RE = _re.compile(
    r"\b(extract|process|convert|parse|structure|digiti[sz]e|summari[sz]e|"
    r"all\s+(?:the\s+)?(?:details|fields|line\s*items|information)|"
    r"full\s+details|entire\s+document|everything\s+(?:in|on|about)\s+"
    r"(?:this|that|it|the))\b",
    _re.IGNORECASE,
)


def _looks_like_narrow_factual_question(goal: str) -> bool:
    if not goal:
        return False
    return (
        bool(_NARROW_QUESTION_CUE_RE.search(goal))
        and not bool(_BULK_PROCESSING_CUE_RE.search(goal))
    )


def _task_input_references_upload(task: dict, input_key: str = "text") -> bool:
    """
    True if this task's RAW (pre-resolution) input for input_key is a
    reference to an uploaded-file task (task_id >= UPLOAD_TASK_ID_BASE,
    e.g. "$task_9001.text"), as opposed to referencing some other task's
    output (a web crawl, a knowledge-base lookup, etc). Deliberately scoped
    to uploads only: content-sniffing every summarize call for financial
    keywords risks a false positive on, say, a crawled news article ABOUT
    invoices; restricting the auto-JSON trigger to text that came directly
    from something the user actually uploaded removes that risk entirely.
    """
    raw_val = (task.get("inputs", {}) or {}).get(input_key)
    if not isinstance(raw_val, str):
        return False
    for m in _TASK_REF_TOKEN_RE.finditer(raw_val):
        if int(m.group(1)) >= UPLOAD_TASK_ID_BASE:
            return True
    return False


def _enrichment_rules_only() -> str:
    """
    Just the safety-net rules block -- explicit scope/comparability/
    recency/tabular-data/date-check/financial-document/OCR rules -- with NO
    base_focus text prefixed onto it. Failure modes seen in testing
    motivated each rule:
      - Substituting or reporting on entities that happen to appear in the
        source material but weren't actually asked about -- persisted even
        after a softer version of this rule was added, so it's now a hard
        exclusion instruction rather than a preference.
      - Comparing figures measured over different time periods or
        baselines as if they were directly comparable, and using
        stale/outdated source data without flagging it.
      - Comparing a "total return" figure (includes dividends/distributions
        reinvested) against a "price return" figure (price change only) as
        if they were the same kind of number -- caught inconsistently in
        testing: flagged for one entity's conflicting figures in the same
        answer, but missed for a different entity's total-return-vs-price
        mismatch sitting right next to it. Same underlying comparability
        problem as the time-period rule, different dimension, so it gets
        its own explicit callout rather than assuming the time-period rule
        covers it by implication.
      - Misattributing values in densely-formatted comparison tables where
        a label is repeated once per entity followed by one value per
        entity -- correct data for the requested entities was present in
        exactly this format and got skipped in favour of an easier-to-read
        (but off-topic) prose figure elsewhere in the same source material.
      - Treating a "YTD"/"this year" figure as current without checking
        the source's OWN stated publication date against today. A softer
        "prefer recent data" rule wasn't specific enough to catch this, so
        it's now an explicit date-checking instruction with a worked
        example.

    IMPORTANT: this text is now sent to the summarize agent as its OWN
    input field (`focus_rules`), separate from `focus` -- it used to be
    concatenated directly onto base_focus and sent as a single `focus`
    string (see _build_enriched_focus below, kept for the one remaining
    caller that still wants the combined form). That concatenation was a
    real bug: the summarize agent's list/extraction cue-detection regexes
    run against the entire `focus` string, and this rules block happens to
    contain the word "list" (see the financial-document paragraph below,
    "...logistics documents that list items and quantities...") -- so
    EVERY summarize call, not just document-extraction ones, was being
    wrongly flagged as a list/ranking request and forced through
    list-reformatting, regardless of what the task actually asked for.
    Keeping this block in its own field means the agent's cue-detection
    only ever sees the task's real, unenriched focus text.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    current_year = datetime.now(timezone.utc).year
    return (
        f"Additional rules for this answer (today's date is {today}):\n"
        f"- HARD EXCLUSION: if the source material mentions entities that are NOT named "
        f"in the request above, do not report their figures at all, even briefly, even to "
        f"say they outperformed something. An unrequested entity having a more dramatic or "
        f"interesting number is not a reason to mention it. Only ever report figures for "
        f"the exact entities named in the request.\n"
        f"- CRITICAL DATE CHECK: before treating ANY period-based figure (YTD, 'this "
        f"year', 'this quarter', 'year-to-date') as current, check whether the source "
        f"itself states a publication date, byline date, or contextual year (e.g. "
        f"referencing a specific past year's levels or events). If that date is not "
        f"{current_year}, the figure is NOT current no matter what words like 'YTD' or "
        f"'this year' are used in the source -- those words describe the article's OWN "
        f"year, not necessarily today's year. For example, an article titled 'FTSE 100 "
        f"at Five-Year High, 11% YTD Gain' published in 2019, whose own text says '2019 "
        f"YTD', is reporting 2019 data -- do not present that 11% as this year's figure. "
        f"Either exclude such a figure entirely, or explicitly label it with its real "
        f"year and state it is not current.\n"
        f"- When comparing two or more entities, only compare figures that are measured "
        f"over the SAME time period and the same baseline (e.g. don't compare a "
        f"year-to-date figure against a 'gain since some other date' figure as if "
        f"they were the same kind of measurement). If the available figures use "
        f"different periods or bases, say so explicitly instead of computing a "
        f"difference between them.\n"
        f"- RETURN-TYPE CHECK: a 'total return' figure (includes dividends or "
        f"distributions reinvested) is NOT the same kind of number as a 'price return' "
        f"or plain price-change figure (excludes dividends). Before comparing two "
        f"entities' figures, check whether each source specifies which type it's "
        f"reporting (look for words like 'total return', 'dividend-reinvested', 'price "
        f"return', or 'price change'). For example, if one entity's figure is explicitly "
        f"a dividend-reinvested total return and another entity's figure is a price-only "
        f"ETF-tracked return, computing a difference between them overstates the gap -- "
        f"say so explicitly instead, the same way you would flag a time-period mismatch. "
        f"Apply this check to EVERY entity being compared, not just the one where the "
        f"mismatch is most obvious.\n"
        f"- Some sources present data as a table where a label is stated once per "
        f"entity being compared, immediately followed by one value per entity in the "
        f"SAME left-to-right order the entities were introduced (e.g. \"YTD YTD "
        f"+8.50% +21.30%\" means the first-named entity gets +8.50% and the "
        f"second-named entity gets +21.30%). This kind of densely-formatted table is "
        f"just as valid as a plain sentence, and is often MORE precise -- do not skip "
        f"it in favour of an easier-to-read prose figure for a different entity. When "
        f"you see repeated labels followed by a run of values, carefully count "
        f"position to match each value to its entity; do not guess.\n"
        f"- NEVER invent a number for an entity just because the request expects one. "
        f"If, after reviewing everything provided, you cannot find an actual, explicit "
        f"figure for a named entity, say plainly that it could not be found -- a "
        f"fabricated but plausible-looking number is worse than an honest gap, because "
        f"it cannot be told apart from a real one.\n"
        f"- FINANCIAL DOCUMENT CHECK: if the source material is an invoice, receipt, "
        f"quote, purchase order, or bill, always explicitly state the vendor, the "
        f"document's own invoice/reference/PO number (the specific alphanumeric ID "
        f"printed on the document itself, e.g. \"INV-2026-00981\" or \"PO-4471\" -- this "
        f"is easy to drop when more than one such document is being handled at once, "
        f"so treat it as mandatory, not optional, every single time), EVERY individual "
        f"line item together with its own price (not just the aggregate totals with "
        f"per-item prices left out), the subtotal, the tax/GST amount, and the final "
        f"TOTAL DUE / amount payable -- not just a subtotal or first line-item total "
        f"you happen to find. If the requested length is short, keep each item's "
        f"detail brief (e.g. just its name and price, one clause each) so every item "
        f"still fits alongside the required totals and reference number, rather than "
        f"omitting items to save space. If a discount was applied, state it and "
        f"confirm whether tax was computed before or after the discount. Do not "
        f"substitute a subtotal for the total due; if both appear, report both and "
        f"clearly label which one is the final payable amount. This check describes "
        f"what to ADD when pricing/tax/total fields ARE present -- it does NOT mean "
        f"other document types (delivery orders, goods receipt notes, packing "
        f"slips, and similar logistics documents that list items and quantities but "
        f"no prices) should be treated as unextractable or refused. Those documents "
        f"still have real content -- item codes, descriptions, quantities, "
        f"reference/PO/GRN/DO numbers, dates, and party names -- and that content "
        f"must always be reported; simply omit the pricing/subtotal/tax/total "
        f"fields for a document type that never had them, rather than treating "
        f"their absence as a reason to extract nothing at all.\n"
        f"- OCR TABLE ROW-ALIGNMENT CHECK: the source text below was produced by OCR "
        f"and may have its table rows visually reflowed, wrapped, or reordered "
        f"relative to how they appeared on the original page -- a row's item code, "
        f"its description, and its price do not always stay adjacent in the raw "
        f"text. Before pairing an item code with a description, confirm the "
        f"description you are about to use actually makes sense as a description of "
        f"that code's product category (e.g. a code you are treating as a cable "
        f"should have a description that is actually about a cable, not a mouse or a "
        f"charger). Do not simply take the nearest line of description text above or "
        f"below a code and assume it belongs to that code merely because of "
        f"proximity in the raw OCR text -- if a code and the nearby description text "
        f"describe two different kinds of product, look further in the source for "
        f"the description that actually matches, rather than pairing mismatched ones "
        f"together. If a table has more description lines than item codes (or vice "
        f"versa) such that a clean one-to-one pairing genuinely cannot be determined, "
        f"say so explicitly for that specific item rather than guessing a pairing.\n"
        f"- FIELD IS BLANK, NOT MISSING FROM YOU TO INFER: if a specific numeric field "
        f"(a unit price, a line total, a subtotal, a GST amount, or a grand total) has "
        f"no digits printed next to its label anywhere in the source text -- i.e. the "
        f"label appears but the value column after it is empty -- report that field as "
        f"blank/not stated in the source, and say so explicitly (e.g. \"the total due "
        f"is not filled in on this document\"). Do NOT compute, estimate, or otherwise "
        f"produce a number for a field that has no digits printed for it anywhere in "
        f"the source, even if other fields on the same document DO have numbers and a "
        f"plausible-looking total could be inferred from them -- an inferred number for "
        f"a blank field is exactly the kind of fabrication the rule above already "
        f"forbids for missing entities, and it applies with equal force to missing "
        f"numeric fields on partially-filled documents."
    )


def _build_enriched_focus(base_focus: str) -> str:
    """
    Combined form -- base_focus text followed by _enrichment_rules_only().
    Kept only for _build_synthesis_focus_with_gap_check below, which builds
    a prompt for the controller's OWN final-synthesis LLM call rather than
    a summarize-agent task dispatch, so it never goes through the
    cue-detection regexes that made concatenating these two things risky
    for per-task summarize calls (see _enrichment_rules_only's docstring).
    Do not reuse this combined form for anything that gets dispatched to
    byoa-summarize-agent -- send `focus` and `focus_rules` separately
    instead (see the text-summarize dispatch block further below).
    """
    return f"{base_focus}\n\n{_enrichment_rules_only()}"


def _build_synthesis_focus_with_gap_check(goal: str) -> str:
    """
    Same as _build_enriched_focus, plus one more instruction: end the
    response with a machine-parseable marker line listing any explicitly
    named entity that has NO actual current data in the material provided.
    This is what drives the gap-fill retry in synthesize_final_answer --
    only used for the synthesis call, not per-task summaries, since only
    the final synthesis step is in a position to trigger a supplementary
    search.
    """
    return (
        f"{_build_enriched_focus(goal)}\n\n"
        f"- Finally, end your entire response with one line, on its own, in exactly "
        f"this format: \"MISSING_DATA: <comma-separated list>\" listing any entity or "
        f"topic explicitly named in the original request that the material above "
        f"contains NO substantive information about at all -- or \"MISSING_DATA: none\" "
        f"if every named entity is covered. An entity is NOT missing just because the "
        f"material about it is encyclopedic, historical, or lacks a specific kind of "
        f"figure -- if there is real content about the entity, it is covered. Only "
        f"flag an entity when the material genuinely contains nothing about it. This "
        f"must be the literal last line, exactly as specified, so it can be parsed "
        f"programmatically."
    )


def synthesize_final_answer(goal: str, tasks: list, completed: dict, agent_index: dict) -> dict:
    """
    One guaranteed final LLM call: given the full original goal and the raw
    content gathered across every task (regardless of how many chains the
    Goal Expander split the goal into), produce one coherent final answer.

    Includes one adaptive retry: if the model's own MISSING_DATA marker
    flags a named entity with no real data in the gathered material, run
    ONE targeted supplementary search+crawl for just that entity, then
    re-synthesize with the enriched context. Only one retry is attempted,
    regardless of what the second pass reports, to bound cost/latency.

    Returns {"success": bool, "answer": str|None, "error": str|None}.
    """
    agent = _find_summarize_agent(agent_index)
    if not agent:
        return {"success": False, "answer": None, "error": "No text-summarize agent registered",
                "missing_entities_detected": [], "gap_fill_triggered": False, "supplementary_search_query": None, "supplementary_content_found": False, "supplementary_sources_succeeded": 0, "supplementary_content_preview": None}

    context = _build_synthesis_context(tasks, completed, agent_index)
    if not context.strip():
        return {"success": False, "answer": None, "error": "No successful task content to synthesize",
                "missing_entities_detected": [], "gap_fill_triggered": False, "supplementary_search_query": None, "supplementary_content_found": False, "supplementary_sources_succeeded": 0, "supplementary_content_preview": None}

    def _call_summarize(text_input: str):
        url = agent["endpoint_url"].rstrip("/") + agent["invoke_path"]
        ts = int(time.time())
        tid = "final_synthesis"
        msg = f"{tid}:{goal}:text-summarize:{ts}".encode()
        sig = _hmac.new(get_agent_secret(agent["agent_id"]).encode(), msg, hashlib.sha256).hexdigest()
        json_body = {
            "task_id": tid,
            "goal_id": goal,
            "capability": "text-summarize",
            "inputs": {"text": text_input, "focus": _build_synthesis_focus_with_gap_check(goal)},
            "timeout_secs": AGENT_CALL_TIMEOUT,
            "hmac_sig": sig,
            "timestamp": ts,
        }
        resp = req.post(
            url, json=json_body,
            headers={"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"},
            timeout=AGENT_CALL_TIMEOUT,
        )
        body = resp.json()
        summary = body.get("outputs", {}).get("summary")
        if resp.status_code < 400 and summary:
            return summary, None
        return None, body.get("error_msg") or f"HTTP {resp.status_code}"

    try:
        _synth_t0 = time.time()
        summary, error = _call_summarize(context)
        print(f"[TIMING] final-synthesis context_chars={len(context)} "
              f"elapsed={time.time() - _synth_t0:.2f}s "
              f"{'FAILED: ' + error if error else 'ok'}", flush=True)
        if error:
            return {"success": False, "answer": None, "error": error,
                    "missing_entities_detected": [], "gap_fill_triggered": False, "supplementary_search_query": None, "supplementary_content_found": False, "supplementary_sources_succeeded": 0, "supplementary_content_preview": None}

        clean_answer, missing_entities = _parse_missing_data_marker(summary)

        gap_fill_triggered = False
        supplementary_search_query = None
        supplementary_content_found = False
        supplementary_sources_succeeded = 0
        supplementary_content_preview = None

        # if missing_entities:
        #     supplementary = _run_supplementary_search_and_crawl(missing_entities, agent_index, goal)
        #     supplementary_search_query = supplementary["query"]
        #     supplementary_content_found = supplementary["found"]
        #     supplementary_sources_succeeded = supplementary["sources_succeeded"]
        #
        #     if supplementary["found"]:
        #         supplementary_content_preview = supplementary["text"][:500]
        #         enriched_context = (
        #             context
        #             + f"\n\n[Supplementary search for missing data: {', '.join(missing_entities)}]\n"
        #             + supplementary["text"]
        #         )
        #         if len(enriched_context) > SYNTHESIS_MAX_CONTEXT_CHARS:
        #             enriched_context = enriched_context[:SYNTHESIS_MAX_CONTEXT_CHARS] + "\n...[truncated]"
        #
        #         retry_summary, retry_error = _call_summarize(enriched_context)
        #         if not retry_error and retry_summary:
        #             gap_fill_triggered = True
        #             clean_answer, _ = _parse_missing_data_marker(retry_summary)
        #         # If the retry call itself failed, fall back to the first
        #         # pass's answer rather than losing the result entirely.
        #     # If supplementary["found"] is False, the entity genuinely
        #     # isn't findable via a targeted search either -- deliberately
        #     # do NOT retry synthesis in that case, since re-prompting with
        #     # no new information risks the model inventing a number just
        #     # because it was told a search happened. The first pass's
        #     # honest "missing" answer is kept as-is.

        return {
            "success": True,
            "answer": clean_answer,
            "error": None,
            "missing_entities_detected": missing_entities,
            "gap_fill_triggered": gap_fill_triggered,
            "supplementary_search_query": supplementary_search_query,
            "supplementary_content_found": supplementary_content_found,
            "supplementary_sources_succeeded": supplementary_sources_succeeded,
            "supplementary_content_preview": supplementary_content_preview,
        }
    except Exception as e:
        return {"success": False, "answer": None, "error": str(e),
                "missing_entities_detected": [], "gap_fill_triggered": False, "supplementary_search_query": None, "supplementary_content_found": False, "supplementary_sources_succeeded": 0, "supplementary_content_preview": None}


# ── Flask app (Cell 7, unchanged) ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}), 200


# ── Search-results fan-out crawl (new) ─────────────────────────────────────
# Rationale: every ranking heuristic tried so far (keyword relevance, numeric
# density, thin-content penalty) has eventually been fooled by some search
# result shape we didn't anticipate -- marketing pages, date-heavy
# dashboards, keyword-stuffed thin pages. Rather than keep patching an
# inherently fuzzy "pick the one best URL" judgment call, this crawls EVERY
# result a web-search-capable agent returned and combines the text, removing
# the selection problem entirely. The downstream summarize/synthesis steps
# already handle sorting useful signal from noise across multiple sources.

FANOUT_MAX_URLS = 8
FANOUT_TOTAL_MAX_CHARS = 20000

# Per-source budget is rank-weighted, not flat. A flat cap (e.g. 4000 chars
# for every source regardless of relevance) meant the single MOST relevant
# source -- the one most likely to contain the actual answer, e.g. a full
# "top 10" ranking table -- got cut off at the exact same point as every
# other, less relevant source. In practice this chopped ranking tables and
# lists in half partway through, even when that source was clearly the best
# one crawled. Instead, the top few ranks (after relevance sorting) get much
# larger slices, tapering down for merely-supplementary sources further down
# the list. FANOUT_TOTAL_MAX_CHARS below still applies as a hard safety net,
# so overall combined size stays roughly the same as before.
#
# LATENCY NOTE: FANOUT_MAX_URLS was previously 15. The cumulative budget for
# just the top 4 ranks (8000+6000+4000+3000=21000) already exceeds
# FANOUT_TOTAL_MAX_CHARS -- meaning, whenever top-ranked sources have
# substantial real content (the normal case, since that's *why* they ranked
# highest), everything crawled beyond roughly rank 4-6 gets silently
# truncated away before summarize ever sees it. We were paying the full
# network cost of crawling up to 15 sources concurrently for content that
# usually never survived to the final combined text anyway. Cutting to 8
# keeps a comfortable margin above where contribution realistically stops
# (covering the case where top sources turn out unusually short), while
# eliminating the tail that math shows rarely if ever mattered.
FANOUT_PER_SOURCE_BUDGET_BY_RANK = [8000, 6000, 4000, 3000, 2000]
FANOUT_PER_SOURCE_DEFAULT_BUDGET = 1500  # rank 6+ (supplementary sources)


def _fanout_source_budget(rank: int) -> int:
    if rank < len(FANOUT_PER_SOURCE_BUDGET_BY_RANK):
        return FANOUT_PER_SOURCE_BUDGET_BY_RANK[rank]
    return FANOUT_PER_SOURCE_DEFAULT_BUDGET


def _detect_fanout_crawl(task: dict, completed: dict, agent_index: dict, crawl_agent: dict):
    """
    If this web-crawl task's 'url' input references a web-search-capable
    upstream task's 'results' array (e.g. "$task_1.results[0].url"),
    return that upstream task's full results list (and the search query
    that produced them) so we can crawl ALL of them instead of resolving
    to a single URL. Returns None if this task doesn't match that pattern.
    """
    if "web-crawl" not in crawl_agent.get("capabilities", []):
        return None

    url_template = task.get("inputs", {}).get("url")
    if not isinstance(url_template, str) or not url_template.startswith("$task_"):
        return None

    m = _re.match(r"^\$task_(\d+)\.results", url_template)
    if not m:
        return None

    upstream_id = int(m.group(1))
    upstream_result = completed.get(upstream_id)
    if not upstream_result or not upstream_result.get("success"):
        return None

    upstream_agent = agent_index.get(upstream_result.get("agent_id"))
    if not upstream_agent or "web-search" not in upstream_agent.get("capabilities", []):
        return None

    response = upstream_result.get("response")
    outputs = response.get("outputs") if isinstance(response, dict) else None
    results = outputs.get("results") if isinstance(outputs, dict) else None
    if not isinstance(results, list) or not results:
        return None

    query = outputs.get("query") if isinstance(outputs, dict) else None
    return results, query


def _crawl_one_for_fanout(crawl_agent: dict, target_url: str, max_links, task_id, goal: str):
    url = crawl_agent["endpoint_url"].rstrip("/") + crawl_agent["invoke_path"]
    ts = int(time.time())
    tid = f"{task_id}_fanout"
    msg = f"{tid}:{goal}:web-crawl:{ts}".encode()
    sig = _hmac.new(get_agent_secret(crawl_agent["agent_id"]).encode(), msg, hashlib.sha256).hexdigest()
    json_body = {
        "task_id": tid,
        "goal_id": goal,
        "capability": "web-crawl",
        "inputs": {"url": target_url, "max_links": max_links},
        "timeout_secs": AGENT_CALL_TIMEOUT,
        "hmac_sig": sig,
        "timestamp": ts,
    }
    try:
        resp = req.post(
            url, json=json_body,
            headers={"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"},
            timeout=AGENT_CALL_TIMEOUT,
        )
        body = resp.json()
        outputs = body.get("outputs", {})
        text = outputs.get("text")
        if resp.status_code < 400 and text:
            return {"url": target_url, "title": outputs.get("title", ""), "text": text}
    except Exception:
        pass
    return None


def _diversify_by_keyword_coverage(crawled: list, keywords: list) -> list:
    """
    Takes an already relevance-sorted list of crawled sources and reorders
    it so that, for each distinct query keyword, the single best-scoring
    source that actually covers that keyword is pulled forward -- even if
    that source's OVERALL relevance score (summed across ALL keywords) was
    lower than a broader source that shallowly mentions several keywords.

    Rationale: the underlying relevance score rewards raw keyword-match
    COUNT, which structurally favours a source mentioning several
    requested entities (e.g. "[Index A] vs [Index B]") over a source
    that's entirely about just ONE of them (e.g. "How [Index C]
    performed in 2025") -- even when that single-entity source is exactly
    what's needed to answer the part of the request covering that entity.
    Without this, a genuinely on-topic deep-dive source for a smaller
    entity gets buried behind several shallower multi-entity sources and
    can fall outside the truncation cap entirely.
    """
    if not keywords:
        return crawled

    covered = set()
    reordered = []
    seen_ids = set()

    for kw in keywords:
        if kw in covered:
            continue
        for item in crawled:
            item_id = id(item)
            if item_id in seen_ids:
                continue
            haystack = (item.get("title", "") + " " + item.get("text", "")[:1000]).lower()
            if _re.search(r"\b" + _re.escape(kw) + r"\b", haystack):
                reordered.append(item)
                seen_ids.add(item_id)
                covered.add(kw)
                break

    for item in crawled:
        if id(item) not in seen_ids:
            reordered.append(item)

    return reordered


def crawl_all_search_results(crawl_agent: dict, results: list, query: str, task: dict, goal: str) -> dict:
    """Crawl every result in a web-search agent's results list concurrently,
    order them by relevance to the search query, combine the text, and
    return an outputs dict shaped like a normal webcrawl result so nothing
    downstream needs to change.

    Ordering matters: with up to 15 sources crawled concurrently but only
    FANOUT_TOTAL_MAX_CHARS worth of combined text kept, whichever sources
    happen to finish crawling first (essentially random, since it's
    thread-completion order) would otherwise decide what survives
    truncation -- a genuinely relevant source that's merely a bit slower to
    fetch could get pushed past the cutoff by an off-topic one that
    happened to respond faster. Sorting by the same relevance heuristic
    used elsewhere ensures the most on-topic content is placed first.

    Ordering also feeds _fanout_source_budget(): the top-ranked source gets
    a much larger per-source character slice than lower-ranked ones (see
    FANOUT_PER_SOURCE_BUDGET_BY_RANK), since it's the source most likely to
    contain the actual answer -- e.g. a complete ranking table -- and a flat
    per-source cap was chopping such tables/lists off partway through even
    when that source was clearly the best one crawled.
    """
    max_links = task.get("inputs", {}).get("max_links", 0)
    to_crawl = results[:FANOUT_MAX_URLS]

    _t0 = time.time()
    crawled = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_crawl_one_for_fanout, crawl_agent, r.get("url"), max_links, task["task_id"], goal): r
            for r in to_crawl if r.get("url")
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                crawled.append(res)
    _crawl_elapsed = time.time() - _t0
    print(f"[TIMING] fanout-crawl task={task.get('task_id')} "
          f"attempted={len(to_crawl)} succeeded={len(crawled)} "
          f"fetch_elapsed={_crawl_elapsed:.2f}s", flush=True)

    keywords = _query_keywords(query) if query else []
    recency_query = bool(query and _RECENCY_QUERY_SIGNAL_RE.search(query))

    def _crawled_item_score(c):
        scoring_input = {
            "title": c.get("title", ""),
            "snippet": "",
            "text_preview": c.get("text", "")[:500],
            "text_length": len(c.get("text", "")),
        }
        return _score_search_result(scoring_input, keywords, recency_query)

    crawled.sort(key=_crawled_item_score, reverse=True)
    crawled = _diversify_by_keyword_coverage(crawled, keywords)

    sections = []
    for rank, c in enumerate(crawled):
        text = c["text"][:_fanout_source_budget(rank)]
        sections.append(f"[Source: {c['title']} ({c['url']})]\n{text}")
    combined_text = "\n\n".join(sections)
    if len(combined_text) > FANOUT_TOTAL_MAX_CHARS:
        combined_text = combined_text[:FANOUT_TOTAL_MAX_CHARS] + "\n...[truncated]"
    print(f"[TIMING] fanout-crawl task={task.get('task_id')} "
          f"combined_chars={len(combined_text)} (cap={FANOUT_TOTAL_MAX_CHARS})", flush=True)

    return {
        "text": combined_text,
        "text_length": len(combined_text),
        "title": f"Combined crawl of {len(crawled)} sources",
        "status_code": 200,
        "links": [],
        "sources_attempted": len(to_crawl),
        "sources_succeeded": len(crawled),
        "sources_crawled": [c["url"] for c in crawled],
    }



# Matches a "key: value" frontmatter line -- same format and same
# reasoning as summarize_agent.py's own _FRONTMATTER_KV_RE (kept as a
# separate local copy rather than imported, matching that file's own
# stated convention of not coupling the two services together over a
# shared regex). Used below only to pull "source" specifically, not
# any arbitrary frontmatter key.
_MD_FRONTMATTER_KV_RE = _re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*(?:#.*)?$")

# Same page-boundary marker this OCR pipeline's own MD output always
# uses (see summarize_agent.py's _PAGE_MARKER_RE) -- frontmatter never
# appears past the first page, so this bounds the scan.
_MD_PAGE_MARKER_RE = _re.compile(r"^##\s*Page\s+\d+\s*$")


def _extract_md_source_field(text: str) -> "str | None":
    """
    Deterministically pulls the "source" value out of an OCR'd MD
    document's own frontmatter (the lines before the first "## Page N"
    marker) -- every MD this pipeline produces has one. Used to
    guarantee the final JSON extraction always carries the correct
    "source" filename, rather than leaving it to the model's own
    extraction, which can drop, rename, or mis-transcribe it like any
    other field. Returns None if no such line is found (never raises).
    """
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if _MD_PAGE_MARKER_RE.match(stripped):
            break
        if stripped in ("---", ""):
            continue
        m = _MD_FRONTMATTER_KV_RE.match(stripped)
        if m and m.group(1).lower() == "source":
            value = m.group(2).strip()
            if value:
                return value
    return None


def _dispatch_one_task(task: dict, goal: str, completed_snapshot: dict, agent_index: dict) -> tuple:
    """
    Runs the full dispatch logic for ONE task and returns (lines, result)
    instead of yielding directly -- this lets it run inside a worker thread
    (threads can't yield into a generator on another thread), while being
    byte-for-byte the same logic that used to live inline in execute_tasks'
    main loop. `completed_snapshot` is a plain dict copy of already-finished
    tasks at the moment this task became ready -- safe to read from a
    worker thread since it's never mutated after being handed off, and a
    task's inputs only ever reference dependencies that are guaranteed
    already-finished before this task was marked ready.
    """
    lines = []
    _dispatch_t0 = time.time()

    if task["agent_id"] == COMBINE_AGENT_ID:
        # Controller-native step -- no HTTP call, no registry entry.
        # Still emits the exact same task_result shape as every other
        # task so the frontend needs no special-casing to render it.
        try:
            resolved_inputs = resolve_inputs(
                task, {f"task_{k}": v for k, v in completed_snapshot.items()}, agent_index
            )
            outputs = _run_combine_task(resolved_inputs)
            result = {
                "task_id": task["task_id"],
                "task_description": task.get("task_description", ""),
                "agent_id": COMBINE_AGENT_ID,
                "success": True,
                "status_code": 200,
                "response": {"status": "success", "error_msg": None, "outputs": outputs},
                "error": None,
            }
        except InputResolutionError as e:
            result = {
                "task_id": task["task_id"],
                "task_description": task.get("task_description", ""),
                "agent_id": COMBINE_AGENT_ID,
                "success": False,
                "status_code": None,
                "response": None,
                "error": f"Input resolution failed: {e}",
            }
        except ValueError as e:
            result = {
                "task_id": task["task_id"],
                "task_description": task.get("task_description", ""),
                "agent_id": COMBINE_AGENT_ID,
                "success": False,
                "status_code": None,
                "response": None,
                "error": str(e),
            }
        lines.append(json.dumps({"task_result": result}) + "\n")
        print(f"[TIMING] task={task['task_id']} agent={COMBINE_AGENT_ID} cap=combine "
              f"elapsed={time.time() - _dispatch_t0:.2f}s", flush=True)
        return lines, result

    agent = agent_index.get(task["agent_id"])
    if not agent:
        result = {
            "task_id": task["task_id"],
            "task_description": task.get("task_description", ""),
            "agent_id": task["agent_id"],
            "success": False,
            "response": None,
            "error": f"Agent not found: {task['agent_id']}",
        }
        lines.append(json.dumps({"task_result": result}) + "\n")
        print(f"[TIMING] task={task['task_id']} agent={task['agent_id']} cap=unresolved "
              f"elapsed={time.time() - _dispatch_t0:.2f}s (agent not found)", flush=True)
        return lines, result

    fanout_detection = _detect_fanout_crawl(task, completed_snapshot, agent_index, agent)
    if fanout_detection is not None:
        fanout_results, fanout_query = fanout_detection
        outputs = crawl_all_search_results(agent, fanout_results, fanout_query, task, goal)
        result = {
            "task_id": task["task_id"],
            "task_description": task.get("task_description", ""),
            "agent_id": task["agent_id"],
            "success": True,
            "status_code": 200,
            "response": {"status": "success", "error_msg": None, "outputs": outputs},
            "error": None,
        }
        lines.append(json.dumps({"task_result": result}) + "\n")
        print(f"[TIMING] task={task['task_id']} agent={task['agent_id']} cap=fanout-crawl "
              f"elapsed={time.time() - _dispatch_t0:.2f}s", flush=True)
        return lines, result

    try:
        resolved_inputs = resolve_inputs(task, {f"task_{k}": v for k, v in completed_snapshot.items()}, agent_index)
    except InputResolutionError as e:
        result = {
            "task_id": task["task_id"],
            "task_description": task.get("task_description", ""),
            "agent_id": task["agent_id"],
            "success": False,
            "status_code": None,
            "response": None,
            "error": f"Input resolution failed: {e}",
        }
        lines.append(json.dumps({"task_result": result}) + "\n")
        print(f"[TIMING] task={task['task_id']} agent={task['agent_id']} cap=unresolved "
              f"elapsed={time.time() - _dispatch_t0:.2f}s (input resolution failed)", flush=True)
        return lines, result

    # Short-circuit: if this task's SOLE dependency is a knowledge-retrieve
    # task that itself reported fallback_recommended=True, skip the actual
    # summarize LLM call AND end this branch entirely -- no filler "no
    # confident match" sentence in the final answer either. The summary is
    # left as an empty string, which _run_combine_task's existing
    # empty-entry filter (`entries = [e for e in entries if e[0]]`) already
    # drops silently, so a KB miss simply contributes nothing to the final
    # combined answer instead of cluttering it with an unhelpful aside.
    # This also means zero Ollama cost for a call that, under concurrent
    # load, could otherwise take minutes competing with every OTHER
    # summarize call across all in-flight goals for the same capacity.
    # Only fires for the "report the KB's own findings" leaf pattern (a
    # summarize task depending on nothing but the knowledge-retrieve task);
    # any summarize task with additional dependencies or a different shape
    # still goes through normally.
    if "text-summarize" in agent.get("capabilities", []):
        deps = task.get("depends_on", []) or []
        if len(deps) == 1:
            dep_result = completed_snapshot.get(deps[0])
            if dep_result and dep_result.get("success"):
                dep_outputs = ((dep_result.get("response") or {}).get("outputs") or {})
                if dep_outputs.get("fallback_recommended") is True:
                    fallback_reason = dep_outputs.get("fallback_reason") or "no confident match was found."
                    result = {
                        "task_id": task["task_id"],
                        "task_description": task.get("task_description", ""),
                        "agent_id": task["agent_id"],
                        "success": True,
                        "status_code": 200,
                        "response": {
                            "status": "success",
                            "error_msg": None,
                            "outputs": {
                                "gap_fill_triggered": False,
                                "input_length": 0,
                                "model": None,
                                "summary": "",
                                "short_circuited": True,
                                "skip_reason": fallback_reason,
                            },
                        },
                        "error": None,
                    }
                    lines.append(json.dumps({"task_result": result}) + "\n")
                    print(f"[TIMING] task={task['task_id']} agent={task['agent_id']} cap=text-summarize "
                          f"elapsed={time.time() - _dispatch_t0:.2f}s "
                          f"(short-circuited: KB fallback_recommended=True, branch ended, no LLM call made)", flush=True)
                    return lines, result

    if "text-summarize" in agent.get("capabilities", []):
        # Prefer this task's own description over the full top-level goal.
        # When the Goal Expander splits a goal into parallel branches
        # (e.g. "general updates" + "Topic A" + "Topic B" as three independent
        # search->crawl->summarize chains), each branch's summarize call
        # previously got the ENTIRE combined goal as focus, causing it to
        # comment on topics from OTHER branches it never actually saw
        # content for (e.g. "this text doesn't cover Topic B" on
        # a task whose only job was ever Topic A). Scoping focus
        # to the task's own description keeps each summary narrowly
        # relevant to what that task actually fetched.
        #
        # NOTE: this used to only run when the Goal Expander's own JSON
        # omitted a "focus" field entirely ("focus" not in resolved_inputs).
        # That let generic LLM-authored focus text (e.g. "Summarize the key
        # details and content of this document" from the inline-upload
        # example template) through completely unenriched -- no date check,
        # no entity scoping, no financial-document check. Enrichment now
        # always runs, wrapping whatever focus the Goal Expander supplied
        # (falling back to task_description/goal only if it supplied none),
        # so every summarize call gets the same safety-net rules regardless
        # of what focus text the LLM happened to generate.
        base_focus = (
            resolved_inputs.get("focus")
            or task.get("task_description", "").strip()
            or goal
        )
        # `focus` stays exactly this -- the task's own real ask, nothing
        # appended -- so byoa-summarize-agent's list/extraction cue-
        # detection (which runs on `focus` alone) only ever sees what the
        # task actually requested. The always-on safety-net rules go in
        # their own field instead of being concatenated in; see
        # _enrichment_rules_only's docstring for the bug this fixes.
        resolved_inputs["focus"] = base_focus
        resolved_inputs["focus_rules"] = _enrichment_rules_only()

        # TEMP DEBUG (2026-08): verifying focus_rules actually reaches the
        # summarize agent as its own field, separate from `focus`, after
        # the list-mode-bug fix. Safe to remove once confirmed in
        # journalctl across a few real goals -- this is diagnostic only,
        # not behavior.
        print(f"[FOCUS_RULES_DEBUG] task={task.get('task_id')} "
              f"focus_len={len(resolved_inputs['focus'])} "
              f"focus_preview={resolved_inputs['focus'][:120]!r} "
              f"focus_rules_len={len(resolved_inputs['focus_rules'])} "
              f"focus_rules_preview={resolved_inputs['focus_rules'][:120]!r}",
              flush=True)

        # JSON-output requests get dropped by the Goal Expander's task
        # decomposition (it authors `focus` as a prose instruction template
        # and doesn't reliably carry a format constraint from the goal into
        # it -- see _wants_json_output above). Checked against the ORIGINAL
        # goal, not the task's own focus, so this survives regardless of
        # what the expander generated for this specific task.
        #
        # ALSO auto-triggers with no goal-phrasing requirement at all when
        # this task's text came directly from an uploaded file AND that
        # text itself looks like a scanned business document (invoice, PO,
        # delivery order, GRN, put-away report, etc) -- see
        # _looks_like_financial_document / _task_input_references_upload
        # above. This is what lets a goal like "process this" or "what's in
        # this file" get structured JSON for a financial document without
        # the user ever having to say "json" -- the document's own content
        # decides, not the phrasing of the ask. Scoped to uploads only
        # (not every summarize call) so a crawled article that merely
        # mentions "invoice" a few times can't misfire this.
        #
        # Either branch sets an explicit `output_format` input (which
        # byoa-summarize-agent >=2.1.0 branches on to switch its own prompt
        # and response parsing into strict-JSON mode) as well as appending
        # a plain-language instruction to `focus` itself (deliberately
        # `focus`, not `focus_rules` -- an older agent version that
        # predates the `focus_rules` field would never see it there, and
        # this fallback exists specifically so behavior degrades gracefully
        # to a JSON-esque prose answer on exactly that kind of older
        # agent). This text doesn't contain any of the cue-detection
        # trigger words, so appending it to `focus` here doesn't
        # reintroduce the list-mode bug.
        _resolved_text_for_detection = str(resolved_inputs.get("text", "") or "")
        _auto_json_from_upload = (
            _task_input_references_upload(task)
            and _looks_like_financial_document(_resolved_text_for_detection)
            # A narrow single-fact question (checked against the ORIGINAL
            # goal, same reasoning as _wants_json_output above -- the
            # Goal Expander's own generated `focus` for this task is
            # unreliable, see the inline-example-anchoring note where this
            # task gets built) should get a direct prose answer, not the
            # full structured object, even though the source document
            # itself looks financial.
            and not _looks_like_narrow_factual_question(goal)
        )
        if _wants_json_output(goal) or _auto_json_from_upload:
            resolved_inputs["output_format"] = "json"
            _format_reason = (
                "the requester explicitly asked for this answer as JSON"
                if _wants_json_output(goal) else
                "this is a scanned business document (invoice/PO/delivery "
                "order/GRN/etc), which this platform always structures as "
                "JSON regardless of how the request was phrased"
            )
            resolved_inputs["focus"] = (
                resolved_inputs["focus"] + "\n\n"
                f"- OUTPUT FORMAT: {_format_reason}. "
                "Your entire response must be a single valid JSON object "
                "(no prose before or after it, no markdown code fences) whose fields "
                "cover the same information the instructions above ask for -- e.g. "
                "vendor/customer names, document reference number, dates, a "
                "line_items array (each with its own description, quantity, and "
                "price), subtotal, tax amount, and total. Use null for any field "
                "that is genuinely blank or not stated in the source -- never "
                "invent a value to fill a field just because the schema expects one."
            )

    url = agent["endpoint_url"].rstrip("/") + agent["invoke_path"]
    ctype = agent.get("invoke_content_type", "multipart/form-data")

    capability = task.get("capability", "")
    if not capability:
        caps = agent.get("capabilities", [])
        capability = caps[0] if caps else ""

    try:
        if capability in ASYNC_CAPABILITIES:
            # These capabilities run long enough, often enough, to
            # occasionally (data-matching: under concurrent vLLM load;
            # text-summarize: on long crawled text) exceed Cloudflare
            # Tunnel's ~100s edge timeout when called via a single
            # blocking request -- see ASYNC_CAPABILITIES and
            # call_agent_async's docstring for the confirmed production
            # failure this fixed for data-matching specifically. Route
            # through submit-then-poll instead -- every individual HTTP
            # request completes in well under a second, so none of them
            # can ever hit that ceiling. All other capabilities are
            # untouched and keep using the existing synchronous paths
            # below.
            async_result = call_agent_async(
                agent, capability, task["task_id"], goal, resolved_inputs,
                max_wait=AGENT_CALL_TIMEOUT,
            )
            result = {
                "task_id": task["task_id"],
                "task_description": task.get("task_description", ""),
                "agent_id": task["agent_id"],
                "success": async_result["success"],
                "status_code": async_result["status_code"],
                "response": async_result["response"],
                "error": async_result["error"],
            }
        elif "application/json" in ctype:
            ts = int(time.time())
            tid = str(task["task_id"])
            msg = f"{tid}:{goal}:{capability}:{ts}".encode()
            sig = _hmac.new(get_agent_secret(agent["agent_id"]).encode(), msg, hashlib.sha256).hexdigest()

            json_body = {
                "task_id": tid,
                "goal_id": goal,
                "capability": capability,
                "inputs": resolved_inputs,
                "timeout_secs": AGENT_CALL_TIMEOUT,
                "hmac_sig": sig,
                "timestamp": ts,
            }
            resp = req.post(
                url, json=json_body,
                headers={"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"},
                timeout=AGENT_CALL_TIMEOUT,
            )
            try:
                final = resp.json()
            except Exception:
                final = {"raw": resp.text[:500]}

            result = {
                "task_id": task["task_id"],
                "task_description": task.get("task_description", ""),
                "agent_id": task["agent_id"],
                "success": resp.status_code < 400,
                "status_code": resp.status_code,
                "response": final,
                "error": None,
            }
        else:
            schema = agent.get("input_schema", {})
            files, fields = {}, {}
            for k, v in resolved_inputs.items():
                if schema.get(k, {}).get("type") == "file":
                    files[k] = (k, v)
                else:
                    fields[k] = str(v)
            if not files:
                files = {"_dummy": ("", b"")}

            with req.post(
                url, data=fields, files=files,
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=AGENT_CALL_TIMEOUT, stream=True,
            ) as resp:
                agent_lines = []
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                        lines.append(json.dumps({
                            "task_id": task["task_id"], "agent_id": task["agent_id"], "stream": decoded,
                        }) + "\n")
                        agent_lines.append(decoded)

                final = None
                for line in reversed(agent_lines):
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "keepalive" not in parsed and "msg" not in parsed:
                            final = parsed
                            break
                    except Exception:
                        pass

                result = {
                    "task_id": task["task_id"],
                    "task_description": task.get("task_description", ""),
                    "agent_id": task["agent_id"],
                    "success": resp.status_code < 400,
                    "status_code": resp.status_code,
                    "response": final,
                    "error": None,
                }

    except req.exceptions.Timeout:
        result = {
            "task_id": task["task_id"],
            "task_description": task.get("task_description", ""),
            "agent_id": task["agent_id"],
            "success": False,
            "status_code": None,
            "response": None,
            "error": f"Timeout after {AGENT_CALL_TIMEOUT}s",
        }
    except Exception as e:
        result = {
            "task_id": task["task_id"],
            "task_description": task.get("task_description", ""),
            "agent_id": task["agent_id"],
            "success": False,
            "status_code": None,
            "response": None,
            "error": str(e),
        }

    # Deterministic "always include source" guarantee: every MD this OCR
    # pipeline produces carries its own "source" line in its frontmatter
    # (see _extract_md_source_field above), so rather than leave it to
    # the model's own extraction -- which, like any other field, can
    # drop it, rename it, or get it wrong -- pull it directly from the
    # task's own resolved `text` input and stamp it onto the result
    # here, in-controller, after the call. Deliberately OVERWRITES
    # whatever (if anything) the model itself put under "source": the
    # frontmatter value is ground truth, and a model-produced copy of
    # it is exactly as error-prone as any other extracted field would
    # be. Wrapped in a blanket try/except -- this is a best-effort
    # enrichment on top of an already-successful task result, so a bug
    # here must never turn a good result into a failure; `resolved_inputs`
    # itself may not even exist yet if the exception above fired before
    # it was ever assigned, which the bare except below also covers.
    try:
        outputs = (result.get("response") or {}).get("outputs")
        if isinstance(outputs, dict) and outputs.get("output_format") == "json":
            structured = outputs.get("structured")
            if isinstance(structured, dict):
                md_source = _extract_md_source_field(str(resolved_inputs.get("text", "") or ""))
                if md_source:
                    structured["source"] = md_source
    except Exception:
        pass

    lines.append(json.dumps({"task_result": result}) + "\n")
    print(f"[TIMING] task={task['task_id']} agent={task['agent_id']} cap={capability} "
          f"elapsed={time.time() - _dispatch_t0:.2f}s", flush=True)
    return lines, result


def _is_auto_approve_eligible(tasks: list, agent_index: dict, doc_refs: list = None) -> bool:
    """
    True when the plan is safe to run without a human approval step.

    Two independent ways to qualify:

    1. DETERMINISTIC JSON-COUNT RULE (checked first): if `doc_refs`
       contains more than one uploaded/available JSON document, this is
       a cross-document matching goal by construction -- see the
       "cross-document matching only triggers for JSON documents" design
       decision -- so it auto-approves outright, regardless of exactly
       how the Goal Expander happened to shape the plan's tasks/agent_id
       assignments. This exists because the per-task capability check
       below, while correct, is only as good as the LLM-produced plan
       matching it exactly; a goal with 2+ JSON files was occasionally
       falling through to the manual approval gate on a technicality
       (an extra task, a capability-set mismatch, an agent_id typo) even
       though the goal was unambiguously a read-only comparison. Counting
       JSON files is a fact about the INPUT, not the LLM's plan, so it
       can't drift the way the capability check can.
       NOTE: this rule does not itself inspect task capabilities, so it
       relies on the "matching-only-for-JSON" invariant holding -- if a
       write/submit-capable task type is ever added to this controller's
       task universe, this rule must be revisited so it can't be used to
       smuggle a submission through alongside a multi-JSON upload.

    2. CAPABILITY-SUBSET RULE (fallback): EVERY task in the plan is
       dispatched to an agent whose declared capabilities are entirely
       within AUTO_APPROVE_CAPABILITIES -- i.e. the whole plan, not just
       part of it, is safe to run unattended. A plan that mixes a
       data-matching task with anything else (e.g. a summarise task
       explaining the result, or -- the case this specifically must
       never allow -- a submission/write task) does NOT qualify under
       this rule; mixed plans keep requiring the normal approval gate.

    An empty task list is never eligible (nothing to safely skip approval
    FOR), regardless of doc_refs.
    """
    if not tasks:
        return False

    json_doc_count = sum(1 for d in (doc_refs or []) if d.get("is_json"))
    if json_doc_count > 1:
        return True

    for t in tasks:
        agent = agent_index.get(t.get("agent_id"))
        if not agent:
            return False
        caps = set(agent.get("capabilities", []))
        if not caps or not caps.issubset(AUTO_APPROVE_CAPABILITIES):
            return False
    return True


def execute_tasks(goal: str, tasks: list, agent_index: dict, preseeded: dict = None,
                  result_sink: dict = None):
    """
    Wave-based concurrent dispatcher. Tasks whose dependencies are all
    already completed are considered "ready" and dispatched together, up
    to MAX_PARALLEL_AGENTS at a time -- genuinely concurrent HTTP calls to
    different agents, not the previous strictly-sequential task-by-task
    loop (which processed even fully-independent tasks, like two parallel
    search branches with empty depends_on, one at a time in list order,
    silently serializing what the plan itself already marked as
    independent). Each task's own dispatch logic is unchanged, byte for
    byte -- see _dispatch_one_task above -- only the scheduling around it
    is new.

    `preseeded`, when present, seeds `completed` with pseudo-task entries
    (see preseed_uploaded_docs()) BEFORE any real task runs -- so a task
    referencing "$task_9001.text" (reserved IDs starting at
    UPLOAD_TASK_ID_BASE) resolves immediately, no agent call involved,
    exactly like any other already-finished dependency.

    `result_sink`, when passed a dict, gets `completed` written into it
    (result_sink["completed"] = completed) right before this generator's
    final yields -- this is a generator, so its local `completed` dict
    isn't otherwise reachable by the caller once the loop finishes. Used
    by approve_goal() to extract a final answer for conversation history
    (see _extract_history_answer) without changing this function's
    yield/return contract for its only other caller.
    """
    completed = dict(preseeded or {})
    tasks_by_id = {t["task_id"]: t for t in tasks}
    remaining = set(tasks_by_id.keys())
    in_flight = {}  # future -> task_id

    def is_ready(tid):
        return all(dep in completed for dep in tasks_by_id[tid].get("depends_on", []))

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as pool:
        while remaining or in_flight:
            newly_ready = [tid for tid in list(remaining) if is_ready(tid)]
            for tid in newly_ready:
                remaining.discard(tid)
                # Snapshot completed-so-far at launch time -- safe to read
                # from the worker thread since it's never mutated again.
                snapshot = dict(completed)
                fut = pool.submit(_dispatch_one_task, tasks_by_id[tid], goal, snapshot, agent_index)
                in_flight[fut] = tid

            if not in_flight:
                # Nothing ready and nothing in flight, but tasks remain --
                # a malformed/circular dependency. Fail them explicitly
                # rather than hanging forever.
                for tid in remaining:
                    result = {
                        "task_id": tid,
                        "task_description": tasks_by_id[tid].get("task_description", ""),
                        "agent_id": tasks_by_id[tid].get("agent_id", ""),
                        "success": False,
                        "status_code": None,
                        "response": None,
                        "error": "Unresolvable dependency (circular or missing task_id reference)",
                    }
                    completed[tid] = result
                    yield json.dumps({"task_result": result}) + "\n"
                remaining.clear()
                break

            # Wait for at least one in-flight task to finish, but re-check
            # periodically so a keepalive goes out if everything currently
            # running is taking a while -- same purpose the old per-dep
            # keepalive loop served, just tied to real waiting this time.
            done, _ = wait(in_flight.keys(), timeout=5, return_when="FIRST_COMPLETED")
            if not done:
                yield json.dumps({"keepalive": True}) + "\n"
                continue

            for fut in done:
                tid = in_flight.pop(fut)
                lines, result = fut.result()
                completed[tid] = result
                for line in lines:
                    yield line

    if result_sink is not None:
        result_sink["completed"] = completed

    yield json.dumps({"status": "COMPLETED"}) + "\n"

    yield json.dumps({"done": True, "goal": goal, "total_tasks": len(tasks)}) + "\n"


def _extract_history_answer(tasks: list, completed: dict, agent_index: dict) -> str:
    """
    Pulls a single, reasonably-sized "the answer this turn gave" string out
    of a finished plan's results, for conversation history (see
    append_conversation_turn). This mirrors -- rather than duplicates -- the
    same leaf logic _inject_combine_task already used to decide what the
    frontend's actual final answer is made of:

      - If the plan has a byoa-combine-agent task (i.e. expand_goal found
        more than one independent leaf), that task's "combined" output IS
        the final answer the frontend renders -- use it directly.
      - Otherwise there is at most one leaf (a task nothing else in the
        plan depends on). Use whatever that leaf's own task produced
        (its "summary" field if it has one, else "text", else a compact
        dump of its outputs) -- same preference order as
        _extract_raw_content_for_synthesis, reused here for consistency.
      - If neither is found (e.g. every task failed), fall back to a short
        note saying so rather than silently storing an empty answer, so a
        later turn's history block doesn't imply an answer was given when
        none actually was.

    Truncation to MAX_HISTORY_ANSWER_CHARS happens in
    append_conversation_turn, not here -- this only picks WHICH text to use.
    """
    combine_task = next((t for t in tasks if t.get("agent_id") == COMBINE_AGENT_ID), None)
    if combine_task:
        result = completed.get(combine_task["task_id"])
        if result and result.get("success"):
            combined = result.get("response", {}).get("outputs", {}).get("combined")
            if combined:
                return str(combined)

    depended_on = set()
    for t in tasks:
        depended_on.update(t.get("depends_on", []) or [])
    leaves = [t for t in tasks if t["task_id"] not in depended_on and t.get("agent_id") != COMBINE_AGENT_ID]

    for leaf in leaves:
        result = completed.get(leaf["task_id"])
        if result and result.get("success"):
            content = _extract_raw_content_for_synthesis(result)
            if content:
                return content

    return "(no answer was produced for this turn)"


# ── Async goal-planning job store ────────────────────────────────────────────
# POST /goal used to block synchronously on expand_goal() (an Ollama call)
# until the plan was ready, with nothing sent to the client in the meantime.
# Under concurrent /goal requests, Ollama serializes inference on the single
# local GPU, so requests queue behind each other -- and since this endpoint
# had no keepalive mechanism (unlike /goal/<plan_id>/approve's streaming
# generator), a queued request sitting silently past Cloudflare Tunnel's
# ~100s edge timeout produced a 524 before the plan was ever returned, even
# though expand_goal() may have still been working fine in the background.
#
# Fixed the same way as the summarize-agent timeout: submit returns
# immediately with a job_id, and the actual planning work (registry fetch +
# expand_goal + pending-plan creation) runs in a background thread. The
# caller polls /goal/<job_id>/status every few seconds -- each individual
# HTTP request (submit + each poll) completes in well under a second, so
# none of them can ever hit the tunnel's edge timeout, no matter how long
# Ollama's queue backs up under concurrent load.
_goal_jobs = {}
_goal_jobs_lock = threading.Lock()
_GOAL_JOB_TTL_SECS = 3600  # stale/completed planning jobs older than this are dropped on next access


def _prune_stale_goal_jobs():
    cutoff = time.time() - _GOAL_JOB_TTL_SECS
    with _goal_jobs_lock:
        stale = [jid for jid, j in _goal_jobs.items()
                 if j.get("status") in ("done", "error") and j.get("completed_at", cutoff) < cutoff]
        for jid in stale:
            del _goal_jobs[jid]


# ── Completed-goal store (feedback loop) ────────────────────────────────────
# plan_id is popped from _pending_plans the moment approve_goal() runs (see
# pop_pending_plan) -- correct for the plan-execution lifecycle, but it
# means there's nothing left to look up once a user tries to RATE that same
# plan_id afterward. This is a separate, longer-lived record (see
# COMPLETED_GOAL_TTL_SECS) written once execution finishes, keyed by the
# same plan_id, holding exactly what POST /feedback needs to build a
# feedback-memory row: the goal text, the final answer, which
# agents/capabilities the DAG actually used, and a short human-readable
# summary of the approach taken.
_completed_goals = {}
_completed_goals_lock = threading.Lock()


def _prune_stale_completed_goals():
    cutoff = time.time() - COMPLETED_GOAL_TTL_SECS
    with _completed_goals_lock:
        stale = [pid for pid, g in _completed_goals.items() if g.get("completed_at", cutoff) < cutoff]
        for pid in stale:
            del _completed_goals[pid]


def _summarize_dag(tasks: list, agent_index: dict) -> str:
    """
    Short, human-readable description of the approach a completed goal's
    DAG took -- e.g. "1. Search for X (byoa-websearch-agent) -> 2. Fetch
    top result (byoa-webcrawl-agent) -> 3. Summarise (byoa-summarize-agent)".
    Stored alongside a feedback rating so a 5-star match can be surfaced to
    the Goal Expander as a reusable pattern, and a 1-star match's comment
    can be read next to the approach that actually produced it.
    """
    parts = []
    for t in sorted(tasks, key=lambda t: t.get("task_id", 0)):
        agent = agent_index.get(t.get("agent_id"), {})
        display = agent.get("display_name") or t.get("agent_id", "?")
        parts.append(f'{t.get("task_id")}. {t.get("task_description", "")} ({display})')
    return " -> ".join(parts)


def _capabilities_used(tasks: list, agent_index: dict) -> list:
    caps = set()
    for t in tasks:
        agent = agent_index.get(t.get("agent_id"), {})
        caps.update(agent.get("capabilities", []))
    return sorted(caps)


def _build_experience_block(goal: str) -> str:
    """
    Queries feedback memory for past goals similar to this one and builds
    a prompt block for the Goal Expander -- this is the actual "learn from
    wins AND mistakes automatically" mechanism, fully unattended, no human
    review required. It's an enhancement, not a dependency: any failure
    (feedback service down, slow, empty collection) falls back to "" so a
    new/cold platform, or a temporarily unreachable feedback service,
    degrades to exactly today's behavior rather than failing the goal.

    GOOD exemplars are injected per-match, same as always -- each reinforces
    a specific pattern that worked, with its rating shown (a high number
    next to "this worked" is safe, reinforcing signal).

    BAD feedback is handled differently, based on what real testing on
    this platform actually showed: a SINGLE low-rated similar goal is
    unreliable on its own -- even with the exact wrong task structure
    stripped out of what gets shown, one "rated 1/5" entry sitting next to
    a near-identical goal still correlated with the model building a LESS
    structured plan next time (skipping a step that was actually needed).
    Rather than disable automatic use of bad feedback entirely, it's now
    gated on CORROBORATION: bad feedback only becomes an active signal once
    at least FEEDBACK_BAD_CORROBORATION_MIN similar goals were independently
    rated poorly. Below that threshold, it's stored (and visible via
    /feedback/list) but not yet trusted. Once corroborated, every matching
    bad rating's comment is aggregated into ONE affirmative requirement --
    not shown as N separate "here's a failure" anecdotes with rating
    numbers attached. Two reasons: (1) LLMs are weak at in-context negation,
    so stating what TO do lands more reliably than showing what NOT to do;
    (2) dropping the numeric rating removes the specific "low score next to
    a near-identical goal" anchor that correlated with harm even after the
    task-structure detail was already removed.
    """
    try:
        r = req.post(
            f"{FEEDBACK_BASE_URL}{FEEDBACK_QUERY_PATH}",
            json={"goal_text": goal, "top_k": 10},
            timeout=FEEDBACK_QUERY_TIMEOUT_SECS,
        )
        r.raise_for_status()
        matches = r.json().get("matches", [])
    except Exception:
        return ""

    if not matches:
        return ""

    # Drop anything below the similarity floor BEFORE bucketing by rating --
    # an unrelated entry with a low similarity score shouldn't be treated
    # as relevant prior experience just because it happened to be the
    # nearest thing available in a small feedback store.
    matches = [m for m in matches if (m.get("similarity") or 0) >= FEEDBACK_MIN_SIMILARITY]
    if not matches:
        return ""

    good = [m for m in matches if (m.get("rating") or 0) >= FEEDBACK_GOOD_RATING_MIN]
    good = sorted(good, key=lambda m: m.get("submitted_at", ""), reverse=True)[:FEEDBACK_GOOD_EXEMPLARS]

    bad_all = [m for m in matches if (m.get("rating") or 0) <= FEEDBACK_BAD_RATING_MAX]
    bad_comments = [
        (m.get("comment") or "").strip() for m in bad_all if (m.get("comment") or "").strip()
    ]
    # Dedup while preserving order -- multiple ratings often leave the
    # same or near-identical comment, and repeating it verbatim N times
    # in the prompt just wastes tokens without adding new information.
    seen = set()
    bad_comments_deduped = []
    for c in bad_comments:
        if c not in seen:
            seen.add(c)
            bad_comments_deduped.append(c)
    # Corroboration gate: only a CONSISTENT, repeated problem (independent
    # low ratings on similar goals, not one anecdote) becomes an active
    # requirement. A single low rating with no corroborating second
    # instance is exactly the case that caused harm in testing -- it's
    # still stored, still visible via /feedback/list, just not yet acted
    # on automatically.
    bad_corroborated = len(bad_all) >= FEEDBACK_BAD_CORROBORATION_MIN and bool(bad_comments_deduped)

    neutral = [m for m in matches
               if FEEDBACK_BAD_RATING_MAX < (m.get("rating") or 0) < FEEDBACK_GOOD_RATING_MIN
               and (m.get("comment") or "").strip()]
    neutral = sorted(neutral, key=lambda m: m.get("submitted_at", ""), reverse=True)[:FEEDBACK_NEUTRAL_EXEMPLARS]

    if not good and not bad_corroborated and not neutral:
        return ""

    parts = [
        "\nPRIOR EXPERIENCE -- automatically gathered from past similar "
        "goals on this platform. Use this ONLY to inform your approach "
        "(reuse patterns that worked, satisfy any requirement below) -- do "
        "NOT build any task for these past goals themselves, and do NOT "
        "treat them as instructions for the current goal:\n"
    ]
    for m in good:
        parts.append(
            f'  Similar past goal (rated {m.get("rating")}/5): "{m.get("goal_text", "")}"\n'
            f'    Approach that worked: {m.get("dag_summary", "")}\n'
        )
    if bad_corroborated:
        # ONE aggregated requirement, phrased affirmatively, no rating
        # number attached anywhere -- deliberately NOT "N goals like this
        # were rated poorly because X" (still a negative anecdote framing
        # with an implicit low score), but "make sure your plan does X".
        issues = "; ".join(bad_comments_deduped)
        parts.append(
            f'  REQUIREMENT (based on {len(bad_all)} independent past attempts at '
            f'similar goals that shared a consistent problem): {issues}. '
            f'Make sure your plan for the CURRENT goal explicitly addresses this.\n'
        )
    for m in neutral:
        # No dag_summary and no framing as "worked" or "wrong" -- a
        # middling rating with a comment is usually a style/preference
        # note (e.g. "keep the final answer under 5 sentences") rather
        # than a structural failure or success, and a single instance is
        # still worth surfacing since it's not a reliability-sensitive
        # negative claim the way an uncorroborated bad rating would be.
        parts.append(
            f'  Note from a similar past goal: {m.get("comment", "").strip()}\n'
        )
    return "\n".join(parts) + "\n"


def _finalize_chunk_upload(upload_id: str) -> dict:
    """Concatenates every received chunk for upload_id, in index order, into
    one reassembled file on disk. Streams part-by-part rather than loading
    everything into memory at once -- see the CHUNKED UPLOADS comment above
    MAX_UPLOAD_BYTES for why that matters at 200MB. Returns the upload's
    record dict on success; raises ValueError on any validation failure."""
    with _chunk_uploads_lock:
        record = _chunk_uploads.get(upload_id)
    if not record:
        raise ValueError(f"Unknown or expired upload_id: {upload_id}")
    if len(record["received"]) != record["total_chunks"]:
        missing = sorted(set(range(record["total_chunks"])) - record["received"])
        raise ValueError(
            f"Upload {upload_id} is incomplete: {len(record['received'])}/{record['total_chunks']} "
            f"chunks received (missing indices: {missing[:10]}{'...' if len(missing) > 10 else ''})"
        )

    upload_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
    final_path = os.path.join(upload_dir, "_assembled")
    total_size = 0
    with open(final_path, "wb") as out:
        for i in range(record["total_chunks"]):
            part_path = os.path.join(upload_dir, f"{i:06d}.part")
            with open(part_path, "rb") as part:
                while True:
                    buf = part.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
                    total_size += len(buf)

    if total_size > MAX_UPLOAD_BYTES:
        os.remove(final_path)
        raise ValueError(f"{record['filename']} exceeds {MAX_UPLOAD_BYTES} byte limit ({total_size} bytes)")

    # Raw .part files are no longer needed once concatenated -- drop them
    # immediately rather than waiting for TTL cleanup, since they're pure
    # duplication of what's now sitting in _assembled.
    for i in range(record["total_chunks"]):
        part_path = os.path.join(upload_dir, f"{i:06d}.part")
        try:
            os.remove(part_path)
        except OSError:
            pass

    with _chunk_uploads_lock:
        record["status"] = "ready"
        record["size"] = total_size
        record["final_path"] = final_path
    return record


@app.route("/upload/chunk", methods=["POST"])
def upload_chunk():
    """Receives ONE piece of a larger file being split client-side to stay
    under Cloudflare's ~100MB edge cap (see CHUNKED UPLOADS comment above).
    Expected multipart/form-data fields:
      upload_id     -- client-generated UUID, same value on every chunk of
                        this file (e.g. via crypto.randomUUID() / uuid4())
      filename      -- original filename, same on every chunk
      chunk_index   -- 0-based index of THIS chunk
      total_chunks  -- total number of chunks for this file
      chunk         -- the raw bytes for this piece (file field)
    Chunks may arrive out of order or with some retried/duplicated -- this
    just overwrites the same index's part file, so a retry is always safe.
    """
    gate = _controller_gate()
    if gate:
        return gate

    _prune_stale_chunk_uploads()

    upload_id = (request.form.get("upload_id") or "").strip()
    filename = (request.form.get("filename") or "").strip()
    try:
        chunk_index = int(request.form.get("chunk_index", ""))
        total_chunks = int(request.form.get("total_chunks", ""))
    except ValueError:
        return jsonify({"error": "chunk_index and total_chunks must be integers"}), 400

    if not upload_id or not _re.fullmatch(r"[a-zA-Z0-9\-]{1,64}", upload_id):
        return jsonify({"error": "upload_id is required and must be a simple id/UUID string"}), 400
    if not filename:
        return jsonify({"error": "filename is required"}), 400
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return jsonify({"error": f"Unsupported file type: {filename}"}), 400
    if total_chunks <= 0 or not (0 <= chunk_index < total_chunks):
        return jsonify({"error": "chunk_index/total_chunks out of range"}), 400
    if "chunk" not in request.files:
        return jsonify({"error": "Missing file field: chunk"}), 400

    f = request.files["chunk"]
    content = f.stream.read()
    if len(content) == 0:
        return jsonify({"error": "chunk is empty"}), 400
    if len(content) > CHUNK_UPLOAD_MAX_CHUNK_BYTES:
        return jsonify({"error": f"chunk exceeds {CHUNK_UPLOAD_MAX_CHUNK_BYTES} byte limit"}), 400

    with _chunk_uploads_lock:
        record = _chunk_uploads.get(upload_id)
        if record is None:
            record = {
                "filename": secure_filename(filename) or "upload",
                "total_chunks": total_chunks,
                "received": set(),
                "status": "receiving",
                "created_at": time.time(),
            }
            _chunk_uploads[upload_id] = record
        elif record["total_chunks"] != total_chunks:
            return jsonify({"error": "total_chunks mismatch with earlier chunks for this upload_id"}), 400

    upload_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    part_path = os.path.join(upload_dir, f"{chunk_index:06d}.part")
    with open(part_path, "wb") as out:
        out.write(content)

    with _chunk_uploads_lock:
        record["received"].add(chunk_index)
        received_count = len(record["received"])

    return jsonify({
        "upload_id": upload_id,
        "received_chunks": received_count,
        "total_chunks": total_chunks,
        "complete": received_count == total_chunks,
    }), 200


@app.route("/upload/finalize/<upload_id>", methods=["POST"])
def upload_finalize(upload_id):
    """Call once every chunk has been POSTed to /upload/chunk. Reassembles
    them into the original file on disk and returns the upload_id to pass
    to /goal's "file_upload_ids" field. Idempotent: calling this again after
    a successful finalize just returns the same already-assembled result."""
    gate = _controller_gate()
    if gate:
        return gate

    with _chunk_uploads_lock:
        existing = _chunk_uploads.get(upload_id)
        if existing and existing.get("status") == "ready":
            return jsonify({
                "upload_id": upload_id, "filename": existing["filename"],
                "size": existing["size"], "status": "ready",
            }), 200

    try:
        record = _finalize_chunk_upload(upload_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "upload_id": upload_id, "filename": record["filename"],
        "size": record["size"], "status": "ready",
    }), 200


def _upload_names_all_json(uploaded_files, file_upload_ids) -> bool:
    """
    True when EVERY file in this /goal request (inline files and chunk-upload
    ids alike) has a .json name. Cheap, name-only pre-check used by
    goal_endpoint to decide whether the MAX_UPLOAD_FILES cap can wait until
    the plan is known. The data-matching rule only ever accepts JSON
    documents (see GOAL_EXPANDER_MATCH_RULE), so an upload containing
    anything else can never be a legitimate over-the-cap matching request and
    is still rejected up front, exactly as before.
    """
    for f in uploaded_files:
        if not (getattr(f, "filename", "") or "").lower().endswith(".json"):
            return False
    with _chunk_uploads_lock:
        for uid in file_upload_ids:
            record = _chunk_uploads.get(uid) or {}
            if not (record.get("filename") or "").lower().endswith(".json"):
                return False
    return True


def _plan_has_match_task(tasks: list, agents: list) -> bool:
    """True if any task in the plan is dispatched to an agent that declares
    the data-matching capability (same test _ensure_upload_content_used
    uses to find match tasks)."""
    match_agent_ids = {a["agent_id"] for a in agents if "data-matching" in a.get("capabilities", [])}
    return any(t.get("agent_id") in match_agent_ids for t in tasks)


@app.route("/goal", methods=["POST"])
def goal_endpoint():
    gate = _controller_gate()
    if gate:
        return gate

    # Accept both the original JSON body (back-compat) and multipart/form-data
    # (goal + inputs-as-JSON-string + one or more "files"). Anything sent as
    # multipart without files still works exactly like the JSON path.
    is_multipart = bool(request.content_type) and "multipart/form-data" in request.content_type

    if is_multipart:
        goal = request.form.get("goal", "").strip()
        raw_inputs = request.form.get("inputs", "{}") or "{}"
        try:
            inputs = json.loads(raw_inputs)
        except json.JSONDecodeError:
            return jsonify({"error": "inputs field must be valid JSON"}), 400
        uploaded_files = request.files.getlist("files")
        # Files too big for one multipart body (Cloudflare's ~100MB edge
        # cap) go through /upload/chunk + /upload/finalize first, and are
        # referenced here by upload_id instead of being re-sent -- see the
        # CHUNKED UPLOADS comment above MAX_UPLOAD_BYTES. Accepts either a
        # JSON array string ('["id1","id2"]') or repeated form fields.
        raw_upload_ids = request.form.get("file_upload_ids", "")
        if raw_upload_ids:
            try:
                file_upload_ids = json.loads(raw_upload_ids)
                if not isinstance(file_upload_ids, list):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                return jsonify({"error": "file_upload_ids must be a JSON array of upload_id strings"}), 400
        else:
            file_upload_ids = request.form.getlist("file_upload_ids")
        conversation_id = (request.form.get("conversation_id") or "").strip() or None
    else:
        body = request.get_json(silent=True) or {}
        goal = body.get("goal", "").strip()
        inputs = body.get("inputs", {})
        uploaded_files = []
        file_upload_ids = body.get("file_upload_ids", []) or []
        conversation_id = (body.get("conversation_id") or "").strip() or None

    if not goal:
        return jsonify({"error": "Missing required field: goal"}), 400

    # First message of a new chat thread -- the frontend has no
    # conversation_id yet. Mint one here and hand it back in every response
    # below (initial 202, and the job-status result once done) so the
    # frontend can echo it on every later /goal call in this same thread.
    # Follow-up turns send it back, so this only fires once per thread.
    if not conversation_id:
        conversation_id = uuid.uuid4().hex

    _prune_stale_goal_jobs()

    job_id = str(uuid.uuid4())  # doubles as the ingestion session_id for any uploaded files

    # Ingestion itself (the slow, network-bound part) runs in _worker()
    # below. But Flask's FileStorage objects in `uploaded_files` are tied
    # to this request's lifecycle -- Werkzeug closes their underlying
    # stream as soon as the response is sent, which now happens BEFORE the
    # background worker gets a chance to read them (first hit this as
    # "seek of closed file" the first time this shipped). Reading each
    # file's bytes into memory here is cheap (no network, bounded by
    # MAX_UPLOAD_BYTES) and produces plain bytes that survive into the
    # worker thread just fine.
    file_payloads = []
    # Over the cap is normally a hard 400. The one exception: an all-JSON
    # upload may go over it IF the plan later turns out to be an
    # auto-approved MATCHING plan (see AUTO_APPROVE_CAPABILITIES -- a
    # data-matching task takes every document in one call, no batching, and
    # runs without the approval step). The plan doesn't exist yet at this
    # point, so the check is deferred to the worker below; any other plan is
    # rejected there with the same "Too many files" error.
    defer_file_limit = False
    if len(uploaded_files) + len(file_upload_ids) > MAX_UPLOAD_FILES:
        if _upload_names_all_json(uploaded_files, file_upload_ids):
            defer_file_limit = True
        else:
            return jsonify({"error": f"Too many files (max {MAX_UPLOAD_FILES})"}), 400
    if uploaded_files:
        for f in uploaded_files:
            filename = f.filename or ""
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_UPLOAD_EXTS:
                return jsonify({"error": f"Unsupported file type: {filename}"}), 400
            content = f.stream.read()
            if len(content) == 0:
                return jsonify({"error": f"{filename} is empty"}), 400
            if len(content) > MAX_UPLOAD_BYTES:
                return jsonify({"error": f"{filename} exceeds {MAX_UPLOAD_BYTES} byte limit"}), 400
            file_payloads.append({"filename": filename, "mimetype": f.mimetype, "content": content})

    # Large files that went through /upload/chunk + /upload/finalize instead
    # of being attached inline -- read the already-reassembled bytes off
    # disk here (once), then clean up the temp dir immediately. From this
    # point on these behave exactly like any small directly-attached file.
    for upload_id in file_upload_ids:
        with _chunk_uploads_lock:
            record = _chunk_uploads.get(upload_id)
        if not record or record.get("status") != "ready":
            return jsonify({"error": f"upload_id not found or not finalized: {upload_id}"}), 400
        with open(record["final_path"], "rb") as fh:
            content = fh.read()
        if len(content) == 0:
            return jsonify({"error": f"{record['filename']} is empty"}), 400
        if len(content) > MAX_UPLOAD_BYTES:
            return jsonify({"error": f"{record['filename']} exceeds {MAX_UPLOAD_BYTES} byte limit"}), 400
        file_payloads.append({"filename": record["filename"], "mimetype": None, "content": content})
        shutil.rmtree(os.path.join(CHUNK_UPLOAD_DIR, upload_id), ignore_errors=True)
        with _chunk_uploads_lock:
            _chunk_uploads.pop(upload_id, None)

    # The stored/displayed goal is exactly what the user typed -- no
    # appended text. Upload guidance is now delivered via the "0. PRIORITY"
    # rule (extra_rules/extra_examples in build_expander_prompt, spliced in
    # BEFORE rule 10), not by modifying the goal string itself -- see
    # _build_upload_rule for why.

    with _goal_jobs_lock:
        _goal_jobs[job_id] = {
            "status": "ingesting" if file_payloads else "planning",
            "result": None, "error": None, "completed_at": None,
        }

    def _worker():
        # See the comment above where file_payloads is built -- this must
        # receive plain bytes, not the original FileStorage list, or this
        # blows up with "seek of closed file" as soon as it's read here.
        #
        # IMPORTANT: files are now tagged/scoped by conversation_id, NOT
        # job_id. job_id is a fresh uuid on every single /goal call, so
        # tagging with it (the old behavior) meant a file could only ever
        # be seen by the ONE turn that uploaded it -- retrieve tasks and
        # the "0. PRIORITY" upload rule are scoped by this same value (see
        # _build_upload_rule / retrieve_agent's session_id filter), so
        # whatever it's tagged with is exactly what later turns CAN'T see
        # unless it matches. Tagging with conversation_id instead makes an
        # uploaded file visible to every later turn in the same chat
        # thread, not just the turn that attached it.
        new_doc_refs = []
        if file_payloads:
            try:
                new_doc_refs = ingest_uploaded_files(conversation_id, file_payloads,
                                                     enforce_file_limit=not defer_file_limit)
            except ValueError as e:
                with _goal_jobs_lock:
                    _goal_jobs[job_id] = {"status": "error", "result": None,
                                           "error": str(e), "completed_at": time.time(),
                                           "conversation_id": conversation_id}
                return
            except RuntimeError as e:
                with _goal_jobs_lock:
                    _goal_jobs[job_id] = {"status": "error", "result": None,
                                           "error": str(e), "completed_at": time.time(),
                                           "conversation_id": conversation_id}
                return
            with _goal_jobs_lock:
                _goal_jobs[job_id]["status"] = "planning"
            # Still recorded against the conversation (append_conversation_docs
            # below) even though this turn doesn't READ that history back --
            # see the comment on doc_refs just below for why: this keeps the
            # data available for a future revert without needing a second,
            # separate storage path built later.
            append_conversation_docs(conversation_id, new_doc_refs)

        # STRICT CURRENT-GOAL-ONLY SCOPE (temporary, by request): doc_refs is
        # now ONLY this turn's freshly-ingested upload(s) -- NOT
        # get_conversation_docs(conversation_id)'s full cross-turn history.
        # This deliberately gives up the "summarize that PDF again" style
        # follow-up (an older upload is no longer visible to the Goal
        # Expander at all once the turn that attached it ends), in exchange
        # for a hard guarantee that a goal can NEVER pull in a file it wasn't
        # itself given -- a stronger, simpler fix for the snowballing bug
        # than the doc-id-scoped force_use_doc_ids restriction below, which
        # still left older docs visible (just not force-used).
        #
        # append_conversation_docs is still called above, so the history
        # keeps accumulating in storage even though it's unused here --
        # reverting to cross-turn continuity later is just restoring the
        # `doc_refs = get_conversation_docs(conversation_id)` line, with no
        # backfill needed.
        doc_refs = new_doc_refs

        # Every doc in doc_refs is now, by construction, this turn's own
        # upload -- so all of it is eligible for _ensure_upload_content_used's
        # "force it into the plan" backstop (see expand_goal's
        # force_use_doc_ids docstring). Kept as an explicit set (rather than
        # passing None for "no restriction") so this line doesn't silently
        # change meaning if doc_refs's source above is ever reverted to
        # include older docs again.
        fresh_doc_ids = {d.get("doc_id") for d in doc_refs if d.get("doc_id")}

        try:
            agents = fetch_active_agents()
        except Exception as e:
            with _goal_jobs_lock:
                _goal_jobs[job_id] = {"status": "error", "result": None,
                                       "error": f"Registry fetch failed: {str(e)}",
                                       "completed_at": time.time(),
                                       "conversation_id": conversation_id}
            return

        if not agents:
            with _goal_jobs_lock:
                _goal_jobs[job_id] = {"status": "error", "result": None,
                                       "error": "No active agents in registry",
                                       "completed_at": time.time(),
                                       "conversation_id": conversation_id}
            return

        # Pull this chat thread's prior turns (if any) BEFORE calling the
        # Goal Expander, so a follow-up like "translate that to French" or
        # "now do the same for next week" can be resolved against what was
        # actually asked/answered earlier in this same conversation_id. A
        # brand-new conversation_id (first turn) simply returns [].
        history = get_conversation_history(conversation_id)

        try:
            tasks = expand_goal(goal, agents, doc_refs=doc_refs,
                                session_id=conversation_id if doc_refs else None,
                                history=history, force_use_doc_ids=fresh_doc_ids)
        except Exception as e:
            with _goal_jobs_lock:
                _goal_jobs[job_id] = {"status": "error", "result": None,
                                       "error": f"Goal expansion failed: {str(e)}",
                                       "completed_at": time.time(),
                                       "conversation_id": conversation_id}
            return

        # Deferred file cap (see goal_endpoint): only a plan that contains a
        # data-matching task AND is auto-approved (the same
        # _is_auto_approve_eligible decision that skips the approval gate
        # further down) may exceed MAX_UPLOAD_FILES. Anything else gets the
        # same error the up-front check would have given. The match-task
        # test is kept on purpose: _is_auto_approve_eligible's JSON-count
        # rule alone approves ANY upload of 2+ JSON files, so on its own it
        # would lift the cap for non-matching goals too.
        if defer_file_limit and not (
            _plan_has_match_task(tasks, agents)
            and _is_auto_approve_eligible(tasks, build_agent_index(agents), doc_refs)
        ):
            with _goal_jobs_lock:
                _goal_jobs[job_id] = {"status": "error", "result": None,
                                       "error": f"Too many files (max {MAX_UPLOAD_FILES}) -- "
                                                f"more files are only accepted for an auto-approved matching goal",
                                       "completed_at": time.time(),
                                       "conversation_id": conversation_id}
            return

        _save_goal_plan(job_id, goal, tasks, conversation_id)

        if not tasks:
            with _goal_jobs_lock:
                _goal_jobs[job_id] = {
                    "status": "done", "error": None, "completed_at": time.time(),
                    "result": {
                        "goal": goal, "expanded_tasks": [], "requires_approval": False,
                        "message": "No relevant agents found for this goal",
                        "uploaded_docs": doc_refs,
                        "conversation_id": conversation_id,
                    },
                }
            return

        if inputs:
            for task in tasks:
                task["inputs"] = {**task.get("inputs", {}), **inputs}

        # session_id=conversation_id ONLY when there are actually docs
        # attached (this turn's and/or earlier turns'). No plan-lifecycle
        # purge is tied to this anymore -- see _cleanup_expired_plans and
        # approve_goal/reject_goal -- files persist for the whole
        # conversation and are only purged in _prune_stale_conversations
        # (TTL) or clear_conversation() (explicit "new chat"). preseeded
        # holds any small-enough-to-inline uploads' full text, keyed with
        # reserved integer task_ids (UPLOAD_TASK_ID_BASE+), so the LLM's
        # $task_9001.text-style references resolve without any agent call
        # at execution time -- this now includes docs from earlier turns
        # too, not just this turn's fresh uploads. conversation_id is
        # carried on the plan so approve_goal() knows which thread to
        # append this turn's final answer to once execution finishes (see
        # append_conversation_turn / _extract_history_answer).
        preseeded = preseed_uploaded_docs(doc_refs) if doc_refs else {}
        agent_index = build_agent_index(agents)

        # AUTO-APPROVE PATH -- see AUTO_APPROVE_CAPABILITIES for the
        # narrow, deliberate exception this is to the platform's normal
        # "AI recommends, a human who clicks POST is accountable" gate.
        # A plan qualifies either because the goal has 2+ JSON files
        # attached (deterministic matching-goal rule -- see
        # _is_auto_approve_eligible) or because EVERY task is a read-only/
        # diagnostic capability (currently just data-matching); either way
        # it skips the pending-plan step and runs immediately, right here
        # in the background worker, instead of waiting on POST
        # /goal/<id>/approve. Everything else (mixed plans, any write/
        # submit-capable task, single-JSON-or-fewer plans that don't pass
        # the capability check) falls through to the unchanged
        # pending-plan path below.
        if _is_auto_approve_eligible(tasks, agent_index, doc_refs):
            plan_id = str(uuid.uuid4())
            result_sink = {}
            for _ in execute_tasks(goal, tasks, agent_index, preseeded=preseeded,
                                    result_sink=result_sink):
                pass  # no streaming client here -- just drain the generator to completion

            answer = _extract_history_answer(tasks, result_sink.get("completed", {}), agent_index)

            # Structured, machine-consumable form of every task's raw
            # output -- built specifically so an external caller (e.g. a
            # partner backend consuming a match result) can read
            # comparisons/overall_match/etc directly as real JSON, without
            # having to parse the narrative final_answer string (which,
            # for a capability like data-matching with no "summary"/"text"
            # field, is itself a stringified dump PLUS a human-readable
            # version tag appended -- fine for chat history, brittle for
            # a program to depend on). Keyed by task_id (as a string,
            # since JSON object keys must be strings) so a caller with
            # multiple tasks in one plan can find the one they want.
            task_outputs = {}
            for t in tasks:
                res = result_sink.get("completed", {}).get(t["task_id"])
                if res and res.get("success"):
                    task_outputs[str(t["task_id"])] = res.get("response", {}).get("outputs", {})

            _prune_stale_completed_goals()
            with _completed_goals_lock:
                _completed_goals[plan_id] = {
                    "goal": goal,
                    "answer": answer,
                    "dag_summary": _summarize_dag(tasks, agent_index),
                    "capabilities_used": _capabilities_used(tasks, agent_index),
                    "conversation_id": conversation_id,
                    "completed_at": time.time(),
                }

            if conversation_id:
                append_conversation_turn(conversation_id, goal, answer)

            with _goal_jobs_lock:
                _goal_jobs[job_id] = {
                    "status": "done", "error": None, "completed_at": time.time(),
                    "result": {
                        "plan_id": plan_id,
                        "goal": goal,
                        "expanded_tasks": tasks,
                        "requires_approval": False,
                        "auto_approved": True,
                        "final_answer": f"{answer}\n\n[Answered by BYOA Controller {CONTROLLER_VERSION}]",
                        "task_outputs": task_outputs,
                        "uploaded_docs": doc_refs,
                        "conversation_id": conversation_id,
                    },
                }
            return

        plan_id = create_pending_plan(goal, tasks, agents,
                                       session_id=conversation_id if doc_refs else None,
                                       preseeded=preseeded,
                                       conversation_id=conversation_id)

        with _goal_jobs_lock:
            _goal_jobs[job_id] = {
                "status": "done", "error": None, "completed_at": time.time(),
                "result": {
                    "plan_id": plan_id,
                    "goal": goal,
                    "expanded_tasks": tasks,
                    "requires_approval": True,
                    "expires_in_secs": PLAN_TTL_SECS,
                    "approve_url": f"/goal/{plan_id}/approve",
                    "reject_url": f"/goal/{plan_id}/reject",
                    "uploaded_docs": doc_refs,
                    "conversation_id": conversation_id,
                },
            }

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({
        "job_id": job_id,
        "status_url": f"/goal/{job_id}/status",
        # Echoed back immediately (not just once the job finishes) so the
        # frontend can attach it to this chat thread's state right away and
        # send it back on the very next /goal call, even if the user sends
        # a second message before this one's plan is approved/executed.
        "conversation_id": conversation_id,
        # uploaded_docs isn't known yet at this point -- ingestion now runs
        # in the background worker along with everything else. Poll
        # status_url; once status leaves "ingesting"/"planning", the result
        # (or error, if ingestion failed) will include uploaded_docs.
    }), 202


@app.route("/goal/<job_id>/status", methods=["GET"])
def goal_job_status(job_id):
    with _goal_jobs_lock:
        job = _goal_jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown or expired job_id"}), 404
    return jsonify(job), 200


@app.route("/goal/<plan_id>/approve", methods=["POST"])
def approve_goal(plan_id):
    gate = _controller_gate()
    if gate:
        return gate

    plan = pop_pending_plan(plan_id)
    if not plan:
        return jsonify({"error": "Unknown, expired, or already-actioned plan_id"}), 404

    body = request.get_json(silent=True) or {}
    tasks = body.get("tasks") or plan["tasks"]
    agents = plan["agents"]
    goal = plan["goal"]
    agent_index = build_agent_index(agents)

    if body.get("tasks"):
        errors = validate_task_references(tasks, agent_index)
        if errors:
            return jsonify({"error": "Edited task list is invalid", "details": errors}), 400

    def generate():
        yield json.dumps({"expanded_tasks": tasks, "approved": True}) + "\n"
        result_sink = {}
        for line in execute_tasks(goal, tasks, agent_index, preseeded=plan.get("preseeded"),
                                  result_sink=result_sink):
            yield line

        # Extracted once here and reused for two different purposes below:
        # the CLEAN text (no version tag) is what gets remembered as this
        # turn's answer for conversation history -- a future turn resolving
        # "that" or "it" against history shouldn't have to see a
        # "[Controller v7]" watermark baked into the context it's reasoning
        # over. The WATERMARKED copy is only for the final line actually
        # sent to the client below.
        answer = _extract_history_answer(tasks, result_sink.get("completed", {}), agent_index)

        # Stash what /feedback will need to rate this goal later, keyed by
        # the same plan_id the client already has (it was handed back in
        # the original /goal response as part of expanded_tasks' plan).
        # This is a SEPARATE record from the now-popped pending plan -- see
        # the comment above _completed_goals for why one isn't a substitute
        # for the other.
        _prune_stale_completed_goals()
        with _completed_goals_lock:
            _completed_goals[plan_id] = {
                "goal": goal,
                "answer": answer,
                "dag_summary": _summarize_dag(tasks, agent_index),
                "capabilities_used": _capabilities_used(tasks, agent_index),
                "conversation_id": plan.get("conversation_id"),
                "completed_at": time.time(),
            }

        # Record this turn in the chat thread's history (if any) now that
        # execution has actually finished, so the NEXT /goal call in this
        # conversation_id can resolve references back to it. Only fires for
        # a plan that was actually approved and run -- reject_goal and
        # unapproved-plan expiry never reach here, since there's no real
        # answer to remember in those cases.
        if plan.get("conversation_id"):
            append_conversation_turn(plan["conversation_id"], goal, answer)

        # Always the last content line of the stream (only "status":
        # "COMPLETED" and "done": true, already sent by execute_tasks
        # above, follow it) -- see CONTROLLER_VERSION above for why this
        # exists: which physical deployment (v5/v6/v7...) actually produced
        # this answer isn't otherwise visible from the response alone.
        yield json.dumps({
            "final_answer": f"{answer}\n\n[Answered by BYOA Controller {CONTROLLER_VERSION}]",
        }) + "\n"

        # NOTE: no KB purge here anymore. plan["session_id"] is now the
        # conversation_id (files are tagged/scoped at the conversation
        # level -- see the /goal worker), so purging it after just THIS
        # plan finishes would delete every file attached anywhere in the
        # chat thread, including ones a later turn still needs. Files are
        # only purged in _prune_stale_conversations() (TTL) or
        # clear_conversation() (explicit "new chat").

    return app.response_class(generate(), mimetype="application/x-ndjson")


@app.route("/goal/<plan_id>/reject", methods=["POST"])
def reject_goal(plan_id):
    plan = pop_pending_plan(plan_id)
    if not plan:
        return jsonify({"error": "Unknown, expired, or already-actioned plan_id"}), 404

    # NOTE: no KB purge here either, same reasoning as approve_goal above --
    # rejecting this one plan shouldn't delete files still attached to the
    # rest of the conversation.

    return jsonify({"status": "rejected", "plan_id": plan_id, "goal": plan["goal"]}), 200


@app.route("/feedback", methods=["POST"])
def feedback_endpoint():
    """
    Records a user's rating (1-5, required) and free-text comment
    (optional) for a completed goal, closing the self-improvement loop:
    this row becomes retrievable context for the NEXT similar goal via
    _build_experience_block, injected into the Goal Expander prompt.

    Body: {"plan_id": "...", "rating": 1-5, "comment": "optional text"}
    plan_id is the SAME id the client received back from POST /goal
    (as expanded_tasks' plan_id) -- looked up here against
    _completed_goals, not the (already-popped) pending-plan store.
    """
    body = request.get_json(silent=True) or {}
    plan_id = (body.get("plan_id") or "").strip()
    rating = body.get("rating")
    comment = (body.get("comment") or "").strip()

    if not plan_id:
        return jsonify({"error": "Missing required field: plan_id"}), 400
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be an integer 1-5"}), 400
    if rating < 1 or rating > 5:
        return jsonify({"error": "rating must be between 1 and 5"}), 400

    _prune_stale_completed_goals()
    with _completed_goals_lock:
        record = _completed_goals.get(plan_id)

    if not record:
        return jsonify({
            "error": f"Unknown or expired plan_id: {plan_id} "
                     f"(completed-goal records are kept for {COMPLETED_GOAL_TTL_SECS}s after execution)"
        }), 404

    try:
        r = req.post(
            f"{FEEDBACK_BASE_URL}{FEEDBACK_SUBMIT_PATH}",
            json={
                "goal_text": record["goal"],
                "rating": rating,
                "comment": comment,
                "dag_summary": record["dag_summary"],
                "capabilities_used": record["capabilities_used"],
                "response_excerpt": record["answer"],
                "plan_id": plan_id,
                "conversation_id": record.get("conversation_id") or "",
                "controller_version": CONTROLLER_VERSION,
            },
            timeout=FEEDBACK_SUBMIT_TIMEOUT_SECS,
        )
        r.raise_for_status()
        stored = r.json()
    except Exception as e:
        return jsonify({"error": f"Failed to store feedback: {e}"}), 502

    return jsonify({"status": "recorded", "plan_id": plan_id, **stored}), 200


@app.route("/conversation/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Read-only view of a conversation thread's stored turns -- mainly for
    frontend debugging (e.g. confirming what context a plan was actually
    built with)."""
    turns = get_conversation_history(conversation_id)
    return jsonify({"conversation_id": conversation_id, "turns": turns}), 200


@app.route("/conversation/<conversation_id>", methods=["DELETE"])
def clear_conversation(conversation_id):
    """
    Drops a chat thread's stored turn history, e.g. for a frontend "new
    chat" / "clear context" action. The conversation_id itself can still be
    reused afterwards -- it just starts back at zero prior turns rather
    than needing a freshly-minted id. Idempotent: clearing an
    unknown/already-empty conversation_id is not an error.
    """
    with CONVERSATIONS_LOCK:
        conv = CONVERSATIONS.pop(conversation_id, None)
        existed = conv is not None
    # Any files attached across this thread's turns are now unreachable --
    # purge them from the knowledge base rather than leaving them to linger
    # until CONVERSATION_TTL_SECS naturally expires.
    if existed:
        purge_session_docs(conversation_id)
    return jsonify({"status": "cleared", "conversation_id": conversation_id, "existed": existed}), 200


@app.route("/admin/clear_all_conversations", methods=["POST"])
def clear_all_conversations():
    """
    Bulk version of clear_conversation() -- drops EVERY conversation's
    stored turn history and purges EVERY conversation's uploaded goal
    files from the knowledge base in one call.

    Snapshots all current conversation_ids under the lock, then empties
    CONVERSATIONS in the same critical section (so nothing new sneaks in
    between the snapshot and the wipe), and only THEN does the purge_session_docs
    calls outside the lock -- these are outbound HTTP calls to the knowledge
    base service and shouldn't hold CONVERSATIONS_LOCK while they run.

    Best-effort per conversation, same as purge_session_docs() itself: one
    failed purge (KB service down, network blip) is logged and does not
    stop the rest from being attempted, and does not fail the whole request.
    """
    with CONVERSATIONS_LOCK:
        conversation_ids = list(CONVERSATIONS.keys())
        CONVERSATIONS.clear()

    purged, failed = [], []
    for cid in conversation_ids:
        ok = purge_session_docs(cid)
        (purged if ok else failed).append(cid)

    return jsonify({
        "status": "cleared_all",
        "conversations_cleared": len(conversation_ids),
        "docs_purged_ok": purged,
        "docs_purge_failed": failed,
    }), 200


@app.route("/agents", methods=["GET"])
def list_agents():
    try:
        agents = fetch_active_agents()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify([{
        "agent_id": a["agent_id"],
        "display_name": a["display_name"],
        "capabilities": a["capabilities"],
        "description": a["description"],
    } for a in agents]), 200


# ── Dev Controls (Cell 7.5, adapted for shared systemd Ollama) ───────────────
CONTROLLER_ACTIVE = True
_controller_lock = threading.Lock()
_ollama_lock = threading.Lock()


def _controller_gate():
    if not CONTROLLER_ACTIVE:
        return jsonify({"error": "Controller is stopped. POST /dev/start to resume."}), 503
    return None


@app.route("/dev/stop", methods=["POST"])
def dev_stop():
    global CONTROLLER_ACTIVE
    with _controller_lock:
        CONTROLLER_ACTIVE = False
    return jsonify({
        "status": "stopped",
        "note": "Ollama stays running (shared systemd service). /goal and /goal/<id>/approve return 503 until /dev/start.",
    }), 200


@app.route("/dev/start", methods=["POST"])
def dev_start():
    global CONTROLLER_ACTIVE
    with _controller_lock:
        CONTROLLER_ACTIVE = True
    return jsonify({"status": "started"}), 200


@app.route("/dev/prompt", methods=["GET"])
def dev_get_prompt():
    return jsonify({"prompt": GOAL_EXPANDER_PROMPT_TEMPLATE}), 200


def _flush_all_pending_plans() -> int:
    """Used by /dev/acceptPrompt and /dev/resetPrompt, which discard every
    pending plan since edits to the prompt template invalidate them.

    NOTE: no longer purges KB docs here. plan["session_id"] is now a
    conversation_id shared across that whole chat thread, so purging it
    just because ITS pending plan got flushed would delete files other
    turns in the same conversation still rely on. Flushing a plan only
    discards the plan/task graph, never the underlying uploaded files."""
    with PENDING_PLANS_LOCK:
        flushed_plans = list(PENDING_PLANS.values())
        PENDING_PLANS.clear()
    return len(flushed_plans)


@app.route("/dev/acceptPrompt", methods=["POST"])
def dev_accept_prompt():
    global GOAL_EXPANDER_PROMPT_TEMPLATE
    body = request.get_json(silent=True) or {}
    new_prompt = body.get("prompt", "")

    if not new_prompt.strip():
        return jsonify({"error": "Missing required field: prompt"}), 400

    missing = [ph for ph in ("{goal}", "{agents_block}") if ph not in new_prompt]
    if missing:
        return jsonify({"error": f"Prompt template is missing required placeholders: {missing}"}), 400

    GOAL_EXPANDER_PROMPT_TEMPLATE = new_prompt

    flushed = _flush_all_pending_plans()

    return jsonify({"status": "ok", "prompt_length": len(new_prompt), "flushed_pending_plans": flushed}), 200


@app.route("/dev/resetPrompt", methods=["POST"])
def dev_reset_prompt():
    global GOAL_EXPANDER_PROMPT_TEMPLATE
    GOAL_EXPANDER_PROMPT_TEMPLATE = _DEFAULT_GOAL_EXPANDER_PROMPT_TEMPLATE
    flushed = _flush_all_pending_plans()
    return jsonify({"status": "reset_to_default", "flushed_pending_plans": flushed}), 200


@app.route("/dev/restartOllama", methods=["POST"])
def dev_restart_ollama():
    """
    Restarts the shared `ollama` systemd service (NOT a subprocess this
    process owns -- summarize agent depends on it too).
    Requires a passwordless sudo rule for exactly this command -- see README.
    """
    if not _ollama_lock.acquire(blocking=False):
        return jsonify({"error": "A restart is already in progress"}), 409

    try:
        try:
            subprocess.run(["sudo", "systemctl", "restart", "ollama"], check=True, timeout=30)
        except Exception as e:
            return jsonify({"status": "error", "error": f"systemctl restart failed: {e}"}), 500

        ready = False
        for _ in range(30):
            try:
                if req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2).status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1)

        if not ready:
            return jsonify({"status": "error", "error": "Ollama did not respond within 30s of restart"}), 500

        warm_ok = False
        for attempt in range(3):
            try:
                r = req.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": "hi"}],
                          "stream": False, "options": {"temperature": 0, "think": False}},
                    timeout=600,
                )
                if r.status_code == 200:
                    warm_ok = True
                    break
            except Exception:
                time.sleep(5)

        return jsonify({
            "status": "restarted" if warm_ok else "restarted_but_warmup_failed",
            "warm": warm_ok,
        }), (200 if warm_ok else 500)

    finally:
        _ollama_lock.release()


@app.route("/dev/status", methods=["GET"])
def dev_status():
    ollama_responsive = False
    try:
        ollama_responsive = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2).status_code == 200
    except Exception:
        pass

    with PENDING_PLANS_LOCK:
        pending_count = len(PENDING_PLANS)

    _prune_stale_conversations()
    with CONVERSATIONS_LOCK:
        active_conversations = len(CONVERSATIONS)

    return jsonify({
        "controller_active": CONTROLLER_ACTIVE,
        "controller_version": CONTROLLER_VERSION,
        "ollama_responsive": ollama_responsive,
        "ollama_model": OLLAMA_MODEL,
        "prompt_length": len(GOAL_EXPANDER_PROMPT_TEMPLATE),
        "pending_plans": pending_count,
        "active_conversations": active_conversations,
    }), 200


if __name__ == "__main__":
    verify_ollama_ready()

    print("=" * 60)
    print("BYOA Platform Controller is live")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  POST {AGENT_ENDPOINT}/goal")
    print(f"  GET  {AGENT_ENDPOINT}/agents")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/dev/status")
    print(f"  POST {AGENT_ENDPOINT}/feedback")
    print("=" * 60)

    from waitress import serve
    serve(app, host="0.0.0.0", port=PORT, threads=8)

