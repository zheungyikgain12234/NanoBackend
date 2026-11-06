"""
json-match-agent
~~~~~~~~~~~~~~~~~
Matches two (or more) OCR-extracted JSON documents that describe the same
transaction but were produced by different sources / documents (e.g. a
Tax Invoice vs a Purchase Order vs a Delivery Order) and therefore use
different field names for the same underlying data.

Design decision (resolves an open question on byoa-match-agent):
  AI is used ONLY to semantically align fields and line items across
  documents with different naming conventions. It is NEVER trusted to
  do the actual numeric comparison. All qty/amount matching, tolerance
  checks, and diffs are computed deterministically in Python after
  alignment. This is a hybrid approach: AI for "which fields correspond",
  code for "do the numbers actually match".

Rationale: LLMs are unreliable at arithmetic and cannot be audited the
way a fixed tolerance check can. Since qty/amount accuracy is explicitly
the most important requirement, that logic must be deterministic.

Model backend: calls a self-hosted qwen3.6-35b-fp8 via vLLM's
OpenAI-compatible /v1/chat/completions endpoint, matching the pattern
already used in summarize_agent.py (same VLLM_BASE_URL env var, same
error-handling shape, same JSON-fence-stripping approach for parsing
model output).
"""
import copy
import json
import os
import re
import threading
import time
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from waitress import serve

from byoa_adk import Agent, attach_async_routes
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")

# ---------------------------------------------------------------------------
# Testing-only: dump every match result to disk so runs can be inspected
# after the fact. Controlled by an env var so it's easy to disable outside
# of testing without another code change.
# ---------------------------------------------------------------------------
TEST_RESULT_DUMP_DIR = os.environ.get(
    "MATCH_RESULT_DUMP_DIR", "/opt/v6/byoa-match/byoa-match-agent"
)
TEST_RESULT_DUMP_ENABLED = os.environ.get("MATCH_RESULT_DUMP_ENABLED", "1") != "0"


def _dump_match_result(task_id: str, result: dict) -> None:
    """
    Best-effort write of a single run's outputs to TEST_RESULT_DUMP_DIR.
    Never raises -- a failure to write the debug dump should never fail
    the actual task, so any exception here is swallowed after logging.
    """
    if not TEST_RESULT_DUMP_ENABLED:
        return
    try:
        os.makedirs(TEST_RESULT_DUMP_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id or "unknown"))
        filename = f"match_result_{ts}_{safe_task_id}.json"
        path = os.path.join(TEST_RESULT_DUMP_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - best-effort debug aid
        print(f"[json-match-agent] WARNING: failed to write match result dump: {exc}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tolerance for considering two amounts "matching" (absolute, in currency units).
# Kept as an explicit constant rather than baked into the AI prompt, so it's
# auditable and can be made per-request (payload.inputs) later if needed.
DEFAULT_AMOUNT_TOLERANCE = Decimal("0.01")
DEFAULT_QTY_TOLERANCE = Decimal("0")  # quantities should match exactly by default

# Upstream retrieval currently exports quantities with float noise (e.g.
# 999.9999999999999 for 1000, 359.99999999999994 for 360). Quantities in this
# business are always whole numbers, so every quantity-like field is rounded to
# the nearest integer once, at the entry point (see _normalize_quantities),
# BEFORE any matching/netting happens. Set QTY_NORMALIZE_TO_INT=0 to disable
# (e.g. once the retrieval backend is fixed, or if fractional units appear).
QTY_NORMALIZE_TO_INT = os.environ.get("QTY_NORMALIZE_TO_INT", "1") != "0"

# Business default currency, used only for narrative display when a
# document doesn't state its own currency anywhere (see
# _infer_doc_currency) -- never affects matching logic itself.
DEFAULT_CURRENCY = "S$"

# Same env var name as summarize_agent.py, so both agents point at the same
# vLLM server by default without needing separate configuration.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")

# Shared between _find_structured (to detect "is this level already the
# right one") and _get_line_items (to actually retrieve the list).
# "lines" added for the po.json/invoice.json/grn*.json convention --
# see _find_structured's docstring for the bug this key list omission
# used to cause.
_LINE_ITEM_KEYS = ("line_items", "items", "delivered_items", "order_items", "lines")


# ---------------------------------------------------------------------------
# Helpers: pulling structured data out of varying wrapper shapes
# ---------------------------------------------------------------------------

def _find_structured(doc: dict) -> dict:
    """
    OCR-MD pipeline outputs vary in wrapper shape (some nest under
    result -> structured, some are already flat). Recursively search
    for a "structured" key; fall back to the doc itself if none found.

    IMPORTANT FIX: previously this always tried descending into the
    first nested dict value before checking whether the CURRENT level
    already had its own line items. That silently broke a real document
    shape (po.json / invoice.json / grn*.json's "header" + "lines"
    convention): a doc like {"header": {...}, "lines": [...]} would
    recurse into "header" (itself a dict with no further nested dicts,
    so it satisfied the fallback "return doc" case), and that non-empty
    header dict would then be returned as THE structured document --
    permanently losing "lines" as a sibling, so downstream
    _get_line_items always found zero line items and every comparison
    against a document in this shape silently had nothing to compare.
    Now: if this level already has a recognized line-items key, it's
    treated as the right level immediately, no descent needed.

    Also flattens a nested "header" dict's fields up to the top level
    (see _flatten_header) so header-wrapped shapes (invoice.json,
    grn*.json) and flat shapes (po.json, which keeps its monetary
    fields alongside "lines" with no "header" wrapper at all) look
    identical to every downstream consumer (AI field discovery,
    monetary_fields access, credit-note auto-detection) without any
    per-document-type special-casing anywhere else in this file.
    """
    if not isinstance(doc, dict):
        return {}
    if any(k in doc and isinstance(doc[k], list) for k in _LINE_ITEM_KEYS):
        return _flatten_header(doc)
    if "structured" in doc and isinstance(doc["structured"], dict):
        return _flatten_header(doc["structured"])
    for v in doc.values():
        if isinstance(v, dict):
            found = _find_structured(v)
            if found:
                return found
    return _flatten_header(doc)


def _flatten_header(doc: dict) -> dict:
    """
    Splices a nested "header" dict's fields up to the top level, if
    present. See _find_structured's docstring for why this exists --
    some document shapes nest identifying/monetary fields one level
    down under "header" (invoice.json, grn*.json) while others
    (po.json) keep them flat alongside "lines". Top-level fields win on
    any key collision (defensive; not expected to occur in practice).
    """
    if "header" in doc and isinstance(doc["header"], dict):
        merged = dict(doc["header"])
        merged.update({k: v for k, v in doc.items() if k != "header"})
        return merged
    return doc


def _get_line_items(doc: dict) -> list:
    """
    Finds the line-items list under whatever key this document uses.
    Same caveat as _extract_qty_amount: fixed key list, not AI-informed.
    Extend this list as new document types surface new naming.
    """
    for key in _LINE_ITEM_KEYS:
        if key in doc and isinstance(doc[key], list):
            return doc[key]
    return []


def _is_probable_non_item_row(item: dict, code_field: Optional[str] = None,
                               qty_field: Optional[str] = None) -> bool:
    """
    Detects rows that OCR extracted into the line_items array but that
    aren't actually product lines -- GST/tax lines, rounding adjustments,
    credit/rebate lines, etc. These show up because the source document
    has them positioned like a line item, but they carry no product
    identity at all.

    Heuristic: a genuine product row always has EITHER a recognizable
    product code (article_number/model/item_no/... or the AI-discovered
    code_field for this document) OR a quantity (via qty_field or the
    hardcoded fallback list). A row with NEITHER is treated as a
    non-item row and excluded before any matching happens -- not
    compared, not reported as "unmatched", just excluded with the
    reason recorded for audit purposes.

    This is deliberately conservative: it only excludes rows missing
    BOTH signals, so a real product row with a code but a missing/OCR'd
    -out quantity (or vice versa) is still treated as a real item.
    Observed on real data: a GST line and a "Fuid Credits" line both had
    article_number=None AND qty=None, while all 7 real product rows in
    the same document had both a code and a qty.
    """
    has_code = bool(_extract_codes(item, code_field))
    qty = _extract_qty_amount(item, qty_field=qty_field)["qty"]
    return not has_code and qty is None


def _partition_line_items(items: list, code_field: Optional[str] = None,
                           qty_field: Optional[str] = None) -> dict:
    """
    Splits a document's line items into real product rows vs. probable
    non-item rows (see _is_probable_non_item_row), preserving each row's
    ORIGINAL index so downstream reporting (doc_a_index/doc_b_index)
    still refers to positions in the source document, not a renumbered
    filtered list -- important for anyone auditing against the raw JSON.
    """
    real_indices, excluded = [], []
    for i, item in enumerate(items):
        if _is_probable_non_item_row(item, code_field, qty_field):
            excluded.append({
                "index": i,
                "description": item.get("description"),
                "amount": item.get("amount_sgd") or item.get("amount") or item.get("total"),
                "reason": "no product code and no quantity -- likely a tax/fee/credit row, not a line item",
            })
        else:
            real_indices.append(i)
    return {"real_indices": real_indices, "excluded": excluded}


# Fields that commonly hold a product/SKU code identical across documents
# regardless of what the document calls the field itself (e.g. doc_a's
# "article_number" and doc_b's "model" can hold the exact same code).
# Order matters only as a hint list per item -- every key present is tried.
_CODE_KEYS = ["article_number", "model", "item_no", "sku", "part_no",
              "product_code", "item_code", "no",
              # The VENDOR's own code for the item, as stored on our side
              # (Business Central: "Vendor Item No." on the PO line). This
              # is what lets a PO line match the vendor invoice's "model"
              # exactly, with no AI guess involved.
              "vendorItemNo", "vendor_item_no", "vendorItemNumber",
              # Barcodes (the vendor invoice's jan_code vs a GTIN/EAN on our side)
              "jan_code", "gtin", "GTIN", "ean", "barcode"]


def _normalize_code(value: Any) -> Optional[str]:
    """Case/whitespace-insensitive normalization for exact code matching."""
    if value is None:
        return None
    s = str(value).strip().upper()
    return s if s else None


def _extract_codes(item: dict, discovered_field: Optional[str] = None) -> list:
    """
    All non-null, normalized code-like values present on one line item.

    discovered_field is the field name the AI identified for THIS
    document as holding a product/SKU code (see
    _call_vllm_field_discovery) -- tried first if given, since it's
    specific to this document's actual naming rather than a guess from
    a fixed list. The hardcoded _CODE_KEYS are still checked too (a
    document can genuinely have more than one code-like field, and this
    also covers cases where discovery returned null or was wrong).
    """
    codes = []
    if discovered_field:
        norm = _normalize_code(item.get(discovered_field))
        if norm is not None:
            codes.append(norm)
    for k in _CODE_KEYS:
        norm = _normalize_code(item.get(k))
        if norm is not None and norm not in codes:
            codes.append(norm)
    return codes


def _deterministic_code_prematch(items_a: list, items_b: list,
                                  code_field_a: Optional[str] = None,
                                  code_field_b: Optional[str] = None) -> dict:
    """
    Pairs line items by EXACT code match (article_number, model, item_no,
    etc.) before any AI involvement. This exists because real documents
    often carry a shared product code under different field names per
    document type -- when that code matches exactly, there is no
    ambiguity to resolve and no reason to let an LLM guess at pairing,
    which is unnecessary risk on the one part of this task (WHICH items
    correspond) that can be fully deterministic when a code overlaps.

    code_field_a/code_field_b are the AI-discovered field names for each
    document's product code (see _call_vllm_field_discovery) -- tried
    first, alongside the hardcoded _CODE_KEYS fallback list, via
    _extract_codes.

    Only items with NO code match here get passed to the AI pairing
    step for semantic/fuzzy pairing -- this reduces what the AI has to
    guess at, and removes the highest-confidence pairs from its judgment
    entirely, rather than asking it to re-derive something checkable.

    Returns the same shape run_reconciliation expects for line item
    pairs, plus which indices remain for the AI step to attempt.
    """
    pairs = []
    used_b = set()

    for ia, item_a in enumerate(items_a):
        codes_a = set(_extract_codes(item_a, code_field_a))
        if not codes_a:
            continue
        for ib, item_b in enumerate(items_b):
            if ib in used_b:
                continue
            codes_b = set(_extract_codes(item_b, code_field_b))
            if codes_a & codes_b:  # any exact code overlap
                pairs.append({
                    "doc_a_index": ia,
                    "doc_b_index": ib,
                    "match_confidence": 1.0,
                    "reason": f"exact code match: {sorted(codes_a & codes_b)[0]}",
                })
                used_b.add(ib)
                break

    matched_a = {p["doc_a_index"] for p in pairs}
    matched_b = {p["doc_b_index"] for p in pairs}
    remaining_a = [i for i in range(len(items_a)) if i not in matched_a]
    remaining_b = [i for i in range(len(items_b)) if i not in matched_b]

    return {"pairs": pairs, "remaining_a": remaining_a, "remaining_b": remaining_b}


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Best-effort numeric coercion. Returns None if not parseable."""
    if value is None:
        return None
    try:
        cleaned = str(value).replace(",", "").replace("$", "").strip()
        if cleaned == "" or cleaned.lower() == "null":
            return None
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


# Keys that hold a QUANTITY (quantity, quantityInvoiced, quantityReceived,
# qty_received, qty, ...). Money/price-like keys are excluded even if they
# happen to contain the word, so only true unit counts are ever rounded.
_QTY_KEY_RE = re.compile(r"qty|quantit", re.IGNORECASE)
_NON_QTY_KEY_RE = re.compile(r"price|amount|cost|rate|total|value|measure", re.IGNORECASE)


def _round_qty_value(value: Any) -> Any:
    """Round one numeric quantity to the nearest whole number (half-up).
    Non-numeric / bool / missing values are returned untouched, and the
    original type is preserved (int stays int, str stays str) so nothing
    downstream sees a different shape. Tiny negatives like -6E-14 become 0."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    try:
        if isinstance(value, str):
            d = _to_decimal(value)
            if d is None or not d.is_finite():
                return value
            return str(int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP)))
        if isinstance(value, (float, Decimal)):
            d = Decimal(str(value))
            if not d.is_finite():
                return value
            return int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, ArithmeticError):
        return value
    return value


def _normalize_quantities(obj: Any) -> Any:
    """Returns a deep-copied structure with every quantity-like field
    rounded to the nearest integer (see QTY_NORMALIZE_TO_INT). Walks the
    whole document, so it works regardless of wrapper shape (flat,
    result->structured, header+lines) and covers line-item quantities as
    well as header fields such as qty_received. Amounts/prices are left
    exactly as given -- they already compare within the amount tolerance."""
    if not QTY_NORMALIZE_TO_INT:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if (isinstance(k, str) and _QTY_KEY_RE.search(k)
                    and not _NON_QTY_KEY_RE.search(k)
                    and not isinstance(v, (dict, list))):
                out[k] = _round_qty_value(v)
            else:
                out[k] = _normalize_quantities(v)
        return out
    if isinstance(obj, list):
        return [_normalize_quantities(v) for v in obj]
    return obj


# Strips a ```json ... ``` or bare ``` ... ``` fence the model sometimes
# wraps its output in despite being told not to. Same approach as
# summarize_agent.py's _strip_json_fence -- tried AFTER json.loads() fails
# plain, rather than instead of the "no fences" instruction, since
# instruction-following on formatting asks isn't guaranteed.
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_json_fence(text: str) -> str:
    m = _JSON_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# AI step: semantic alignment only (never numeric comparison)
# ---------------------------------------------------------------------------

FIELD_DISCOVERY_PROMPT = """You are inspecting two extracted business \
documents (e.g. invoice, purchase order, delivery order, goods receipt \
note) that describe the same transaction but use different field names \
for the same concepts.

Document A fields (top-level, non-line-item):
{doc_a_fields}

Document B fields (top-level, non-line-item):
{doc_b_fields}

Document A sample line items (a few examples, to show its field naming):
{doc_a_sample_items}

Document B sample line items (a few examples, to show its field naming):
{doc_b_sample_items}

Your task has two parts:

1. Identify EVERY top-level monetary/dollar-amount field in each \
document (not just the grand total -- also subtotals, tax/GST amounts, \
shipping or handling fees, credits, rounding adjustments, or any other \
field holding a currency amount), and pair each one in Document A with \
its corresponding field in Document B where a genuine counterpart \
exists (e.g. "invoice_amount" in A may correspond to "grand_total" in \
B; "total_excluding_gst" in A may correspond to "order_total_before_tax" \
in B). Give each pair a short label describing what it represents (e.g. \
"grand total incl. tax", "subtotal excl. GST", "GST amount"). If a \
monetary field in one document has NO real counterpart in the other \
(e.g. a Delivery Order has no dollar fields at all, or one document has \
a fee the other doesn't track), still include it with the other side's \
field set to null, so it's visible rather than silently dropped.

2. For EACH document independently, identify the field NAME used on its \
line items (not the value, the key name itself) for: a product/SKU code \
identifier, a quantity, and a line amount/total. Look at the sample line \
items to find the consistent key name each document uses -- a document \
uses the same key name across all of its own line items even if that \
name differs from the other document's. Use null for any that don't \
exist in that document at all (e.g. a document with no prices has no \
line-item amount field).

Do NOT compare or judge any actual qty/amount VALUES -- only identify \
which field NAMES hold this information, in each document.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "monetary_fields": [
    {{"label": "<short description>", "doc_a_field": "<field name or null>", "doc_b_field": "<field name or null>"}}
  ],
  "line_item_fields": {{
    "doc_a": {{"code_field": "<field name or null>", "qty_field": "<field name or null>", "amount_field": "<field name or null>"}},
    "doc_b": {{"code_field": "<field name or null>", "qty_field": "<field name or null>", "amount_field": "<field name or null>"}}
  }}
}}
"""

PAIRING_PROMPT = """You are pairing line items between two extracted \
business documents that describe the same transaction but use different \
field names.

Document A line items:
{doc_a_items}

Document B line items:
{doc_b_items}

Identify which line items in A correspond to which line items in B \
(match by description similarity, item/model number, and position -- NOT \
by comparing the qty/amount values themselves, since your pairing will \
be used to check whether those values agree or disagree).

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "line_item_pairs": [
    {{"doc_a_index": 0, "doc_b_index": 0, "match_confidence": 0.0-1.0, "reason": "<short reason>"}}
  ],
  "unmatched_doc_a_indices": [],
  "unmatched_doc_b_indices": []
}}
"""


# Confirmed production failure this section fixes: a match job with
# multiple documents (which fires several _call_vllm_json calls in
# short succession -- field discovery + pairing, potentially more than
# once if run concurrently with OTHER match/summarize jobs hitting the
# same shared vLLM server) failed with a completely empty result and
# no error surfaced to the caller -- but the SAME job, run alone with
# nothing else competing for the GPU, succeeded every time. This is
# the identical failure mode already fixed once in summarize_agent.py
# (see that file's history) -- this file's _call_vllm_json had no
# retry or concurrency protection at all, so a transient
# connection-refused/timeout/429/5xx from vLLM being momentarily
# overloaded by concurrent requests propagated straight up as a hard
# failure instead of being absorbed.
#
# Two independent mechanisms, both needed:
# 1. A concurrency-limiting semaphore -- caps how many requests THIS
#    process sends to vLLM at once, so a multi-document match job's own
#    several calls don't pile onto an already-loaded server; excess
#    calls wait for a free slot (up to VLLM_QUEUE_WAIT_SECS) rather
#    than firing all at once and each getting a degraded response.
# 2. Bounded retry-with-backoff, but ONLY for TRANSIENT failures --
#    connection errors, timeouts, and 429/5xx status codes, all of
#    which can genuinely be different on a retried attempt once the
#    server has a free moment. Retrying a NON-transient failure (the
#    model returning something that isn't valid JSON) is pointless and
#    NOT done here: this call runs at temperature=0 specifically for
#    determinism, so the exact same prompt would almost certainly
#    produce the exact same malformed output again -- that failure
#    means something about the PROMPT or MODEL is wrong, not that the
#    server was momentarily busy, and retrying would just waste time
#    before failing anyway.
#
# Same env var names as summarize_agent.py's equivalent fix, so ops
# tooling/runbooks referencing one apply to both services.
VLLM_CONCURRENCY_LIMIT = int(os.environ.get("VLLM_CONCURRENCY_LIMIT", "3"))
MAX_TRANSIENT_MODEL_RETRIES = int(os.environ.get("MAX_TRANSIENT_MODEL_RETRIES", "5"))
VLLM_QUEUE_WAIT_SECS = float(os.environ.get("VLLM_QUEUE_WAIT_SECS", "90"))

_vllm_semaphore = threading.Semaphore(VLLM_CONCURRENCY_LIMIT)

# ---------------------------------------------------------------------------
# Parallel ("batch") matching.
#
# A match used to run strictly one model-backed comparison after another:
# for N documents that is N-1 reference-vs-other comparisons (each one a
# field-discovery call plus, usually, a pairing call), so 10 or 30 files
# took 10x or 30x as long as 2. Those comparisons do not depend on each
# other, so for a LARGE upload they now run in parallel batches:
#
#   PARALLEL_MATCH_MIN_DOCS   parallel mode only starts when `documents` has
#                             MORE than this many entries. At or below it the
#                             old sequential code path runs, unchanged.
#   MAX_PARALLEL_MATCH_WORKERS how many comparisons run at once. Defaults to
#                             VLLM_CONCURRENCY_LIMIT on purpose: more workers
#                             than vLLM slots would only queue behind
#                             _vllm_semaphore, and a worker that waits longer
#                             than VLLM_QUEUE_WAIT_SECS for a slot fails.
#                             Raise VLLM_CONCURRENCY_LIMIT first if the model
#                             server has headroom, then raise this.
#
# Two kinds of parallelism, never nested (nesting would multiply the number
# of workers fighting for the same vLLM slots):
#   1. several independent TRANSACTIONS in one upload (clearly separate POs)
#      are split into batches that run side by side -- see
#      _split_into_transaction_sets;
#   2. otherwise, the comparisons inside the one match run side by side.
# ---------------------------------------------------------------------------
PARALLEL_MATCH_MIN_DOCS = int(os.environ.get("PARALLEL_MATCH_MIN_DOCS", "5"))
MAX_PARALLEL_MATCH_WORKERS = max(
    1, int(os.environ.get("MAX_PARALLEL_MATCH_WORKERS", str(VLLM_CONCURRENCY_LIMIT)))
)


def _map_ordered(fn, items: list, max_workers: int) -> list:
    """
    fn(item) for every item, up to `max_workers` at a time, results in the
    SAME ORDER as `items` regardless of which finishes first. With
    max_workers <= 1 (or a single item) it is just a plain loop in the
    calling thread -- the old sequential behaviour, same exceptions in the
    same order. In parallel mode the first failure is re-raised and any
    work that has not started yet is cancelled.
    """
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
        try:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return results


def _call_vllm_json(prompt: str, timeout: float = 120) -> dict:
    """
    Shared low-level caller: posts a prompt to the self-hosted
    qwen3.6-35b-fp8 model via vLLM's OpenAI-compatible
    /v1/chat/completions endpoint, temperature=0 for determinism (this
    is a structured extraction/matching task, not a creative one --
    summarize_agent.py uses 0.3 for prose summarization, but that
    rationale doesn't apply here), and parses the JSON response
    (tolerating a ```json fence the model sometimes adds anyway).

    Used by both _call_vllm_field_discovery and _call_vllm_pairing --
    those differ only in which prompt they build and what shape they
    expect back; the HTTP/error/parsing/retry/concurrency handling is
    identical, so it lives here once.

    Concurrency-limited (see VLLM_CONCURRENCY_LIMIT above): acquires
    _vllm_semaphore before sending anything, waiting up to
    VLLM_QUEUE_WAIT_SECS for a free slot if this process is already at
    its concurrency cap. If no slot frees up in that time, raises
    RuntimeError rather than blocking indefinitely -- a genuinely stuck
    queue should surface as a failure eventually, just not on the very
    first sign of contention.

    Retries up to MAX_TRANSIENT_MODEL_RETRIES times, with exponential
    backoff (1s, 2s, 4s, ...), but ONLY for transient failures:
    connection errors, timeouts, and HTTP 429/5xx. A non-transient
    failure (a non-2xx status that isn't 429/5xx, or a 2xx response
    that isn't valid JSON) is raised immediately on the first attempt
    -- see the module-level comment above for why retrying those is
    pointless at temperature=0.
    """
    acquired = _vllm_semaphore.acquire(timeout=VLLM_QUEUE_WAIT_SECS)
    if not acquired:
        raise RuntimeError(
            f"Timed out after {VLLM_QUEUE_WAIT_SECS}s waiting for a free vLLM request slot "
            f"(VLLM_CONCURRENCY_LIMIT={VLLM_CONCURRENCY_LIMIT}) -- the server appears to be "
            f"persistently saturated, not just momentarily busy."
        )
    try:
        last_exc = None
        for attempt in range(MAX_TRANSIENT_MODEL_RETRIES):
            try:
                r = requests.post(
                    f"{VLLM_BASE_URL}/v1/chat/completions",
                    json={
                        "model": VLLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 2048,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=timeout,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < MAX_TRANSIENT_MODEL_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Could not reach vLLM at {VLLM_BASE_URL} after "
                    f"{MAX_TRANSIENT_MODEL_RETRIES} attempts: {exc}"
                ) from exc

            if not r.ok:
                try:
                    body = r.json()
                    detail = (
                        body.get("error", {}).get("message") if isinstance(body.get("error"), dict)
                        else body.get("error") or body.get("message") or json.dumps(body)
                    )
                except Exception:
                    detail = (r.text or "")[:2000]

                is_transient = r.status_code == 429 or r.status_code >= 500
                if is_transient and attempt < MAX_TRANSIENT_MODEL_RETRIES - 1:
                    last_exc = RuntimeError(
                        f"{r.status_code} {r.reason} from {VLLM_BASE_URL}/v1/chat/completions "
                        f"-- vLLM said: {detail}"
                    )
                    time.sleep(2 ** attempt)
                    continue

                raise RuntimeError(
                    f"{r.status_code} {r.reason} from {VLLM_BASE_URL}/v1/chat/completions "
                    f"-- vLLM said: {detail}"
                )

            text = r.json()["choices"][0]["message"]["content"].strip()
            cleaned = _strip_json_fence(text)
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError) as exc:
                # NOT transient -- see docstring. Same prompt at
                # temperature=0 would almost certainly fail the same
                # way again, so this fails on the first attempt rather
                # than burning retries and time on it.
                raise RuntimeError(
                    f"Model did not return valid JSON output. "
                    f"Raw output (first 500 chars): {cleaned[:500]!r}"
                ) from exc

        # Only reached if every attempt hit a transient failure.
        raise RuntimeError(
            f"vLLM call failed after {MAX_TRANSIENT_MODEL_RETRIES} attempts, all transient "
            f"(connection/timeout/429/5xx): {last_exc}"
        ) from last_exc
    finally:
        _vllm_semaphore.release()


def _call_vllm_field_discovery(doc_a_top: dict, doc_b_top: dict,
                                items_a: list, items_b: list, timeout: float = 120) -> dict:
    """
    Stage 1: asks the model to identify (a) which top-level fields hold
    the total/subtotal amount in each document, and (b) which line-item
    field names each document uses for product code / quantity / amount.

    This replaces what used to be a set of hardcoded key lists
    (_CODE_KEYS, and fixed qty_keys/amount_keys in _extract_qty_amount)
    -- those existed because a document might call its amount field
    "total", "amount_sgd", "line_total", etc., and every new naming
    convention required a code change to recognize. Asking the model to
    identify the field NAME (never the value) removes that maintenance
    burden while keeping the actual numeric comparison fully
    deterministic: this call never judges whether values match, only
    which keys to look at. The hardcoded lists remain as a fallback --
    see _extract_qty_amount and _deterministic_code_prematch -- for
    when discovery is inconclusive (null) or wrong.

    Only a few sample items per document are sent (not the full list),
    since field NAMING is consistent within one document -- this keeps
    the call fast regardless of how many line items exist.
    """
    sample_a = items_a[:3]
    sample_b = items_b[:3]
    prompt = FIELD_DISCOVERY_PROMPT.format(
        doc_a_fields=json.dumps(doc_a_top, indent=2),
        doc_b_fields=json.dumps(doc_b_top, indent=2),
        doc_a_sample_items=json.dumps(sample_a, indent=2),
        doc_b_sample_items=json.dumps(sample_b, indent=2),
    )
    return _call_vllm_json(prompt, timeout=timeout)


def _call_vllm_pairing(items_a: list, items_b: list, timeout: float = 120) -> dict:
    """
    Stage 2: asks the model to pair line items between two documents.
    Only called with items that survived deterministic exact-code
    pre-matching (see _deterministic_code_prematch) -- i.e. only the
    genuinely ambiguous remainder, not the whole line-item list. If
    nothing remains after pre-matching, this is skipped entirely (see
    _align_documents), saving a full round-trip for the common case
    where every item has a matching code across documents.

    Returns pair indices relative to the items_a/items_b lists passed
    in here -- NOT the original document's full item list. Callers are
    responsible for remapping (see _align_documents). Never used for
    numeric judgments -- see module docstring.
    """
    prompt = PAIRING_PROMPT.format(
        doc_a_items=json.dumps(items_a, indent=2),
        doc_b_items=json.dumps(items_b, indent=2),
    )
    return _call_vllm_json(prompt, timeout=timeout)


# ---------------------------------------------------------------------------
# Deterministic numeric comparison (the part that must NOT be AI)
# ---------------------------------------------------------------------------

def _repair_ai_pairing_order_crossings(pairs: list, items_a: list) -> list:
    """
    Deterministically un-crosses AI-paired line items whose doc_a side
    shares an IDENTICAL description with at least one other AI-paired
    item -- exactly the scenario the pairing model has no real signal
    to resolve (see PAIRING_PROMPT: paired by description/item-number/
    position, deliberately NEVER by amount, since the pairing is later
    used to judge whether amounts agree -- using them to pair too would
    be circular).

    Confirmed production case this fixes: a PO had two lines both
    literally named "CHAMPS INSTANT WATER HEATER" (different unit costs,
    but identical description text). Paired against an invoice whose
    two corresponding lines had genuinely distinct descriptions (a
    "Gunmetal" variant and a "Matte Black" variant), one run of the
    pairing model matched them in the correct order, and a SEPARATE run
    on the byte-identical input documents matched them CROSSED --
    despite temperature=0, this service has a documented history of
    batch-composition-dependent GPU numerical variance flipping exactly
    this kind of close call (see byoa-match-agent's fixed batch-only
    failure). The crossed run fabricated two large, entirely non-existent
    "discrepancies" -- the same two real numbers, just swapped onto the
    wrong rows -- while burying the one REAL, smaller discrepancy that
    genuinely existed on a different, unambiguous line.

    The fix trusts DOCUMENT ORDER over the model's guess whenever it's
    ambiguous: real invoices/POs list corresponding line items in the
    same relative sequence as each other (that's how humans generate
    and reconcile these documents in the first place). For any group of
    doc_a items sharing byte-identical description text (whitespace/
    case-insensitive) that were each AI-paired to some doc_b index,
    this reassigns those pairs' doc_b_index values into the SAME
    relative order as their doc_a_index -- i.e. if the group's doc_a
    indices are i1 < i2 < ... < in, their matched doc_b indices are
    re-sorted into ascending order too, undoing any crossing the model
    introduced. The affected pairs' `reason` field gets a note appended
    so a reviewer can see a pairing was order-corrected rather than
    silently changed.

    Deliberately conservative:
    - Only regroups items that actually share a description -- pairs
      for differently-described items are never touched.
    - Only reassigns doc_b_index WITHIN a group; the overall multiset of
      doc_a<->doc_b pairs doesn't change, only which doc_a in the group
      ends up paired with which doc_b.
    - A group where any member is missing a doc_b_index, or where the
      description can't be found under any of the common field names
      tried, is left completely untouched rather than guessed at.
    - A group of size 1 has nothing to cross with, so it's already
      unambiguous and left alone.

    Only operates on AI-paired items -- deterministic exact-code
    prematches (see _deterministic_code_prematch) are unambiguous by
    construction and are never passed through this function.
    """
    _DESCRIPTION_FIELD_CANDIDATES = ["description", "desc", "item_description", "product_description"]

    def _norm_description(item: dict) -> "str | None":
        for key in _DESCRIPTION_FIELD_CANDIDATES:
            value = item.get(key)
            if value:
                return re.sub(r"\s+", " ", str(value)).strip().lower()
        return None

    groups: dict = {}
    for idx, pair in enumerate(pairs):
        ia = pair.get("doc_a_index")
        if ia is None or ia >= len(items_a):
            continue
        desc = _norm_description(items_a[ia])
        if desc is None:
            continue
        groups.setdefault(desc, []).append(idx)

    for pair_indices in groups.values():
        if len(pair_indices) < 2:
            continue  # a single occurrence has nothing to cross with

        group_pairs = [pairs[i] for i in pair_indices]
        if any(p.get("doc_b_index") is None for p in group_pairs):
            continue  # incomplete group -- don't guess, leave it as the model returned it

        group_pairs_by_a = sorted(group_pairs, key=lambda p: p["doc_a_index"])
        sorted_b_indices = sorted(p["doc_b_index"] for p in group_pairs)

        for pair, new_b in zip(group_pairs_by_a, sorted_b_indices):
            if pair["doc_b_index"] != new_b:
                pair["doc_b_index"] = new_b
                pair["reason"] = (
                    (pair.get("reason") or "").rstrip()
                    + " [order-corrected: reassigned to preserve document order among "
                      "identically-described line items, since description gave the "
                      "pairing step no real signal to disambiguate them]"
                ).strip()

    return pairs


_UNIT_PRICE_KEYS = ("unit_price", "unitPrice", "unitCost", "unit_cost", "price_each", "price")


def _unit_price_of(item: dict, qty_field: Optional[str], amount_field: Optional[str]) -> Optional[Decimal]:
    for k in _UNIT_PRICE_KEYS:
        v = _to_decimal(item.get(k))
        if v is not None:
            return v
    vals = _extract_qty_amount(item, qty_field, amount_field)
    if vals["total"] is not None and vals["qty"]:
        return vals["total"] / vals["qty"]
    return None


def _repair_ai_pairing_by_unit_price(ai_pairs: list, items_a: list, items_b: list,
                                     fields_a: dict, fields_b: dict,
                                     free_a: set, free_b: set) -> list:
    """
    Safety net for the pairing model picking the wrong one of two
    look-alike lines. Confirmed case: an invoice line
    "POWER SUPPLY/AC CHARGER/MPA-ACAS01BFBK" (43.98) was paired with the
    PO's "30W ... WALL CHARGER" (16.46) instead of the "100W ... WALL
    CHARGER" (43.98), and the invoice's other charger line
    (MPA-ACAS03BFBK, 16.46) was left unmatched -- producing a fake
    amount mismatch AND a fake "billed but not received" in one go.

    The pairing prompt deliberately ignores money (pairing is later used
    to CHECK the money), so this only steps in when the model's pairing
    is visibly wrong AND the fix is unambiguous:
      - a pair whose unit prices disagree, and
      - EXACTLY ONE still-unpaired line on the other side has exactly the
        same unit price (then the freed line is re-paired the same way,
        again only if exactly one candidate fits), or
      - two pairs that are simply crossed (swapping makes both prices agree).
    A genuine overcharge on a line with no exactly-priced look-alike
    sitting unpaired is never touched, so it still gets flagged.
    """
    tol = Decimal("0.005")
    price_a = {i: _unit_price_of(it, fields_a.get("qty_field"), fields_a.get("amount_field")) for i, it in enumerate(items_a)}
    price_b = {i: _unit_price_of(it, fields_b.get("qty_field"), fields_b.get("amount_field")) for i, it in enumerate(items_b)}

    def same(x, y):
        return x is not None and y is not None and abs(x - y) <= tol

    def mismatched(p):
        pa, pb = price_a.get(p["doc_a_index"]), price_b.get(p["doc_b_index"])
        return pa is not None and pb is not None and not same(pa, pb)

    pairs = [dict(p) for p in ai_pairs]
    note = " [re-paired by unit price: the model's pairing had mismatched unit prices]"

    # 1) crossed pairs
    for x in range(len(pairs)):
        for y in range(x + 1, len(pairs)):
            p, q = pairs[x], pairs[y]
            if (mismatched(p) and mismatched(q)
                    and same(price_a[p["doc_a_index"]], price_b[q["doc_b_index"]])
                    and same(price_a[q["doc_a_index"]], price_b[p["doc_b_index"]])):
                p["doc_b_index"], q["doc_b_index"] = q["doc_b_index"], p["doc_b_index"]
                for r in (p, q):
                    r["reason"] = (r.get("reason") or "") + note

    # 2) pair with an exactly-priced unpaired line on the other side
    free_a, free_b = set(free_a), set(free_b)
    for p in pairs:
        if not mismatched(p):
            continue
        cands = [b for b in free_b if same(price_b[b], price_a[p["doc_a_index"]])]
        if len(cands) != 1:
            continue
        old_b = p["doc_b_index"]
        p["doc_b_index"] = cands[0]
        p["reason"] = (p.get("reason") or "") + note
        free_b.discard(cands[0])
        back = [a for a in free_a if same(price_a[a], price_b[old_b])]
        if len(back) == 1:
            pairs.append({"doc_a_index": back[0], "doc_b_index": old_b, "match_confidence": p.get("match_confidence"),
                          "reason": "freed by unit-price re-pairing" + note})
            free_a.discard(back[0])
        else:
            free_b.add(old_b)
    return pairs


def _align_documents(doc_a: dict, doc_b: dict, timeout: float = 120) -> dict:
    """
    Full alignment pipeline for one document pair:

      1. AI field discovery (_call_vllm_field_discovery) -- identifies,
         once per document pair: which top-level fields hold the total/
         subtotal amount, and which line-item field names each document
         uses for product code / quantity / amount. This replaces what
         used to be fixed hardcoded key lists for line-item qty/amount/
         code fields -- a real bug this replaced: a document using
         "amount_sgd" for its line amounts was silently invisible to
         comparison because that key wasn't on the hardcoded list.
         Only a few sample items are sent (naming is consistent within
         one document), keeping this call fast and cheap.
      2. Filter out probable non-item rows (GST lines, credit/rebate
         adjustments, etc. that OCR mis-extracted into line_items) --
         see _is_probable_non_item_row, now using the discovered code/
         qty fields alongside the hardcoded fallback lists. These never
         reach prematch or pairing; they're reported separately as
         excluded rows rather than silently vanishing.
      3. Deterministic exact-code pre-match, using the discovered code
         field (plus the hardcoded fallback list) -- zero AI judgment,
         zero ambiguity, for any REAL line items sharing an exact code.
      4. AI pairing (_call_vllm_pairing) ONLY for whatever real items
         remain unmatched after step 3, and ONLY if any remain on both
         sides -- if every item prematched by code, this call is
         skipped entirely, saving a full round-trip for what turns out
         to be the common case.
      5. Remap indices at each stage back to the ORIGINAL document's
         line_items positions, so doc_a_index/doc_b_index in the output
         always refer to a position in the source JSON.

    Removing unambiguous (exact-code) and not-actually-items cases from
    what the AI has to guess at removes two observed failure modes: a
    wrong pairing scoring the same confidence as a correct one, and a
    tax/fee row being compared as if it were a product line.

    Returns discovered field names alongside the match results, so
    run_reconciliation can use them for its own qty/amount extraction
    instead of guessing from a fixed list independently of this call.
    """
    items_a = _get_line_items(doc_a)
    items_b = _get_line_items(doc_b)

    doc_a_top = {k: v for k, v in doc_a.items() if k != "line_items" and not isinstance(v, list)}
    doc_b_top = {k: v for k, v in doc_b.items() if k not in ("line_items", "items") and not isinstance(v, list)}

    discovery = _call_vllm_field_discovery(doc_a_top, doc_b_top, items_a, items_b, timeout=timeout)
    line_item_fields = discovery.get("line_item_fields", {})
    fields_a = line_item_fields.get("doc_a", {}) or {}
    fields_b = line_item_fields.get("doc_b", {}) or {}
    code_field_a, qty_field_a, amount_field_a = fields_a.get("code_field"), fields_a.get("qty_field"), fields_a.get("amount_field")
    code_field_b, qty_field_b, amount_field_b = fields_b.get("code_field"), fields_b.get("qty_field"), fields_b.get("amount_field")

    partition_a = _partition_line_items(items_a, code_field_a, qty_field_a)
    partition_b = _partition_line_items(items_b, code_field_b, qty_field_b)

    # REAL coordinate space: only the rows that passed the non-item filter.
    real_indices_a = partition_a["real_indices"]
    real_indices_b = partition_b["real_indices"]
    real_items_a = [items_a[i] for i in real_indices_a]
    real_items_b = [items_b[i] for i in real_indices_b]

    prematch = _deterministic_code_prematch(real_items_a, real_items_b, code_field_a, code_field_b)

    # REMAINING coordinate space: real items with no exact-code prematch.
    remaining_a_items = [real_items_a[i] for i in prematch["remaining_a"]]
    remaining_b_items = [real_items_b[i] for i in prematch["remaining_b"]]

    def _remaining_to_original_a(pos: int) -> int:
        return real_indices_a[prematch["remaining_a"][pos]]

    def _remaining_to_original_b(pos: int) -> int:
        return real_indices_b[prematch["remaining_b"][pos]]

    remapped_ai_pairs = []
    if remaining_a_items and remaining_b_items:
        pairing_result = _call_vllm_pairing(remaining_a_items, remaining_b_items, timeout=timeout)
        for pair in pairing_result.get("line_item_pairs", []):
            sub_ia, sub_ib = pair.get("doc_a_index"), pair.get("doc_b_index")
            if sub_ia is None or sub_ib is None:
                continue
            if sub_ia >= len(prematch["remaining_a"]) or sub_ib >= len(prematch["remaining_b"]):
                continue  # model returned an out-of-range index; skip rather than crash
            remapped_ai_pairs.append({
                **pair,
                "doc_a_index": _remaining_to_original_a(sub_ia),
                "doc_b_index": _remaining_to_original_b(sub_ib),
            })
    # else: nothing left to pair -- every real item resolved by exact
    # code match, so the pairing call is skipped entirely.

    # Deterministic exact-code prematches are unambiguous by
    # construction, so only the AI-guessed pairs need this correction.
    remapped_ai_pairs = _repair_ai_pairing_order_crossings(remapped_ai_pairs, items_a)

    _taken_a = {real_indices_a[p["doc_a_index"]] for p in prematch["pairs"]} | {p["doc_a_index"] for p in remapped_ai_pairs}
    _taken_b = {real_indices_b[p["doc_b_index"]] for p in prematch["pairs"]} | {p["doc_b_index"] for p in remapped_ai_pairs}
    remapped_ai_pairs = _repair_ai_pairing_by_unit_price(
        remapped_ai_pairs, items_a, items_b, fields_a, fields_b,
        free_a=set(real_indices_a) - _taken_a, free_b=set(real_indices_b) - _taken_b)

    # Remap the deterministic prematch's REAL-space indices back to ORIGINAL.
    remapped_prematch_pairs = [
        {**p, "doc_a_index": real_indices_a[p["doc_a_index"]], "doc_b_index": real_indices_b[p["doc_b_index"]]}
        for p in prematch["pairs"]
    ]

    all_pairs = remapped_prematch_pairs + remapped_ai_pairs
    matched_a = {p["doc_a_index"] for p in all_pairs}
    matched_b = {p["doc_b_index"] for p in all_pairs}

    # Unmatched candidates are only drawn from REAL rows -- an excluded
    # (non-item) row should never show up as "unmatched", since it was
    # deliberately never a candidate for matching in the first place.
    unmatched_a = [i for i in real_indices_a if i not in matched_a]
    unmatched_b = [i for i in real_indices_b if i not in matched_b]

    return {
        "monetary_fields": discovery.get("monetary_fields", []),
        "line_item_fields": {"doc_a": fields_a, "doc_b": fields_b},
        "line_item_pairs": all_pairs,
        "unmatched_doc_a_indices": unmatched_a,
        "unmatched_doc_b_indices": unmatched_b,
        "excluded_doc_a_rows": partition_a["excluded"],
        "excluded_doc_b_rows": partition_b["excluded"],
        "deterministic_match_count": len(remapped_prematch_pairs),
        "ai_match_count": len(remapped_ai_pairs),
    }

def _extract_qty_amount(item: dict, qty_field: Optional[str] = None,
                         amount_field: Optional[str] = None) -> dict:
    """
    Pulls qty/unit_price/total from a line item, tolerant of naming.

    qty_field/amount_field are the AI-discovered field names for THIS
    document (see _call_vllm_field_discovery) -- tried first if given.
    The hardcoded qty_keys/amount_keys lists below remain as a fallback
    for when discovery returned null or was wrong, so a document is
    never left with zero extraction just because one discovery call
    was inconclusive.
    """
    qty_keys = ["qty", "quantity", "qty_ordered", "qty_delivered", "delivered_qty",
                "quantity_delivered", "qty_shipped", "shipped_qty"]
    amount_keys = ["total", "line_amount", "amount", "line_total", "amount_sgd"]
    unit_keys = ["unit_price", "price_each", "price"]

    def first_present(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    qty_val = item.get(qty_field) if qty_field and item.get(qty_field) is not None else first_present(item, qty_keys)
    amount_val = (
        item.get(amount_field) if amount_field and item.get(amount_field) is not None
        else first_present(item, amount_keys)
    )

    qty = _to_decimal(qty_val)
    unit_price = _to_decimal(first_present(item, unit_keys))
    total = _to_decimal(amount_val)

    # OCR sometimes drops a single qty cell (e.g. SI1202997 line 4: qty
    # blank, but unit_price 10.96 and amount 10.96 are both there). When
    # that happens, work the qty back out from amount / unit_price -- but
    # ONLY when it comes out as a clean whole number, so a price/amount
    # mismatch never gets quietly "explained away" as some odd fraction.
    # The line is marked qty_derived so the output says it was inferred.
    qty_derived = False
    if qty is None and unit_price is not None and total is not None:
        # Wrapped so a strange value (e.g. OCR text "NaN"/"Infinity", which
        # Decimal accepts) can never crash the whole match -- worst case the
        # qty just stays blank, exactly like before this feature existed.
        try:
            if unit_price.is_finite() and total.is_finite() and unit_price > 0:
                guess = total / unit_price
                rounded = guess.to_integral_value()
                if rounded > 0 and abs(guess - rounded) <= Decimal("0.001"):
                    qty = rounded
                    qty_derived = True
        except (InvalidOperation, ArithmeticError):
            pass

    return {
        "qty": qty,
        "qty_derived": qty_derived,
        "unit_price": unit_price,
        "total": total,
        "description": item.get("description") or item.get("item_desc") or item.get("item_description"),
    }


# ---------------------------------------------------------------------------
# Document role inference + comparison mode
# ---------------------------------------------------------------------------
#
# Three-way AP matching isn't one symmetric "does A match B" question --
# it's two DIFFERENT questions depending on which two document types are
# being compared:
#   - Invoice vs GRN ("fulfillment check"): both represent ACTUAL events
#     (what was delivered, what's being billed) and should tie out
#     exactly. Any difference is a genuine discrepancy.
#   - PO vs Invoice/GRN ("commitment check"): the PO is a COMMITMENT, not
#     a forecast of a single delivery -- partial shipments are normal.
#     The invoiced/received quantity being LESS than what was ordered is
#     an open-order status, not an error; being MORE than what was
#     ordered is still a real problem (billed/shipped beyond commitment).
# Getting this distinction right is the difference between correctly
# reporting "3,870 of 20,000 ordered units received so far" and wrongly
# reporting a flat, unexplained mismatch for a perfectly normal partial
# delivery.

def _describe_line_item_issue(item: dict, mode: str, role_a: str, role_b: str, po_is_a: bool,
                               currency_label: str = DEFAULT_CURRENCY) -> Optional[str]:
    """
    One short phrase naming exactly what's wrong with a single discrepant
    line item -- e.g. "Pear: Invoice shows 10, GRN/Delivery shows 5" --
    so a comparison's summary can say WHICH item and WHAT the numbers
    were instead of just "discrepancies found". Returns None for a line
    with nothing wrong (both qty_status and amount_status CONSISTENT/
    NOT_APPLICABLE/PARTIAL_FULFILLMENT).
    """
    if item.get("qty_status") != "DISCREPANCY" and item.get("amount_status") != "DISCREPANCY":
        return None
    description = item.get("doc_a_description") or item.get("doc_b_description") or "item"
    label_a = _ROLE_LABELS.get(role_a, "Doc A")
    label_b = _ROLE_LABELS.get(role_b, "Doc B")

    parts = []
    item = {**item, "qty_a": item.get("qty_a") or "blank", "qty_b": item.get("qty_b") or "blank"}
    if item.get("qty_status") == "DISCREPANCY":
        if mode == "commitment_check":
            po_label, actual_label = (label_a, label_b) if po_is_a else (label_b, label_a)
            ordered, actual = (item.get("qty_a"), item.get("qty_b")) if po_is_a else (item.get("qty_b"), item.get("qty_a"))
            parts.append(f"{po_label} allows {ordered}, {actual_label} shows {actual}")
        elif mode == "fulfillment_check":
            parts.append(f"{label_a} shows {item.get('qty_a')}, {label_b} shows {item.get('qty_b')}")
        else:
            parts.append(f"{label_a} shows {item.get('qty_a')}, {label_b} shows {item.get('qty_b')}")
    if item.get("amount_status") == "DISCREPANCY" and item.get("price_note"):
        parts.append(f"unit price: {item['price_note']}")
    elif item.get("amount_status") == "DISCREPANCY":
        parts.append(f"amount {_format_amount(item.get('amount_a'), currency_label)} vs "
                     f"{_format_amount(item.get('amount_b'), currency_label)}")
    return f"{description}: {'; '.join(parts)}"


def _classify_mismatch_label(status: str, line_item_results: list, monetary_comparisons: list) -> Optional[str]:
    """
    A more specific companion to "status": when status is DISCREPANCY,
    names WHAT KIND of mismatch it is -- "QTY_MISMATCH", "AMOUNT_
    MISMATCH", "QTY_AND_AMOUNT_MISMATCH" when a discrepant line has both
    problems (or different lines each have a different one), or "OTHER_
    MISMATCH" for a DISCREPANCY that isn't tied to any qty/amount line
    comparison at all (e.g. a role/generic mismatch, or a discrepancy
    whose only evidence is missing data rather than a numeric
    disagreement). Returns None when status isn't DISCREPANCY -- there's
    nothing to further classify about a CONSISTENT, PARTIAL_FULFILLMENT,
    or NOT_APPLICABLE result.
    """
    if status != "DISCREPANCY":
        return None
    has_qty = any(li.get("qty_status") == "DISCREPANCY" for li in line_item_results)
    has_amount = (any(li.get("amount_status") == "DISCREPANCY" for li in line_item_results)
                  or any(mc.get("status") == "DISCREPANCY" for mc in monetary_comparisons))
    if has_qty and has_amount:
        return "QTY_AND_AMOUNT_MISMATCH"
    if has_qty:
        return "QTY_MISMATCH"
    if has_amount:
        return "AMOUNT_MISMATCH"
    return "OTHER_MISMATCH"


def _combine_mismatch_labels(labels: list) -> Optional[str]:
    """
    Combines several already-computed mismatch labels (e.g. one per
    comparison in a multi-document call, or one per invoice in a
    timeline) into a single aggregate label, using the same QTY/AMOUNT/
    BOTH/OTHER vocabulary as _classify_mismatch_label. None entries
    (comparisons with no mismatch) are ignored; an all-None input
    returns None.
    """
    present = [l for l in labels if l]
    if not present:
        return None
    has_qty = any(l in ("QTY_MISMATCH", "QTY_AND_AMOUNT_MISMATCH") for l in present)
    has_amount = any(l in ("AMOUNT_MISMATCH", "QTY_AND_AMOUNT_MISMATCH") for l in present)
    if has_qty and has_amount:
        return "QTY_AND_AMOUNT_MISMATCH"
    if has_qty:
        return "QTY_MISMATCH"
    if has_amount:
        return "AMOUNT_MISMATCH"
    return "OTHER_MISMATCH"


_ROLE_LABELS = {"PO": "PO", "INVOICE": "Invoice", "GRN": "GRN/Delivery",
                "CREDIT_NOTE": "Credit Note", "UNKNOWN": "Document"}

_PO_KEY_MARKERS = {"ponumber", "purchaseorderno"}
_INVOICE_KEY_MARKERS = {"invoiceno", "invoicenumber"}
_GRN_KEY_MARKERS = {"grnno", "deliverynoteno", "dono", "goodsreceiptno"}


def _normalize_key(k: str) -> str:
    return k.lower().replace("_", "")


def _infer_document_role(struct: dict) -> str:
    """
    Best-effort classification of a document as PO / INVOICE / GRN /
    CREDIT_NOTE / UNKNOWN, used ONLY to decide comparison MODE (see
    _comparison_mode below) -- this never changes which fields get
    extracted or how AI alignment works, only how a mismatch gets
    interpreted afterward. Two detection layers, most authoritative
    first:

      1. An explicit document_type/doc_type/type field, if present
         (e.g. the older OCR pipeline's "Tax Invoice"/"Delivery Note"/
         "goods_receipt_note" convention) -- checked by substring.
      2. Self-identifying ID field NAMES, for document shapes with no
         type field at all (e.g. po.json/invoice.json/grn*.json, which
         carry no document_type anywhere): a top-level "poNumber" key
         means "this document IS a PO", as opposed to a field like
         "customer_po" or "po_ref" which merely REFERENCES one --
         only self-identifying markers are checked, deliberately, to
         avoid misclassifying a document just because it mentions
         another document's number.

    Returns "UNKNOWN" when neither layer matches -- a comparison
    involving an UNKNOWN-role document falls back to mode "generic"
    (the original strict/symmetric behavior), so an unrecognized
    document shape never silently gets weakened matching.
    """
    doc_type = str(
        struct.get("document_type") or struct.get("doc_type") or struct.get("type") or ""
    ).lower()
    if "credit" in doc_type:
        return "CREDIT_NOTE"
    if "invoice" in doc_type:
        return "INVOICE"
    if "delivery" in doc_type or "grn" in doc_type or "goods receipt" in doc_type or "goods_receipt" in doc_type:
        return "GRN"
    if "purchase order" in doc_type or "purchase_order" in doc_type or doc_type == "po":
        return "PO"

    keys = {_normalize_key(k) for k in struct if isinstance(k, str)}
    if keys & _PO_KEY_MARKERS:
        return "PO"
    if keys & _INVOICE_KEY_MARKERS:
        return "INVOICE"
    if keys & _GRN_KEY_MARKERS:
        return "GRN"
    return "UNKNOWN"


def _infer_doc_id(struct: dict) -> Optional[str]:
    """
    Best-effort extraction of a document's OWN identifying number (its
    invoice number, PO number, GRN number, etc.) for display purposes
    only -- this never affects role inference, comparison mode, or any
    matching logic. It exists purely so output can show which real-world
    documents were actually involved (e.g. "PPINV-B10097157") instead of
    just a bare index.

    Two layers, most authoritative first:
      1. An explicit doc_id/document_id field, if the OCR pipeline
         already tags the document with its own identifier directly.
      2. The same self-identifying marker fields _infer_document_role
         uses (e.g. a PO's "poNumber", an Invoice's "invoiceNo", a
         GRN's "grnNo") -- deliberately reused so a field that merely
         REFERENCES another document (e.g. "customer_po" on an
         Invoice) is never mistaken for this document's own ID, same
         as role inference avoids that mistake.

    Returns None if no candidate field is found or present values are
    empty -- callers should treat that as "unknown", never fabricate one.
    """
    for key in ("doc_id", "document_id", "docid", "documentid"):
        val = struct.get(key)
        if val not in (None, ""):
            return str(val)

    normalized = {_normalize_key(k): k for k in struct if isinstance(k, str)}
    for marker_set in (_PO_KEY_MARKERS, _INVOICE_KEY_MARKERS, _GRN_KEY_MARKERS):
        for marker in marker_set:
            if marker in normalized:
                val = struct.get(normalized[marker])
                if val not in (None, ""):
                    return str(val)
    return None


def _comparison_mode(role_a: str, role_b: str) -> str:
    """
    "fulfillment_check": Invoice vs GRN, either order -- strict, both
    sides are actual events.
    "commitment_check": PO vs (Invoice or GRN) -- lenient on quantity/
    amount shortfalls (partial fulfillment is normal), still flags
    overages.
    "generic": anything else (a role is UNKNOWN, or the two roles don't
    form a recognized AP pair) -- unchanged strict behavior, identical
    to how this file always compared everything before this feature.
    """
    roles = {role_a, role_b}
    if roles == {"INVOICE", "GRN"}:
        return "fulfillment_check"
    if "PO" in roles and roles & {"INVOICE", "GRN"}:
        return "commitment_check"
    return "generic"


def _compare_amounts(a: Optional[Decimal], b: Optional[Decimal], tolerance: Decimal) -> dict:
    if a is None and b is None:
        # Neither document has a value for this field at all -- they
        # AGREE it's absent (e.g. a Delivery Note and another document
        # that both correctly carry no total/GST amount). This is not a
        # discrepancy and must never fail overall_match.
        return {"match": True, "reason": "both_missing", "diff": None}
    if a is None or b is None:
        # Exactly ONE side has a value -- this IS a genuine discrepancy
        # (one document reports an amount the other doesn't have at all).
        return {"match": False, "reason": "missing_value", "diff": None}
    diff = abs(a - b)
    return {"match": diff <= tolerance, "diff": str(diff)}


def _classify_comparison(a: Optional[Decimal], b: Optional[Decimal], tolerance: Decimal,
                          mode: str, po_is_a: bool = False,
                          qty_is_short: Optional[bool] = None) -> dict:
    """
    Wraps _compare_amounts with a richer STATUS classification, aware of
    the commitment-vs-fulfillment distinction (see _comparison_mode).
    _compare_amounts itself is left completely UNCHANGED -- this only
    interprets its raw match/diff result differently depending on
    `mode`, so the underlying strict-equality primitive stays simple,
    tested, and reusable on its own; every key _compare_amounts already
    returns (match/diff/reason) is still present here, PLUS a new
    "status" key: one of "CONSISTENT", "PARTIAL_FULFILLMENT",
    "DISCREPANCY", "NOT_APPLICABLE".

    mode == "fulfillment_check" or "generic": any mismatch is
    DISCREPANCY -- identical to the old strict behavior, byte-for-byte,
    for every comparison that isn't a recognized PO-vs-Invoice/GRN pair.

    mode == "commitment_check": po_is_a says which side is the "ordered"
    value. If the "actual" (non-PO) side is LESS than "ordered", that's
    normally PARTIAL_FULFILLMENT (expected -- the order isn't fully
    closed out yet, not an error). If "actual" EXCEEDS "ordered", that's
    still DISCREPANCY (shipped/billed beyond what was ever committed to
    is a real problem, not a benign partial-order state).

    `qty_is_short` disambiguates WHY the actual side is lower, for the
    two different things this function is used to classify:
      - When classifying a QUANTITY comparison itself, leave this None:
        a qty shortfall's leniency is unconditional -- that's the whole
        point of "partial fulfillment" (fewer units delivered than
        ordered so far).
      - When classifying an AMOUNT comparison, pass in whether the
        associated line's quantity was ALSO short (`qty_is_short=True`)
        or fully matched (`qty_is_short=False`). A lower amount only
        reflects a genuine "not fully delivered/invoiced yet" state
        when it lines up with fewer units actually moving -- e.g. 2 of
        5 ordered units invoiced at a proportionally lower amount is a
        real open order. A lower amount on a line where the FULL
        ordered quantity was already delivered/invoiced (qty_is_short
        =False) isn't an open order at all; the units are all
        accounted for and the price itself doesn't match, so that's a
        real billing/price DISCREPANCY, not a benign partial-fulfillment
        state. Leave as None (the old, unconditional behavior) for
        callers with no associated per-line quantity to check against,
        such as header-level monetary fields (GST, subtotal, etc.).
    """
    raw = _compare_amounts(a, b, tolerance)
    if raw.get("reason") == "both_missing":
        status = "NOT_APPLICABLE"
    elif raw["match"]:
        status = "CONSISTENT"
    elif raw.get("reason") == "missing_value":
        status = "DISCREPANCY"
    elif mode == "commitment_check":
        ordered, actual = (a, b) if po_is_a else (b, a)
        is_shortfall = (ordered is not None and actual is not None and actual < ordered)
        if is_shortfall and (qty_is_short is None or qty_is_short):
            status = "PARTIAL_FULFILLMENT"
        else:
            status = "DISCREPANCY"
    else:
        status = "DISCREPANCY"
    return {**raw, "status": status}


def _aggregate_status(statuses: list) -> str:
    """DISCREPANCY beats PARTIAL_FULFILLMENT beats CONSISTENT beats
    NOT_APPLICABLE -- one real discrepancy anywhere is enough to flag
    the whole comparison, even if everything else lines up."""
    if "DISCREPANCY" in statuses:
        return "DISCREPANCY"
    if "PARTIAL_FULFILLMENT" in statuses:
        return "PARTIAL_FULFILLMENT"
    if "CONSISTENT" in statuses:
        return "CONSISTENT"
    return "NOT_APPLICABLE"


def _build_comparison_summary(role_a: str, role_b: str, mode: str, status: str,
                               po_qty_total: Optional[Decimal], actual_qty_total: Optional[Decimal],
                               discrepancy_directions: Optional[set] = None,
                               discrepant_lines: Optional[list] = None,
                               netted_pools: bool = False) -> str:
    """Plain-English one-liner for a single pairwise comparison -- e.g.
    "Invoice matches GRN/Delivery -- fully consistent." or "PO shows
    partial fulfillment: 3870 of 20000 ordered units received/invoiced
    -- not a discrepancy, an open-order status."

    `discrepancy_directions` (only meaningful when status == "DISCREPANCY"
    under commitment_check mode) is a set of {"overage", "shortfall",
    "missing_data"} describing WHY the discrepancy happened, collected
    across every discrepant line/field:
      - "overage": actual qty or amount EXCEEDS what was ordered --
        the only case this wording used to cover, back when a lower
        amount always got the lenient PARTIAL_FULFILLMENT treatment
        unconditionally.
      - "shortfall": actual amount is LESS than what was ordered even
        though that line's quantity was already fully accounted for --
        the case _classify_comparison's qty_is_short param now also
        classifies as DISCREPANCY (a real price/billing mismatch, not
        an open order -- see that function's docstring). Confirmed
        production case this fixes: an invoice billed $229.38 against a
        $283.38 PO line, quantity fully matched on both sides -- calling
        that "exceeds what was ordered" would have been backwards; the
        vendor billed LESS, not more.
      - "missing_data": one side has no value at all to compare against
        the other for that line/field.
    Falls back to the old unconditional "exceeds what was ordered"
    wording when the set is empty/None, so any caller that hasn't been
    updated to pass it keeps its previous behavior exactly.

    `discrepant_lines` (only meaningful when status == "DISCREPANCY"), if
    given, is a list of already-formatted per-line phrases (see
    _describe_line_item_issue) -- appended so the summary names WHICH
    item(s) and what the actual numbers were, instead of stopping at
    "discrepancies found". Omitted/empty list keeps the old, plain wording.
    """
    label_a = _ROLE_LABELS.get(role_a, "Document A")
    label_b = _ROLE_LABELS.get(role_b, "Document B")
    detail = f" Specifically: {'; '.join(discrepant_lines)}." if (status == "DISCREPANCY" and discrepant_lines) else ""

    if mode == "fulfillment_check":
        if status == "DISCREPANCY":
            return f"{label_b} does not match {label_a} -- discrepancies found.{detail}"
        if status == "PARTIAL_FULFILLMENT":
            # Only reachable with netted_pools=True (story mode): the
            # invoice bills fewer units than are still received-but-not-
            # yet-invoiced on the GRN -- normal split invoicing.
            grn_label, inv_label = (label_a, label_b) if role_a == "GRN" else (label_b, label_a)
            if po_qty_total is not None and actual_qty_total is not None and po_qty_total > 0:
                return (f"{inv_label} bills {actual_qty_total} of the {po_qty_total} units received on "
                        f"{grn_label} and not yet invoiced -- partial invoicing, not a discrepancy.")
            return (f"{inv_label} bills less than is still uninvoiced on {grn_label} -- partial "
                    f"invoicing, not a discrepancy.")
        return f"{label_b} matches {label_a} -- fully consistent."

    if mode == "commitment_check":
        po_label, other_label = (label_a, label_b) if role_a == "PO" else (label_b, label_a)
        if status == "DISCREPANCY":
            directions = discrepancy_directions or set()
            if directions == {"overage"} or not directions:
                return (f"{other_label} does not reconcile with {po_label}: at least one line "
                        f"exceeds what was ordered.{detail}")
            if directions == {"shortfall"}:
                return (f"{other_label} does not reconcile with {po_label}: at least one line's "
                        f"amount is LESS than what was ordered even though the quantity is fully "
                        f"accounted for -- a price/billing mismatch, not an open order.{detail}")
            if directions == {"price"}:
                return (f"{other_label} does not reconcile with {po_label}: at least one line is billed "
                        f"at a different unit price than the PO.{detail}")
            if directions == {"missing_data"}:
                return (f"{other_label} does not reconcile with {po_label}: at least one line is "
                        f"missing data needed to compare against what was ordered.{detail}")
            # Mixed causes across different lines/fields -- name them all
            # rather than asserting a single direction that isn't true of
            # every discrepant line.
            parts = []
            if "overage" in directions:
                parts.append("some exceeding what was ordered")
            if "shortfall" in directions:
                parts.append("some billed for less despite full quantity")
            if "missing_data" in directions:
                parts.append("some missing data to compare")
            if "price" in directions:
                parts.append("some billed at a different unit price")
            return (f"{other_label} does not reconcile with {po_label}: multiple lines disagree "
                    f"with what was ordered ({'; '.join(parts)}).{detail}")
        if status == "PARTIAL_FULFILLMENT":
            if po_qty_total is not None and actual_qty_total is not None and po_qty_total > 0:
                if netted_pools:
                    # Story mode: the PO passed in is what's LEFT after earlier
                    # invoices, so "ordered" would be the wrong word here.
                    return (f"{other_label} covers {actual_qty_total} of the {po_qty_total} units "
                            f"still open on {po_label} -- not a discrepancy, an open-order status.")
                return (f"{po_label} shows partial fulfillment: {actual_qty_total} of "
                        f"{po_qty_total} ordered units received/invoiced -- not a "
                        f"discrepancy, an open-order status.")
            return (f"{po_label} shows partial fulfillment against {other_label} -- an "
                    f"open-order status, not a discrepancy.")
        return f"{other_label} fully matches {po_label} -- order complete, no discrepancies."

    # generic (role unrecognized on at least one side) -- unchanged tone
    if status == "DISCREPANCY":
        return f"{label_b} does not match {label_a}.{detail}"
    return f"{label_b} matches {label_a}."


def run_reconciliation(doc_a: dict, doc_b: dict, tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
                        timeout: float = 120, line_items_only: bool = True,
                        netted_pools: bool = False,
                        header_tolerance_extra: Decimal = Decimal("0")) -> dict:
    struct_a = _find_structured(doc_a)
    struct_b = _find_structured(doc_b)

    role_a = _infer_document_role(struct_a)
    role_b = _infer_document_role(struct_b)
    doc_a_id = _infer_doc_id(struct_a)
    doc_b_id = _infer_doc_id(struct_b)
    mode = _comparison_mode(role_a, role_b)
    po_is_a = mode == "commitment_check" and role_a == "PO"

    # netted_pools (set only by run_po_invoice_story): the PO/GRN passed in
    # are what's still LEFT after earlier invoices were subtracted. In that
    # case an Invoice-vs-GRN pair behaves like a ceiling check: billing
    # FEWER units than are still uninvoiced on the GRN is normal split
    # invoicing (PARTIAL_FULFILLMENT); billing MORE is still a DISCREPANCY.
    # Everywhere else (a plain one-invoice call) Invoice-vs-GRN stays strict.
    grn_ceiling = netted_pools and mode == "fulfillment_check"
    line_classify_mode = "commitment_check" if grn_ceiling else mode
    ceiling_is_a = (role_a == "GRN") if grn_ceiling else po_is_a

    alignment = _align_documents(struct_a, struct_b, timeout=timeout)

    items_a = _get_line_items(struct_a)
    items_b = _get_line_items(struct_b)

    line_item_fields = alignment.get("line_item_fields", {})
    qty_field_a = (line_item_fields.get("doc_a") or {}).get("qty_field")
    amount_field_a = (line_item_fields.get("doc_a") or {}).get("amount_field")
    qty_field_b = (line_item_fields.get("doc_b") or {}).get("qty_field")
    amount_field_b = (line_item_fields.get("doc_b") or {}).get("amount_field")

    # Document-level signal: does this document carry amount data on ANY
    # line item at all? A Delivery Order structurally has no $ amounts on
    # any row -- that's different from one row's amount being missing due
    # to a genuine data/OCR gap. If a whole document has zero extractable
    # amounts across all its items, amount comparison against it is not
    # applicable and must not count as a mismatch; if the document DOES
    # carry amounts elsewhere but this one row doesn't, that's still a
    # real gap worth flagging.
    doc_a_has_amounts = any(_extract_qty_amount(it, qty_field_a, amount_field_a)["total"] is not None for it in items_a)
    doc_b_has_amounts = any(_extract_qty_amount(it, qty_field_b, amount_field_b)["total"] is not None for it in items_b)
    amounts_applicable = doc_a_has_amounts and doc_b_has_amounts

    # Only accumulated for mode == "commitment_check", to give
    # _build_comparison_summary real numbers for a partial-fulfillment
    # summary line (e.g. "3870 of 20000 ordered units received").
    po_qty_total = Decimal("0")
    actual_qty_total = Decimal("0")
    qty_totals_present = False

    # Collected across every discrepant qty/amount comparison, so the
    # summary (see _build_comparison_summary) can describe WHY status
    # ended up DISCREPANCY instead of always assuming an overage --
    # that assumption stopped being universally true once a lower
    # amount on a fully-matched quantity also became a DISCREPANCY
    # cause (see _classify_comparison's qty_is_short param).
    discrepancy_directions = set()

    def _note_discrepancy_direction(cmp_result: dict, ordered, actual) -> None:
        if cmp_result["status"] != "DISCREPANCY":
            return
        if ordered is None or actual is None:
            discrepancy_directions.add("missing_data")
        elif actual > ordered:
            discrepancy_directions.add("overage")
        elif actual < ordered:
            discrepancy_directions.add("shortfall")
        # actual == ordered can't reach DISCREPANCY (that's CONSISTENT),
        # so no third branch is needed here.

    line_item_results = []
    for pair in alignment.get("line_item_pairs", []):
        ia, ib = pair["doc_a_index"], pair["doc_b_index"]
        if ia >= len(items_a) or ib >= len(items_b):
            continue
        va = _extract_qty_amount(items_a[ia], qty_field_a, amount_field_a)
        vb = _extract_qty_amount(items_b[ib], qty_field_b, amount_field_b)

        qty_cmp = _classify_comparison(va["qty"], vb["qty"], DEFAULT_QTY_TOLERANCE,
                                       line_classify_mode, ceiling_is_a)
        if amounts_applicable:
            # Tell the amount classification whether THIS line's quantity
            # was also short, so a lower amount on a fully-delivered
            # quantity is correctly flagged as a price/billing
            # DISCREPANCY rather than being waved through as
            # PARTIAL_FULFILLMENT just because the number is smaller.
            # See _classify_comparison's docstring for the full
            # rationale (qty_is_short param).
            amt_cmp = _classify_comparison(va["total"], vb["total"], tolerance,
                                            line_classify_mode, ceiling_is_a,
                                            qty_is_short=(qty_cmp["status"] == "PARTIAL_FULFILLMENT"))
        else:
            amt_cmp = {"match": None, "diff": None, "status": "NOT_APPLICABLE"}

        # UNIT-PRICE CHECK on partially billed lines. A lower amount on a
        # lower quantity was always waved through as PARTIAL_FULFILLMENT --
        # but that also let an over-priced split invoice through (2 HDMI
        # cables at 26.50 when the PO says 24.72: 53.00 is still "less
        # than" the 98.88 left on the PO line). So: the amount must also be
        # in proportion -- (amount left on PO) x (qty billed / qty left).
        price_note = None
        if (amounts_applicable and line_classify_mode == "commitment_check"
                and qty_cmp["status"] == "PARTIAL_FULFILLMENT" and amt_cmp["status"] == "PARTIAL_FULFILLMENT"):
            o_q, a_q = (va["qty"], vb["qty"]) if ceiling_is_a else (vb["qty"], va["qty"])
            o_amt, a_amt = (va["total"], vb["total"]) if ceiling_is_a else (vb["total"], va["total"])
            if None not in (o_q, a_q, o_amt, a_amt) and o_q > 0 and a_q > 0:
                expected = o_amt * a_q / o_q
                if abs(a_amt - expected) > tolerance + Decimal("0.005") * a_q:
                    amt_cmp = {**amt_cmp, "status": "DISCREPANCY", "reason": "unit_price_mismatch"}
                    price_note = (f"billed {a_q} for {a_amt} ({(a_amt / a_q).quantize(Decimal('0.01'))}/unit), "
                                  f"PO rate is {(o_amt / o_q).quantize(Decimal('0.01'))}/unit")
                    discrepancy_directions.add("price")

        if mode == "commitment_check":
            qty_ordered, qty_actual = (va["qty"], vb["qty"]) if po_is_a else (vb["qty"], va["qty"])
            _note_discrepancy_direction(qty_cmp, qty_ordered, qty_actual)
            if amounts_applicable and price_note is None:
                amt_ordered, amt_actual = (va["total"], vb["total"]) if po_is_a else (vb["total"], va["total"])
                _note_discrepancy_direction(amt_cmp, amt_ordered, amt_actual)

        if (mode == "commitment_check" or grn_ceiling) and va["qty"] is not None and vb["qty"] is not None:
            ordered, actual = (va["qty"], vb["qty"]) if ceiling_is_a else (vb["qty"], va["qty"])
            po_qty_total += ordered
            actual_qty_total += actual
            qty_totals_present = True

        line_item_results.append({
            "doc_a_index": ia,
            "doc_b_index": ib,
            "match_confidence": pair.get("match_confidence"),
            "doc_a_description": va["description"],
            "doc_b_description": vb["description"],
            "qty_a": str(va["qty"]) if va["qty"] is not None else None,
            "qty_b": str(vb["qty"]) if vb["qty"] is not None else None,
            "qty_match": qty_cmp["match"],
            "qty_status": qty_cmp["status"],
            "amount_a": str(va["total"]) if va["total"] is not None else None,
            "amount_b": str(vb["total"]) if vb["total"] is not None else None,
            "amount_applicable": amounts_applicable,
            "amount_match": amt_cmp["match"],
            "amount_diff": amt_cmp["diff"],
            "amount_status": amt_cmp["status"],
            **({"qty_a_derived": True} if va["qty_derived"] else {}),
            **({"price_note": price_note} if price_note else {}),
            **({"qty_b_derived": True} if vb["qty_derived"] else {}),
        })

    # Under commitment_check, a PO-side row with NO counterpart at all in
    # the other document (never picked up by alignment -- see
    # unmatched_doc_*_indices) is exactly as much an open-order signal as
    # a matched pair with a lower quantity: the PO committed to it and
    # nothing has been invoiced/received against it in THIS document yet.
    # Without this, such a row was only visible via unmatched_doc_a/b_
    # indices and silently excluded from both the aggregate status and
    # the po_qty_total/actual_qty_total summary numbers -- which could
    # make an invoice covering only PART of a multi-line PO look like
    # "fully matches PO" instead of the partial fulfillment it actually
    # is.
    #
    # Under fulfillment_check (Invoice vs GRN, both sides real events),
    # the same idea applies but ONLY in the invoice-side direction: an
    # INVOICE row with no GRN counterpart means goods were billed before
    # they were ever received -- a genuine problem (billing_ahead_of_
    # delivery), not a benign timing gap, so it gets synthesized the same
    # way as a PO-side row above. The reverse -- a GRN row with nothing
    # invoiced against it yet -- is deliberately left untouched: goods
    # received but not yet billed is a normal, temporary state, not
    # something to flag as a discrepancy.
    #
    # Both directions are left in unmatched_doc_a/b_indices as well (not
    # removed) so existing consumers of that list see no change there.
    unmatched_authoritative_role = {"commitment_check": "PO", "fulfillment_check": "INVOICE"}.get(mode)
    if unmatched_authoritative_role == "PO":
        authoritative_is_a = po_is_a
    elif unmatched_authoritative_role == "INVOICE":
        authoritative_is_a = (role_a == "INVOICE")
    else:
        authoritative_is_a = None

    if unmatched_authoritative_role is not None:
        auth_items, auth_qty_field = (items_a, qty_field_a) if authoritative_is_a else (items_b, qty_field_b)
        auth_unmatched = (alignment.get("unmatched_doc_a_indices", []) if authoritative_is_a
                          else alignment.get("unmatched_doc_b_indices", []))
        for idx in auth_unmatched:
            if idx >= len(auth_items):
                continue
            auth_val = _extract_qty_amount(auth_items[idx], auth_qty_field)
            auth_qty = auth_val["qty"]
            if auth_qty is None:
                continue  # nothing numeric to compare -- leave as a plain unmatched row
            counterpart_qty = Decimal("0")
            a_qty, b_qty = (auth_qty, counterpart_qty) if authoritative_is_a else (counterpart_qty, auth_qty)
            qty_cmp = _classify_comparison(a_qty, b_qty, DEFAULT_QTY_TOLERANCE, mode, po_is_a)
            _note_discrepancy_direction(qty_cmp, auth_qty, counterpart_qty)
            if mode == "commitment_check":
                po_qty_total += auth_qty
                actual_qty_total += counterpart_qty
                qty_totals_present = True
            note = ("PO line has no counterpart in this document -- not yet invoiced/received here."
                    if unmatched_authoritative_role == "PO"
                    else "Invoice line has no counterpart in this document -- billed for goods not yet received.")
            line_item_results.append({
                "doc_a_index": idx if authoritative_is_a else None,
                "doc_b_index": idx if not authoritative_is_a else None,
                "match_confidence": None,
                "doc_a_description": auth_val["description"] if authoritative_is_a else None,
                "doc_b_description": auth_val["description"] if not authoritative_is_a else None,
                "qty_a": str(a_qty),
                "qty_b": str(b_qty),
                "qty_match": qty_cmp["match"],
                "qty_status": qty_cmp["status"],
                "amount_a": None,
                "amount_b": None,
                "amount_applicable": False,
                "amount_match": None,
                "amount_diff": None,
                "amount_status": "NOT_APPLICABLE",
                "note": note,
            })

    # Compare EVERY discovered monetary field, not just one designated
    # "total" -- subtotal, GST/tax amount, shipping fees, credits, etc.
    # all get checked the same way. Each entry is only "applicable" (and
    # can only fail overall_match) if BOTH documents actually have a
    # field for that concept; a field only one document tracks is still
    # reported (for visibility) but never counted as a mismatch.
    #
    # `line_items_only` (default True) skips this block entirely: only
    # the line items decide the match/status, and header-level monetary
    # fields (GST, subtotal, grand total, etc.) are never compared and
    # never appear in the output. Set it to False to also compare those
    # header fields as before.
    monetary_comparisons = []
    if not line_items_only:
        for field_pair in alignment.get("monetary_fields", []):
            label = field_pair.get("label", "")
            field_a = field_pair.get("doc_a_field")
            field_b = field_pair.get("doc_b_field")
            val_a = _to_decimal(struct_a.get(field_a)) if field_a else None
            val_b = _to_decimal(struct_b.get(field_b)) if field_b else None
            applicable = bool(field_a) and bool(field_b)
            # header_tolerance_extra: set by story mode only. A PO's total
            # incl. GST is rounded once; each split invoice rounds its own
            # GST, so after N earlier invoices the "left on the PO" total
            # can differ from the last invoice by up to ~1 cent per invoice
            # purely from rounding.
            cmp_result = (_classify_comparison(val_a, val_b, tolerance + header_tolerance_extra, mode, po_is_a)
                          if applicable
                          else {"match": None, "diff": None, "status": "NOT_APPLICABLE"})
            if mode == "commitment_check" and applicable:
                ordered, actual = (val_a, val_b) if po_is_a else (val_b, val_a)
                _note_discrepancy_direction(cmp_result, ordered, actual)
            monetary_comparisons.append({
                "label": label,
                "applicable": applicable,
                "doc_a_field": field_a,
                "doc_b_field": field_b,
                "doc_a_value": str(val_a) if val_a is not None else None,
                "doc_b_value": str(val_b) if val_b is not None else None,
                "match": cmp_result["match"],
                "diff": cmp_result["diff"],
                "status": cmp_result["status"],
            })

    comparison_status = _aggregate_status(
        [r["qty_status"] for r in line_item_results]
        + [r["amount_status"] for r in line_item_results]
        + [mc["status"] for mc in monetary_comparisons]
    )
    # overall_match kept for backward compatibility with existing
    # consumers (the controller's task_outputs, any saved test payloads):
    # True unless there's a REAL discrepancy. Under mode "generic"/
    # "fulfillment_check" this is IDENTICAL to the old
    # `all_line_items_match and all_monetary_fields_match` computation
    # (a partial-fulfillment status can never occur there, so the only
    # way to end up not-DISCREPANCY is everything actually matching).
    # Under "commitment_check", a PARTIAL_FULFILLMENT status now also
    # counts as overall_match=True, since it's an expected open-order
    # state, not an error -- that's the one behavior change here.
    overall_match = comparison_status != "DISCREPANCY"

    # Currency for narrative display only (never affects matching):
    # whichever side states one explicitly, falling back to the other
    # side, falling back to DEFAULT_CURRENCY if neither document says.
    currency = _infer_doc_currency(struct_a) or _infer_doc_currency(struct_b) or DEFAULT_CURRENCY

    mismatch_label = _classify_mismatch_label(comparison_status, line_item_results, monetary_comparisons)

    discrepant_lines = [
        phrase for phrase in (
            _describe_line_item_issue(li, mode, role_a, role_b, po_is_a, currency) for li in line_item_results
        )
        if phrase
    ]

    summary = _build_comparison_summary(
        role_a, role_b, mode, comparison_status,
        po_qty_total if qty_totals_present else None,
        actual_qty_total if qty_totals_present else None,
        discrepancy_directions,
        discrepant_lines,
        netted_pools=netted_pools,
    )
    derived_notes = []
    for li in line_item_results:
        for side, role in (("a", role_a), ("b", role_b)):
            if li.get(f"qty_{side}_derived"):
                desc = li.get(f"doc_{side}_description") or "item"
                derived_notes.append(f"{desc}: {_ROLE_LABELS.get(role, 'Document')} qty was blank, "
                                     f"read as {li[f'qty_{side}']} from amount / unit price")
    if derived_notes:
        summary = f"{summary} Note: {'; '.join(derived_notes)}."

    return {
        "overall_match": overall_match,
        "status": comparison_status,
        "label": mismatch_label,
        "currency": currency,
        "mode": mode,
        "doc_a_role": role_a,
        "doc_b_role": role_b,
        "doc_a_id": doc_a_id,
        "doc_b_id": doc_b_id,
        "summary": summary,
        "monetary_fields": monetary_comparisons,
        "line_items": line_item_results,
        "line_item_fields": alignment.get("line_item_fields", {}),
        "unmatched_doc_a_indices": alignment.get("unmatched_doc_a_indices", []),
        "unmatched_doc_b_indices": alignment.get("unmatched_doc_b_indices", []),
        "excluded_doc_a_rows": alignment.get("excluded_doc_a_rows", []),
        "excluded_doc_b_rows": alignment.get("excluded_doc_b_rows", []),
        "tolerance_used": str(tolerance),
    }


# ---------------------------------------------------------------------------
# Document grouping (split shipments) and credit-note netting
# ---------------------------------------------------------------------------
#
# Both features route through the SAME merge primitive
# (_merge_document_group), because they're the same underlying
# operation with different signs: "this shipment was split across two
# GRNs" is "sum these documents together" (all signs +1); "this credit
# note reduces the invoice" is also "sum these documents together" but
# with the credit note's contribution SUBTRACTED (sign -1). Keeping one
# code path means a bug fix or a new summable-field pattern only needs
# to be made once.

# Field-name heuristic for what gets SUMMED across group members vs what
# just keeps its first non-null value. "qty"/"quantity"/"amount" cover
# the observed conventions (quantity, quantityInvoiced, lineAmount,
# amount, amountInclVat, amount_sgd, qty). "unit" is excluded even
# though "unitCost"/"unit_price" would otherwise match "amount"-like
# intent -- a per-unit RATE must stay constant across partial shipments
# of the same order line, never get added together.
_SUMMABLE_FIELD_HINTS = ("qty", "quantity", "amount")
_NON_SUMMABLE_FIELD_HINTS = ("unit",)


def _is_summable_field(key: str) -> bool:
    k = key.lower()
    if any(h in k for h in _NON_SUMMABLE_FIELD_HINTS):
        return False
    return any(h in k for h in _SUMMABLE_FIELD_HINTS)


# Priority order for matching one group member's line item to the
# corresponding line item in another member of the SAME group. Unlike
# _align_documents (which pairs items ACROSS different document types
# with an AI call, since field naming genuinely differs), group members
# are the same document type/schema by definition -- a deterministic key
# lookup is correct and needs no AI call.
_MERGE_KEY_FIELDS = ("lineNo", "line_no", "orderLineNo", "itemNo",
                     "article_number", "model", "item_no", "sku")


def _line_item_merge_key(item: dict, position: int):
    for k in _MERGE_KEY_FIELDS:
        if item.get(k) is not None:
            return (k, str(item[k]).strip().upper())
    return ("__pos__", position)  # last resort: same position in each member's list


def _filter_doc_to_keys(doc: dict, keys: set) -> dict:
    """
    Returns a copy of `doc` (as its flattened structured form) with its
    line items reduced to only those whose _line_item_merge_key is in
    `keys`. Used to keep cross-role netting (a credit note or a prior
    invoice being subtracted from a PO/GRN) from ever introducing a
    brand-new negative-quantity line for a product the target document
    never had -- that would be a phantom entry, not a real reduction of
    something that actually existed on the PO/GRN. A line the invoice
    bills that the PO/GRN never had at all is a genuine discrepancy for
    THAT invoice's own comparison to surface, not something to silently
    fold into the running ledger for every invoice after it.
    """
    struct = _find_structured(doc)
    items = _get_line_items(struct)
    filtered = [it for pos, it in enumerate(items) if _line_item_merge_key(it, pos) in keys]
    item_key = next((k for k in _LINE_ITEM_KEYS if k in struct), "line_items")
    return {**struct, item_key: filtered}


def _consumed_doc_from_comparison(target_doc: dict, comparison: dict) -> dict:
    """
    Turns one Invoice-vs-(PO or GRN) comparison into "what this invoice
    used up", written in the TARGET document's own line keys and field
    names, so it can be subtracted from that target with
    _net_matching_items.

    Why this exists: the old story mode subtracted the raw invoice from
    the PO/GRN by matching line keys (lineNo / itemNo / model ...). That
    only works when both sides use the same key -- true for the ERP
    exports (PPINV/INV-INVSPLIT have lineNo), but NOT for an OCR'd vendor
    invoice, whose lines are keyed by "model" while the PO's are keyed by
    "lineNo". Nothing matched, nothing was subtracted, and every split
    invoice was compared to the FULL PO (double billing went unnoticed).
    The comparison already knows which invoice line paired with which
    PO/GRN line (the AI alignment), so this reuses those pairs instead.
    """
    struct = _find_structured(target_doc)
    items = _get_line_items(struct)
    fields = (comparison.get("line_item_fields") or {}).get("doc_b") or {}
    qty_keys = ("qty", "quantity", "qty_ordered", "qty_delivered", "delivered_qty", "quantity_delivered")
    amt_keys = ("lineAmount", "line_amount", "amount", "total", "line_total", "amount_sgd")

    used = {}  # target line position -> [qty, amount or None]
    for li in comparison.get("line_items", []):
        ib, ia = li.get("doc_b_index"), li.get("doc_a_index")
        if ib is None or ia is None or ib >= len(items):
            continue  # synthesized unmatched row -- nothing on the target to reduce
        q = _to_decimal(li.get("qty_a"))
        if q is None:
            continue
        slot = used.setdefault(ib, [Decimal("0"), Decimal("0")])
        slot[0] += q
        # Use up the amount at the TARGET's own rate (amount left / qty left),
        # not at whatever the invoice charged. An over-priced invoice is
        # already flagged on its own step; consuming its inflated amount
        # would also wrongly flag the NEXT, correctly-priced invoice (it
        # would look like there isn't enough money left on the PO line).
        t = _extract_qty_amount(items[ib], fields.get("qty_field"), fields.get("amount_field"))
        if t["total"] is not None and t["qty"] is not None and t["qty"] > 0:
            slot[1] += t["total"] * q / t["qty"]
        else:
            slot[1] += _to_decimal(li.get("amount_a")) or Decimal("0")

    out = []
    for pos, it in enumerate(items):
        key_field, _ = _line_item_merge_key(it, pos)
        row = {} if key_field == "__pos__" else {key_field: it[key_field]}
        q, a = used.get(pos, (Decimal("0"), Decimal("0")))
        qf = fields.get("qty_field") if fields.get("qty_field") in it else next((k for k in qty_keys if k in it), None)
        af = fields.get("amount_field") if fields.get("amount_field") in it else next((k for k in amt_keys if k in it), None)
        if qf:
            row[qf] = str(q)
        if af and it.get(af) is not None:
            row[af] = str(a)
        out.append(row)
    item_key = next((k for k in _LINE_ITEM_KEYS if k in struct), "line_items")
    return {item_key: out}


def _net_matching_items(target_doc: dict, subtractor_doc: dict) -> dict:
    """
    Subtracts, from target_doc, only the line items in subtractor_doc
    that correspond to something ALREADY on target_doc (see
    _filter_doc_to_keys for why). This is the primitive
    run_po_invoice_story uses to progressively reduce the running
    PO/GRN pools as each invoice in the timeline is processed.
    """
    target_keys = set()
    for pos, it in enumerate(_get_line_items(_find_structured(target_doc))):
        target_keys.add(_line_item_merge_key(it, pos))
    filtered_subtractor = _filter_doc_to_keys(subtractor_doc, target_keys)
    return _merge_document_group([target_doc, filtered_subtractor], signs=[1, -1])


# ---------------------------------------------------------------------------
# Best-effort date parsing -- used only to ORDER invoices for the
# multi-invoice "story"/timeline output (run_po_invoice_story) and to
# display each one's date in that narrative. Never used for anything that
# affects match/no-match logic itself -- see that function's module-level
# comment for why an explicit invoice_under_review_index remains the
# authoritative way to say which invoice a decision hinges on when dates
# are ambiguous or unparseable.
# ---------------------------------------------------------------------------

_DATE_FIELD_NAMES = ("date", "invoice_date", "invoiceDate", "doc_date", "docDate",
                     "document_date", "documentDate", "grn_date", "grnDate",
                     "receipt_date", "receiptDate", "received_date", "receivedDate", "po_date", "poDate",
                     "order_date", "orderDate", "do_date", "doDate",
                     "delivery_date", "deliveryDate", "shipment_date", "shipmentDate")
# Deliberately excluded: postingDate/posting_date (when a document was
# posted to the books -- an ERP/accounting event, not the date ON the
# document) and dueDate/due_date (a payment deadline, not when the
# document was issued). Both are real fields real documents carry, and
# both were tried here at different points -- but the decision is to
# treat a document with no genuine date field of its own as UNDATED
# rather than borrow either of these: matching still runs the same
# either way (see _grn_pool_as_of's fallback), only the story's date
# display and cross-invoice ordering are affected, and guessing at the
# wrong substitute date there is worse than plainly having none.

_ORDINAL_SUFFIX_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)

_DATE_FORMATS = (
    "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d.%m.%Y",
)


def _parse_date_string(raw: str) -> Optional[date]:
    """
    Best-effort parse of a wide variety of human/OCR date strings ("20th
    Jan 2026", "2026-01-20", "20/01/2026", ...) using only the standard
    library -- no dateutil dependency assumed in the deployment
    environment. Returns None (never raises) on anything it can't
    confidently parse; callers must treat that as "unknown", not guess.
    """
    if not raw:
        return None
    s = _ORDINAL_SUFFIX_RE.sub(r"\1", raw.strip())
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(s.split("T")[0])
    except ValueError:
        return None


def _infer_doc_date(struct: dict) -> Optional[date]:
    """
    Best-effort extraction of a document's own date (invoice date, GRN
    date, etc.), for display/ordering purposes ONLY -- see this section's
    module-level comment. Tries known field names in order; returns None
    if none are present or none parse, rather than fabricating a date.
    """
    for key in _DATE_FIELD_NAMES:
        val = struct.get(key)
        if val not in (None, ""):
            parsed = _parse_date_string(str(val))
            if parsed:
                return parsed
    return None


# Field names a document might use to state its own currency explicitly.
_CURRENCY_FIELD_NAMES = ("currency", "currency_code", "currencyCode", "ccy")


def _infer_doc_currency(struct: dict) -> Optional[str]:
    """
    Best-effort extraction of a document's own stated currency, for
    display purposes only -- never affects matching logic (a discrepancy
    is a discrepancy regardless of what currency it's denominated in).
    Returns None (not DEFAULT_CURRENCY) when nothing is found, so callers
    can distinguish "this document said nothing" from "this document
    said S$" -- e.g. to prefer one document's explicit currency over
    another's before falling back to the default themselves.
    """
    for key in _CURRENCY_FIELD_NAMES:
        val = struct.get(key)
        if val not in (None, ""):
            return str(val).strip()
    return None


def _format_amount(value_str: Optional[str], currency_label: str) -> str:
    """
    Renders a stored amount (a plain decimal string, as amount_a/amount_b
    are stored in line_item_results) as a currency-prefixed display
    string for narrative text ONLY -- e.g. "S$4.00" or "USD 4.00". The
    underlying numeric fields elsewhere in the output are left as plain
    decimal strings with no currency prefix, so existing numeric
    consumers are unaffected; this formatting exists solely for
    human-readable text. A purely alphabetic label (a currency CODE like
    "USD", "SGD") gets a separating space; anything else (a symbol like
    "S$", "$", "€") is prefixed directly with none, matching how each is
    conventionally written.
    """
    if value_str is None:
        return "none"
    separator = " " if currency_label.isalpha() else ""
    try:
        d = Decimal(value_str).quantize(Decimal("0.01"))
    except InvalidOperation:
        return f"{currency_label}{separator}{value_str}"
    return f"{currency_label}{separator}{d}"


def _merge_document_group(raw_docs: list, signs: Optional[list] = None) -> dict:
    """
    Merges 2+ raw documents that represent PARTS of the same underlying
    document (e.g. two partial GRNs for one shipment, or an Invoice plus
    a Credit Note netting against it) into ONE synthetic document, so
    the rest of this file (run_reconciliation / run_reconciliation_multi)
    can treat the group as a single comparison target exactly like an
    ungrouped document -- nothing else in this file needs to know
    grouping happened at all.

    signs (one per raw_docs entry, default all +1) lets a caller
    SUBTRACT a document's numeric contribution instead of adding it --
    this is how a credit note is netted against its target document
    (see _resolve_document_groups): pass -1 for the credit note's
    position.

    Only fields judged "summable" (see _is_summable_field) are
    accumulated across members; every other field (description, item
    identifiers, dates, document numbers, unit rates) keeps the first
    non-null value encountered, in raw_docs order -- these are expected
    to be identical (or at least not meaningfully combinable) across
    partial shipments of the same order line.

    len(raw_docs) == 1 returns that single document completely
    unchanged (not even passed through _find_structured) -- this is
    what keeps grouping/credit-notes fully backward compatible: a
    document that isn't part of any group or credit-note relationship
    is untouched all the way through.
    """
    if len(raw_docs) == 1:
        return raw_docs[0]

    if signs is None:
        signs = [1] * len(raw_docs)
    if len(signs) != len(raw_docs):
        raise ValueError("signs must have the same length as raw_docs")

    structs = [_find_structured(d) for d in raw_docs]

    # --- top-level (header) fields ---
    merged_top: dict = {}
    for struct, sign in zip(structs, signs):
        # Only ADDING (sign > 0) members decide which header fields exist.
        # A subtracted member (credit note / prior invoice) may only reduce
        # fields the target already has -- if its own field names leaked in
        # (e.g. an invoice's "invoiceNo" landing on a netted GRN, even with
        # a None value), _infer_document_role would re-classify the result.
        if sign <= 0:
            continue
        for k in struct:
            if k in _LINE_ITEM_KEYS and isinstance(struct[k], list):
                continue  # line items handled separately below
            merged_top.setdefault(k, None)

    for k in list(merged_top.keys()):
        if _is_summable_field(k):
            total = Decimal("0")
            any_present = False
            for struct, sign in zip(structs, signs):
                val = _to_decimal(struct.get(k))
                if val is not None:
                    total += val * sign
                    any_present = True
            merged_top[k] = str(total) if any_present else None
        else:
            # A SUBTRACTED member's (credit note's) own identity/type
            # fields must never leak into the target document's identity
            # -- only its numeric contributions apply. Restricting this
            # to sign > 0 members fixes a real bug: netting a credit
            # note (which happens to be the only member carrying a
            # "document_type" key) against a PO with no such key of its
            # own previously caused the ENTIRE merged/netted PO to be
            # mislabeled with the credit note's own "document_type":
            # "Credit Note" value -- silently breaking role inference
            # for the merged document downstream.
            for struct, sign in zip(structs, signs):
                if sign > 0 and struct.get(k) is not None:
                    merged_top[k] = struct[k]
                    break

    # --- line items ---
    items_lists = [_get_line_items(struct) for struct in structs]
    merged_items_map: dict = {}
    for items, sign in zip(items_lists, signs):
        for pos, item in enumerate(items):
            key = _line_item_merge_key(item, pos)
            merged_items_map.setdefault(key, []).append((item, sign))

    merged_items = []
    for key, sourced_items in merged_items_map.items():
        # Same sign restriction as above: start from a POSITIVE-sign
        # member if one exists (never seed the merged line item's
        # identity from a subtracted/credit-note member), falling back
        # to whatever's available only if every source for this key
        # happens to be a subtracted member (shouldn't normally happen,
        # but stay defined rather than crash).
        base_source = next((item for item, sign in sourced_items if sign > 0), sourced_items[0][0])
        base = dict(base_source)
        # Same rule for line-item fields: only fields an ADDING member
        # actually carries. Otherwise a field only the subtracted member
        # has (e.g. an invoice's "lineAmount" netted against a GRN that has
        # no amounts at all) shows up as a phantom negative value (-300).
        # Falls back to every source's fields only when no adding member
        # exists for this line.
        field_sources = [item for item, sign in sourced_items if sign > 0] or [i for i, _ in sourced_items]
        all_field_names = set()
        for item in field_sources:
            all_field_names.update(item.keys())
        for field_name in all_field_names:
            if _is_summable_field(field_name):
                total = Decimal("0")
                any_present = False
                for item, sign in sourced_items:
                    val = _to_decimal(item.get(field_name))
                    if val is not None:
                        total += val * sign
                        any_present = True
                base[field_name] = str(total) if any_present else None
            else:
                for item, sign in sourced_items:
                    if sign > 0 and item.get(field_name) is not None:
                        base[field_name] = item[field_name]
                        break
        merged_items.append(base)

    # Preserve whichever line-item key name the first group member
    # actually used (e.g. "lines" vs "line_items"), so _get_line_items
    # still finds the merged result the same way it would have found
    # any ungrouped document.
    item_key = next((k for k in _LINE_ITEM_KEYS if k in structs[0]), "line_items")
    merged_top[item_key] = merged_items
    return merged_top


def _document_schema_signature(struct: dict) -> tuple:
    """
    A lightweight fingerprint of a document's shape: its top-level field
    names (excluding the line-items list itself) plus its first line
    item's field names. Two documents that are genuinely different PARTS
    of the same split shipment (e.g. two partial GRNs produced by the
    same upstream system) share this signature exactly -- same header
    fields, same per-line fields, just different quantities. A PO and an
    Invoice, even describing the same order, do NOT share it (different
    field names/conventions between document types). This is what makes
    the signature a safe, deterministic proxy for "these came from the
    same document-producing process, so summing them is meaningful" --
    without ever needing to know what a "GRN" or "shipment" IS.
    """
    top_keys = frozenset(k for k in struct if k not in _LINE_ITEM_KEYS)
    items = _get_line_items(struct)
    item_keys = frozenset(items[0].keys()) if items else frozenset()
    return (top_keys, item_keys)


def _auto_group_by_schema(docs: list, excluded_idx: set) -> list:
    """
    Deterministically clusters documents by _document_schema_signature,
    entirely from document CONTENT -- no caller-supplied "groups" input
    needed, no goal text to interpret, no AI call. This is what lets
    grouping work with ZERO changes anywhere upstream (the Goal Expander
    can keep building a plain "documents" array exactly as it always
    has): a shipment/order split across N files of the same schema is
    grouped automatically the moment they're compared, whether or not
    the goal/caller ever mentions a split at all.

    EVERY document not in excluded_idx (credit notes) is eligible for
    clustering, INCLUDING whichever one happens to be index 0. An
    earlier version of this function excluded index 0 specifically,
    reasoning that grouping "the thing everything else compares
    against" felt like a more surprising default -- but that exclusion
    was itself a real bug: if a PO happens to be split across two files
    and one of those files is uploaded first (making it index 0), the
    old exclusion made it permanently ineligible to merge with its own
    sibling PO file, leaving both unmerged and silently comparing
    PO-vs-PO as if they were unrelated documents. There is no principled
    reason index 0 specifically should be immune to a schema match that
    would merge any other pair -- so it isn't. Which GROUP ends up
    "the reference" is still decided afterward, by _resolve_document_
    groups' existing "sort by minimum original index" step; this
    function only decides which documents belong together, not which
    resulting group is the reference.

    Returns index-groups in the same shape "groups" takes as an
    explicit input (a list of lists of original indices) so it can feed
    directly into the same resolution path an explicit "groups" input
    would. excluded_idx (credit notes, handled via their own netting
    path -- see _resolve_document_groups) are skipped entirely: never
    clustered, never included in the returned groups.
    """
    clusters: dict = {}
    for i, d in enumerate(docs):
        if i in excluded_idx:
            continue
        sig = _document_schema_signature(_find_structured(d))
        clusters.setdefault(sig, []).append(i)

    groups = []
    for sig, idxs in clusters.items():
        groups.append(idxs)  # a cluster of 1 is just a normal singleton group
    return groups


# ---------------------------------------------------------------------------
# Cross-invoice netting: catches double-invoicing across the LIFETIME of a
# PO, not just within one match call.
#
# Problem this solves: a PO can be invoiced in more than one batch over time
# (e.g. INV1 bills 2 of the 3 ordered lines today, INV2 bills the remaining
# line months later). Each new invoice must be checked against what's LEFT
# on the PO/GRN, not against the PO/GRN's original full quantities --
# otherwise a line already fully invoiced by INV1 looks identical to one
# that was simply never invoiced, and re-billing it in INV2 (double
# invoicing) is indistinguishable from a normal partial fulfillment.
#
# This agent is stateless (no memory between /run calls), so the caller
# (platform controller) is responsible for supplying every invoice raised
# against the PO so far -- prior ones as historical record, plus the new
# one -- and for saying explicitly which one is being decided on right now
# via "invoice_under_review_index". Every OTHER document with an INVOICE
# role is then a "prior invoice": its qty/amount fields are SUBTRACTED
# (sign -1, via the same _merge_document_group primitive credit notes
# already use) from every PO and GRN group before the invoice under review
# is compared against them -- so those comparisons see only what's actually
# still available, not the original full amounts.
#
# Deliberately explicit, not auto-detected by date: OCR'd date fields come
# in too many formats to sort reliably without risking a wrong guess on
# exactly the kind of decision (can we pay this invoice?) that must not be
# wrong. With exactly one INVOICE-role document present, there is nothing
# to disambiguate and it's used automatically -- fully backward compatible
# with every existing single-invoice call.
# ---------------------------------------------------------------------------

def _resolve_invoice_under_review(docs: list, credit_note_idx: set,
                                   invoice_under_review_index_input) -> tuple:
    """
    Returns (invoice_under_review_idx, prior_invoice_idx_set).

    invoice_under_review_idx is None when there is no INVOICE-role document
    in `docs` at all (the feature is simply not applicable -- e.g. a plain
    PO-vs-GRN call). prior_invoice_idx_set is always a set (possibly empty).
    """
    n = len(docs)
    invoice_idx = [
        i for i in range(n)
        if i not in credit_note_idx and _infer_document_role(_find_structured(docs[i])) == "INVOICE"
    ]

    if invoice_under_review_index_input is not None:
        idx = invoice_under_review_index_input
        if not isinstance(idx, int) or not (0 <= idx < n):
            raise ValueError(f"invoice_under_review_index is out of range: {idx!r}")
        if idx not in invoice_idx:
            raise ValueError(
                f"invoice_under_review_index {idx} does not refer to a document recognized "
                f"with an INVOICE role"
            )
        return idx, set(i for i in invoice_idx if i != idx)

    if len(invoice_idx) == 0:
        return None, set()
    if len(invoice_idx) == 1:
        return invoice_idx[0], set()
    raise ValueError(
        f"documents contain {len(invoice_idx)} INVOICE-role documents ({invoice_idx}) -- "
        f"pass \"invoice_under_review_index\" to say which one is being decided on now; "
        f"the rest will be treated as prior invoices already recorded against this PO"
    )


def _resolve_document_groups(docs: list, groups_input, credit_note_indices_input,
                              credit_note_targets_input, invoice_under_review_index_input=None) -> tuple:
    """
    Applies the optional "groups" / "credit_note_indices" /
    "credit_note_targets" task inputs to produce the final ordered list
    of documents handed to run_reconciliation_multi.

    BEHAVIOR CHANGE (not fully backward compatible, deliberately): when
    groups_input is NOT given, this no longer treats every document as
    its own singleton group -- it now calls _auto_group_by_schema to
    detect and merge split-shipment-shaped documents automatically, from
    content alone (see that function's docstring). Explicitly passing
    "groups" (even an empty list, []) OVERRIDES auto-detection entirely
    and falls back to the old singleton-per-document behavior for
    anything not named in it -- so a caller that wants the pre-grouping
    behavior back can always force it with "groups": []. A caller that
    passes real groups gets exactly what it asked for, with no
    auto-detection layered on top.

    Ordering: groups are sorted by their MINIMUM original index, so
    "the first document is the reference" (the existing, documented
    convention) generalizes naturally to "the first GROUP is the
    reference" without the caller needing to think about it -- put the
    reference document's index anywhere in `documents` and whichever
    group contains the smallest such index becomes group 0.

    Credit notes: identified either explicitly via credit_note_indices
    (a list of indices into the ORIGINAL `documents` array), or
    auto-detected by inspecting each document's own document_type/
    doc_type/type field for the substring "credit" (case-insensitive)
    -- covers "Credit Note", "Credit Memo", etc. NOTE: auto-detection
    only works for document shapes that actually carry such a field;
    a shape with no type identifier at all (as seen in some real
    documents) requires the explicit credit_note_indices override.
    Each credit note is removed from the normal group pool and instead
    merged (sign -1) into its TARGET document's group -- explicit via
    credit_note_targets (a dict of {credit_note_index: target_index}),
    defaulting to whichever document has an INVOICE role when not
    specified (this is a role-based default, not positional -- it
    doesn't matter which document was uploaded first), falling back to
    the reference group only if no document is recognized as an Invoice.

    Prior invoices: see _resolve_invoice_under_review's module-level
    comment block for the full rationale. In short, when more than one
    INVOICE-role document is present, "invoice_under_review_index" says
    which one is being decided on now; every other invoice is removed
    from the normal group pool (same treatment as a credit note) and
    instead netted (sign -1) into EVERY PO and GRN group -- not just one
    target, unlike credit notes -- so the invoice under review is
    compared against what's actually still left on the PO/GRN, not
    their original full quantities.

    Returns (resolved_docs, reference_index): reference_index is the
    position, in resolved_docs, of the group containing the invoice
    under review (or 0, the existing default, when the feature doesn't
    apply -- zero or one INVOICE-role documents present).
    """
    n = len(docs)

    credit_note_idx = set()
    if credit_note_indices_input:
        for i in credit_note_indices_input:
            if not isinstance(i, int) or not (0 <= i < n):
                raise ValueError(f"credit_note_indices contains an out-of-range index: {i!r}")
            credit_note_idx.add(i)
    else:
        for i, d in enumerate(docs):
            struct = _find_structured(d)
            doc_type = str(
                struct.get("document_type") or struct.get("doc_type") or struct.get("type") or ""
            ).lower()
            if "credit" in doc_type:
                credit_note_idx.add(i)

    targets_input = credit_note_targets_input or {}
    credit_note_targets = {int(k): int(v) for k, v in targets_input.items()}

    invoice_under_review_idx, prior_invoice_idx = _resolve_invoice_under_review(
        docs, credit_note_idx, invoice_under_review_index_input
    )
    netting_excluded_idx = credit_note_idx | prior_invoice_idx

    if groups_input:
        explicit_groups_input = groups_input
    else:
        # No explicit groups given -- auto-detect from document schema
        # instead of defaulting to "every document is its own group".
        # See _auto_group_by_schema's docstring for the rationale.
        # Prior invoices are excluded from the auto-grouping pool for
        # the same reason credit notes are: they're netted separately,
        # never summed together with a sibling document by schema --
        # two invoices sharing an OCR template must NOT be silently
        # merged into one, since each is a separate billing event that
        # needs its own can-we-pay-this decision.
        explicit_groups_input = _auto_group_by_schema(docs, excluded_idx=netting_excluded_idx)
    seen = set()
    validated_groups = []
    for g in explicit_groups_input:
        if not isinstance(g, list) or not g:
            raise ValueError(f"each entry in groups must be a non-empty list of indices, got {g!r}")
        for i in g:
            if not isinstance(i, int) or not (0 <= i < n):
                raise ValueError(f"groups contains an out-of-range index: {i!r}")
            if i in seen:
                raise ValueError(f"index {i} appears in more than one group")
            if i in credit_note_idx:
                raise ValueError(f"index {i} is a credit note (per credit_note_indices/auto-detection) "
                                  f"and cannot also appear in groups")
            if i in prior_invoice_idx:
                raise ValueError(f"index {i} is a prior invoice (every INVOICE-role document other than "
                                  f"invoice_under_review_index) and cannot also appear in groups")
            seen.add(i)
        validated_groups.append(list(g))

    # Every remaining index (not explicitly grouped, not a credit note,
    # not a prior invoice) is its own singleton group.
    for i in range(n):
        if i not in seen and i not in netting_excluded_idx:
            validated_groups.append([i])
            seen.add(i)

    validated_groups.sort(key=min)
    group_of = {idx: g for g in validated_groups for idx in g}

    # Default credit-note netting target: prefer whichever document has
    # an INVOICE role -- that's how credit notes actually work in real
    # AP practice (they reduce what's being billed), regardless of
    # upload/document order. Previously this defaulted to "whichever
    # document happens to be the reference (index 0 / first group)",
    # which meant the SAME three real-world documents could net the
    # credit note against the PO instead of the Invoice purely because
    # of which file got uploaded first -- an order-dependent result for
    # a question that has one right answer regardless of order. Falls
    # back to the old reference-group behavior only when no document is
    # recognized as an Invoice at all (e.g. netting a credit note
    # against a PO/GRN pair with no invoice present in this comparison).
    # Prefer the invoice under review itself (the one actually being
    # decided on) over any other invoice-role document -- when there's
    # more than one invoice in play, a credit note should net against
    # the invoice on the table right now, not a settled historical one.
    # Falls back to the old "any INVOICE-role document" search (now
    # also excluding prior invoices, which aren't in group_of at all)
    # only when invoice_under_review_idx wasn't resolved to anything.
    if invoice_under_review_idx is not None and invoice_under_review_idx in group_of:
        default_target_group = group_of[invoice_under_review_idx]
    else:
        invoice_idx = next(
            (i for i in range(n) if i not in netting_excluded_idx
             and _infer_document_role(_find_structured(docs[i])) == "INVOICE"),
            None,
        )
        default_target_group = (
            group_of[invoice_idx] if invoice_idx is not None
            else (validated_groups[0] if validated_groups else [])
        )

    # The reference is only ever pinned to the invoice under review when
    # this feature is actually in play (an explicit invoice_under_review_
    # index was given, or there was more than one invoice to disambiguate
    # and net). A single-invoice call with nothing to net stays on the
    # existing, unrelated "first GROUP by minimum original index" default
    # (reference_index 0) exactly as before this feature existed -- this
    # matters because which document is the reference decides WHICH PAIRS
    # get compared (e.g. Invoice-vs-GRN's strict fulfillment_check only
    # happens if the Invoice ends up the reference); a caller who has
    # always relied on that existing convention for a plain PO/GRN/Invoice
    # call must see it completely unchanged.
    pin_reference_to_invoice = invoice_under_review_index_input is not None or bool(prior_invoice_idx)

    resolved_docs = []
    reference_index = 0
    for pos, g in enumerate(validated_groups):
        members = [(docs[i], 1) for i in g]

        # Keys this group's own (positive-sign) members actually carry --
        # used to keep any cross-role netting below (credit notes, prior
        # invoices) from introducing a phantom line for a product this
        # group never had at all. See _filter_doc_to_keys's docstring.
        group_keys = set()
        for i in g:
            for it_pos, it in enumerate(_get_line_items(_find_structured(docs[i]))):
                group_keys.add(_line_item_merge_key(it, it_pos))

        for cn_idx in credit_note_idx:
            target_idx = credit_note_targets.get(cn_idx)
            target_group = group_of.get(target_idx) if target_idx is not None else default_target_group
            if target_group is g:
                members.append((_filter_doc_to_keys(docs[cn_idx], group_keys), -1))
        if prior_invoice_idx:
            # Unlike a credit note (one target), a prior invoice nets
            # against EVERY PO and GRN group -- it must reduce both what
            # the PO still commits to and what the GRN still has left to
            # invoice, since the invoice under review is checked against
            # both. Role is read off the group's own (unmerged) first
            # member, since merging hasn't happened yet at this point.
            group_role = _infer_document_role(_find_structured(docs[g[0]]))
            if group_role in ("PO", "GRN"):
                for pi_idx in prior_invoice_idx:
                    members.append((_filter_doc_to_keys(docs[pi_idx], group_keys), -1))
        if pin_reference_to_invoice and invoice_under_review_idx is not None and invoice_under_review_idx in g:
            reference_index = pos
        raw_docs = [m[0] for m in members]
        signs = [m[1] for m in members]
        resolved_docs.append(_merge_document_group(raw_docs, signs=signs))

    return resolved_docs, reference_index


def run_reconciliation_multi(docs: list, tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
                              timeout: float = 120, line_items_only: bool = True,
                              reference_index: int = 0, netted_pools: bool = False,
                              header_tolerance_extra: Decimal = Decimal("0"),
                              parallel: Optional[bool] = None) -> dict:
    """
    Compares 2+ documents. docs[reference_index] (default 0, e.g. the
    Invoice) is treated as the reference document and every other document
    is checked against it -- this is the standard shape of AP 3-way
    matching (Invoice checked against PO, and separately against DO), and
    needs N-1 LLM alignment calls for N documents rather than the C(N,2)
    calls a full all-pairs comparison would need. Documents other than the
    reference are NOT compared directly against each other.

    reference_index only needs to be anything other than 0 when cross-
    invoice netting is in play (see _resolve_invoice_under_review /
    _resolve_document_groups): the invoice under review isn't necessarily
    the first document in the resolved list (the PO or a GRN group can
    sort ahead of it), but it must still be the one everything else is
    compared against, since it's the one being decided on.

    If your matching needs actually require all-pairs comparison (e.g.
    catching a PO/DO discrepancy that doesn't involve the Invoice at
    all), that's a different function -- ask before assuming this covers
    it, since it doubles the LLM calls for 3 documents and grows
    quadratically beyond that.
    """
    if len(docs) < 2:
        raise ValueError("run_reconciliation_multi requires at least 2 documents.")
    if not (0 <= reference_index < len(docs)):
        raise ValueError(f"reference_index {reference_index} is out of range for {len(docs)} documents.")

    # The N-1 reference-vs-other comparisons are independent of each other
    # (run_reconciliation only READS its two documents), so for a large
    # call they run in parallel batches -- see PARALLEL_MATCH_MIN_DOCS. The
    # output list keeps the same order as before. `parallel=None` means
    # "decide from the document count"; run_po_invoice_story and the batch
    # runner pass an explicit True/False.
    if parallel is None:
        parallel = len(docs) > PARALLEL_MATCH_MIN_DOCS

    def _compare_one(i: int) -> dict:
        return {
            "reference_index": reference_index,
            "compared_index": i,
            **run_reconciliation(docs[reference_index], docs[i], tolerance=tolerance, timeout=timeout,
                                  line_items_only=line_items_only, netted_pools=netted_pools,
                                  header_tolerance_extra=header_tolerance_extra),
        }

    comparisons = _map_ordered(
        _compare_one,
        [i for i in range(len(docs)) if i != reference_index],
        MAX_PARALLEL_MATCH_WORKERS if parallel else 1,
    )

    overall_status = _aggregate_status([c["status"] for c in comparisons])
    overall_match = overall_status != "DISCREPANCY"  # same relationship as the per-comparison level
    label = _combine_mismatch_labels([c["label"] for c in comparisons])
    # Currency: prefer whichever comparison found an explicit one over
    # documents that only fell back to DEFAULT_CURRENCY, so one PO/GRN/
    # Invoice set that states its currency isn't overridden by another
    # document in the same call that simply didn't say.
    explicit_currencies = [c["currency"] for c in comparisons if c["currency"] != DEFAULT_CURRENCY]
    currency = explicit_currencies[0] if explicit_currencies else DEFAULT_CURRENCY

    document_roles = {i: _infer_document_role(_find_structured(d)) for i, d in enumerate(docs)}
    document_ids = {i: _infer_doc_id(_find_structured(d)) for i, d in enumerate(docs)}

    # One combined plain-English readout across every comparison, e.g.
    # "Invoice matches GRN/Delivery -- fully consistent. PO shows
    # partial fulfillment: 3870 of 20000 ordered units received/
    # invoiced -- not a discrepancy, an open-order status."
    summary = " ".join(c["summary"] for c in comparisons)

    # A match gets an explicit "MATCH" label instead of null, so a caller
    # can read final_label on its own without also checking
    # final_overall_match. Discrepancy labels (QTY_MISMATCH, ...) are
    # unchanged. "Match" here = final_overall_match is true, which
    # includes PARTIAL_FULFILLMENT (a correct part-invoice of an open PO).
    if overall_match and not label:
        label = "MATCH"

    return {
        "final_overall_match": overall_match,
        "final_status": overall_status,
        "final_label": label,
        "currency": currency,
        "final_summary": summary,
        "reference_index": reference_index,
        "document_count": len(docs),
        "document_roles": document_roles,
        "document_ids": document_ids,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# PO invoice "story": walks every invoice raised against one PO, in
# chronological order, narrating each one's own outcome at the point it
# happened.
#
# This is a different question from "can we pay the one invoice under
# review" (run_reconciliation_multi + the invoice_under_review_index
# netting above): here the caller wants the full history read back --
# "In INV1 (22 Jan 2026), partial fulfillment... In INV2 (2 Jun 2026),
# discrepancy..." -- so every invoice needs its OWN comparison result,
# not just the most recent one.
#
# Dates are used here ONLY to order and label the story -- never to
# decide which invoices to net against which (that would make the
# match/no-match verdict depend on how reliably an OCR'd date string
# happens to parse, which is exactly the risk _resolve_invoice_under_review
# avoids by requiring an explicit index instead). Sorting the narrative
# is a much lower-stakes use of a parsed date than deciding a payment, so
# it's fine for that ordering to fall back to document order when a date
# doesn't parse -- each entry says plainly when that happened.
#
# GRN timing: each invoice's fulfillment check only sees the GRNs dated
# on or before that invoice's own date (see _grn_pool_as_of) -- a GRN
# dated after the invoice wasn't there yet, so an invoice billing goods
# whose GRN hasn't arrived yet is correctly flagged as underdelivery
# against GRN, even while it matches the PO's committed quantities fine.
# A GRN with no parseable date of its own defaults to included, so an
# OCR date gap on the receipt never manufactures a false underdelivery
# flag on its own.
# ---------------------------------------------------------------------------

def run_po_invoice_story(docs: list, tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
                          timeout: float = 120, line_items_only: bool = True,
                          credit_note_indices_input=None, credit_note_targets_input=None,
                          invoice_under_review_index_input=None,
                          parallel: Optional[bool] = None) -> dict:
    n = len(docs)
    # The invoices themselves MUST be walked one after another -- each
    # one is checked against what the earlier ones left on the PO/GRN, and
    # that netting comes from the earlier invoice's own comparison result.
    # What CAN run side by side is the independent work inside one step (the
    # PO comparison and the GRN comparison) and the posted-invoice
    # alignments below, so that is what `parallel` switches on.
    story_parallel = (n > PARALLEL_MATCH_MIN_DOCS) if parallel is None else parallel
    structs = [_find_structured(d) for d in docs]
    roles = [_infer_document_role(s) for s in structs]

    credit_note_idx = set()
    if credit_note_indices_input:
        for i in credit_note_indices_input:
            if not isinstance(i, int) or not (0 <= i < n):
                raise ValueError(f"credit_note_indices contains an out-of-range index: {i!r}")
            credit_note_idx.add(i)
    else:
        for i, s in enumerate(structs):
            doc_type = str(s.get("document_type") or s.get("doc_type") or s.get("type") or "").lower()
            if "credit" in doc_type:
                credit_note_idx.add(i)

    invoice_idx = [i for i in range(n) if i not in credit_note_idx and roles[i] == "INVOICE"]
    po_idx = [i for i in range(n) if roles[i] == "PO"]
    grn_idx = [i for i in range(n) if roles[i] == "GRN"]

    if not po_idx:
        raise ValueError("no PO-role document found among `documents` -- the invoice story needs a "
                          "PO to check each invoice's commitment against")
    if not invoice_idx:
        raise ValueError("no INVOICE-role document found among `documents` -- nothing to build a "
                          "story for")

    # POSTED-INVOICE EXCLUSION: an invoice document whose OWN id starts
    # with "P" (e.g. "PPINV-B10097417") is the AP system's own record
    # that a vendor invoice was already posted/processed -- not a second,
    # independent invoice raised against the PO. If both a posted
    # invoice and the vendor's original invoice (e.g. "SI1202997") are
    # present together, matching BOTH against the PO double-consumes the
    # same purchase (the posted invoice's quantities get subtracted from
    # the PO, then the vendor invoice's do too), producing a false
    # QTY_MISMATCH on a purchase that was never actually short.
    #
    # So: only when BOTH kinds are present, the posted invoice(s) are
    # dropped from the timeline entirely -- never independently compared
    # against the PO/GRN -- and used ONLY to mark which of the vendor
    # invoice's own line items they confirm as already paid (see
    # "already_paid" on each affected line_items entry below, added via
    # the same AI line-item alignment _align_documents already uses to
    # pair a PO's line wording against an Invoice's differently-worded
    # line for the same product). If a posted invoice is the ONLY
    # invoice present, there's no vendor invoice to defer to, so it's
    # matched normally -- this exclusion never fires in that case.
    posted_idx = [i for i in invoice_idx
                  if (_infer_doc_id(structs[i]) or "").strip().upper().startswith("P")]
    vendor_idx = [i for i in invoice_idx if i not in posted_idx]
    paid_lines_by_vendor_idx = {}
    if posted_idx and vendor_idx:
        align_jobs = []   # (vendor_idx, posted_idx) pairs that need an alignment call
        for vi in vendor_idx:
            vi_ids = {re.sub(r"[^A-Z0-9]", "", str(structs[vi].get(k) or "").upper())
                      for k in ("doc_id", "invoice_number", "invoiceNo", "invoice_no", "inv_number")
                      if structs[vi].get(k)}
            for pi in posted_idx:
                # A posted record naming ITS vendor invoice (PPINV's
                # vendor_invoice_no) only confirms that one. Without this,
                # every posted record was aligned against every vendor
                # invoice, so PPINV-for-SI-A marked the HDMI line on SI-C
                # as "already paid" just because both have an HDMI cable.
                pref = next((structs[pi].get(k) for k in ("vendor_invoice_no", "vendorInvoiceNo",
                                                            "vendor_invoice_number") if structs[pi].get(k)), None)
                if pref and re.sub(r"[^A-Z0-9]", "", str(pref).upper()) not in vi_ids:
                    continue
                align_jobs.append((vi, pi))
        # The alignments are independent of each other, so they run in
        # parallel for a large call (same order of results either way).
        aligns = _map_ordered(
            lambda job: _align_documents(structs[job[1]], structs[job[0]], timeout=timeout),
            align_jobs,
            MAX_PARALLEL_MATCH_WORKERS if story_parallel else 1,
        )
        for vi in vendor_idx:
            paid_lines_by_vendor_idx[vi] = set()
        for (vi, _pi), align in zip(align_jobs, aligns):
            paid_lines_by_vendor_idx[vi].update(p["doc_b_index"] for p in align["line_item_pairs"])
        invoice_idx = vendor_idx

    po_doc = _merge_document_group([docs[i] for i in po_idx]) if len(po_idx) > 1 else docs[po_idx[0]]

    # Fold each credit note into its target invoice's own line items
    # (sign -1) before the timeline walk, so that invoice's billed
    # quantities already reflect the credit memo. Default target: the
    # explicit map, else the first invoice by original document order --
    # pass credit_note_targets explicitly if that's not the right one
    # when more than one invoice is present.
    targets_input = credit_note_targets_input or {}
    credit_note_targets = {int(k): int(v) for k, v in targets_input.items()}
    invoice_docs = {i: docs[i] for i in invoice_idx}
    def _norm_ref(v):
        return re.sub(r"[^A-Z0-9]", "", str(v or "").upper())
    _CN_REF_FIELDS = ("applies_to_invoice", "applied_to_invoice", "original_invoice", "original_invoice_no",
                      "invoice_ref", "reference_invoice", "against_invoice", "appliesToDocNo")
    def _invoice_ids(s):
        return {_norm_ref(s.get(k)) for k in ("doc_id", "invoice_number", "invoiceNo", "invoice_no", "inv_number")
                if s.get(k)} | {_norm_ref(_infer_doc_id(s))}
    for cn_idx in credit_note_idx:
        # Explicit input wins; then the credit note's own "applies to invoice"
        # reference; only then the old fallback (first invoice in upload
        # order), which silently netted CN-0007 against the wrong invoice.
        target_idx = credit_note_targets.get(cn_idx)
        if target_idx is None:
            ref = next((_norm_ref(structs[cn_idx].get(f)) for f in _CN_REF_FIELDS if structs[cn_idx].get(f)), None)
            if ref:
                target_idx = next((iv for iv in invoice_idx if ref in _invoice_ids(structs[iv])), None)
        if target_idx is None:
            target_idx = invoice_idx[0]
        if target_idx in invoice_docs:
            invoice_docs[target_idx] = _merge_document_group(
                [invoice_docs[target_idx], docs[cn_idx]], signs=[1, -1]
            )
        else:
            raise ValueError(f"credit_note_targets maps credit note {cn_idx} to {target_idx}, which "
                              f"is not one of the INVOICE-role documents in `documents`")

    # Order invoices chronologically -- but only when EVERY invoice has a
    # usable date. Sorting on a PARTIAL set of dates is worse than not
    # sorting at all: pushing every undated invoice to the end (or any
    # other fixed slot) silently assumes it happened last, which is
    # exactly the kind of wrong guess that corrupts the netting order --
    # an undated invoice that actually happened FIRST would otherwise get
    # compared against an already-fully-netted PO/GRN instead of the
    # original quantities, and everything after it in the fallback
    # ordering would inherit the mistake. So: if dates are missing or
    # unparseable for ANY invoice, all invoices keep their original
    # position in `documents` instead (the order the caller supplied
    # them in) -- each entry's OWN date is still parsed and shown in its
    # narrative/date_confidence for display, just not used to reorder
    # anything. Only once every invoice's date parses does date order
    # actually decide the sequence.
    dated = []
    for original_pos, i in enumerate(invoice_idx):
        d = _infer_doc_date(_find_structured(invoice_docs[i]))
        dated.append((d, original_pos, i))
    all_dates_known = all(d is not None for d, _, _ in dated)
    if all_dates_known:
        dated.sort(key=lambda t: (t[0], t[1]))
    else:
        dated.sort(key=lambda t: t[1])  # preserve original document order

    # Each GRN's own date, parsed once up front (used per-invoice below).
    grn_dates = {i: _infer_doc_date(_find_structured(docs[i])) for i in grn_idx}

    def _grn_pool_as_of(invoice_date: Optional[date]):
        """
        Merges only the GRNs actually available BY invoice_date -- a
        receipt dated after the invoice wasn't there yet when the
        invoice was raised, and including it would hide a genuine
        "billed goods not yet received" problem (see this function's
        module-level comment on case 4/5-style underdelivery). A GRN
        with no parseable date of its own defaults to INCLUDED rather
        than excluded: an OCR date gap on the receipt itself is a data
        quality issue, not evidence the goods weren't there -- excluding
        it would manufacture a false underdelivery flag from a missing
        field, not a real timing problem. If invoice_date itself isn't
        known, there's no anchor to filter against at all, so every GRN
        is included (same fallback as the rest of this file when a date
        needed for a decision isn't available).
        """
        if invoice_date is None:
            included = grn_idx
        else:
            included = [i for i in grn_idx if grn_dates[i] is None or grn_dates[i] <= invoice_date]
        if not included:
            return None
        return _merge_document_group([docs[i] for i in included]) if len(included) > 1 else docs[included[0]]

    timeline = []
    # Running total of every invoice already processed in this timeline
    # (sign +1 each) -- subtracted from BOTH the PO and the as-of-date
    # GRN pool before comparing the NEXT invoice, so each invoice is
    # checked against what's actually still available, not the original
    # full quantities. None until the first invoice has been processed.
    # What earlier invoices already used up, kept separately in the PO's
    # own line keys and in the GRN pool's own line keys (see
    # _consumed_doc_from_comparison for why the raw invoice can't be used).
    consumed_po = None
    consumed_grn = None
    consumed_po_header = {}   # PO header field name -> total already billed by earlier invoices
    _invoices_counted = 0     # invoices already taken off the PO (duplicates don't count)
    _po_struct_orig = _find_structured(po_doc)
    _po_line_total = None     # PO's original line total at its own prices (set on the first PO comparison)
    seen_invoice_ids = {}   # normalized invoice number -> document index first processed with it
    for date_val, _original_pos, i in dated:
        inv_doc = invoice_docs[i]

        # DUPLICATE INVOICE NUMBER. The same vendor invoice uploaded twice
        # (re-scanned, sent by email AND post, ...) used to be matched
        # again as if it were new: the copy got flagged, but it ALSO used
        # up PO/GRN quantity, so the next, perfectly good invoice was
        # flagged too. Now the copy is flagged once and uses up nothing.
        _dup_id = _infer_doc_id(_find_structured(inv_doc)) or f"document[{i}]"
        _norm = re.sub(r"[^A-Z0-9]", "", _dup_id.upper())
        if _norm and _norm in seen_invoice_ids:
            _date_str = date_val.isoformat() if date_val else "date unknown"
            timeline.append({
                "document_index": i, "invoice_id": _dup_id, "date": _date_str,
                "date_confidence": "parsed" if date_val else "unknown_used_document_order",
                "status": "DISCREPANCY", "label": "DUPLICATE_INVOICE",
                "currency": _infer_doc_currency(_find_structured(inv_doc)) or DEFAULT_CURRENCY,
                "overall_match": False,
                "duplicate_of_document_index": seen_invoice_ids[_norm],
                "narrative": (f"In {_dup_id}, this invoice number was already processed earlier in this "
                              f"batch -- treated as a duplicate and not counted again. Check whether it is a "
                              f"re-scan of the same invoice or a new invoice reusing the number."),
                "comparisons": [],
            })
            continue
        seen_invoice_ids[_norm] = i

        available_po = po_doc if consumed_po is None else _net_matching_items(po_doc, consumed_po)
        # Header totals too (only used when line_items_only is False):
        # earlier invoices' subtotal / total are taken off the PO's header
        # totals, the same way their lines are taken off the PO's lines.
        # Without this, every later invoice was compared with the PO's
        # FULL total, so the last invoice of a completed order showed
        # PARTIAL_FULFILLMENT ("covers 9 of the 9 units still open").
        if consumed_po_header:
            available_po = copy.deepcopy(_find_structured(available_po))
            for _f, _used in consumed_po_header.items():
                _orig = _to_decimal(available_po.get(_f))
                if _orig is not None:
                    available_po[_f] = str(_orig - _used)
        grn_pool = _grn_pool_as_of(date_val)
        available_grn = (grn_pool if (grn_pool is None or consumed_grn is None)
                          else _net_matching_items(grn_pool, consumed_grn))

        compare_docs = [available_po] + ([available_grn] if available_grn is not None else []) + [inv_doc]
        ref_idx = len(compare_docs) - 1
        step = run_reconciliation_multi(compare_docs, tolerance=tolerance, timeout=timeout,
                                         line_items_only=line_items_only, reference_index=ref_idx,
                                         netted_pools=True,
                                         header_tolerance_extra=Decimal("0.01") * _invoices_counted,
                                         parallel=story_parallel)

        # already_paid: only set when this invoice has a posted-invoice
        # reference (see POSTED-INVOICE EXCLUSION above) -- absent from
        # line_items entirely otherwise, so callers that don't know about
        # this feature see no shape change. The invoice is always doc_a
        # in these comparisons (see doc_a_role in the output), so its own
        # line_items positions are doc_a_index.
        if i in paid_lines_by_vendor_idx:
            paid_lines = paid_lines_by_vendor_idx[i]
            for comp in step["comparisons"]:
                if comp.get("doc_a_role") == "INVOICE":
                    for li in comp.get("line_items", []):
                        # rows with no invoice line (a PO/GRN line this invoice
                        # doesn't bill) have nothing to be "paid" -- leave them out
                        if li.get("doc_a_index") is not None:
                            li["already_paid"] = li["doc_a_index"] in paid_lines

        doc_id = _infer_doc_id(_find_structured(inv_doc)) or f"document[{i}]"
        date_str = date_val.isoformat() if date_val else "date unknown"
        # The story text only mentions the date when one was actually
        # found -- an invoice with no parseable date still gets matched
        # normally (qty/amount comparison never depends on having a
        # date; see _grn_pool_as_of's fallback), so there's nothing
        # wrong to call out by saying "(date unknown)" in the narrative.
        # The structured "date"/"date_confidence" fields below still
        # record it, for any caller that wants to know programmatically.
        narrative = (f"In {doc_id} ({date_str}), {step['final_summary']}".strip() if date_val
                     else f"In {doc_id}, {step['final_summary']}".strip())

        timeline.append({
            "document_index": i,
            "invoice_id": doc_id,
            "date": date_str,
            "date_confidence": "parsed" if date_val else "unknown_used_document_order",
            "status": step["final_status"],
            "label": step["final_label"],
            "currency": step["currency"],
            "overall_match": step["final_overall_match"],
            "narrative": narrative,
            "comparisons": step["comparisons"],
        })

        for comp in step["comparisons"]:
            if comp.get("compared_index") == 0:
                used = _consumed_doc_from_comparison(available_po, comp)
                consumed_po = used if consumed_po is None else _merge_document_group([consumed_po, used])
                if _po_line_total is None:
                    _pf = ((comp.get("line_item_fields") or {}).get("doc_b") or {}).get("amount_field")
                    if _pf:
                        _po_line_total = sum((_to_decimal(it.get(_pf)) or Decimal("0"))
                                             for it in _get_line_items(_po_struct_orig)) or None
                # Take the header totals off at the PO's OWN prices, worked
                # out from the PO lines this invoice used up -- not from the
                # invoice's own header. Two real reasons:
                #  - an over-priced invoice is flagged on its own step; taking
                #    its inflated total off the PO would wrongly flag the NEXT,
                #    correctly priced invoice;
                #  - a credit note reduces the invoice's LINES but not its
                #    printed subtotal/total, so the invoice header overstates
                #    what was really billed.
                # consumed[field] = (PO-price value of the lines used)
                #                   x (PO field / PO line total), e.g. the PO's
                #                   incl.-GST total uses its own GST ratio.
                _amt_f = ((comp.get("line_item_fields") or {}).get("doc_b") or {}).get("amount_field")
                _po_sum = (sum((_to_decimal(it.get(_amt_f)) or Decimal("0"))
                               for it in _get_line_items(_find_structured(used))) if _amt_f else None)
                for _mf in comp.get("monetary_fields") or []:
                    _f = _mf.get("doc_b_field")
                    _val = _to_decimal(_mf.get("doc_a_value"))
                    if not (_mf.get("applicable") and _f and _val is not None):
                        continue
                    _po_f = _to_decimal(_po_struct_orig.get(_f))
                    if _po_sum is not None and _po_f is not None and _po_line_total:
                        _used_f = _po_sum * _po_f / _po_line_total
                    else:
                        _used_f = _val   # no line amounts to work from -- fall back to the invoice header
                    consumed_po_header[_f] = consumed_po_header.get(_f, Decimal("0")) + _used_f.quantize(Decimal("0.01"))
            elif comp.get("compared_index") == 1 and available_grn is not None:
                used = _consumed_doc_from_comparison(available_grn, comp)
                consumed_grn = used if consumed_grn is None else _merge_document_group([consumed_grn, used])
        _invoices_counted += 1

    if invoice_under_review_index_input is not None:
        current = next((t for t in timeline if t["document_index"] == invoice_under_review_index_input), None)
        if current is None:
            raise ValueError(f"invoice_under_review_index {invoice_under_review_index_input} does not "
                              f"refer to one of the INVOICE-role documents in `documents`")
    else:
        current = timeline[-1]  # the most recent invoice by date -- the one this call is really about

    final_summary = "\n".join(t["narrative"] for t in timeline)

    result = {
        "final_overall_match": current["overall_match"],
        "final_status": current["status"],
        "final_label": current["label"],
        "currency": current["currency"],
        "current_invoice_index": current["document_index"],
        "final_summary": final_summary,
        "timeline": timeline,
        "document_roles": {i: roles[i] for i in range(n)},
        "document_ids": {i: _infer_doc_id(structs[i]) for i in range(n)},
    }
    # Only present when the POSTED-INVOICE EXCLUSION above actually
    # fired -- lists which documents were excluded from the timeline and
    # used only as an already_paid reference, by their own doc_id.
    if paid_lines_by_vendor_idx:
        result["posted_invoices_excluded"] = [_infer_doc_id(structs[i]) or f"document[{i}]" for i in posted_idx]
    return result


# ---------------------------------------------------------------------------
# Batch matching across independent transactions.
#
# One upload can hold several unrelated purchases (PO-1 with its GRNs and
# invoices, PO-2 with its own, ...). Matched as one big set they would be
# merged into a single PO/GRN pool, which is wrong; matched one set after
# another they are slow. When -- and ONLY when -- the documents split into
# separate transactions unambiguously, each transaction becomes its own
# batch and the batches run in parallel.
#
# "Unambiguously" means every document carries a PO reference (a PO's own
# number counts) and documents are linked purely by sharing a PO number.
# If any document has no PO reference, a credit note is present, or the
# links form a single group, nothing is split and the old single-set path
# runs instead -- a wrong split would corrupt the result, a missed split
# only costs speed.
# ---------------------------------------------------------------------------
_PO_REF_KEY_NAMES = {"poref", "ponumber", "pono", "customerpo", "purchaseorderno",
                      "purchaseordernumber", "orderreference"}


def _po_keys_of(struct: dict) -> dict:
    """{normalized PO number: as printed} for every PO-number-like field."""
    keys = {}
    for k, v in struct.items():
        if not isinstance(k, str) or _normalize_key(k) not in _PO_REF_KEY_NAMES:
            continue
        if isinstance(v, (dict, list)) or v in (None, ""):
            continue
        norm = re.sub(r"[^A-Z0-9]", "", str(v).upper())
        if norm:
            keys[norm] = str(v).strip()
    return keys


def _split_into_transaction_sets(docs: list) -> Optional[list]:
    """
    Returns [{"document_indices": [...], "batch_key": str}, ...] (two or more
    sets, each of 2+ documents, ordered by their first document) or None
    when the documents should NOT be split. See the block comment above.
    """
    n = len(docs)
    structs = [_find_structured(d) for d in docs]
    roles = [_infer_document_role(st) for st in structs]
    if any(r == "CREDIT_NOTE" for r in roles):
        return None
    per_doc = [_po_keys_of(st) for st in structs]
    # A PO's OWN number is its PO reference too -- including when the
    # extraction stored it only as doc_id rather than poNumber/po_number.
    for i, st in enumerate(structs):
        if roles[i] == "PO":
            own = _infer_doc_id(st)
            norm = re.sub(r"[^A-Z0-9]", "", str(own).upper()) if own else ""
            if norm:
                per_doc[i].setdefault(norm, str(own).strip())
    if any(not keys for keys in per_doc):
        return None  # a document with no PO reference cannot be placed

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for i, keys in enumerate(per_doc):
        for norm in keys:
            if norm in owner:
                parent[find(i)] = find(owner[norm])
            else:
                owner[norm] = i

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    groups = sorted(comps.values(), key=lambda c: c[0])
    if len(groups) < 2 or any(len(c) < 2 for c in groups):
        return None

    sets = []
    for idxs in groups:
        label = None
        for i in idxs:
            if roles[i] == "PO":
                label = _infer_doc_id(structs[i])
                break
        if not label:
            norm_to_printed = {}
            for i in idxs:
                norm_to_printed.update(per_doc[i])
            label = norm_to_printed[min(norm_to_printed)]
        sets.append({"document_indices": idxs, "batch_key": str(label)})
    return sets


class _MatchFailure(Exception):
    """A match that could not be completed; str() is the user-facing message."""


def _match_documents(docs: list, inputs: dict, tolerance: Decimal, timeout: float,
                     line_items_only: bool, parallel: Optional[bool] = None) -> dict:
    """
    ONE match over `docs` (a PO story when a PO sits alongside an invoice,
    otherwise grouping + reference-vs-others). This is the logic run() used
    inline before, unchanged; the failure messages are identical too.
    `parallel` None = decide from the document count, False = force
    sequential (used inside parallel batches so pools never nest).
    """
    # Story mode kicks in whenever a PO is present alongside at least one
    # INVOICE-role document (see run()'s original comment, kept below).
    invoice_count = sum(
        1 for d in docs
        if _infer_document_role(_find_structured(d)) == "INVOICE"
    )
    po_present = any(_infer_document_role(_find_structured(d)) == "PO" for d in docs)

    if invoice_count >= 1 and po_present:
        try:
            return run_po_invoice_story(
                docs, tolerance=tolerance, timeout=timeout, line_items_only=line_items_only,
                credit_note_indices_input=inputs.get("credit_note_indices"),
                credit_note_targets_input=inputs.get("credit_note_targets"),
                invoice_under_review_index_input=inputs.get("invoice_under_review_index"),
                parallel=parallel,
            )
        except Exception as exc:
            raise _MatchFailure(f"Invoice story failed: {exc}") from exc

    try:
        resolved_docs, reference_index = _resolve_document_groups(
            docs,
            inputs.get("groups"),
            inputs.get("credit_note_indices"),
            inputs.get("credit_note_targets"),
            inputs.get("invoice_under_review_index"),
        )
    except ValueError as exc:
        raise _MatchFailure(f"Invalid groups/credit_note_*/invoice_under_review_index inputs: {exc}") from exc

    if len(resolved_docs) < 2:
        raise _MatchFailure("After resolving groups/credit notes, fewer than 2 documents remain to compare.")

    try:
        return run_reconciliation_multi(resolved_docs, tolerance=tolerance, timeout=timeout,
                                         line_items_only=line_items_only,
                                         reference_index=reference_index, parallel=parallel)
    except Exception as exc:
        raise _MatchFailure(f"Reconciliation failed: {exc}") from exc


def _run_match_batches(docs: list, sets: list, tolerance: Decimal, timeout: float,
                       line_items_only: bool) -> dict:
    """
    Runs every transaction set from _split_into_transaction_sets as its own
    independent match, in parallel, and combines the results. Each set's
    full result (exactly what a call with only that set's documents would
    have returned) is kept under "batches"; the top-level fields keep the
    usual names so a caller that only reads final_* still works.

    One set failing does not lose the others: that batch carries an "error"
    and counts as a DISCREPANCY (OTHER_MISMATCH) so it can't be mistaken for
    a clean match. Inside a batch the comparisons run sequentially -- the
    parallelism is ACROSS batches, so pools never nest.
    """
    def _one(batch: dict) -> dict:
        idxs = batch["document_indices"]
        entry = {"batch_key": batch["batch_key"], "document_indices": idxs,
                 "index_scope": "batch_local"}
        try:
            entry.update(_match_documents([docs[i] for i in idxs], {}, tolerance, timeout,
                                           line_items_only, parallel=False))
        except Exception as exc:
            entry["error"] = str(exc)
        return entry

    batches = _map_ordered(_one, sets, MAX_PARALLEL_MATCH_WORKERS)
    for b, entry in enumerate(batches):
        entry["batch_index"] = b

    statuses = ["DISCREPANCY" if e.get("error") else e["final_status"] for e in batches]
    final_status = _aggregate_status(statuses)
    mismatch_labels = [
        "OTHER_MISMATCH" if e.get("error") else e.get("final_label")
        for e in batches
    ]
    combined = _combine_mismatch_labels([l for l in mismatch_labels if l and l != "MATCH"])
    final_overall_match = final_status != "DISCREPANCY"
    final_label = combined or ("MATCH" if final_overall_match else "OTHER_MISMATCH")
    explicit_currencies = [e["currency"] for e in batches
                           if e.get("currency") and e["currency"] != DEFAULT_CURRENCY]
    structs = [_find_structured(d) for d in docs]
    summary_lines = [
        f"[{e['batch_key']}] could not be matched: {e['error']}" if e.get("error")
        else f"[{e['batch_key']}] {e['final_summary']}"
        for e in batches
    ]
    return {
        "final_overall_match": final_overall_match,
        "final_status": final_status,
        "final_label": final_label,
        "currency": explicit_currencies[0] if explicit_currencies else DEFAULT_CURRENCY,
        "final_summary": "\n".join(summary_lines),
        "document_count": len(docs),
        "document_roles": {i: _infer_document_role(st) for i, st in enumerate(structs)},
        "document_ids": {i: _infer_doc_id(st) for i, st in enumerate(structs)},
        "batch_count": len(batches),
        "batches": batches,
    }


# ---------------------------------------------------------------------------
# ADK Agent
# ---------------------------------------------------------------------------

class JsonMatchAgent(Agent):
    name         = "json-match-agent"
    version      = "1.7.0"
    description  = ("Matches qty/amount data across 2+ OCR-extracted JSON documents with differing "
                     "field names (e.g. Invoice/PO/DO 3-way matching). Uses AI for field alignment, "
                     "deterministic checks for numeric comparison. By default only line items are "
                     "compared -- header-level monetary fields (GST, subtotal, grand total, etc.) are "
                     "ignored; pass \"line_items_only\": false to also compare those. The first "
                     "document in the list is treated as the reference; all others are compared "
                     "against it. Documents sharing an identical schema (e.g. a shipment split across "
                     "two GRNs) are auto-grouped and summed before comparison; pass \"groups\" to "
                     "override. PO-vs-Invoice/GRN "
                     "comparisons are treated as a lenient \"commitment check\" (a lower quantity/"
                     "amount than the PO is a normal open-order status, not a discrepancy; a higher "
                     "one still is), while Invoice-vs-GRN comparisons stay a strict \"fulfillment "
                     "check\". Also supports credit-note netting (\"credit_note_indices\"/"
                     "\"credit_note_targets\") to subtract a credit note's amounts from its target "
                     "document before comparison. When `documents` contains a PO-role document "
                     "alongside at least one INVOICE-role document, the response shape changes "
                     "automatically: instead of one \"comparisons\" list, you get a chronological "
                     "\"final_summary\"/\"timeline\" -- each invoice is checked against what was still left on "
                     "the PO and against only the GRNs already dated on or before that invoice's own "
                     "date (a GRN dated after it wasn't there yet, so billing goods before their GRN "
                     "arrives is correctly flagged even for a single invoice); earlier invoices' "
                     "quantities are netted out first, so a line already fully invoiced is correctly "
                     "flagged DISCREPANCY -- double invoicing -- if billed again, rather than read as a "
                     "normal partial fulfillment. \"invoice_under_review_index\" optionally names which "
                     "invoice's result should be surfaced at the top level (\"final_overall_match\"/"
                     "\"final_status\"); it defaults to the most recent invoice by date. With no PO present at "
                     "all (e.g. a plain Invoice-vs-GRN or PO-vs-GRN call), the response is unchanged "
                     "from before this existed. Large uploads (more than PARALLEL_MATCH_MIN_DOCS documents, "
                     "default 5) are matched in parallel: when every document carries a PO reference "
                     "and they form two or more separate PO transactions, each transaction is matched "
                     "as its own batch (see \"batches\"), batches running in parallel; otherwise the "
                     "comparisons inside the one match run in parallel.")
    capabilities = ["data-matching"]  # see note below on capability naming

    def run(self, payload: TaskPayload) -> TaskResult:
        docs = payload.inputs.get("documents")
        if not isinstance(docs, list) or len(docs) < 2:
            return TaskResult(
                task_id=payload.task_id,
                status=TaskStatus.FAILURE,
                error_msg="inputs.documents must be a list of at least 2 JSON documents.",
            )

        # Snap float-noisy quantities (999.9999999999999 -> 1000) to whole
        # numbers before ANY matching/netting -- see _normalize_quantities.
        # Works on a deep copy; payload.inputs is never mutated.
        docs = _normalize_quantities(docs)

        tolerance_raw = payload.inputs.get("amount_tolerance", "0.01")
        try:
            tolerance = Decimal(str(tolerance_raw))
        except InvalidOperation:
            tolerance = DEFAULT_AMOUNT_TOLERANCE

        # Default True: compare line items only, skipping header-level
        # monetary fields (GST, subtotal, grand total, etc.) entirely.
        # Pass "line_items_only": false to also compare those fields.
        # FORCED: always line items only. The controller's Goal Planner has
        # been seen adding "line_items_only": false to the task on its own,
        # so any value sent in the inputs is ignored here on purpose.
        # To allow header-total checks again, restore:
        #   line_items_only = payload.inputs.get("line_items_only", True)
        line_items_only = True

        # Use the caller's requested timeout for each vLLM alignment call,
        # instead of a hardcoded value. Note: for N documents this makes
        # N-1 sequential calls, each given this same budget -- worst-case
        # wall-clock is (N-1) x timeout_secs, not a single shared budget.
        # That's fine for today's 2-doc case; revisit if 3-way matching
        # in practice needs an overall budget instead of a per-call one.
        timeout = payload.timeout_secs if payload.timeout_secs and payload.timeout_secs > 0 else 120

        # Everything about HOW one match is done (story mode when a PO sits
        # alongside an invoice, otherwise grouping + reference-vs-others) now
        # lives in _match_documents, unchanged -- see the comments there.
        #
        # LARGE uploads (more than PARALLEL_MATCH_MIN_DOCS documents) are
        # processed in parallel batches instead of one long serial chain:
        #   - if the documents clearly split into independent transactions
        #     (see _split_into_transaction_sets), each transaction is matched
        #     as its own batch, batches in parallel;
        #   - otherwise the independent comparisons inside the single match
        #     run in parallel (see run_reconciliation_multi).
        # An upload at or below the threshold, or one where the caller passed
        # groups / credit-note / invoice_under_review inputs (those index into
        # the WHOLE list, so it must stay one set), takes the old path.
        explicit_structure = any(
            payload.inputs.get(k) is not None
            for k in ("groups", "credit_note_indices", "credit_note_targets",
                      "invoice_under_review_index")
        )
        sets = None
        if len(docs) > PARALLEL_MATCH_MIN_DOCS and not explicit_structure:
            try:
                sets = _split_into_transaction_sets(docs)
            except Exception:
                sets = None  # the split is an optimisation; never let it break a match

        if sets:
            try:
                result = _run_match_batches(docs, sets, tolerance, timeout, line_items_only)
            except Exception as exc:
                return TaskResult(
                    task_id=payload.task_id,
                    status=TaskStatus.FAILURE,
                    error_msg=f"Batch reconciliation failed: {exc}",
                )
        else:
            try:
                result = _match_documents(docs, payload.inputs, tolerance, timeout, line_items_only)
            except _MatchFailure as exc:
                return TaskResult(
                    task_id=payload.task_id,
                    status=TaskStatus.FAILURE,
                    error_msg=str(exc),
                )

        # Testing aid: persist this run's outputs to disk (see
        # _dump_match_result / MATCH_RESULT_DUMP_* env vars above).
        _dump_match_result(payload.task_id, result)

        return TaskResult(
            task_id=payload.task_id,
            status=TaskStatus.SUCCESS,
            outputs=result,
        )


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 5001))
    agent = JsonMatchAgent()
    agent.generate_manifest(
        endpoint=os.environ.get("AGENT_ENDPOINT", f"http://localhost:{port}"),
        input_schema={
            "documents": {"type": "array", "required": True, "minItems": 2},
            "amount_tolerance": {"type": "string", "required": False},
            "line_items_only": {
                "type": "boolean", "required": False,
                "description": "IGNORED in this build -- always true (forced in run()). "
                                "When true, only line items decide the match/status -- "
                                "header-level monetary fields (GST, subtotal, grand total, shipping, "
                                "etc.) are never compared and never appear in the output. Set to false "
                                "to also compare those header fields, as in versions before this flag.",
            },
            "groups": {
                "type": "array", "required": False,
                "description": "List of index-lists into `documents`. Each inner list's documents "
                                "are merged (qty/amount fields summed) into ONE comparison target "
                                "before matching -- e.g. [[2, 3]] merges documents[2] and "
                                "documents[3] (two partial GRNs for one shipment) into a single "
                                "combined GRN. OPTIONAL: when omitted, documents sharing an "
                                "identical schema (same field names) are auto-grouped this way "
                                "automatically -- pass \"groups\": [] explicitly to force the old "
                                "no-auto-grouping behavior instead. Groups are ordered by their "
                                "smallest member index, so the existing 'first document is the "
                                "reference' convention extends to 'first group is the reference'.",
            },
            "credit_note_indices": {
                "type": "array", "required": False,
                "description": "List of indices into `documents` identifying which are credit "
                                "notes. Each credit note's qty/amount fields are SUBTRACTED from "
                                "its target document (see credit_note_targets) rather than being "
                                "compared as an independent document. If omitted, credit notes are "
                                "auto-detected via a document_type/doc_type/type field containing "
                                "'credit' -- auto-detection only works for document shapes that "
                                "carry such a field.",
            },
            "credit_note_targets": {
                "type": "object", "required": False,
                "description": "Optional {credit_note_index: target_index} map (both indices into "
                                "`documents`) controlling which document each credit note nets "
                                "against. A credit note not listed here defaults to netting against "
                                "whichever document has an INVOICE role (regardless of upload order) "
                                "-- falling back to the reference document only if no Invoice is "
                                "present among the documents being compared.",
            },
            "invoice_under_review_index": {
                "type": "integer", "required": False,
                "description": "Index into `documents` of the invoice being decided on right now. "
                                "Required when `documents` contains more than one INVOICE-role "
                                "document (e.g. a PO invoiced in more than one batch over time) -- "
                                "omitted with exactly one INVOICE-role document, that one is used "
                                "automatically. Every OTHER INVOICE-role document is treated as a "
                                "prior invoice already recorded against this PO: its qty/amount "
                                "fields are subtracted from every PO and GRN document/group before "
                                "the invoice under review is compared against them, so a line "
                                "already fully invoiced earlier is flagged DISCREPANCY (double "
                                "invoicing) if billed again, instead of looking like a normal open "
                                "order.",
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "final_overall_match": {"type": "boolean",
                                   "description": "The headline verdict for this whole matching "
                                                   "session. True unless at least one comparison has "
                                                   "status DISCREPANCY. A PARTIAL_FULFILLMENT status "
                                                   "(a PO not yet fully delivered/invoiced) does NOT "
                                                   "make this false -- that's an expected open-order "
                                                   "state, not an error."},
                "final_status": {"type": "string",
                           "description": "The headline status for this whole matching session: "
                                           "CONSISTENT, PARTIAL_FULFILLMENT, DISCREPANCY, or "
                                           "NOT_APPLICABLE. In the multi-invoice response, this is "
                                           "whichever invoice's status is surfaced as current (see "
                                           "current_invoice_index) -- not necessarily the most severe "
                                           "one across the whole timeline."},
                "final_label": {"type": ["string", "null"],
                          "description": "The headline label for this whole matching session, "
                                          "present alongside final_overall_match/final_status. "
                                          "\"MATCH\" when final_overall_match is true; when "
                                          "final_status is DISCREPANCY, one of: "
                                          "\"QTY_MISMATCH\" (a quantity disagreement, no amount "
                                          "problem), \"AMOUNT_MISMATCH\" (amounts disagree but "
                                          "quantities line up), \"QTY_AND_AMOUNT_MISMATCH\" (both -- "
                                          "either the same line has both problems, or different lines "
                                          "have different ones), or \"OTHER_MISMATCH\" (a DISCREPANCY "
                                          "not tied to a specific qty/amount line comparison at all, "
                                          "e.g. a role mismatch or missing-data-only cause). Combines "
                                          "the union of every underlying comparison: any QTY plus any "
                                          "AMOUNT anywhere makes this QTY_AND_AMOUNT_MISMATCH even if "
                                          "no single line had both. Each entry inside \"comparisons\" "
                                          "and \"timeline\" also carries its OWN, separate \"label\" -- "
                                          "that one describes just that one entry, not the session."},
                "currency": {"type": "string",
                             "description": "Present alongside \"final_label\" wherever it appears. "
                                             "Whichever currency a document in the comparison stated "
                                             "explicitly (a \"currency\"/\"currency_code\" field, e.g. "
                                             "\"USD\", \"SGD\"), or \"S$\" if none of them said. "
                                             "Display-only -- shown as a prefix on dollar amounts "
                                             "inside \"final_summary\" text (e.g. \"S$4.00\", "
                                             "\"USD 4.00\"); never affects matching logic, and the "
                                             "numeric amount_a/amount_b fields elsewhere in the output "
                                             "stay plain, unprefixed decimal strings."},
                "final_summary": {"type": "string",
                            "description": "The headline, human-readable readout for this whole "
                                            "matching session -- this is the field to display as the "
                                            "main match result. In the single-comparison-set response "
                                            "(no \"timeline\"), it's a combined plain-English readout "
                                            "of every comparison, e.g. 'Invoice matches GRN/Delivery "
                                            "-- fully consistent. PO shows partial fulfillment: 3870 "
                                            "of 20000 ordered units received/invoiced -- not a "
                                            "discrepancy, an open-order status.' In the multi-invoice "
                                            "response (see \"timeline\"), it's instead a line-per-"
                                            "invoice narrative in chronological order, e.g. 'In INV1 "
                                            "(2026-01-22), ... partial fulfillment ...\\nIn INV2 "
                                            "(2026-06-02), ... discrepancy ...', joined from each "
                                            "timeline entry's own \"narrative\"."},
                "reference_index": {"type": "integer"},
                "document_count": {"type": "integer"},
                "document_roles": {"type": "object",
                                    "description": "{document_index: role}, role one of PO, INVOICE, "
                                                    "GRN, CREDIT_NOTE, UNKNOWN -- best-effort, used to "
                                                    "decide which comparisons are lenient "
                                                    "(commitment_check, PO vs Invoice/GRN) vs strict "
                                                    "(fulfillment_check, Invoice vs GRN)."},
                "document_ids": {"type": "object",
                                  "description": "{document_index: doc_id}, each document's own "
                                                  "identifying number (invoice no., PO no., GRN no., "
                                                  "etc.), e.g. \"PPINV-B10097157\" -- best-effort, null "
                                                  "when no self-identifying field is found. Display-only, "
                                                  "never affects matching logic."},
                "comparisons": {
                    "type": "array",
                    "description": "One entry per non-reference document, each with overall_match, "
                                   "status (CONSISTENT/PARTIAL_FULFILLMENT/DISCREPANCY/NOT_APPLICABLE), "
                                   "mode (fulfillment_check/commitment_check/generic), doc_a_role, "
                                   "doc_b_role, doc_a_id, doc_b_id, summary (plain-English one-liner), "
                                   "monetary_fields (list, each now also carrying a status), line_items "
                                   "(list, each now also carrying qty_status/amount_status), "
                                   "unmatched_doc_a_indices, unmatched_doc_b_indices, "
                                   "excluded_doc_a_rows, excluded_doc_b_rows. Only present when "
                                   "`documents` had no PO-role document alongside an INVOICE-role one -- "
                                   "see \"timeline\" below for that response shape instead.",
                },
                "current_invoice_index": {"type": "integer",
                                           "description": "Only present in the multi-invoice response "
                                                           "(see \"timeline\"). Index into `documents` "
                                                           "of whichever invoice's result is surfaced "
                                                           "in the top-level final_overall_match/"
                                                           "final_status -- either "
                                                           "invoice_under_review_index, or the most "
                                                           "recent invoice by date if that wasn't "
                                                           "given."},
                "timeline": {
                    "type": "array",
                    "description": "Only present in the multi-invoice response: one entry per "
                                   "INVOICE-role document. Ordered by each invoice's own date field "
                                   "ONLY if every invoice in `documents` has one that parses; if even "
                                   "one is missing/unparseable, all entries instead keep the order "
                                   "their documents were given in (date_confidence flags which applies "
                                   "per entry) -- partial date coverage is never used to reorder, since "
                                   "guessing at an undated invoice's position would also corrupt the "
                                   "running PO/GRN netting for every invoice after it. Each entry has "
                                   "document_index, invoice_id, date "
                                   "(ISO string, or \"date unknown\"), date_confidence "
                                   "(\"parsed\"/\"unknown_used_document_order\"), status, overall_match, "
                                   "narrative (one-line human-readable verdict for this invoice alone), "
                                   "and comparisons (this invoice's own PO/GRN comparisons, in the same "
                                   "shape as the single-invoice \"comparisons\" field, checked against "
                                   "what was still left on the PO/GRN after every earlier invoice in "
                                   "the timeline had already been netted out). An invoice document whose "
                                   "own id starts with \"P\" (e.g. \"PPINV-B10097417\") never gets its "
                                   "own entry here when at least one other, non-\"P\" invoice is also "
                                   "present -- see \"posted_invoices_excluded\" and \"already_paid\".",
                },
                "batch_count": {"type": "integer",
                                "description": "Only present when a large upload was split into independent "
                                               "PO transactions and matched in parallel batches."},
                "batches": {
                    "type": "array",
                    "description": "Only present together with batch_count: one entry per transaction, "
                                   "each with batch_key (its PO number), document_indices (positions in "
                                   "`documents`), index_scope (\"batch_local\": document indices INSIDE "
                                   "an entry refer to that batch's own document order, map them through "
                                   "document_indices), and that transaction's full normal result "
                                   "(final_status, comparisons or timeline, ...) -- or an \"error\" string "
                                   "if that one batch could not be matched, which counts as a DISCREPANCY "
                                   "in the top-level verdict.",
                },
                "posted_invoices_excluded": {
                    "type": "array",
                    "description": "Only present when `documents` contained BOTH an invoice whose own "
                                   "id starts with \"P\" (a posted/system record, e.g. "
                                   "\"PPINV-B10097417\") and at least one that doesn't (the vendor's "
                                   "original invoice, e.g. \"SI1202997\"). Lists the excluded posted "
                                   "invoice(s)' own doc_id(s) -- they are never compared against the "
                                   "PO/GRN or given their own \"timeline\" entry, since they represent "
                                   "the SAME purchase as the vendor invoice already posted/paid, not a "
                                   "second one; comparing both would double-consume the PO's quantities. "
                                   "They're used only to populate \"already_paid\" (see \"comparisons\" "
                                   "and \"timeline\"). If a posted invoice is the only invoice present, "
                                   "this field is absent and it's matched normally like any other "
                                   "invoice.",
                },
            },
        },
    )


    # extra_protected_paths tells the HMAC middleware to also guard
    # /run_async from the moment the app is built, rather than the
    # route being added afterwards where the middleware never sees it.
    #
    # ── Async job-queue routes ───────────────────────────────────────
    # Added after a confirmed production failure: under concurrent
    # vLLM load (see this file's own retry/concurrency-limiting fix,
    # VLLM_CONCURRENCY_LIMIT etc.), a match job can legitimately take
    # well over 100 seconds to complete -- long enough to exceed
    # Cloudflare Tunnel's ~100s edge timeout on the synchronous /run
    # path this agent used to serve exclusively via agent.serve(...).
    # The job succeeded server-side every time (this agent's own logs
    # showed TaskStatus.SUCCESS and a 200 on /run), but the response
    # never reached the controller, which recorded an empty result
    # instead -- the edge proxy silently killed the connection well
    # before either side's own configured timeout ever mattered.
    #
    # /run_async returns immediately with a job_id and runs this
    # agent's run() in a background thread; the controller then polls
    # /status/<job_id> every few seconds. Each individual HTTP request
    # (submit + each poll) completes in well under a second, so none
    # of them can ever hit the tunnel's edge timeout, no matter how
    # long the underlying matching actually takes. This mirrors
    # byoa-summarize-agent's identical fix for the identical failure
    # mode -- attach_async_routes comes from byoa_adk (v1.1.0+) as a
    # reusable addon, not something hand-rolled per agent, so this is
    # the same call summarize_agent.py makes, just applied here too.
    #
    # Replaces the previous agent.serve(port=port) convenience call,
    # which built and served the Flask app in one step with no hook
    # for attaching these extra routes.
    app = build_flask_app(agent, BYOA_SECRET, extra_protected_paths={"/run_async"})
    attach_async_routes(app, agent)
    serve(app, host="0.0.0.0", port=port, threads=4)
