"""
BYOA Summarize Agent -- standalone GX10 service.

Same SummarizeAgent logic as the Kaggle notebook (adk-summary-agent.ipynb,
Cell 4), unchanged. What's different from the Kaggle version:
  - Does NOT install/start Ollama. GX10 runs `ollama` as its own systemd
    service, shared with the platform controller. This process only
    verifies it's reachable and the configured model is pulled.
  - No pyngrok. Public exposure is handled by the systemd-managed
    `cloudflared` unit on this box, proxying
    https://summarize.zackloh.dev -> http://localhost:PORT.
  - No `while True: sleep(60)` keepalive cell -- systemd keeps this alive.
  - Config comes from a .env file instead of hardcoded notebook cell values.

PERFORMANCE NOTE (added after a latency review across many real plans):
list_instruction and extraction_override are narrow "IMPORTANT EXCEPTION"
blocks that only ever matter when the request is literally asking for a
list/ranking, or literally asking to bare-extract one entity with nothing
else attached. Across every real summarize call logged this session, the
large majority never matched either exception -- meaning the model was
paying to read ~400 words of instructions describing exceptions that could
never have applied to that particular call. Both are now only attached when
a cheap keyword heuristic suggests they're actually relevant, biased toward
over-including rather than ever risking a missed case (verified against
every real focus string seen in production before this rollout). This
changes NOTHING about correctness for calls where the heuristic includes
them -- it only removes dead-weight instruction text for calls where it
provably never applied. focus_instruction and check_instruction (the
gap-fill honesty mechanism) are untouched and always included when a focus
is present, since those aren't overhead -- they're the actual behavior.
"""
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests as _req
from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent, DAG, Step, attach_async_routes
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5400"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://summarize.zackloh.dev")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")

# Caps how many model calls this agent fires CONCURRENTLY at once (per-row
# extraction, per-chunk label backfill classification, etc.) -- a resource
# bound, not a correctness one. vLLM (unlike the Ollama backend this
# service migrated away from) serves concurrent requests natively, so
# these calls are safe to run in parallel; this cap just avoids one
# large document opening an unbounded number of simultaneous connections
# against the shared vLLM instance other agents/requests also use.
#
# NOTE: this only bounds concurrency WITHIN a single document's own
# ThreadPoolExecutor (see pool_size at the two call sites below). It does
# NOT bound total concurrency across multiple documents being processed
# at once by this same waitress process (threads=4, see serve() at the
# bottom of this file) -- e.g. platform_controller.py's MAX_PARALLEL_AGENTS
# dispatching several documents to this agent concurrently, each of which
# then opens its own MAX_CONCURRENT_MODEL_CALLS-sized pool against vLLM.
# That multiplication (concurrent docs x per-doc pool size) is what
# VLLM_CONCURRENCY_LIMIT / _vllm_semaphore below actually caps, at the
# single choke point (_call_ollama) every model call already passes
# through regardless of which document or thread pool it came from.
MAX_CONCURRENT_MODEL_CALLS = int(os.environ.get("MAX_CONCURRENT_MODEL_CALLS", "3"))

# True process-wide cap on simultaneous requests in flight to vLLM, across
# ALL concurrently-processed documents (not just within one). Tune this to
# whatever this vLLM instance can actually sustain without individual
# requests queueing past their read timeout -- start conservative and raise
# it once you've confirmed stress-test batches stay clean. Independent of
# MAX_CONCURRENT_MODEL_CALLS: that one bounds fan-out per document, this one
# bounds the real total load this process puts on the shared backend.
VLLM_CONCURRENCY_LIMIT = int(os.environ.get("VLLM_CONCURRENCY_LIMIT", "6"))
_vllm_semaphore = threading.BoundedSemaphore(VLLM_CONCURRENCY_LIMIT)

# How long a call is willing to wait for a free vLLM slot, in seconds.
# Deliberately NOT derived from that call's own timeout_secs (see
# _call_ollama) -- per-row extraction calls in particular get a very small
# timeout_secs slice (e.g. 8-15s for a document with many rows, computed at
# line ~4707), and tying the queue-wait budget to that slice meant a
# document with more rows got LESS time to find a free slot per row, right
# when overall contention (many documents each fanning out many rows) is
# what's actually driving the wait. Queue depth is a property of total
# concurrent load on this process, not of how small any one caller's own
# request budget happens to be, so this is a single, separately-tunable
# knob instead.
VLLM_QUEUE_WAIT_SECS = float(os.environ.get("VLLM_QUEUE_WAIT_SECS", "60"))

# How many total attempts (first try + retries) a transient vLLM-saturation
# failure gets before the caller gives up on that unit of work (one row, one
# page chunk, etc.) and reports it as a real failure. Bounded deliberately --
# NOT unlimited. An unbounded retry-until-success sounds appealing under
# saturation ("it'll get in eventually"), but every attempt still waits up to
# VLLM_QUEUE_WAIT_SECS for a slot, and that wait is invisible to
# _timed_run()'s own hard deadline on the WHOLE document (payload.timeout_secs,
# enforced from outside this thread -- see the comment in _call_ollama).
# Retrying forever just delays hitting that outer deadline instead of
# avoiding it, and when it hits, the entire document silently disappears
# with no result at all -- worse than today's single marked-incomplete page,
# not better. This many attempts, each waiting up to VLLM_QUEUE_WAIT_SECS,
# rides out realistic contention spikes while still leaving real headroom
# under the document's overall budget for whatever work comes after it.
MAX_TRANSIENT_MODEL_RETRIES = int(os.environ.get("MAX_TRANSIENT_MODEL_RETRIES", "3"))

# ---------------------------------------------------------------------------
# AP Document Store integration (see LLM extraction integration guide).
# Base URL/key come from .env, not hardcoded, since the tunnel URL can
# change whenever the backend team's network path is reconfigured --
# treat AP_STORE_BASE_URL like VLLM_BASE_URL above, not like a
# constant.
# ---------------------------------------------------------------------------
AP_STORE_BASE_URL = os.environ.get("AP_STORE_BASE_URL", "").rstrip("/")
AP_STORE_API_KEY = os.environ.get("AP_STORE_API_KEY", "")
# Submission is a no-op (skipped, logged) rather than a hard failure when
# unconfigured, so this agent still works standalone (e.g. local dev,
# before the backend team hands over a URL/key) without every extraction
# call erroring out.
AP_STORE_ENABLED = bool(AP_STORE_BASE_URL and AP_STORE_API_KEY)
if not AP_STORE_ENABLED:
    print(
        "WARN AP_STORE_BASE_URL / AP_STORE_API_KEY not set -- "
        "extracted documents will NOT be submitted to the AP Document Store."
    )

# When True, a markdown rendering of the *extracted, structured JSON*
# (not the original OCR source) is attached to the submission payload
# under the "MD_Version" key, alongside the parsed fields -- gives the
# AP Document Store side a human-readable view of exactly what's being
# submitted. Off by default since it roughly doubles the submission
# payload size.
AP_STORE_INCLUDE_MD_VERSION = os.environ.get(
    "AP_STORE_INCLUDE_MD_VERSION", "false"
).strip().lower() in ("1", "true", "yes")

# When True, the MD_Version markdown is ALSO written to disk under
# AP_STORE_MD_VERSION_DIR, one file per doc_id -- independent of
# AP_STORE_ENABLED/whether the submission itself succeeds, so it's
# usable purely for local inspection (e.g. while the AP Document Store
# endpoint is unreachable). Requires AP_STORE_INCLUDE_MD_VERSION.
AP_STORE_SAVE_MD_VERSION_LOCALLY = os.environ.get(
    "AP_STORE_SAVE_MD_VERSION_LOCALLY", "false"
).strip().lower() in ("1", "true", "yes")
AP_STORE_MD_VERSION_DIR = os.environ.get(
    "AP_STORE_MD_VERSION_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "md_versions"),
)

# doc_id assignment tries, in order: (1) a model-based lookup that
# reads the document's own RENDERED MARKDOWN content (see
# _find_document_reference_from_md) to identify its printed reference
# number -- schema-agnostic, no fixed field-name list to keep up to
# date, (2) if that misses after every retry but there's an
# upload_id, a hash of it, cached so the same upload always resolves
# to the same doc_id, and only (3), when both come back empty, a
# running per-prefix sequence (_next_ap_doc_sequence) -- restored as
# the absolute last resort: every real AP document is expected to
# carry its own reference number somewhere, so this step should
# essentially never fire in practice.
# It still has the two caveats a shared mutable counter always has --
# no idempotency across reprocessing runs (the same document
# reprocessed twice gets two different numbers) and no protection
# against duplicate hand-outs if this service ever runs as more than
# one process/replica -- but those only matter for documents that
# already defeated the model's own attempt to read their own printed
# number straight off the content, which should be rare enough that the
# tradeoff is acceptable as a last resort rather than the primary path
# it used to be.
AP_DOC_COUNTER_FILE = os.environ.get(
    "AP_DOC_COUNTER_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ap_doc_counters.json"),
)
_ap_doc_id_lock = threading.Lock()

# upload_id -> doc_id cache, used only by the hash-based fallback path
# (see _assign_ap_doc_id) so that fallback stays idempotent across
# reprocessing runs even when it's not the primary path anymore.
AP_DOC_ID_CACHE_FILE = os.environ.get(
    "AP_DOC_ID_CACHE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ap_doc_id_cache.json"),
)

# TYPE_PREFIX per API doc_type. The integration guide only confirms
# PO- and INV- by example ("PO-2026-0231", "INV-2026-0456"); the rest
# are this service's best guess at a consistent convention and are NOT
# confirmed by the backend team -- worth confirming before relying on
# them for CreditDebitNote / PaymentVoucher / OfficialReceipt /
# Quotation / RFQ / PR in particular.
AP_DOC_TYPE_PREFIXES = {
    "PR": "PR",
    "RFQ": "RFQ",
    "Quotation": "QUO",
    "PO": "PO",
    "DO": "DO",
    "GRN": "GRN",
    "Invoice": "INV",
    "CreditDebitNote": "CDN",
    "PaymentVoucher": "PV",
    "OfficialReceipt": "OR",
}

# Ordered so more specific phrases are checked before shorter/ambiguous
# ones. Matched case-insensitively against whatever free-text
# `document_type` label the extraction model produced -- that label is
# never one of the API's exact ten enum values by construction, so this
# mapping is required before every submission.
_AP_DOC_TYPE_KEYWORDS = [
    ("PR", ("purchase requisition",)),
    ("RFQ", ("request for quotation", "rfq")),
    ("Quotation", ("quotation", "quote")),
    ("GRN", ("goods received note", "goods receipt note", "grn", "put-away", "putaway", "put away")),
    # "delivery note" added alongside "delivery order" -- this is the
    # model-guess mapping list (see _AP_DOC_TYPE_SOURCE_TEXT_KEYWORDS
    # above for the separate raw-source-text scan list, which already
    # had this). Confirmed miss: a sub-document's own `document_type`
    # came back as the literal string "delivery_note" (multi-document
    # split case, each sub-document is nested under a lower_snake_case
    # key matching its own document_type), and with no "delivery note"
    # keyword here it mapped to None -- the WHOLE sub-document was
    # silently skipped ("could not map document_type 'delivery_note'")
    # instead of being submitted as a DO.
    ("DO", ("delivery order", "delivery note", "d.o.", " do ", "^do$")),
    ("PO", ("purchase order", "p.o.", " po ", "^po$")),
    ("CreditDebitNote", ("credit note", "debit note", "credit/debit", "credit-debit")),
    ("PaymentVoucher", ("payment voucher",)),
    ("OfficialReceipt", ("official receipt", "receipt")),
    ("Invoice", ("tax invoice", "invoice", "bill")),
]


def _map_to_ap_doc_type(label) -> "str | None":
    """
    Maps the extraction model's free-text `document_type` guess onto one
    of the AP Document Store's exact ten doc_type strings. Returns None
    if nothing matches confidently -- callers must treat None as "don't
    submit, this needs a human to classify it", never guess a default.

    Normalizes underscores/hyphens to spaces before matching -- multi-
    document sub-documents carry a lower_snake_case `document_type`
    (e.g. "goods_receipt_note", "delivery_note", from the per-page
    nesting convention -- see _extract_sub_documents), which otherwise
    fails to match ANY space-separated keyword phrase below. Confirmed
    production failure mode from that mismatch: "goods_receipt_note"
    matched none of the GRN keywords (all space-separated) but DID
    match bare "receipt" as a raw substring, silently misrouting a
    Goods Receipt Note to OfficialReceipt instead. Normalizing first
    fixes both the false-negative (GRN keywords now match) and the
    false-positive (GRN is checked, and matches, before OfficialReceipt
    is ever reached).
    """
    if not label or not isinstance(label, str):
        return None
    normalized_label = re.sub(r"[_-]+", " ", label.strip().lower())
    normalized_label = re.sub(r"\s+", " ", normalized_label)
    text = f" {normalized_label} "
    for api_type, keywords in _AP_DOC_TYPE_KEYWORDS:
        for kw in keywords:
            if kw.startswith("^") and kw.endswith("$"):
                if text.strip() == kw.strip("^$"):
                    return api_type
            elif kw in text:
                return api_type
    return None


# Human-readable label per AP doc_type enum value, used only to
# rewrite `structured.document_type` / the `summary` text when
# _finalize_ap_submission resolves a DIFFERENT type than the
# extraction model's own free-text guess (header-title short-circuit
# or the pricing tie-break both do this -- see _submit_single_document).
# Without this, a document could be correctly routed as "DO" for AP
# submission purposes while `structured.document_type` and `summary`
# (and anything downstream reading those instead of `ap_submission`,
# e.g. `final_answer`) still displayed the model's original, wrong
# "Invoice" guess -- a split-brain result where routing is fixed but
# every human/display-facing field is not.
AP_DOC_TYPE_DISPLAY_LABELS = {
    "PR": "Purchase Requisition",
    "RFQ": "Request for Quotation",
    "Quotation": "Quotation",
    "PO": "Purchase Order",
    "DO": "Delivery Order / Delivery Note",
    "GRN": "Goods Receipt Note",
    "Invoice": "Invoice",
    "CreditDebitNote": "Credit/Debit Note",
    "PaymentVoucher": "Payment Voucher",
    "OfficialReceipt": "Official Receipt",
}


def _reconcile_document_type_label(doc_parsed: dict, ap_submission: dict) -> str:
    """
    After _finalize_ap_submission has resolved the actual api_doc_type
    (which may differ from the model's own `document_type` guess --
    see doc_type_check.source values like "header_title_literal" or
    "ambiguous_source_text_resolved_by_pricing_check"), rewrites
    `doc_parsed["document_type"]` in place to match the RESOLVED type,
    and returns a display label for use in the `summary` string.

    The model's original guess is preserved (never deleted) under
    `document_type_model_guess` for auditability, so a human reviewer
    can still see what the model said vs. what the pipeline resolved
    to and why (via ap_submission.doc_type_check).

    MULTI-DOCUMENT CONTAINERS (ap_submission has a "documents" list,
    i.e. _extract_sub_documents fired): this function does NOT stamp a
    single top-level `document_type` onto `doc_parsed` in that case --
    a container holding two+ distinct sub-documents (e.g.
    doc_parsed["delivery_note"] and doc_parsed["goods_receipt_note"])
    doesn't have one type as a whole, and doing so previously produced
    a genuinely misleading result (a top-level "document_type":
    "Delivery Order / Delivery Note" sitting alongside BOTH sub-
    documents) and would corrupt `_extract_sub_documents` on any future
    reprocessing pass, since that function treats a non-empty top-level
    `document_type` as "already a single document, don't look for sub-
    documents". Instead, each sub-document dict is reconciled
    individually against its OWN resolved type from
    ap_submission["documents"], matched by "source_key".

    No-ops (returns the original label unchanged) when AP submission
    was skipped or the resolved type doesn't map to a known display
    label, since there's nothing more specific to reconcile to in
    that case.
    """
    sub_results = ap_submission.get("documents")
    if isinstance(sub_results, list):
        for result in sub_results:
            source_key = result.get("source_key")
            resolved_api_type = result.get("api_doc_type")
            if not source_key or not resolved_api_type or resolved_api_type not in AP_DOC_TYPE_DISPLAY_LABELS:
                continue
            sub_doc = doc_parsed.get(source_key)
            if not isinstance(sub_doc, dict):
                continue
            original_label = sub_doc.get("document_type")
            display_label = AP_DOC_TYPE_DISPLAY_LABELS[resolved_api_type]
            if _map_to_ap_doc_type(original_label) != resolved_api_type:
                if original_label:
                    sub_doc["document_type_model_guess"] = original_label
                sub_doc["document_type"] = display_label
        # No single overall label for a multi-document container --
        # summary text falls back to the generic "extraction complete"
        # message at the call site when this returns None.
        return None

    resolved_api_type = ap_submission.get("api_doc_type")
    if not resolved_api_type or resolved_api_type not in AP_DOC_TYPE_DISPLAY_LABELS:
        return doc_parsed.get("document_type")

    original_label = doc_parsed.get("document_type")
    display_label = AP_DOC_TYPE_DISPLAY_LABELS[resolved_api_type]

    if _map_to_ap_doc_type(original_label) != resolved_api_type:
        if original_label:
            doc_parsed["document_type_model_guess"] = original_label
        doc_parsed["document_type"] = display_label

    return doc_parsed.get("document_type")


# Source-text doc_type cross-check -- a sanity check independent of the
# extraction model's own `document_type` guess, since a model can
# misclassify a document even when its own printed header is
# unambiguous (confirmed in production: model said "Purchase Order" on
# a document literally headed "TAX INVOICE"). Deliberately narrower/more
# literal than _AP_DOC_TYPE_KEYWORDS above -- these are only phrases
# expected to appear as an actual printed document title.
_AP_DOC_TYPE_SOURCE_TEXT_KEYWORDS = [
    ("PR", ("purchase requisition",)),
    ("RFQ", ("request for quotation",)),
    ("GRN", ("goods received note", "goods receipt note")),
    # "delivery note" added alongside "delivery order" -- confirmed
    # production miss: a document printed with the literal title
    # "Delivery Note" was not recognized by this scan at all (only
    # "delivery order" was listed), so the scan fell through to
    # finding a single "invoice" match (from an "Invoice No." cross-
    # reference field on the same form) and incorrectly "confirmed"
    # the model's Invoice guess. There is no separate AP doc_type for
    # a delivery note vs. a delivery order in the ten-type enum, so
    # both map onto "DO".
    ("DO", ("delivery order", "delivery note")),
    ("PO", ("purchase order",)),
    ("CreditDebitNote", ("credit note", "debit note")),
    ("PaymentVoucher", ("payment voucher",)),
    ("OfficialReceipt", ("official receipt",)),
    ("Quotation", ("quotation",)),
    ("Invoice", ("tax invoice", "invoice")),
]

# Marks where a printed document's HEADER/TITLE region ends and its
# body (bill-to/ship-to block) begins. Used by
# _scan_header_title_doc_type below to avoid conflating a document's
# own printed title (e.g. "Delivery Note" at the very top) with an
# unrelated field label appearing later in the body (e.g. an
# "Invoice No.:" cross-reference field on a Delivery Note that
# references a separate invoice) -- these are NOT the same signal and
# treating them identically is exactly how the Nespresso delivery-note
# case above got misclassified as an Invoice.
#
# Deliberately excludes "deliver to": confirmed production failure on
# a Purchase Order template where "Deliver To" is a short INLINE
# routing/location-code field sitting on the same header line as, and
# immediately before, the document's own printed title ("Deliver To
# 03ZFN1 PURCHASE ORDER") -- unlike "invoice to"/"bill to"/"ship to"/
# "sold to", which reliably open a multi-line customer address block
# and so reliably mark the end of the vendor's own header, "deliver
# to" varies too much between templates (sometimes a block boundary,
# sometimes a one-field label) to safely cut on. Cutting here truncated
# the header region before ever reaching "PURCHASE ORDER", so the
# header-title scan found nothing and the document fell through to the
# less precise whole-text scan -- see the ambiguous-candidate tie-break
# below for how that additionally went wrong.
_HEADER_REGION_CUTOFF_MARKERS = (
    "invoice to", "bill to", "ship to", "sold to",
)
_HEADER_REGION_MAX_CHARS = 800


def _scan_header_title_doc_type(source_text: str) -> "str | None":
    """
    Zero-inference, zero-model-cost lookup for a document type that is
    EXPLICITLY stated in the document's own printed header/title --
    the common case (invoices, delivery notes, POs, etc. almost always
    say what they are at the top of the page). This is a pure regex
    scan over a narrowed header region, run BEFORE any extraction
    model call, so that when the type is unambiguous the extraction
    prompt can be told the fact directly instead of asking the model
    to spend reasoning budget re-deriving it from the whole document.

    Deliberately narrower than _scan_source_text_doc_types (which
    scans the ENTIRE document and is still used as a fallback/cross-
    check): restricting to the header region avoids false hits from
    field labels that happen to contain a doc-type word further down
    the page (e.g. "Invoice No.:" on a document titled "Delivery
    Note"), which is a different signal than the document's own title.

    Returns a single confident api_doc_type string, or None if the
    header region is empty, unmatched, or names more than one type
    (genuinely a hybrid document, e.g. this one) -- callers must treat
    None as "not decidable this cheaply", and fall back to the
    existing model-classification + whole-text cross-check path,
    never guess a default here.
    """
    if not source_text:
        return None
    normalized = source_text.replace(_OCR_WS_PLACEHOLDER, " ").lower()
    normalized = re.sub(r"\s+", " ", normalized)

    cutoff = len(normalized)
    for marker in _HEADER_REGION_CUTOFF_MARKERS:
        idx = normalized.find(marker)
        if idx != -1:
            cutoff = min(cutoff, idx)
    header_region = normalized[: min(cutoff, _HEADER_REGION_MAX_CHARS)]
    if not header_region.strip():
        return None

    matches = []
    for api_type, phrases in _AP_DOC_TYPE_SOURCE_TEXT_KEYWORDS:
        for phrase in phrases:
            if phrase in header_region and api_type not in matches:
                matches.append(api_type)
                break
    return matches[0] if len(matches) == 1 else None


def _scan_source_text_doc_types(source_text: str) -> list:
    """
    Returns the list of AP doc_types whose literal header phrase appears
    anywhere in the raw OCR'd source text, in the order first
    encountered. An empty list means nothing to cross-check against.
    More than one element means the source text has multiple document-
    type header phrases -- genuinely ambiguous, callers should flag,
    not auto-resolve.
    """
    if not source_text:
        return []
    normalized = source_text.replace(_OCR_WS_PLACEHOLDER, " ").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    found = []
    for api_type, phrases in _AP_DOC_TYPE_SOURCE_TEXT_KEYWORDS:
        for phrase in phrases:
            if phrase in normalized:
                if api_type not in found:
                    found.append(api_type)
                break
    return found


# Field-name canonicalization -- the extraction model is free-running
# text generation, not a fixed schema, so the SAME logical field shows
# up under different spellings across runs/documents. Extend this with
# more canonical_key -> [variants] entries as you spot other fields
# that drift, or to align onto the AP Document Store's OWN expected
# field names per doc_type (see GET /api/schemas/:docType).
AP_FIELD_NAME_ALIASES = {
    "inv_number": [
        "inv_number", "inv_num", "inv_no", "invoice_no", "invoice_num",
        "invoice_number", "tax_invoice_number", "tax_invoice_no",
    ],
}


def _normalize_key(key: str) -> str:
    """Strips everything but lowercase letters/digits so "Inv No.",
    "inv_no", "INV-NO", "invno" all compare equal."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _match_field_alias(d: dict, variants: list, normalized_lookup: dict = None) -> "str | None":
    """
    Two-pass alias match against `d`'s keys for a SINGLE field concept
    (one canonical field's list of expected spellings):
    1. EXACT normalized-key equality against each variant -- tried
       first, wins if found, since it's the least ambiguous match.
    2. Only if no variant matched exactly: substring containment (a
       variant's normalized form found anywhere inside an actual
       field's normalized name), restricted to variants of at least 5
       normalized characters to avoid short/generic aliases (e.g.
       "date", "po") spuriously matching inside unrelated names.

    Returns the ORIGINAL matched key from `d`, or None if nothing
    matched. Extracted out of _canonicalize_field_names (which calls
    this once per canonical field in its alias_map) so the exact same
    matching logic can also be reused for any other single-field alias
    lookup outside that loop that isn't itself a dict of
    canonical->variants.
    """
    if normalized_lookup is None:
        normalized_lookup = {_normalize_key(k): k for k in d.keys()}

    for variant in variants:
        candidate = normalized_lookup.get(_normalize_key(variant))
        if candidate is not None and d[candidate] not in (None, ""):
            return candidate

    for variant in variants:
        norm_variant = _normalize_key(variant)
        if len(norm_variant) < 5:
            continue
        for norm_key, orig_key in normalized_lookup.items():
            if norm_variant not in norm_key or d[orig_key] in (None, ""):
                continue
            # A generic financial alias like "total" is a substring of
            # BOTH "subtotal" (pre-tax, wrong) and "total_due" (post-
            # tax, correct) -- see _canonicalize_field_names' original
            # comment on this exact production miss. Kept here since
            # this helper now owns the substring pass.
            if norm_variant == "total" and "subtotal" in norm_key:
                continue
            return orig_key
    return None


def _canonicalize_field_names(d: dict, alias_map: dict = AP_FIELD_NAME_ALIASES) -> dict:
    """
    Returns a shallow copy of `d` where, for each canonical key in
    `alias_map`, the value from whichever variant spelling is present
    gets ALSO written under the canonical key -- variant keys are left
    in place, nothing deleted. Matching itself (exact-then-substring)
    is done by _match_field_alias, shared with other single-field
    alias lookups elsewhere in this file.
    """
    result = dict(d)
    normalized_lookup = {_normalize_key(k): k for k in d.keys()}

    for canonical, variants in alias_map.items():
        if result.get(canonical) not in (None, ""):
            continue
        match_key = _match_field_alias(d, variants, normalized_lookup)
        if match_key is not None:
            result[canonical] = d[match_key]

    return result


# Per-doc_type "worth having" field lists an external consumer (a
# downstream integration / a partner's own API) expects by these EXACT
# names, regardless of what this pipeline's own extraction happened to
# call the equivalent value. Add/adjust entries here as that contract
# changes -- this list is deliberately separate from AP_FIELD_NAME_ALIASES
# above (which exists to normalize THIS pipeline's own internal drift),
# since the two can evolve independently.
_AP_DOC_TYPE_CANONICAL_FIELDS = {
    "RFQ": ["pr_ref"],
    "PO": ["pr_ref"],
    "DO": ["po_ref"],
    "GRN": ["po_ref", "do_ref"],
    "Invoice": ["po_ref"],
    "CreditDebitNote": ["invoice_ref"],
    "PaymentVoucher": ["invoice_ref"],
    "OfficialReceipt": ["pv_ref"],
}

# Short, plain-language description of what each canonical field
# MEANS -- used only by _infer_canonical_field_via_model below, as the
# last-resort fallback when neither the exact nor substring alias
# match in _canonicalize_field_names finds anything. The alias list
# above is a list of expected SPELLINGS; this is the semantic meaning,
# needed because a model prompt asking "which key means this?" needs
# a description to reason from, not just the bare canonical name.
_AP_CANONICAL_FIELD_DESCRIPTIONS = {
    "pr_ref": "the purchase requisition (PR) number this document references",
    "po_ref": "the purchase order (PO) number this document references",
    "do_ref": "the delivery order / delivery note number",
    "invoice_ref": "the invoice number",
    "pv_ref": "the payment voucher number this document references",
}


_AP_CANONICAL_FIELD_ALIASES = {
    "pr_ref": ["pr_ref", "pr_no", "pr_number", "purchase_requisition_no", "purchase_requisition_number"],
    "po_ref": [
        "po_ref", "po_number", "po_no", "customer_po", "purchase_order_no",
        "purchase_order_number", "order_reference",
    ],
    "do_ref": ["do_ref", "do_no", "do_number", "delivery_order_no", "delivery_note_no", "do"],
    "invoice_ref": ["invoice_ref", "invoice_number", "invoice_no", "tax_invoice_no", "tax_invoice_number"],
    "pv_ref": ["pv_ref", "payment_voucher_no", "pv_no", "voucher_no"],
}


# Short keyword that appears in EACH api_doc_type's own self-
# identifying field name (its own "grn_no", "po_number", etc.) --
# used by _infer_canonical_field_via_model to exclude a document's own
# identifier from being offered as a candidate answer for any "_ref"
# canonical field, since a "_ref" field always points to a DIFFERENT
# document by construction (see _AP_DOC_TYPE_CANONICAL_FIELDS -- no
# doc_type's canonical field list ever asks it to find a reference to
# its own type). Over-filtering here is safe: the worst case is
# leaving a "_ref" field null when a genuine candidate happened to
# also contain the keyword, not returning a wrong value -- so the
# short two-letter keywords ("po", "do", "pr") are kept despite a
# small false-positive substring risk (e.g. "report_by"), since a
# missed-but-safe null is preferable to a confidently wrong self-match.
_AP_DOC_TYPE_SELF_ID_KEYWORDS = {
    "PR": "pr",
    "RFQ": "rfq",
    "Quotation": "quot",
    "PO": "po",
    "DO": "do",
    "GRN": "grn",
    "Invoice": "invoice",
    "CreditDebitNote": "note",
    "PaymentVoucher": "voucher",
    "OfficialReceipt": "receipt",
}
_AP_REF_FIELDS = {"po_ref", "do_ref", "pr_ref", "invoice_ref", "pv_ref"}


def _infer_canonical_field_via_model(
    lookup_source: dict, field: str, api_doc_type: str = None, timeout_secs: float = 10
) -> "tuple | None":
    """
    Last-resort fallback for a canonical field that BOTH the exact and
    substring passes in _canonicalize_field_names failed to match --
    asks the model which of the document's own top-level scalar field
    NAMES (not a re-read of the raw OCR text -- cheap, small prompt)
    represents the concept described in
    _AP_CANONICAL_FIELD_DESCRIPTIONS.

    Deliberately gated to only fire on a genuine miss from the cheap
    deterministic passes, not run for every canonical field on every
    document: those passes resolve the large majority of cases for
    free, and this pipeline's external consumer trusts these values
    directly (no downstream tie-break double-checks them the way
    document-type resolution does), so keeping the model in a narrow,
    last-resort role bounds both the added cost and the blast radius
    of a wrong guess.

    For "_ref" fields (po_ref, do_ref, etc. -- see
    _AP_REF_FIELD_SELF_ID_KEYWORDS), any candidate whose key contains
    the CURRENT document's own type keyword is dropped before the
    model ever sees it: a "_ref" field always points to a DIFFERENT
    document by definition (a GRN's do_ref names the delivery order it
    was generated against, never the GRN itself), so a GRN's own
    "grn_no" can never be a valid answer to "what's the do_ref" no
    matter how confidently a model might otherwise pick it.

    Returns (matched_key, value) if the model named one of the
    document's actual keys with a non-empty value, else None -- a
    hallucinated key name that doesn't exist in `lookup_source`, an
    explicit "none" answer, or a filtered-out self-referential
    candidate, all resolve to None rather than being force-matched to
    something arbitrary.
    """
    description = _AP_CANONICAL_FIELD_DESCRIPTIONS.get(field)
    if not description:
        return None

    candidates = {k: v for k, v in lookup_source.items() if v not in (None, "") and not isinstance(v, (dict, list))}

    # Confirmed production failure without this exclusion: processing
    # a GRN, asked to find "the delivery order / delivery note number"
    # (do_ref), the model was given the GRN's own fields as candidates
    # -- including "grn_no": "GRN-B10097758" -- and picked that as the
    # answer despite it being the GRN's OWN number, not a DO reference
    # at all (the "GRN-" prefix alone makes this an obvious non-match,
    # but the model defaulted to the closest-sounding option rather
    # than correctly answering "none"). A "_ref" field always points
    # to a DIFFERENT document by construction, so the current
    # document's own self-identifying field can never be a valid
    # answer -- drop it from the candidate set entirely before the
    # model ever sees it, rather than relying on the prompt alone to
    # avoid picking it.
    if field in _AP_REF_FIELDS and api_doc_type:
        self_keyword = _AP_DOC_TYPE_SELF_ID_KEYWORDS.get(api_doc_type)
        if self_keyword:
            candidates = {k: v for k, v in candidates.items() if self_keyword not in _normalize_key(k)}

    if not candidates:
        return None

    try:
        prompt = (
            "Below is a JSON object extracted from a business document. "
            f"I'm looking for the field that represents: {description}.\n"
            "Look at the keys and values below and tell me which key holds "
            "this value. Only answer with a key if you are CONFIDENT it "
            "genuinely represents that concept -- do not pick the closest-"
            "sounding option if nothing truly matches. In particular, a "
            "document's own reference/ID number is NOT the same thing as a "
            "reference to a DIFFERENT document (e.g. a goods receipt "
            "note's own GRN number is never the delivery order number it "
            "should be linked to, even though both are reference-shaped "
            "strings). Respond with ONLY the exact key name as printed "
            "below, or the single word 'none' if no key confidently "
            "represents this concept. Do not explain, do not add "
            "punctuation.\n\n"
            + json.dumps(candidates, ensure_ascii=False)
        )
        answer = _call_ollama(prompt, timeout_secs).strip()
    except Exception:
        return None

    if not answer or answer.lower() == "none":
        return None
    if answer in candidates:
        return answer, candidates[answer]
    # Model may echo the key with different casing/quoting -- try a
    # normalized match against the real keys before giving up, rather
    # than discarding an answer that's clearly right but not byte-exact.
    normalized_answer = _normalize_key(answer)
    for key, val in candidates.items():
        if _normalize_key(key) == normalized_answer:
            return key, val
    return None


def _flatten_one_level(doc: dict) -> dict:
    """
    Returns a LOOKUP COPY of `doc` with every nested dict's scalar
    leaves flattened up one level -- both as "<container>_<subkey>"
    and the bare "<subkey>" -- so alias matching (which expects a flat
    "customer_po"/"vendor_name"/etc.) still finds a value some
    extraction runs nested inside a container object (e.g.
    "invoice_details": {"customer_po": ...}). Confirmed production
    miss without this: a document nested "customer_po" inside
    "invoice_details" (a container name outside any fixed expected
    list), so a flat-only lookup never saw it and "po_ref" stayed null
    despite the value being right there. This pipeline's extraction
    keeps inventing new container names run to run -- there's no fixed
    list that stays complete, so flatten generically instead of
    guessing at names.

    The nested dict is removed from the COPY only (never from `doc`
    itself) once flattened -- otherwise, when a container name
    collides with a target field name (e.g. "vendor" the container vs.
    "vendor" the field being resolved), the lookup sees a non-empty
    dict already sitting under that exact key and treats it as
    "already resolved," skipping past the flattened "vendor_name"
    scalar entirely.

    Extracted out of _add_canonical_ap_fields so the same flattening
    is available to any other single-field alias lookup over a
    document's own fields -- e.g. the provenance lookup in
    _find_document_reference_from_md.
    """
    lookup = dict(doc)
    for container, nested in list(doc.items()):
        if not isinstance(nested, dict):
            continue
        for subkey, subval in nested.items():
            if isinstance(subval, (dict, list)):
                continue  # only flatten scalar leaves, not further nesting
            lookup.setdefault(f"{container}_{subkey}", subval)
            lookup.setdefault(subkey, subval)
        lookup.pop(container, None)
    return lookup


def _add_canonical_ap_fields(doc: dict, api_doc_type: str, timeout_secs: float = 20) -> dict:
    """
    Mutates `doc` IN PLACE, adding (never removing/renaming) the small
    set of doc_type-specific field names an external consumer expects
    verbatim -- e.g. it wants "po_ref" and will never look for
    "po_number", "po_no", or "customer_po", even though this pipeline's
    own extraction may have produced any of those spellings depending
    on the run. Each canonical field's VALUE is pulled from whichever
    of this pipeline's own existing fields already holds it (see
    _AP_CANONICAL_FIELD_ALIASES) -- nothing is re-derived, guessed, or
    computed; every original field is left exactly as it was.

    The canonical key is always added, even as None, so a downstream
    consumer checking for the key's PRESENCE (not just a truthy value)
    sees a consistent shape across every document sharing an
    api_doc_type, regardless of which alias spelling this run's
    extraction happened to produce. A non-empty value already sitting
    under the canonical key name itself is never overwritten.

    Some extraction runs nest vendor/customer/etc. info as a
    sub-object (e.g. {"vendor": {"name": ..., "address": ...}}), others
    keep it flat ("vendor_name": ...) -- one level of common containers
    is flattened into a throwaway lookup copy (see _flatten_one_level,
    never written back onto `doc`) so alias matching sees both shapes.

    No-op, returns `doc` unchanged, if api_doc_type has no defined
    canonical field list or `doc` isn't a dict.
    """
    target_fields = _AP_DOC_TYPE_CANONICAL_FIELDS.get(api_doc_type)
    if not target_fields or not isinstance(doc, dict):
        return doc

    lookup_source = _flatten_one_level(doc)

    alias_map = {field: _AP_CANONICAL_FIELD_ALIASES.get(field, [field]) for field in target_fields}
    resolved = _canonicalize_field_names(lookup_source, alias_map)

    for field in target_fields:
        existing = doc.get(field)
        # A dict/list sitting under the canonical name (e.g. a nested
        # "vendor": {"name": ..., "address": ...} object, produced by
        # some extraction runs) isn't a usable scalar value for an
        # external consumer expecting a plain "vendor" string -- treat
        # it the same as missing/empty rather than "already set", so it
        # gets replaced by the resolved scalar below.
        if existing in (None, "") or isinstance(existing, (dict, list)):
            doc[field] = resolved.get(field)

    # Last-resort model fallback -- only for fields BOTH deterministic
    # passes above left null. See _infer_canonical_field_via_model:
    # deliberately narrow (one small call per still-missing field, not
    # per document) so the common case -- most fields resolve via the
    # free alias passes -- pays no model cost at all.
    still_missing = [f for f in target_fields if doc.get(f) in (None, "")]
    for field in still_missing:
        match = _infer_canonical_field_via_model(
            lookup_source, field, api_doc_type=api_doc_type, timeout_secs=max(timeout_secs * 0.1, 6)
        )
        if match is not None:
            doc[field] = match[1]

    return doc


# Looks for an `attachment_id` marker anywhere in the first ~20 lines of
# the source MD (frontend convention: it's placed at the top of the
# file) -- e.g. "attachment_id: abc123" or "Attachment ID: abc123".
_ATTACHMENT_ID_RE = re.compile(
    r"attachment[_\s-]*id\s*[:\-]?[\s`*_]*([A-Za-z0-9._-]+)",
    re.IGNORECASE,
)


# Matches a "key: value" line inside the YAML-style frontmatter block
# that precedes the first "## Page N" marker (e.g. "job_id:
# bdad6012-154d-4fce-bcb7-2c519b7c5954"). This is a plain, deterministic
# regex over the frontmatter's own key -- NOT the probabilistic
# label:value LLM classifier used elsewhere (_ensure_source_labels_
# captured) for OCR body content, which is unreliable for this purpose:
# confirmed that whether "job_id: ..." gets classified as a real field
# varies run to run since it's an LLM call, not a deterministic parse.
# job_id/source/submitted_by/etc. are always present in this exact
# "key: value" shape in the frontmatter this pipeline generates, so a
# fixed regex is both simpler and 100% reliable here.
_FRONTMATTER_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*(?:#.*)?$")


def _extract_frontmatter_fields(source_text: str, keys: tuple = ("job_id",)) -> dict:
    """
    Deterministically pulls the given keys out of the YAML-style
    frontmatter block (the lines before the first "## Page N" marker),
    if present. Returns {} for any key not found or if there's no
    frontmatter at all. Never raises.
    """
    if not source_text:
        return {}
    found = {}
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Page"):
            break
        if stripped in ("---", ""):
            continue
        match = _FRONTMATTER_KV_RE.match(stripped)
        if match:
            key, value = match.group(1), match.group(2).strip()
            if key in keys and value:
                found[key] = value
    return found


def _extract_upload_id(text: str) -> "str | None":
    if not text:
        return None
    head = "\n".join(text.splitlines()[:20])
    match = _ATTACHMENT_ID_RE.search(head)
    if match:
        return match.group(1).strip()
    return None


# How many times _find_document_reference_from_md re-asks the model
# before giving up (see that function's docstring for why retrying is
# worthwhile here, unlike a temperature=0 call).
MAX_REF_NUMBER_ATTEMPTS = int(os.environ.get("MAX_REF_NUMBER_ATTEMPTS", "3"))


def _find_document_reference_from_md(doc: dict, api_doc_type: str, timeout_secs: float = 20,
                                      max_attempts: int = MAX_REF_NUMBER_ATTEMPTS) -> tuple:
    """
    Finds the document's own unique identifying reference number by
    asking the model to read this document's OWN rendered markdown
    (see _json_to_markdown) -- the same content a human reviewer would
    see -- rather than pattern-matching field NAMES against a fixed
    alias list. Schema-agnostic by design: it doesn't matter what key
    the extraction happened to store the value under, whether it's
    nested, or what container it's inside -- the question is simply
    "what identifies this specific document", read off the rendered
    content the same way a person would.

    `doc` is rendered via _json_to_markdown fresh here -- it only ever
    sees THIS document's own fields, never another sub-document's
    content the way a shared, unscoped multi-page source_text could
    (see _submit_single_document's trust_own_type docstring for the
    exact cross-document contamination this file already had to guard
    against once before for document_type detection). Using this
    document's own rendered fields rather than the raw multi-page OCR
    text sidesteps that same risk here.

    Grounded: the returned value must appear verbatim (case-
    insensitive, whitespace-normalized) somewhere in the rendered
    markdown -- guards against the model inventing a plausible-looking
    but nonexistent answer, the same spirit as the old field-existence
    check, just checked against rendered text instead of a field name.

    RETRIES up to `max_attempts` times before giving up, because
    `_call_ollama` here runs at the module's default temperature (0.3,
    not 0) -- an ungrounded/blank answer on one attempt is NOT
    guaranteed to repeat on the next. Each retry beyond the first also
    strengthens the prompt's wording, on the working assumption that a
    genuine reference number is always present somewhere in the
    rendered content, so a null answer is more likely the model missing
    it than a true absence.

    Returns (source, value) as strings, or (None, None) if `doc` isn't
    a dict, has no renderable content, or no grounded answer was found
    after every attempt. `source` is best-effort provenance for
    debugging only: the flattened top-level field name the value was
    also found under verbatim, if any, else "md_content" when the
    value was only found in rendered prose/a table cell with no single
    matching field. Fails silent, never raises -- callers should fall
    back to a synthetic doc_id.
    """
    if not isinstance(doc, dict):
        return None, None

    md_content = _json_to_markdown(doc, title=doc.get("document_type") or api_doc_type)
    if not md_content or not md_content.strip():
        return None, None

    lookup_source = _flatten_one_level(doc)
    normalized_md = re.sub(r"\s+", " ", md_content).lower()

    def _build_prompt(attempt: int) -> str:
        # api_doc_type can be None here (classification hasn't resolved,
        # or never will -- see _submit_single_document): phrase the
        # prompt naturally either way rather than literally embedding
        # the word "None", which would read as a real (wrong) category.
        doc_type_phrase = f'classified as "{api_doc_type}"' if api_doc_type else "of an unknown/unclassified type"
        base = (
            f"Below is the rendered content of a document {doc_type_phrase}. "
            f"Identify the ONE value that is this "
            f"document's OWN unique reference/identifier number, exactly "
            f"as printed on the document itself -- e.g. the invoice "
            f"number on an Invoice, the PO number on a Purchase Order, "
            f"the DO number on a Delivery Order, the GRN number on a "
            f"Goods Receipt Note. This must be the number that uniquely "
            f"identifies THIS specific document -- NOT a customer "
            f"number, vendor number, account number, or a reference to a "
            f"DIFFERENT document (e.g. an Invoice's own number, never the "
            f"PO number it references).\n\n"
            f"Document content:\n{md_content}\n\n"
        )
        if attempt > 0:
            base += (
                f"A previous pass over this SAME content did not find a "
                f"confident answer, but this document is expected to have "
                f"its own identifying number somewhere in the content "
                f"above -- look again carefully, including inside any "
                f"table, before concluding none exists.\n\n"
            )
        base += (
            f"Respond with ONLY a JSON object: {{\"value\": \"...\"}}, or "
            f'{{"value": null}} if truly no such number is present. No '
            f"prose, no markdown fences, just the JSON object."
        )
        return base

    for attempt in range(max_attempts):
        try:
            raw = _call_ollama(_build_prompt(attempt), max(timeout_secs * 0.2, 10))
            result, _ = _try_parse_json(raw)
        except Exception:
            continue  # transient call/parse failure -- worth another attempt

        if not isinstance(result, dict):
            continue
        value = result.get("value")
        if not (isinstance(value, str) and value.strip()):
            continue  # model said null, or gave a non-string answer
        value = value.strip()

        normalized_value = re.sub(r"\s+", " ", value).lower()
        if normalized_value not in normalized_md:
            continue  # not grounded in what was actually rendered

        # Best-effort provenance only -- doc_id assignment doesn't
        # depend on finding one, this is purely for debugging via
        # doc_id_reference_field in the output.
        source_field = None
        for key, val in lookup_source.items():
            if val not in (None, "") and str(val).strip() == value:
                source_field = key
                break

        return (source_field or "md_content"), value

    return None, None


def _structured_has_pricing_values(structured: dict, timeout_secs: float = 10) -> "bool | None":
    """
    Asks the model whether the ALREADY-EXTRACTED structured JSON (not
    the raw OCR source) carries genuine monetary/pricing values -- unit
    prices, line amounts, or a total -- as opposed to a document whose
    pricing-shaped fields are present but blank/null (e.g. a Delivery
    Note that has an "Amount SGD" column header but every cell under it
    is empty), or a document with no pricing concept at all (a Goods
    Receipt Note, Purchase Requisition, etc.).

    Deliberately model-judged rather than a hardcoded field-name check
    (e.g. `structured.get("amount_sgd")`) -- this codebase's extraction
    model is free-running text generation, not a fixed schema (see the
    field-canonicalization notes above), so the same concept can come
    back as "amount_sgd", "amount", "line_total", "total_price", or in
    a different currency's naming entirely; a fixed key list would need
    constant upkeep and silently miss spellings not seen yet.

    Kept cheap despite being model-judged: this call only ever runs in
    the doc-type tie-break branch (see _scan_header_title_doc_type),
    not on every document, and it reasons over the small already-
    extracted JSON object rather than the noisy raw OCR text -- a much
    smaller, more constrained prompt than re-deriving document_type
    from scratch would be.

    Returns True/False, or None on any failure -- callers should NOT
    assume a default on None; fall back to whatever the header-scan/
    model document_type already decided instead of guessing.
    """
    try:
        prompt = (
            "Below is a JSON object extracted from a business document. "
            "Does it contain at least one GENUINE monetary/pricing value "
            "-- a non-blank unit price, line amount, or total -- "
            "anywhere in the object, including inside any line-item "
            "list? A pricing-shaped field that exists but is null, "
            "empty, or zero everywhere does NOT count as having a "
            "pricing value. Respond with ONLY one word: yes or no.\n\n"
            + json.dumps(structured, ensure_ascii=False)
        )
        answer = _call_ollama(prompt, timeout_secs)
    except Exception:
        return None

    answer = answer.strip().lower()
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    return None  # unparseable/ambiguous answer -- don't guess


def _slugify_doc_id_suffix(value: str) -> str:
    """Keeps only letters/digits/dashes from a reference number so it's
    safe to use inside a doc_id (e.g. "PSINV-B10596837" stays as-is;
    "SGD 1,234.56"-style values would collapse oddly, which is fine --
    a genuine reference number never looks like that in the first
    place)."""
    slug = re.sub(r"[^A-Za-z0-9-]", "", value).strip("-")
    return slug.upper() if slug else ""


def _next_ap_doc_sequence(prefix: str) -> int:
    """Absolute-last-resort fallback: returns the next integer sequence
    for `prefix`, persisted in AP_DOC_COUNTER_FILE across restarts,
    keyed by "<prefix>-<year>" so sequences restart at 1 each new year.
    See _assign_ap_doc_id's docstring and the comment above
    AP_DOC_COUNTER_FILE for why this is only reached when every
    deterministic AND model-based attempt to find the document's own
    number has already failed."""
    key = f"{prefix}-{datetime.now().year}"
    with _ap_doc_id_lock:
        counters = {}
        if os.path.exists(AP_DOC_COUNTER_FILE):
            try:
                with open(AP_DOC_COUNTER_FILE, "r") as f:
                    counters = json.load(f)
            except Exception:
                counters = {}
        next_seq = int(counters.get(key, 0)) + 1
        counters[key] = next_seq
        try:
            with open(AP_DOC_COUNTER_FILE, "w") as f:
                json.dump(counters, f)
        except Exception as e:
            print(f"WARN failed to persist AP doc counter file: {e}")
    return next_seq


def _assign_ap_doc_id(api_doc_type: str, parsed: dict, upload_id: "str | None", timeout_secs: float = 20) -> dict:
    """
    Assigns a doc_id, in priority order:

    1. The document's OWN printed reference number, used AS-IS -- no
       synthetic "<PREFIX>-<YEAR>-" wrapper. Found via
       _find_document_reference_from_md: a model-based lookup (with
       retries) that reads this document's own RENDERED MARKDOWN
       content -- the same thing a human reviewer would look at --
       rather than pattern-matching field NAMES against a fixed alias
       list. This is deliberately schema-agnostic: it doesn't matter
       what key the extraction happened to store the value under or
       how it's nested, only what the content itself says identifies
       the document. The backend's integration guide's stated
       convention ("<TYPE_PREFIX>-<YEAR>-<sequence>", e.g.
       "INV-2026-0456") was written for a meaningless SEQUENCE number,
       where the prefix+year genuinely carries information; once the
       suffix is instead a real reference number a human would
       recognize from the document itself (e.g. "PSINV-B10596837"),
       wrapping it in a synthetic prefix only makes it harder to match
       back to what's printed on the paper, for no real benefit. This
       is preferred over any synthetic ID since it's the ID a human
       would actually recognize, and resubmitting the same document
       naturally yields the same doc_id without needing any separate
       idempotency bookkeeping. This is expected to be the path taken
       for essentially every real document -- an AP document always
       carries its own reference number somewhere in its content.
    2. A hash of `upload_id`, wrapped in "<PREFIX>-<YEAR>-" (this
       synthetic suffix has no inherent meaning, so the wrapper earns
       its keep here), cached against upload_id so the SAME upload
       always resolves to the SAME doc_id across repeated runs -- used
       only if step 1 came back empty after every retry.
    3. A running per-prefix sequence (_next_ap_doc_sequence) -- the
       absolute last resort, reached only when step 1 found nothing
       AND there's no upload_id to hash in step 2. Carries a running
       counter's usual caveats (no idempotency across reprocessing
       runs, no protection against duplicate hand-outs across multiple
       processes/replicas) -- see the comment above
       AP_DOC_COUNTER_FILE -- but by the time execution reaches here,
       the model has already tried (with retries) to read the
       document's own printed number directly off its content and
       failed, so this step is expected to be rare.

    Returns {"doc_id": str, "source": "reference_number"|"upload_hash"|
    "sequence", "reference_field": str|None}.
    """
    # api_doc_type can genuinely be None here now -- doc_id assignment
    # runs even when AP doc_type classification hasn't resolved yet or
    # never will (see this function's docstring). dict.get evaluates
    # its default argument eagerly regardless of whether the key is
    # found, so a naive `.get(api_doc_type, api_doc_type.upper()[:3])`
    # crashes on None before .get ever gets a chance to return -- the
    # prefix only matters for the synthetic-ID fallback formats
    # anyway (upload_hash/sequence), so an unclassified document just
    # gets a generic "DOC" prefix there.
    prefix = AP_DOC_TYPE_PREFIXES.get(api_doc_type, api_doc_type.upper()[:3] if api_doc_type else "DOC")

    field, value = _find_document_reference_from_md(parsed, api_doc_type, timeout_secs)
    if field and value:
        suffix = _slugify_doc_id_suffix(value)
        if suffix:
            return {
                "doc_id": suffix,
                "source": "reference_number",
                "reference_field": field,
            }

    if upload_id:
        cache_key = f"{prefix}:{upload_id}"
        with _ap_doc_id_lock:
            cache = {}
            if os.path.exists(AP_DOC_ID_CACHE_FILE):
                try:
                    with open(AP_DOC_ID_CACHE_FILE, "r") as f:
                        cache = json.load(f)
                except Exception:
                    cache = {}
            if cache_key in cache:
                return {"doc_id": cache[cache_key], "source": "upload_hash", "reference_field": None}
            hash_suffix = hashlib.sha256(upload_id.encode("utf-8")).hexdigest()[:8].upper()
            doc_id = f"{prefix}-{datetime.now().year}-{hash_suffix}"
            cache[cache_key] = doc_id
            try:
                with open(AP_DOC_ID_CACHE_FILE, "w") as f:
                    json.dump(cache, f)
            except Exception as e:
                print(f"WARN failed to persist AP doc_id cache file: {e}")
        return {"doc_id": doc_id, "source": "upload_hash", "reference_field": None}

    # Absolute last resort: neither inference attempt found the
    # document's own number, and there's no upload_id to hash either.
    seq = _next_ap_doc_sequence(prefix)
    return {
        "doc_id": f"{prefix}-{datetime.now().year}-{seq:04d}",
        "source": "sequence",
        "reference_field": None,
    }


def _submit_to_ap_store(api_doc_type: str, doc_id: str, structured: dict, upload_id: "str | None") -> dict:
    """
    POSTs the extracted document to the AP Document Store. Never
    raises -- failures are caught and returned as a dict so a
    submission problem degrades the task result rather than failing the
    whole extraction.
    """
    if not AP_STORE_ENABLED:
        return {"status": "skipped", "reason": "AP_STORE_BASE_URL/AP_STORE_API_KEY not configured"}

    url = f"{AP_STORE_BASE_URL}/api/documents/{api_doc_type}"
    headers = {"Authorization": f"Bearer {AP_STORE_API_KEY}"}
    # The API requires `doc_id` and `doc_type` (exact API enum string)
    # INSIDE the metadata body itself -- confirmed by testing: submitting
    # without a literal `doc_type` key 400s with "expected \"PO\"".
    body = dict(structured)
    body["doc_id"] = doc_id
    body["doc_type"] = api_doc_type

    # multipart/form-data is required by the API -- `files=` with
    # (None, value) tuples is what forces `requests` to build genuine
    # multipart instead of x-www-form-urlencoded (confirmed by testing:
    # using `data=` instead silently sent url-encoded, which the
    # backend's multipart parser never saw at all).
    files = {"metadata": (None, json.dumps(body))}
    if upload_id:
        files["originalUploadId"] = (None, upload_id)

    print(f"DEBUG AP_STORE POST url={url}")
    print(f"DEBUG AP_STORE outgoing metadata body:\n{json.dumps(body, indent=2, ensure_ascii=False)}")
    if upload_id:
        print(f"DEBUG AP_STORE outgoing originalUploadId: {upload_id}")

    try:
        resp = _req.post(url, headers=headers, files=files, timeout=30)
    except Exception as e:
        return {"status": "error", "reason": f"request failed: {e}"}

    try:
        body_resp = resp.json()
    except Exception:
        body_resp = {"raw": resp.text[:2000]}

    if resp.status_code == 201:
        return {
            "status": "submitted",
            "http_status": 201,
            "needs_review": body_resp.get("needs_review", False),
            "review_reasons": body_resp.get("review_reasons", []),
            "conflict_status": body_resp.get("conflict_status"),
            "conflicts_with": body_resp.get("conflicts_with"),
            "attachment_error": body_resp.get("attachment_error"),
            "submission_path": body_resp.get("submission"),
        }
    if resp.status_code == 400:
        return {"status": "validation_failed", "http_status": 400, "details": body_resp.get("details", body_resp)}
    if resp.status_code == 401:
        return {"status": "unauthorized", "http_status": 401, "reason": "check AP_STORE_API_KEY"}
    return {"status": "error", "http_status": resp.status_code, "body": body_resp}


def _json_to_markdown(d: dict, title: str = None) -> str:
    """
    Renders a structured extraction dict as markdown -- this is the
    inverse of the OCR-MD -> JSON extraction step, used to build the
    "MD_Version" attached to AP Store submissions (see
    AP_STORE_INCLUDE_MD_VERSION). NOT the original OCR source text --
    that's `source_text` elsewhere in this file.

    Scalars become "**key:** value" lines, lists of dicts (e.g.
    line_items) become a markdown table (columns = union of keys, in
    first-seen order), lists of scalars become bullet lists, and
    nested dicts become their own "###" subsections. Each of these is
    emitted as one block; blocks are joined with blank lines, but
    lines WITHIN a table or list are kept on consecutive lines so the
    markdown table/list syntax doesn't break.
    """
    def _fmt_scalar(v):
        if v is None:
            return ""
        return str(v)

    def _render_table(rows: list) -> str:
        columns = []
        for row in rows:
            if isinstance(row, dict):
                for k in row.keys():
                    if k not in columns:
                        columns.append(k)
        if not columns:
            return "\n".join(f"- {_fmt_scalar(v)}" for v in rows)
        lines = ["| " + " | ".join(columns) + " |",
                  "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            if isinstance(row, dict):
                cells = [_fmt_scalar(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns]
            else:
                cells = [_fmt_scalar(row)] + [""] * (len(columns) - 1)
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _render_dict(obj: dict, heading_level: int) -> list:
        blocks = []
        for key, value in obj.items():
            label = key.replace("_", " ").strip()
            if isinstance(value, dict):
                blocks.append(f"{'#' * heading_level} {label}")
                blocks.extend(_render_dict(value, min(heading_level + 1, 6)))
            elif isinstance(value, list):
                blocks.append(f"{'#' * heading_level} {label}")
                blocks.append(_render_table(value))
            else:
                scalar = _fmt_scalar(value)
                if scalar != "":
                    blocks.append(f"**{label}:** {scalar}")
        return blocks

    body = []
    if title:
        body.append(f"# {title}")
    body.extend(_render_dict(d, heading_level=2))
    return "\n\n".join(body)


def _save_md_version_locally(doc_id: str, md_text: str) -> "str | None":
    """
    Writes `md_text` to AP_STORE_MD_VERSION_DIR/<doc_id>.md so it can be
    inspected on disk regardless of whether the AP Document Store
    submission itself succeeds (e.g. while the tunnel/endpoint is
    down). Never raises -- a local-debug convenience shouldn't be able
    to fail an extraction. Returns the path written, or None on
    failure/if skipped.
    """
    try:
        os.makedirs(AP_STORE_MD_VERSION_DIR, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id or "unknown_doc_id")
        path = os.path.join(AP_STORE_MD_VERSION_DIR, f"{safe_name}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_text)
        return path
    except Exception as e:
        print(f"WARN failed to save MD_Version locally for doc_id={doc_id!r}: {e}")
        return None


def _extract_sub_documents(parsed: dict) -> list:
    """
    Detects the case where a single OCR'd source (e.g. one attachment
    covering multiple pages) actually contains MULTIPLE distinct
    documents -- e.g. a Delivery Order on page 1 with its own Goods
    Receipt Note on page 2, extracted into separate nested objects
    (parsed["delivery_order"], parsed["goods_receipt_note"]) rather
    than a single top-level document_type.

    Returns a list of (key, merged_dict) tuples, one per detected
    sub-document. `merged_dict` is the sub-document's own fields
    backfilled with any top-level fields from `parsed` that aren't
    themselves another sub-document (attachment_id, source, engines,
    mean_confidence, extracted_at, source_backfill, etc.) -- shared OCR
    metadata that would otherwise only live on the un-submittable
    top-level object. The sub-document's own fields always win on a
    name clash (e.g. its own more specific "date").

    Returns [] if `parsed` isn't multi-document shaped: it already has
    its own top-level document_type (single-document case, handled
    separately), or none of its nested dict values look like a
    document.
    """
    if not isinstance(parsed, dict):
        return []
    top_level_doc_type = parsed.get("document_type")
    if isinstance(top_level_doc_type, str) and top_level_doc_type.strip():
        return []

    doc_dict_keys = {
        k for k, v in parsed.items()
        if isinstance(v, dict) and isinstance(v.get("document_type"), str) and v.get("document_type").strip()
    }
    if not doc_dict_keys:
        return []

    # Everything that ISN'T one of the sub-document dicts is shared
    # context -- deliberately excludes the other sibling sub-documents
    # so e.g. the Delivery Order submission doesn't end up carrying a
    # full copy of the Goods Receipt Note's fields, and vice versa.
    shared_fields = {k: v for k, v in parsed.items() if k not in doc_dict_keys}

    sub_documents = []
    for key in doc_dict_keys:
        merged = dict(shared_fields)
        merged.update(parsed[key])
        sub_documents.append((key, merged))
    return sub_documents


def _submit_single_document(doc: dict, source_text: str, timeout_secs: float = 20, trust_own_type: bool = False) -> dict:
    """
    Classifies, doc_id-assigns, canonicalizes, and submits ONE
    already-extracted document object (`doc` must have its own
    top-level `document_type`). This is the body previously inlined in
    `_finalize_ap_submission` -- pulled out so it can be called once
    per sub-document when a single OCR source turns out to contain
    multiple documents (see `_extract_sub_documents`).

    `trust_own_type`: set True when `doc` is ONE of several sub-
    documents sharing a single multi-page `source_text` (the
    _extract_sub_documents case). In that case `source_text` spans
    EVERY sub-document's pages, not just this one -- confirmed
    production failure: a Goods Receipt Note sub-document's tie-break
    saw a "Delivery Note" title AND an "Invoice No." field that both
    actually belonged to the OTHER sub-document's page, and resolved
    to the wrong type as a result. Each sub-document's own
    `document_type` field, by contrast, WAS assigned per-page during
    multi-page extraction (see the "Identify what KIND of document
    THIS page is" instruction in the page-chunking prompt) and is
    already properly scoped -- so when `trust_own_type` is set, that
    mapped type is used directly and the header-title/whole-text scans
    (which read the shared, unscoped `source_text`) are skipped
    entirely rather than risking exactly this kind of cross-document
    contamination.
    """
    model_api_doc_type = _map_to_ap_doc_type(doc.get("document_type"))

    # doc_id assignment happens FIRST, independent of whether AP
    # doc_type classification below succeeds. A document's own printed
    # reference number is a property of its CONTENT (see
    # _find_document_reference_from_md, which reads the rendered
    # document directly, not any doc_type-specific field convention) --
    # it has nothing to do with which AP Document Store bucket the
    # document eventually gets filed under. Confirmed production miss
    # from doing this the other way around: a document whose own
    # document_type classification failed entirely (header-title scan
    # found nothing usable, and the model's own extraction that run
    # happened to omit "document_type"/produce something unmappable)
    # hit the "could not map document_type" skip below and returned
    # immediately -- BEFORE doc_id was ever assigned -- even though the
    # document's own reference number ("document_number": "SIN26-02247")
    # was sitting right there in the successfully-extracted fields the
    # whole time. AP submission genuinely can't proceed without knowing
    # the doc_type (there's no bucket to file it under), but doc_id
    # resolution has no such dependency and shouldn't be silently
    # skipped along with it.
    upload_id = _extract_upload_id(source_text)
    doc_id_info = _assign_ap_doc_id(model_api_doc_type, doc, upload_id, timeout_secs)
    doc_id = doc_id_info["doc_id"]

    # doc_id is a must-have field on the structured JSON itself, same
    # guarantee _add_canonical_ap_fields gives po_ref/invoice_ref/etc.:
    # a downstream consumer checking for the key's PRESENCE should
    # always find it, regardless of which spelling this run's
    # extraction happened to produce it under, or which of
    # _assign_ap_doc_id's fallback steps ultimately resolved it, or
    # whether AP doc_type classification below succeeds at all. Same
    # "never overwrite an existing non-empty scalar" rule as the
    # canonical fields -- if the extraction already happened to
    # produce a literal "doc_id" key (rare, but possible), that value
    # is left alone rather than replaced.
    if doc.get("doc_id") in (None, "") or isinstance(doc.get("doc_id"), (dict, list)):
        doc["doc_id"] = doc_id

    if trust_own_type:
        if not model_api_doc_type:
            return {
                "status": "skipped",
                "reason": f"could not map document_type {doc.get('document_type')!r} to an AP Document Store doc_type",
                "doc_id": doc_id,
                "doc_id_source": doc_id_info["source"],
                "doc_id_reference_field": doc_id_info["reference_field"],
                "api_doc_type": None,
            }
        api_doc_type = model_api_doc_type
        doc_type_check = {"source": "sub_document_own_type", "model_said": model_api_doc_type}
    else:
        # STEP 1 -- cheap, zero-model-cost header-title lookup, checked
        # FIRST, before ever consulting the model's own document_type
        # guess. When the document's printed header/title unambiguously
        # names a single AP doc_type, that fact is used directly and the
        # model's guess is only kept around for logging/comparison -- no
        # extra model call is spent re-deriving something already stated
        # on the page. Only safe/meaningful for a single-document
        # source_text -- see `trust_own_type` above for why sub-
        # documents skip this branch entirely.
        header_doc_type = _scan_header_title_doc_type(source_text)

        if header_doc_type:
            api_doc_type = header_doc_type
            doc_type_check = {
                "source": "header_title_literal",
                "header_said": header_doc_type,
                "model_said": model_api_doc_type,
            }
            if model_api_doc_type and model_api_doc_type != header_doc_type:
                doc_type_check["source"] = "header_title_literal_overrode_model"
                print(
                    f"INFO AP doc_type: header title says {header_doc_type!r}, "
                    f"model said {model_api_doc_type!r} -- using header title {header_doc_type!r}"
                )
        elif not model_api_doc_type:
            return {
                "status": "skipped",
                "reason": f"could not map document_type {doc.get('document_type')!r} to an AP Document Store doc_type",
                "doc_id": doc_id,
                "doc_id_source": doc_id_info["source"],
                "doc_id_reference_field": doc_id_info["reference_field"],
                "api_doc_type": None,
            }
        else:
            # STEP 2 -- header title was empty/unmatched/ambiguous (2+
            # candidates in the header region, e.g. a form titled
            # "Delivery Note" that also carries its own "Invoice No."
            # cross-reference field). Fall back to the model's own guess,
            # cross-checked against a scan of the WHOLE document text
            # (broader/less precise than the header-only scan above, kept
            # as the existing secondary safety net).
            source_scan_matches = _scan_source_text_doc_types(source_text)
            api_doc_type = model_api_doc_type
            doc_type_check = {"source": "model", "model_said": model_api_doc_type}

            if len(source_scan_matches) == 1:
                source_said = source_scan_matches[0]
                if source_said == model_api_doc_type:
                    doc_type_check = {"source": "model_confirmed_by_source_text", "model_said": model_api_doc_type}
                else:
                    api_doc_type = source_said
                    doc_type_check = {
                        "source": "overridden_by_source_text",
                        "model_said": model_api_doc_type,
                        "source_text_said": source_said,
                    }
                    print(
                        f"WARN AP doc_type override: model said {model_api_doc_type!r}, "
                        f"source text header says {source_said!r} -- using {source_said!r}"
                    )
            elif len(source_scan_matches) > 1:
                # STEP 3 -- genuinely hybrid document (both a delivery/DO-
                # type phrase and an "invoice" phrase appear, as with the
                # Nespresso delivery-note-with-invoice-cross-reference
                # case). Rather than leave this as an unresolved hedge,
                # break the tie with a model judgment over the small
                # ALREADY-EXTRACTED structured JSON (see
                # _structured_has_pricing_values): a document carrying no
                # genuine priced values is functionally a delivery/goods-
                # movement record, not an Invoice, regardless of which
                # reference numbers it happens to also carry.
                #
                # Priority order when resolving which candidate to
                # pick: GRN is checked ahead of DO/Invoice when present,
                # since a Goods Receipt Note is the most specific type
                # of the three and its own keyword match is unambiguous
                # by construction (see _AP_DOC_TYPE_SOURCE_TEXT_KEYWORDS)
                # -- there's no analogous "does it have pricing" signal
                # that would ever need to override an explicit GRN
                # title match.
                has_pricing = _structured_has_pricing_values(doc, timeout_secs=max(timeout_secs * 0.15, 8))
                if "GRN" in source_scan_matches:
                    api_doc_type = "GRN"
                    doc_type_check = {
                        "source": "ambiguous_source_text_resolved_by_grn_priority",
                        "model_said": model_api_doc_type,
                        "source_text_candidates": source_scan_matches,
                        "has_pricing_values": has_pricing,
                    }
                elif has_pricing is False and "DO" in source_scan_matches:
                    api_doc_type = "DO"
                    doc_type_check = {
                        "source": "ambiguous_source_text_resolved_by_pricing_check",
                        "model_said": model_api_doc_type,
                        "source_text_candidates": source_scan_matches,
                        "has_pricing_values": False,
                    }
                    print(
                        f"INFO AP doc_type: ambiguous header candidates "
                        f"{source_scan_matches} with no genuine pricing values found "
                        f"-- resolving to DO instead of model's {model_api_doc_type!r}"
                    )
                elif has_pricing is True and "Invoice" in source_scan_matches and "DO" in source_scan_matches:
                    # Confirmed production failure without the "DO" in
                    # source_scan_matches" guard: this branch used to
                    # fire for ANY ambiguous set containing "Invoice",
                    # including {PO, Invoice} -- but a Purchase Order
                    # legitimately, routinely carries unit costs/amounts
                    # too (it's ordering priced goods), so "has pricing"
                    # is not evidence of Invoice over PO the way it is
                    # evidence of Invoice over DO (a delivery note
                    # structurally has no prices at all). A real PO
                    # -- header literally printed "PURCHASE ORDER",
                    # model's own guess correctly said "PO" -- got
                    # overridden to "Invoice" purely because it had
                    # prices, which every PO does. This tie-break is
                    # only valid for the DO-vs-Invoice pair it was
                    # designed for; any other candidate pairing falls
                    # through to the else below and keeps the model's
                    # own (already doc_type-aware) guess instead.
                    api_doc_type = "Invoice"
                    doc_type_check = {
                        "source": "ambiguous_source_text_resolved_by_pricing_check",
                        "model_said": model_api_doc_type,
                        "source_text_candidates": source_scan_matches,
                        "has_pricing_values": True,
                    }
                else:
                    # Pricing check was inconclusive, or the candidate
                    # pair isn't one this heuristic can safely
                    # discriminate (e.g. {PO, Invoice}, {Quotation,
                    # Invoice} -- both legitimately have pricing) --
                    # stay with the existing "flag, don't auto-resolve"
                    # behavior. `api_doc_type` was already defaulted to
                    # `model_api_doc_type` above, so this correctly
                    # keeps the model's own guess rather than forcing
                    # a resolution the available signals don't support.
                    doc_type_check = {
                        "source": "model_ambiguous_source_text",
                        "model_said": model_api_doc_type,
                        "source_text_candidates": source_scan_matches,
                        "has_pricing_values": has_pricing,
                    }

    # Adds the external-facing canonical field names for this
    # api_doc_type (see _AP_DOC_TYPE_CANONICAL_FIELDS), mutating `doc`
    # in place. For a single-document source this IS the same object
    # returned to the caller as `outputs.structured`, so the added
    # fields are visible there too, not just in the submission payload
    # built from it below. For a multi-document sub-document, `doc` is
    # a merged COPY (see _extract_sub_documents), so this mutation
    # reaches the submission payload but needs an explicit write-back
    # in _finalize_ap_submission to also reach the parent's `structured`
    # -- see the sub_documents loop there.
    _add_canonical_ap_fields(doc, api_doc_type, timeout_secs)

    submission_payload = _canonicalize_field_names(doc)
    md_version_local_path = None
    if AP_STORE_INCLUDE_MD_VERSION:
        # Shallow-copy first -- _canonicalize_field_names already returns
        # a fresh dict, but don't rely on that not changing later.
        submission_payload = dict(submission_payload)
        submission_payload["MD_Version"] = _json_to_markdown(
            submission_payload, title=doc.get("document_type")
        )
        if AP_STORE_SAVE_MD_VERSION_LOCALLY:
            # Written BEFORE the submit call so it's on disk even if
            # AP_STORE_ENABLED is False or the POST fails/times out.
            # Filed under this sub-document's OWN doc_id (e.g.
            # GRN-B10096605.md), never the parent source's.
            md_version_local_path = _save_md_version_locally(
                doc_id, submission_payload["MD_Version"]
            )

    result = _submit_to_ap_store(api_doc_type, doc_id, submission_payload, upload_id)
    if md_version_local_path:
        result["md_version_local_path"] = md_version_local_path
    result["doc_id"] = doc_id
    result["doc_id_source"] = doc_id_info["source"]
    result["doc_id_reference_field"] = doc_id_info["reference_field"]
    result["api_doc_type"] = api_doc_type
    result["doc_type_check"] = doc_type_check
    result["upload_id_used"] = upload_id
    return result


def _finalize_ap_submission(parsed: dict, source_text: str, timeout_secs: float = 20) -> dict:
    """
    Shared finishing step for all JSON-extraction paths. Two shapes:

    1. SINGLE document -- `parsed` has its own top-level document_type
       (the common case): classifies doc_type (with a source-text
       cross-check), assigns a doc_id (preferring the document's own
       printed reference number over any synthetic ID), recovers the
       upload_id, canonicalizes field-name variants, and submits.
       Delegates to `_submit_single_document`.

    2. MULTIPLE documents -- `parsed` has NO top-level document_type,
       but one or more nested dict values each have their own (e.g.
       parsed["delivery_order"] and parsed["goods_receipt_note"] from
       a single combined OCR page/attachment): each is submitted
       independently via `_submit_single_document`, and results are
       aggregated under "documents". Previously this shape was
       silently skipped ("could not map document_type None") because
       only the top level was ever checked -- see
       `_extract_sub_documents`.

    Never mutates `parsed`. No-op (skipped) if `parsed` isn't a dict,
    or has neither a usable top-level document_type nor any detectable
    sub-documents.
    """
    if not isinstance(parsed, dict):
        return {"status": "skipped", "reason": "no structured object to submit"}

    top_level_doc_type = parsed.get("document_type")
    if isinstance(top_level_doc_type, str) and top_level_doc_type.strip():
        return _submit_single_document(parsed, source_text, timeout_secs)

    sub_documents = _extract_sub_documents(parsed)
    if not sub_documents:
        return {
            "status": "skipped",
            "reason": f"could not map document_type {top_level_doc_type!r} to an AP Document Store doc_type",
        }

    results = []
    for key, doc in sub_documents:
        result = _submit_single_document(doc, source_text, timeout_secs, trust_own_type=True)
        result["source_key"] = key
        results.append(result)

        # `doc` here is a MERGED COPY (see _extract_sub_documents), not
        # the same object as `parsed[key]` -- so the canonical fields
        # (and doc_id) _submit_single_document just added onto `doc` in
        # place don't automatically reach `parsed`, and by extension
        # `outputs.structured`. Write them back onto the real
        # sub-document explicitly, respecting the same "don't overwrite
        # a non-empty existing value" rule (parsed[key] may itself
        # already hold a correct value under that exact name).
        resolved_api_type = result.get("api_doc_type")
        target_fields = list(_AP_DOC_TYPE_CANONICAL_FIELDS.get(resolved_api_type, []))
        # doc_id is a must-have field the same way the canonical fields
        # are (see _submit_single_document) -- write it back here too,
        # regardless of api_doc_type, since every doc_type needs one.
        target_fields.append("doc_id")
        sub_doc = parsed.get(key)
        if target_fields and isinstance(sub_doc, dict):
            for field in target_fields:
                if sub_doc.get(field) in (None, "") and doc.get(field) not in (None, ""):
                    sub_doc[field] = doc[field]
                elif field not in sub_doc:
                    sub_doc[field] = doc.get(field)

    submitted_doc_ids = [r["doc_id"] for r in results if r.get("doc_id")]
    submitted_api_doc_types = [r["api_doc_type"] for r in results if r.get("api_doc_type")]
    statuses = {r.get("status") for r in results}
    if statuses == {"submitted"}:
        overall_status = "submitted"
    elif "submitted" in statuses:
        overall_status = "multi_document_partial"
    else:
        overall_status = "error"

    return {
        "status": overall_status,
        "reason": f"source contained {len(results)} separate documents -- see 'documents'",
        "document_count": len(results),
        "documents": results,
        # Convenience roll-ups for callers/logging that only look at
        # the top-level doc_id/api_doc_type fields -- the authoritative
        # per-document detail is always in "documents".
        "doc_id": ", ".join(submitted_doc_ids) if submitted_doc_ids else None,
        "api_doc_type": ", ".join(submitted_api_doc_types) if submitted_api_doc_types else None,
    }


def verify_vllm_ready():
    """
    Startup gate: confirms vLLM is reachable and the configured model is
    loaded before this service starts accepting requests.

    Replaces a legacy verify_ollama_ready() that checked real Ollama at
    OLLAMA_BASE_URL (http://localhost:11434) via its /api/tags endpoint.
    That check predates this codebase's migration from Ollama to vLLM
    (see platform history) -- Ollama was later fully stopped and
    disabled on the GX10, but this startup check was never updated to
    match, so every boot hit a hard RuntimeError against a service that
    was never coming back, putting byoa-summarize6 into an unbounded
    systemd restart loop (every request 502ing the whole time, since
    the process crashes before it ever binds its port). _call_ollama
    itself has long since pointed at vLLM's OpenAI-compatible endpoint
    (VLLM_BASE_URL) -- this just brings the startup check in line with
    what the service actually depends on at runtime.

    Uses vLLM's OpenAI-compatible /v1/models endpoint (equivalent to
    Ollama's /api/tags: lists what's currently loaded) rather than
    issuing a real chat completion, so this stays a cheap, side-effect-
    free readiness probe like the check it replaces.
    """
    try:
        r = _req.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"vLLM not reachable at {VLLM_BASE_URL} -- is the vLLM server/container "
            f"running? ({e})"
        )
    model_ids = [m.get("id") for m in r.json().get("data", [])]
    if VLLM_MODEL not in model_ids:
        raise RuntimeError(
            f"Model {VLLM_MODEL!r} is not currently served by vLLM at {VLLM_BASE_URL}.\n"
            f"Currently served: {model_ids}"
        )
    print(f"OK vLLM reachable, model {VLLM_MODEL!r} present.")


# Matches a trailing "CHECK: complete" / "CHECK: incomplete" marker line that
# the model is asked to append to its own draft summary -- this is how the
# agent self-checks against the request/goal (`focus`) without any extra
# tool calls: the same LLM call that produces the draft also judges it.
_CHECK_MARKER_RE = re.compile(r"\n?CHECK:\s*(complete|incomplete)\s*$", re.IGNORECASE)


def _parse_check_marker(draft: str):
    """
    Splits the trailing CHECK marker off a draft summary.
    Returns (clean_summary, incomplete: bool). If no marker is present
    (e.g. no focus was given, so none was requested), treats it as complete.
    """
    match = _CHECK_MARKER_RE.search(draft)
    if not match:
        return draft.strip(), False
    clean = _CHECK_MARKER_RE.sub("", draft).strip()
    return clean, match.group(1).lower() == "incomplete"


# Cheap heuristics deciding whether list_instruction / extraction_override
# are worth attaching to this call's prompt -- see PERFORMANCE NOTE above.
# Deliberately generous: when genuinely uncertain, these lean toward
# INCLUDING the instruction rather than skipping it, since a missed case
# would be a real capability regression, while an unnecessary inclusion is
# just a few hundred extra tokens on that one call.
_LIST_CUES_RE = re.compile(
    r"\b(list|listing|top\s*\d|rank|enumerate|options|reasons|competitors|name the|"
    r"name all|exactly\s+\d+|\d+\s+(distinct|different|key|main|separate)|several|multiple)\b",
    re.IGNORECASE,
)
_EXTRACTION_CUES_RE = re.compile(
    r"\b(identify|name only|extract|which\b|what is the|respond with only)\b",
    re.IGNORECASE,
)
_EXTRACTION_DISQUALIFIERS_RE = re.compile(
    r"\b(assess|evaluate|describe|explain|report|whether|how is|how good|how well|why|should|and)\b",
    re.IGNORECASE,
)
# An explicit prose/length constraint in the focus (e.g. "in 4-5 sentences",
# "in one paragraph") means the requester has already told us the intended
# shape of the answer -- a list-cue word elsewhere in that SAME sentence
# (e.g. "reasons", which is a literal _LIST_CUES_RE trigger) should not
# override an explicit instruction the requester actually wrote. Motivating
# case: focus = "Briefly explain the main economic reasons for X in 4-5
# sentences" contains "reasons" (a list cue) AND an explicit 4-5-sentence
# instruction in the same breath -- without this override, list mode won.
_PROSE_FORMAT_OVERRIDE_RE = re.compile(
    r"\b\d+[\s-]*(?:to|-)?[\s-]*\d*\s+sentences?\b"
    r"|\bin\s+(?:a|one)\s+(?:short\s+|brief\s+)?paragraph\b"
    r"|\bas\s+(?:a|one)\s+paragraph\b"
    r"|\b(?:a\s+)?single\s+paragraph\b"
    r"|\bone\s+paragraph\b",
    re.IGNORECASE,
)

# Splits on sentence-ending punctuation followed by whitespace -- used only
# to COUNT sentences (len() of the split), not to actually split content.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Matches a line that starts with "N." / "N)" / "N:" followed by content --
# used to VERIFY (not just hope) that a list-mode summary actually came back
# as a real numbered list, rather than trusting the model's compliance with
# the prompt instruction alone.
_NUMBERED_LIST_LINE_RE = re.compile(r"(?m)^\s*\d+[\.\)\:]\s+\S")


# ---------------------------------------------------------------------------
# Deterministic row-boundary detection for line-item tables.
#
# Some upstream OCR/MD pipelines insert a genuine blank line between
# consecutive table rows (confirmed on production data from an updated OCR
# pass) even though a single row's own content can still span multiple
# physical lines internally (e.g. a wrapped description). Where that blank-
# line convention is present, it gives a fully deterministic, document-
# agnostic signal for exactly where one row ends and the next begins --
# something no amount of prompt engineering could reliably infer, because
# the model has no way to distinguish "this wrapped continuation belongs to
# the row above" from "this begins a new row" from field content alone.
#
# This block of helpers is intentionally self-contained and makes no
# assumption about column names, header phrasing, or any value specific to
# one document or vendor -- it identifies rows purely from (a) the blank-
# line convention itself, and (b) the presence of a generic date-shaped
# token, which is common to virtually any dated line-item table (invoices,
# purchase orders, delivery orders, goods-received notes, put-away reports,
# etc.), not unique to any one document type.
# ---------------------------------------------------------------------------

_BLANK_LINE_SPLIT_RE = re.compile(r"\n[ \t]*\n+")
_GENERIC_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

# A second candidate anchor for line-item tables that DON'T repeat a date
# per row -- confirmed on real production invoice data: a tax invoice's
# date appears exactly twice, both at the document-header level ("Tax
# Invoice Date", "Order Date"), never once per line item, so the date
# anchor above finds nothing at all on this document type and safely
# falls back to whole-document extraction rather than misfiring. A
# currency amount, by contrast, appears on every real line item of that
# same invoice (a Unit Price / line Total column), making it a viable
# second anchor for exactly the document types (invoices, POs, GRNs)
# where a per-row date doesn't exist.
_GENERIC_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b")

# The amount anchor has a real false-positive mode the date anchor
# doesn't: a document's own totals/summary section (Subtotal, GST,
# Grand Total) is structurally identical to a run of real line items --
# several consecutive lines, each ending in a currency-amount-shaped
# token -- confirmed on the same real invoice sample, where a 3-line
# totals block was found by the amount anchor with the exact same shape
# as the 3 real line items above it. These are universal accounting
# vocabulary (not specific to any one vendor or document instance,
# equivalent in spirit to already relying on this OCR pipeline's own
# fixed "## Detected handwritten..." section marker elsewhere in this
# file) used specifically to tell a summary/total line apart from a
# real line item when the amount anchor is in use; never applied when
# the date anchor already found a qualifying region, since dates don't
# share this false-positive mode.
_SUMMARY_LINE_RE = re.compile(
    r"\b(?:total|subtotal|sub-total|grand\s*total|gst|vat|tax|"
    r"amount\s*due|balance\s*due)\b",
    re.IGNORECASE,
)

# A SECOND, vocabulary-free check for the same false-positive mode
# _SUMMARY_LINE_RE targets -- a totals/summary label this pipeline has
# never seen before (confirmed on production data: "Paid Credits : 0.00"
# uses none of _SUMMARY_LINE_RE's words, so it slipped through and
# became a phantom "row" with every real field null) can't be caught by
# extending a fixed word list forever. Instead of more words, this
# counts how many separate number-LOOKING fields the line actually has:
# a genuine line-item row reports several (an item code and/or qty and/or
# unit price, in addition to its final amount -- confirmed >=3 on every
# real row tested), while a totals/summary line -- REGARDLESS of what
# word is used for its label, in any language -- reports exactly ONE
# (just the total itself). A bare duplicate/repeated-amount line with NO
# label text at all (also confirmed in production: a second physical
# line printing only "54.03" again, no words at all) is caught by the
# same rule for the same reason -- one number, nothing else.
# Deliberately a MINIMUM, not an exact match -- a real row can legally
# have many more than 3 numeric-looking fields (extra reference numbers,
# a discount, etc.); a summary line essentially never has more than one.
_MIN_ROW_NUMERIC_TOKENS = 2

# Any digit-run, loosely shaped (bare integer, comma-grouped, or decimal)
# -- deliberately broader than _GENERIC_AMOUNT_RE (which only matches the
# strict "X.XX currency" shape) because this counts how many separate
# NUMBER-LOOKING fields a line has, not just currency amounts -- a row's
# own quantity ("2") or a plain item code ("111210") don't have 2 decimal
# places, but still count as a field.
_ANY_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# The OCR tool's own inline tags ("[handwriting 17]", "[artifact 9]",
# "[※14]", "[illegible]") carry digits that are NOT document data. They
# used to be counted as "number-looking fields" by the min_numeric_tokens
# check below, which let a lone totals amount (e.g. "[handwriting 18]
# ,S$1,475.33") look like a real line-item row. Stripped ONLY for that
# count -- the line text itself is never changed.
_OCR_TAG_RE = re.compile(
    r"\[(?:handwriting|artifact)\s+\d+\]|\[\u203b\d+\]|\[(?:illegible|unreadable)\]"
)

# The OCR tool appends its own review sections AFTER the page text. They
# are metadata about the scan, not part of the document, but they are full
# of lines that look like table rows (amounts + digits). Row detection must
# only ever look at the page text above them.
_OCR_TRAILER_RE = re.compile(
    r"^##\s*(?:Detected handwritten/colored annotations"
    r"|Per-page OCR confidence breakdown"
    r"|Cross-engine word disagreements)",
    re.MULTILINE,
)


def _body_for_row_detection(text: str) -> str:
    """
    Returns `text` cut off at the first OCR-tool review section, or `text`
    unchanged if there is none. Only a TAIL is removed, so every offset
    found in the result is still valid in the original `text`. Used for
    row detection only -- the model still sees the full text.
    """
    m = _OCR_TRAILER_RE.search(text)
    return text[:m.start()] if m else text


def _split_into_blank_line_blocks(text: str) -> list:
    """
    Splits `text` on blank-line boundaries, returning a list of
    (block_text, start_offset, end_offset) tuples in original document
    order with exact source positions preserved, so later steps can
    reconstruct the surrounding (non-row) text exactly rather than
    reassembling it from scratch.
    """
    blocks = []
    pos = 0
    for part in _BLANK_LINE_SPLIT_RE.split(text):
        start = text.find(part, pos)
        if start == -1:
            start = pos
        end = start + len(part)
        blocks.append((part, start, end))
        pos = end
    return blocks


def _find_table_row_blocks(text: str, max_bridge_gap: int = 2):
    """
    Identifies individual table-row text spans within `text`. Dispatches
    across candidate anchor patterns in priority order -- a date first
    (the more reliable signal when it exists: it repeats per row on
    warehouse/put-away-style documents, and essentially never produces a
    false positive), falling back to a currency-amount pattern (for
    invoices/POs/GRNs whose date only appears at the document-header
    level, confirmed on real production data -- see _GENERIC_AMOUNT_RE)
    if the date anchor finds no qualifying region at all.

    Returns whatever the first anchor that finds a qualifying region
    returns; ([], "", -1, -1) if NEITHER anchor finds one, signaling
    callers to fall back to ordinary whole-document extraction. Never
    tries a later anchor if an earlier one already succeeded -- an
    anchor that already found real rows is not second-guessed just
    because a different anchor might also match something.

    Wrapped-header reconstruction (_reconstruct_wrapped_header_columns)
    is applied ONLY on the date-anchor path, deliberately -- confirmed
    on real production data that it's unsafe on the amount-anchor path:
    for an invoice, the block preceding the localized table is often
    just letterhead/address text, never verified to actually BE a
    column-header block the way it demonstrably is for the date-anchor
    case. Running the reconstruction there produced a real false
    positive -- "LTD" (from a wrapped company name) and an unrelated
    phone number, on completely unrelated lines, coincidentally aligned
    in character position and were wrongly "reunited" as if they were
    one header. Applying this fix to content that was never confirmed
    to be a header at all is a category error, not something a tighter
    distance threshold would reliably fix.
    """
    result = _find_table_row_blocks_with_anchor(text, _GENERIC_DATE_RE, max_bridge_gap)
    if result[0]:
        row_texts, header_context, region_start, region_end = result
        header_context = _reconstruct_wrapped_header_columns(header_context)
        return row_texts, header_context, region_start, region_end
    return _find_table_row_blocks_with_anchor(
        text, _GENERIC_AMOUNT_RE, max_bridge_gap, exclude_re=_SUMMARY_LINE_RE,
        min_numeric_tokens=_MIN_ROW_NUMERIC_TOKENS,
    )


def _reconstruct_wrapped_header_columns(header_context: str) -> str:
    """
    Deterministically reunites a column header that OCR's line-by-line
    reading order split across two SEPARATE physical lines on the page
    (e.g. "Variant" printed alone on one line, with its actual pair
    "Code" landing, out of reading order, as an isolated word on a
    DIFFERENT line entirely) -- confirmed root cause on production
    data of a real extraction bug: a per-row prompt given only the raw,
    un-reunited header text had no way to know a "Variant Code" column
    existed at all, and the model repeatedly invented a bare "code"
    field and populated it with a hallucinated value pulled from
    neighboring rows, rather than correctly recognizing the wrapped
    column and outputting null for it.

    Uses CHARACTER POSITION alignment, not text proximity or word
    matching -- verified against real production data before writing
    this: an isolated word on one physical line and its true pair on a
    DIFFERENT line start at nearly the same character offset (e.g.
    "Variant" at offset 40 and its orphaned "Code" pair at offset 41 on
    a separate line, a 1-character difference -- consistent with both
    being printed in the same page column, just picked up by OCR on
    different reading passes), which is a fundamentally different, more
    reliable signal than "these two words happen to sit close together
    in the reading-order text."

    Deliberately narrow in scope: only reunites an orphan word that has
    a LARGE gap (2+ OCR whitespace-placeholder characters) on BOTH
    sides within its own line -- i.e. a word with no natural same-line
    single-gap partner at all, like the "Code" sitting isolated between
    "Item No" and "Description". A word that already has an immediate
    same-line single-gap neighbor (e.g. the "Code" directly following
    "Quantity") is left alone even if it ALSO happens to sit near a
    cross-line orphan -- resolving that specific ambiguity would risk
    changing behavior for the "quantity"/"unit_of_measure" fields,
    which have extracted correctly on every run tested; this fix
    targets only the confirmed, currently-broken case, not every
    theoretically-possible wrapped header in a document like this one.

    Returns the ORIGINAL `header_context` text with a single appended
    note describing any reunited column found, or the original text
    unchanged if no confident cross-line alignment is found -- never
    invents a reunification without a close (<=3 character) position
    match, and never touches header_context's own text otherwise.
    """
    ph = _OCR_WS_PLACEHOLDER
    lines = header_context.split("\n")

    def _line_tokens(line):
        # (start_offset, word, gap_before, gap_after) for every word in
        # this line, using RAW character positions (not collapsed) --
        # position alignment across lines only means anything if
        # measured in the original, un-collapsed coordinate system.
        tokens = []
        for m in re.finditer(f"[^{ph}]+", line):
            tokens.append([m.start(), m.group(), None, None])
        for i, tok in enumerate(tokens):
            start = tok[0]
            prev_end = tokens[i - 1][0] + len(tokens[i - 1][1]) if i > 0 else 0
            tok[2] = start - prev_end if i > 0 else None
            next_start = tokens[i + 1][0] if i + 1 < len(tokens) else None
            tok[3] = (next_start - (start + len(tok[1]))) if next_start is not None else None
        return tokens

    orphans = []  # (line_idx, start_offset, word)
    for li, line in enumerate(lines):
        for start, word, gap_before, gap_after in _line_tokens(line):
            before_ok = gap_before is None or gap_before >= 2
            after_ok = gap_after is None or gap_after >= 2
            if before_ok and after_ok:
                orphans.append((li, start, word))

    # Only cross-line pairs matter -- an "orphan" alone on its own line
    # with nothing else nearby isn't reunitable with anything.
    best_pair = None
    best_dist = 4  # exclusive upper bound: only accept a close (<=3 char) match
    for i in range(len(orphans)):
        for j in range(i + 1, len(orphans)):
            li_a, pos_a, word_a = orphans[i]
            li_b, pos_b, word_b = orphans[j]
            if li_a == li_b:
                continue
            dist = abs(pos_a - pos_b)
            if dist < best_dist:
                best_dist = dist
                best_pair = (
                    (li_a, word_a) if li_a < li_b else (li_b, word_b),
                    (li_b, word_b) if li_a < li_b else (li_a, word_a),
                )

    if best_pair is None:
        return header_context

    (first_line, first_word), (second_line, second_word) = best_pair
    reunited = f"{first_word} {second_word}"
    note = (
        f"\n\n[NOTE: \"{first_word}\" (on its own line above) and \"{second_word}\" "
        f"(appearing elsewhere, out of reading order) are two halves of ONE "
        f"wrapped column header that prints across separate physical lines: "
        f"\"{reunited}\". This is a REAL column. If this row's own text does "
        f"not show a distinct value for it, output null for it -- do NOT "
        f"reuse, split off, or guess a value from the Description or any "
        f"other column's text to fill it.]"
    )
    return header_context + note


# A value that got cut off mid-code by OCR's line-by-line reading order
# reliably ends in a trailing "-" right before the line break (confirmed
# on production data: "EWH-Q00-33P-" / "EWH-R02-33C-", each missing a
# suffix segment -- "COMW" / "COGM" -- that OCR picked up as an isolated
# word on the FOLLOWING physical line instead, because that suffix sat
# far enough right on the page to fall into the next reading pass). A
# code ending in "-" is never a real, complete value in this pipeline's
# documents -- a dash only appears there as a segment separator, and a
# segment separator with nothing after it means the next segment is
# missing, not that the code legitimately ends in a dash.
_TRAILING_DASH_CODE_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-$")


def _reconstruct_split_row_values(row_text: str) -> str:
    """
    Same underlying OCR failure, and the same character-position-
    alignment fix, as _reconstruct_wrapped_header_columns above -- just
    applied to a VALUE inside one row's own text instead of a column
    header. There, an orphaned header word had no natural same-line
    neighbor; here, the tell is different (and narrower, to avoid
    false-reuniting two values that just happen to sit near each
    other): a code-shaped token ending in a bare trailing "-" on one
    line, paired with an isolated word on a DIFFERENT line inside the
    same row's text whose start offset is a close (<=3 character)
    match -- the same tolerance used for headers, verified against
    production data there.

    Only reunites when a trailing-dash code is present at all (the
    concrete, confirmed failure shape) -- this is deliberately not a
    general "glue any two nearby orphan words together" pass, since
    that would risk merging genuinely unrelated short tokens (e.g. a
    stray annotation and a real value) that happen to land at similar
    offsets by coincidence.

    Concatenates directly with no inserted space or dash -- the
    trailing "-" already IS the segment separator, so "EWH-Q00-33P-" +
    "COMW" becomes "EWH-Q00-33P-COMW", not "EWH-Q00-33P- COMW" or
    "EWH-Q00-33P--COMW".

    Returns the row text unchanged if no trailing-dash code is found,
    or if no confident cross-line match is found for one.
    """
    ph = _OCR_WS_PLACEHOLDER
    lines = row_text.split("\n")

    def _line_tokens(line):
        tokens = []
        for m in re.finditer(f"[^{ph}]+", line):
            tokens.append([m.start(), m.group(), None, None])
        for i, tok in enumerate(tokens):
            start = tok[0]
            prev_end = tokens[i - 1][0] + len(tokens[i - 1][1]) if i > 0 else 0
            tok[2] = start - prev_end if i > 0 else None
            next_start = tokens[i + 1][0] if i + 1 < len(tokens) else None
            tok[3] = (next_start - (start + len(tok[1]))) if next_start is not None else None
        return tokens

    per_line_tokens = [_line_tokens(l) for l in lines]

    dash_codes = []  # (line_idx, start_offset, word)
    for li, toks in enumerate(per_line_tokens):
        for start, word, _gb, _ga in toks:
            if _TRAILING_DASH_CODE_RE.match(word):
                dash_codes.append((li, start, word))

    if not dash_codes:
        return row_text

    orphans = []  # (line_idx, start_offset, word)
    for li, toks in enumerate(per_line_tokens):
        for start, word, gap_before, gap_after in toks:
            before_ok = gap_before is None or gap_before >= 2
            after_ok = gap_after is None or gap_after >= 2
            if before_ok and after_ok:
                orphans.append((li, start, word))

    new_lines = list(lines)
    for code_line, code_start, code_word in dash_codes:
        best = None
        best_dist = 4  # exclusive upper bound: only accept a close (<=3 char) match
        for oli, ostart, oword in orphans:
            if oli == code_line:
                continue
            dist = abs(ostart - code_start)
            if dist < best_dist:
                best_dist = dist
                best = (oli, ostart, oword)
        if best is None:
            continue
        oli, ostart, oword = best
        merged = code_word + oword
        new_lines[code_line] = new_lines[code_line].replace(code_word, merged, 1)
        # Blank out the orphan's own line position rather than deleting
        # the line -- preserves line count/indices, same convention
        # _strip_isolated_noise_char_lines already uses, and leaves
        # every other token on that line (if any) untouched.
        line = new_lines[oli]
        new_lines[oli] = line[:ostart] + ph * len(oword) + line[ostart + len(oword):]

    return "\n".join(new_lines)


def _find_table_row_blocks_with_anchor(text: str, anchor_re, max_bridge_gap: int = 2,
                                        exclude_re=None, min_numeric_tokens: int = 1):
    """
    Identifies individual table-row text spans within `text`, robust to
    the blank-line-between-rows convention being present INCONSISTENTLY
    -- confirmed on production data: the same document, OCR'd on two
    different occasions, produced one pass with a blank line reliably
    between every row, and a separate pass with only ONE blank line
    anywhere inside the whole table (splitting 8 real rows into two
    blank-line-delimited blocks of 5 and 3 rows each, fused together
    with plain single newlines and no blank line between them at all).
    A detector that only ever treated one blank-line block as one row
    correctly found 8 rows on the first pass and silently only 2 on the
    second, with no signal anything had gone wrong.

    Parameterized on `anchor_re` -- the per-row repeating signal to key
    off of (a date, or a currency amount; see _find_table_row_blocks,
    which tries both in priority order) -- and two independent, ADDITIVE
    ways for an otherwise-matching line to still be rejected:
      - `exclude_re`: a fixed-vocabulary check (used to keep a
        document's own totals/summary section, which can share the same
        amount-per-line shape as real line items, from being mistaken
        for row data when it uses one of a few well-known words like
        "Total"/"GST"/"Subtotal").
      - `min_numeric_tokens`: a vocabulary-free structural check for the
        SAME false-positive mode, for when the summary line's label
        ISN'T one of those known words (see _MIN_ROW_NUMERIC_TOKENS'
        comment for why counting numeric-looking fields on the line
        catches this in general, rather than growing exclude_re's word
        list per newly-seen label). Default 1 imposes no requirement
        beyond `anchor_re` itself matching, so this is opt-in per call
        site.
    A line failing EITHER check is treated as if it didn't match at all.

    Two-stage algorithm:

    Stage 1 -- LOCALIZE the table's overall region using blank-line
    blocks and the anchor pattern: find the maximal simple run(s) of
    consecutive blank-line-delimited blocks that each contain a
    qualifying anchor, bridge adjacent qualifying runs across a small
    gap of non-anchored blocks (tolerates a genuine interruption -- e.g.
    a stray annotation line -- inside an otherwise continuous table),
    and take the largest resulting run. This step is still block-based
    because it's doing a DIFFERENT job than row-splitting: telling the
    table's real content apart from preamble/metadata (an isolated
    single anchored block, e.g. "Assignment Date: ...", never qualifies
    as part of a run at all) and from trailing content (a handwriting-
    annotation section, with no anchor matches at all).

    Stage 2 -- SPLIT the localized region into individual rows using
    per-PHYSICAL-LINE anchor occurrence, not block boundaries. Every row
    line demonstrably contains the anchor on every OCR pass tested,
    independent of whether a blank line follows it -- this is the one
    signal that's held up when the blank-line convention itself hasn't.
    Each anchor-bearing line marks where a new row begins; everything
    from just after the previous row's anchor-line up to (not including)
    the next anchor-line belongs to that row -- capturing wrapped
    continuation lines, stray noise lines, and any lines that precede
    the anchor-line but still belong to that same row (e.g. a wrapped
    description printed on the row's line above its own item_no+date
    line), regardless of whether those lines sit inside one blank-line
    block or straddle a block boundary.

    Returns (row_blocks, header_context, start_offset, end_offset): the
    list of per-row text spans in order; the column-header line text
    (used to give per-row extraction calls context on what each value
    means); and the [start_offset, end_offset) span of the original
    `text` the rows collectively cover. Returns ([], "", -1, -1) if no
    blank-line convention is present at all, or if the localized region
    contains fewer than 2 anchor-bearing lines -- callers must treat
    that as "this anchor doesn't support the strong per-row splitting
    for this document."
    """
    if "\n\n" not in text:
        return [], "", -1, -1

    def _line_qualifies(line):
        if not anchor_re.search(line):
            return False
        if exclude_re is not None and exclude_re.search(line):
            return False
        if min_numeric_tokens > 1 and len(_ANY_NUMERIC_TOKEN_RE.findall(_OCR_TAG_RE.sub(" ", line))) < min_numeric_tokens:
            return False
        return True

    # SEPARATE, LOOSER qualification used ONLY for Stage 1's "does a
    # table-like region even exist here" localization -- deliberately
    # excludes the min_numeric_tokens check that _line_qualifies applies
    # for Stage 2. Confirmed real regression from using the same (strict)
    # check for both: an invoice with exactly 2 genuine line items plus a
    # totals section fell straight through Stage 1's 3+-qualifying-lines
    # threshold once the totals-section lines correctly stopped counting
    # as candidate ROWS (Stage 2's job) -- 2 real rows alone don't reach
    # 3, so the ENTIRE table, including its 2 legitimate rows, was missed
    # completely rather than just having its phantom rows removed. Using
    # the loose check here restores the totals section's contribution to
    # "this looks like a real financial table region," while Stage 2
    # below still uses the strict check to keep those same totals lines
    # from becoming individual extracted rows.
    def _line_qualifies_loose(line):
        if not anchor_re.search(line):
            return False
        if exclude_re is not None and exclude_re.search(line):
            return False
        return True

    blocks = _split_into_blank_line_blocks(text)
    line_counts = [
        sum(1 for l in b.split("\n") if _line_qualifies_loose(l)) for b, _, _ in blocks
    ]

    # Stage 1: localize the table region. A run of CONSECUTIVE blocks
    # that each contain at least one qualifying line is a candidate --
    # but the run only qualifies overall if EITHER it spans 2+ distinct
    # blocks (the original signal: a real table's rows are consistently
    # near each other across successive document paragraphs) OR it
    # contains 3+ qualifying lines even within a single block (needed
    # because a short document's ENTIRE table plus its totals section
    # can share one undivided blank-line block with no separator
    # between rows at all -- confirmed on real invoice data). Requiring
    # 3+ for the single-block case, not just 2+, matters concretely:
    # with only a 2+ threshold, two unrelated header fields that each
    # happen to contain a date (e.g. "Tax Invoice Date" and "Order
    # Date", sitting in the same block purely because they're printed
    # near each other) satisfied it and were wrongly treated as if they
    # were 2 real table rows -- confirmed on the same real invoice. Two
    # coincidental matches next to each other is plausible; three is a
    # much stronger repetition signal.
    # Requires 3+ TOTAL qualifying lines in a run, regardless of how
    # many distinct blocks they're spread across -- NOT "2+ distinct
    # blocks" as a separate, weaker bypass. Confirmed real gap in the
    # original "2+ blocks OR 3+ lines-in-one-block" version: two
    # ordinary header fields that each happen to contain a date (e.g.
    # "Date: 5/5/2026" and "Delivery Date: 8/5/2026" on a PO) can land
    # in two adjacent SEPARATE blank-line blocks, one date each --
    # trivially satisfying "2+ distinct blocks" even though it's the
    # exact same false-positive the single-block 3+ rule was written
    # to prevent. Unifying to one 3+-total-lines threshold (however
    # they're distributed across blocks) closes this without weakening
    # detection of genuine multi-row tables, which naturally have 3+
    # rows in the overwhelming majority of real documents anyway.
    raw_runs = []
    cur = []
    for i, count in enumerate(line_counts):
        if count > 0:
            cur.append(i)
        else:
            if sum(line_counts[k] for k in cur) >= 3:
                raw_runs.append(cur)
            cur = []
    if sum(line_counts[k] for k in cur) >= 3:
        raw_runs.append(cur)

    if not raw_runs:
        return [], "", -1, -1

    merged_runs = [raw_runs[0]]
    for run in raw_runs[1:]:
        prev = merged_runs[-1]
        gap = run[0] - prev[-1] - 1
        if gap <= max_bridge_gap:
            merged_runs[-1] = prev + run
        else:
            merged_runs.append(run)

    best_anchored_indices = max(
        merged_runs, key=lambda run: sum(line_counts[k] for k in run)
    )
    best_start = best_anchored_indices[0]
    region_start = blocks[best_anchored_indices[0]][1]
    region_end = blocks[best_anchored_indices[-1]][2]
    header_context = blocks[best_start - 1][0] if best_start > 0 else ""

    # Trim region_end back from Stage 1's loosely-qualified span to the
    # end of the LAST block that has a genuinely STRICT-qualifying line
    # (i.e. a real row, not a totals/summary line that only counted
    # toward the loose 3+ threshold above). Without this, a trailing
    # totals section that Stage 1 correctly recognized as part of "a
    # table-like region exists here" would still get swallowed into the
    # returned row span -- meaning its text is excluded from BOTH row
    # extraction (correctly, since it's not a row) AND from whatever
    # handles everything OUTSIDE the row span (incorrectly -- callers
    # build doc-level/field text as `text[:region_start] + text[region_end:]`,
    # so text still inside [region_start, region_end) is invisible to
    # that path too). Confirmed on production data: a "Total Amount to
    # pay" value was missing from the final output even after the
    # totals-lines-as-phantom-rows bug above was fixed, because it
    # remained trapped inside the returned span with nothing left to
    # extract it. Only trims the END, never region_start -- a table's
    # real rows are never preceded by its own totals section.
    #
    # KNOWN LIMITATION: if the LAST real row's own wrapped continuation
    # text happens to land in a separate blank-line block that contains
    # no anchor-qualifying line of its own (an inconsistent-blank-line
    # OCR pass, per this function's main docstring), this trim can cut
    # that continuation off along with the totals section, since both
    # look identical at the block level (a block with zero strict-
    # qualifying lines). Accepted for now as a real but narrower risk
    # than the confirmed bug this fixes.
    strict_line_counts = [
        sum(1 for l in b.split("\n") if _line_qualifies(l)) for b, _, _ in blocks
    ]
    strict_indices_in_run = [i for i in best_anchored_indices if strict_line_counts[i] > 0]
    if strict_indices_in_run:
        region_end = min(region_end, blocks[strict_indices_in_run[-1]][2])

    # Stage 2: split the localized region into individual rows by
    # per-physical-line anchor occurrence, independent of block boundaries.
    region_text = text[region_start:region_end]
    region_lines = region_text.split("\n")
    anchor_line_positions = [
        i for i, l in enumerate(region_lines) if _line_qualifies(l)
    ]
    if len(anchor_line_positions) < 2:
        return [], "", -1, -1

    # A mixed letter+digit token (e.g. "N9R82B12L1") is a strong,
    # generalizable tell for genuine ROW DATA -- a column-header phrase
    # is plain words, and a wrapped continuation word (e.g. "HEATER") is
    # plain letters, but neither ever looks like a fabricated alnum
    # code. This is what lets the next two fixes tell "real preceding
    # row data" apart from both header text and a wrap-continuation
    # word using shape alone, with no document-specific assumption.
    #
    # Excludes a match directly adjacent to a hyphen or word character
    # on either side -- without this, a source_no field like
    # "PO-TB10093915" (present on literally every row's own date-line)
    # also matches, since "TB10093915" alone has the same letter+digit
    # shape as a genuine bin code; the hyphen immediately before it is
    # what actually distinguishes a reference-number fragment from a
    # standalone code value with real whitespace on both sides.
    _MIXED_ALNUM_CODE_RE = re.compile(r"(?<![\w-])[A-Za-z]+\d[A-Za-z0-9]{3,}(?![\w-])")

    # Some rows print their own data on a line BEFORE their own anchor
    # line (confirmed: at least 3 distinct physical layouts for "the
    # same" row exist across different OCR passes of one document), so
    # whatever precedes the very first anchor line in the region can be
    # a mix of genuine header text AND the first row's own data -- not
    # purely one or the other. Find the LAST header-only line (no code
    # token) before the first code-bearing line; only that prefix is
    # true header context, and everything from the code-bearing line
    # onward is prepended to row 1 instead of being discarded.
    #
    # This is APPENDED to whatever header_context Stage 1 already found
    # from the preceding block, never replacing it -- confirmed bug on
    # production data: a header that wraps across MULTIPLE physical
    # lines (e.g. "Unit of" / "Variant" / "Measure" printed on separate
    # lines, genuinely reuniting into "Variant Code" and "Unit of
    # Measure") sits in that PRECEDING block, correctly captured by
    # Stage 1 -- but this code used to unconditionally overwrite it with
    # only the in-region header line, silently discarding the wrapped
    # fragments entirely. Every per-row extraction call was then given
    # a header context with no mention of "Variant" or "Unit of"/
    # "Measure" at all, and had no way to know that column existed --
    # very likely the direct cause of the model inventing a bare "code"
    # field and populating it with a hallucinated value pulled from
    # neighboring rows' descriptions, since it had no correct name or
    # even acknowledgment of the real (genuinely blank) column to put
    # null in instead.
    row1_prefix = ""
    if anchor_line_positions[0] > 0:
        discarded_lines = region_lines[:anchor_line_positions[0]]
        split_idx = len(discarded_lines)
        for j, l in enumerate(discarded_lines):
            if _MIXED_ALNUM_CODE_RE.search(l):
                split_idx = j
                break
        header_lines = discarded_lines[:split_idx]
        row1_prefix_lines = discarded_lines[split_idx:]
        in_region_header_text = "\n".join(header_lines).strip("\n")
        if in_region_header_text.strip():
            header_context = (
                (header_context + "\n" + in_region_header_text)
                if header_context.strip()
                else in_region_header_text
            )
        if row1_prefix_lines:
            row1_prefix = "\n".join(row1_prefix_lines) + "\n"

    row_texts = []
    for i, line_idx in enumerate(anchor_line_positions):
        end_line_idx = (
            anchor_line_positions[i + 1]
            if i + 1 < len(anchor_line_positions)
            else len(region_lines)
        )
        row_texts.append("\n".join(region_lines[line_idx:end_line_idx]))
    if row1_prefix:
        row_texts[0] = row1_prefix + row_texts[0]

    # The SAME "own data precedes own anchor line" layout also means a
    # row's initial per-line split can accidentally trail off into the
    # NEXT row's preceding data (everything between one anchor line and
    # the next was assigned to the EARLIER row above, which is wrong
    # when that next row prints its own data before its own anchor
    # line). Scan each row's span, starting STRICTLY AFTER that row's
    # OWN anchor line (located explicitly by content, not assumed to
    # sit at a fixed index -- row 1 can have a prefix line before it
    # now, so "index 0" is no longer a safe assumption for where the
    # anchor line itself sits), for the first code-bearing line -- a
    # wrap-continuation word like "HEATER" never matches this, but the
    # next row's own data line does -- and move everything from that
    # point onward into the following row instead.
    #
    # Move points are computed ONCE from the ORIGINAL (pre-mutation)
    # row_texts, then applied in a single pass -- computing and applying
    # a move at the same time cascades: content just moved INTO row i+1
    # would get re-scanned there and match again, shifting it AGAIN into
    # row i+2, and so on all the way down (confirmed: this collapsed
    # rows 5 through 8 into one on an earlier version of this fix).
    move_points = [None] * len(row_texts)
    for i in range(len(row_texts) - 1):
        lines = row_texts[i].split("\n")
        own_anchor_line_idx = next(
            (k for k, l in enumerate(lines) if _line_qualifies(l)), 0
        )
        for j in range(own_anchor_line_idx + 1, len(lines)):
            if _MIXED_ALNUM_CODE_RE.search(lines[j]):
                move_points[i] = j
                break

    rebalanced = []
    carry_in = ""
    for i, row_text in enumerate(row_texts):
        lines = row_text.split("\n")
        mp = move_points[i]
        own_part = "\n".join(lines[:mp]) if mp is not None else row_text
        carry_out = "\n".join(lines[mp:]) if mp is not None else ""
        rebalanced.append((carry_in + "\n" + own_part) if carry_in else own_part)
        carry_in = carry_out
    if carry_in:
        rebalanced[-1] += "\n" + carry_in
    row_texts = rebalanced

    return row_texts, header_context, region_start, region_end


# Placeholder character the upstream OCR/MD normalization step substitutes
# 1:1 for each original whitespace character it couldn't confidently
# place. A whole line whose only content, once collapsed, is a single bare
# letter (e.g. a lone "J" sitting alone in an otherwise-blank table-row
# slot) has been confirmed on production data to be scan noise -- a stray
# mark, dust speck, or bleed-through -- rather than a real transcribed
# word; no legitimate table cell in this pipeline's documents is ever a
# single bare letter standing completely alone on its own line.
_OCR_WS_PLACEHOLDER = "\u2337"
_ISOLATED_NOISE_CHAR_LINE_RE = re.compile(
    r"^" + _OCR_WS_PLACEHOLDER + r"*[A-Za-z]" + _OCR_WS_PLACEHOLDER + r"*$"
)


def _strip_isolated_noise_char_lines(text: str) -> str:
    """
    Blanks out (rather than deletes -- preserving line indices/structure
    for anything downstream that assumes line-position meaning) any line
    matching _ISOLATED_NOISE_CHAR_LINE_RE. A no-op on text that doesn't
    use the OCR placeholder character at all.
    """
    if _OCR_WS_PLACEHOLDER not in text:
        return text
    return "\n".join(
        "" if _ISOLATED_NOISE_CHAR_LINE_RE.match(line) else line
        for line in text.split("\n")
    )


# Matches this OCR pipeline's "[cancelled: ...]" wrapper -- its way of
# flagging text on the page that was struck through / crossed out (e.g.
# a superseded column-header row). Unlike "[handwriting N]", "[illegible]",
# and "[unreadable]", nothing downstream currently knows this marker
# exists, so it was flowing into header_context and row text as literal
# content -- confirmed on production data to have caused a real invoice's
# table header to read "[cancelled: No. Item No. Description Quantity
# Unit Price Amt (SGD)]" verbatim inside a per-row extraction prompt,
# with no indication to the model that "cancelled" was pipeline metadata
# rather than something stated about the table itself.
_CANCELLED_WRAPPER_RE = re.compile(r"\[cancelled:(.*?)\]", re.DOTALL)


def _unwrap_cancelled_annotation(text: str) -> str:
    """
    Removes the "[cancelled: ...]" wrapper while keeping the text it
    wraps exactly where it was -- the wrapped text is real document
    content (whatever was on the page, just struck through), so it must
    not be dropped, only un-tagged. The "[cancelled:" and closing "]"
    markers are replaced with the OCR whitespace placeholder character
    (one placeholder per character removed) rather than deleted
    outright, so this behaves like just another gap in the OCR's own
    whitespace-gap encoding -- preserving character offsets/column
    alignment for every downstream step that reads position
    (_find_table_row_blocks_with_anchor's anchor-line matching,
    _reconstruct_wrapped_header_columns' offset comparison, etc.) rather
    than shifting them the way a normal string deletion would.
    A no-op on text with no such wrapper.
    """
    if "[cancelled:" not in text:
        return text

    def _replace(m: "re.Match") -> str:
        return (
            _OCR_WS_PLACEHOLDER * len("[cancelled:")
            + m.group(1)
            + _OCR_WS_PLACEHOLDER
        )

    return _CANCELLED_WRAPPER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Coloured-bar label recovery.
#
# Some invoices print their field/column labels as WHITE text on a coloured
# bar ("A/C CODE | P.O. NUMBER | TERMS | SALESMAN"). The OCR tool can't read
# white-on-colour text as page text, so it files the labels in its "##
# Detected handwritten/colored annotations" section and leaves only a
# "[handwriting N]" tag in the page body where the bar was. The VALUES
# printed under the bar are read fine, but now sit there with no labels --
# confirmed on a real invoice: the model took the first value under the
# lost "A/C CODE" bar (an account code) for the invoice number, and never
# captured the PO number two cells to its right. Everything after the first
# extraction (doc_id, po_ref) is built from that JSON, so it cannot recover
# a value that never made it in.
#
# This puts the labels back deterministically, ONLY on a page that really
# has a coloured-bar label run (an annotation made of 3+ known header
# words). Any other page -- every normal black-and-white invoice -- comes
# back byte-for-byte unchanged. Two things are done, both additive:
#   1. every annotation containing a known header word is shown inline at
#      its own tag:   [handwriting 6]  ->  [handwriting 6 = "...INVOICE"]
#   2. for a label run, the values line printed under the bar is cut into
#      as many cells as there are labels (at its widest gaps) and a
#      "LABEL: value | LABEL: value" line is added right under it. Nothing
#      is added unless that cut is unambiguous.
# No original line is ever removed or edited except the tag swap in (1).
# ---------------------------------------------------------------------------
_HEADER_LABEL_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    r"A/?C\.?\s*(?:CODE|NO\.?)"
    r"|ACCOUNT\s*(?:CODE|NO\.?|NUMBER)"
    r"|CUST(?:OMER)?\.?\s*(?:CODE|NO\.?)"
    r"|P\.?\s?O\.?\s*(?:NUMBER|NO\.?|REF\.?)"
    r"|PURCHASE\s+ORDER(?:\s*(?:NO\.?|NUMBER))?"
    r"|INVOICE\s*(?:NUMBER|NO\.?)"
    r"|(?:DELIVERY|D\.?O\.?)\s*(?:ORDER\s*)?(?:NO\.?|NUMBER)"
    r"|PAYMENT\s+TERMS|TERMS"
    r"|SALES\s?(?:MAN|PERSON|REP)"
    r"|DUE\s+DATE|DATE"
    r"|PARTICULARS|DESCRIPTION|QUANTITY|QTY|UNIT\s+PRICE|AMOUNT|ITEM\s*NO\.?"
    r"|BILL\s+TO|DELIVERED\s+TO|SHIP\s+TO|TAX\s*INVOICE|INVOICE|NUMBER"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)
_HANDWRITING_TAG_RE = re.compile(r"\[handwriting\s+(\d+)\]")
_ANNOTATION_ENTRY_RE = re.compile(
    r'^-\s*Page\s+(\d+)\s*:\s*handwriting\s+(\d+)\s*(?:\([^)]*\))?\s*:\s*"(.*)"\s*$'
)
_BAR_PAGE_RE = re.compile(r"^##\s*Page\s+(\d+)\s*$")


def _split_label_run(annotation_text: str):
    """
    The list of header labels in `annotation_text` if it is a run of 3+
    known header words with at most a few stray characters left over
    (e.g. "HHO PARTICULARS QUANTITY ..." -- the OCR's garbled "ITEM NO."
    leaves "HHO"), else None.
    """
    matches = list(_HEADER_LABEL_RE.finditer(annotation_text))
    if len(matches) < 3:
        return None
    leftover = _HEADER_LABEL_RE.sub("", annotation_text)
    if len(re.sub(r"[^A-Za-z0-9]", "", leftover)) > 6:
        return None
    return [re.sub(r"\s+", " ", m.group(0)).strip() for m in matches]


def _cut_values_line_into_cells(line: str, k: int):
    """
    Cuts one OCR line into exactly `k` cells at its `k-1` WIDEST gaps, or
    returns None if that cut is not clear-cut. "Net  60 after EOM" has a
    double gap inside one cell, but the gaps BETWEEN cells are far wider,
    so widest-gaps-wins gets the cells right where a fixed "2+ spaces"
    rule would not. Refuses (None) unless every cut gap is at least 3
    wide AND wider than every gap left uncut.
    """
    if k < 2:
        return None
    masked = _OCR_TAG_RE.sub(lambda m: " " * len(m.group(0)), line)
    toks = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[^\u2337\s]+", masked)
            if re.search(r"[A-Za-z0-9]", m.group(0))]  # a lone "-" or "|" is not a value
    if len(toks) < k:
        return None
    gaps = [toks[i + 1][0] - toks[i][1] for i in range(len(toks) - 1)]
    order = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)
    cut = set(order[:k - 1])
    chosen_min = min(gaps[i] for i in cut)
    rest_max = max((gaps[i] for i in order[k - 1:]), default=0)
    if chosen_min < 3 or chosen_min <= rest_max:
        return None
    cells, cur = [], [toks[0][2]]
    for i in range(len(gaps)):
        if i in cut:
            cells.append(" ".join(cur))
            cur = []
        cur.append(toks[i + 1][2])
    cells.append(" ".join(cur))
    return cells if len(cells) == k else None


def _recover_header_bar_labels(text: str) -> str:
    """See the block comment above. Never raises; on any problem returns
    `text` unchanged."""
    try:
        return _recover_header_bar_labels_impl(text)
    except Exception:
        return text


def _recover_header_bar_labels_impl(text: str) -> str:
    marker = "## Detected handwritten/colored annotations"
    cut = text.find(marker)
    if cut == -1:
        return text
    body, tail = text[:cut], text[cut:]

    annotations = {}
    for line in tail.split("\n"):
        m = _ANNOTATION_ENTRY_RE.match(line.strip())
        if m:
            annotations[(int(m.group(1)), int(m.group(2)))] = m.group(3)

    runs = {}
    for key, txt in annotations.items():
        labels = _split_label_run(txt)
        if labels:
            runs[key] = labels
    if not runs:
        return text  # no coloured-bar label run on this document: untouched

    lines = body.split("\n")
    page = 1
    page_of_line = []
    for ln in lines:
        pm = _BAR_PAGE_RE.match(ln.strip())
        if pm:
            page = int(pm.group(1))
        page_of_line.append(page)

    tag_lines = {}
    for i, ln in enumerate(lines):
        for m in _HANDWRITING_TAG_RE.finditer(ln):
            tag_lines.setdefault((page_of_line[i], int(m.group(1))), []).append(i)

    additions = {}  # line index -> lines to insert right after it
    for key, labels in runs.items():
        occ = tag_lines.get(key, [])
        if len(occ) != 1:
            continue
        def _restored(cells):
            # a cell with fewer than 2 letters/digits is OCR debris (e.g. an
            # account code read as "J"): say so rather than present it as a value
            shown = [c if len(re.sub(r"[^A-Za-z0-9]", "", c)) >= 2 else "[unreadable]" for c in cells]
            return (f"[colour-bar labels restored from annotation {key[1]}] "
                    + " | ".join(f"{lab}: {cell}" for lab, cell in zip(labels, shown)))

        # 1) values printed on the SAME line as the bar's tag (the tag sits in
        #    the middle of the row). If that line doesn't cut cleanly it is
        #    skipped, not final -- the values may be on a later line instead.
        own_visible = _OCR_TAG_RE.sub("", lines[occ[0]]).replace(_OCR_WS_PLACEHOLDER, "").strip()
        own_cells = _cut_values_line_into_cells(lines[occ[0]], len(labels)) if own_visible else None
        if own_cells:
            additions.setdefault(occ[0], []).append(_restored(own_cells))
            continue
        # 2) otherwise the first non-empty line under the bar
        for j in range(occ[0] + 1, min(occ[0] + 5, len(lines))):
            visible = _OCR_TAG_RE.sub("", lines[j]).replace(_OCR_WS_PLACEHOLDER, "").strip()
            if not visible:
                continue
            cells = _cut_values_line_into_cells(lines[j], len(labels))
            if cells:
                additions.setdefault(j, []).append(_restored(cells))
            break  # only the FIRST non-empty line under the bar is ever tried

    out = []
    for i, ln in enumerate(lines):
        def _inline(m, _page=page_of_line[i]):
            txt = annotations.get((_page, int(m.group(1))))
            if txt and _HEADER_LABEL_RE.search(txt):
                return f'[handwriting {m.group(1)} = "{txt}"]'
            return m.group(0)
        out.append(_HANDWRITING_TAG_RE.sub(_inline, ln))
        out.extend(additions.get(i, []))
    return "\n".join(out) + tail


# Fixed section marker this OCR pipeline's own MD output always uses for
# page boundaries (same kind of format-native marker the "## Detected
# handwritten/colored annotations" section already relies on elsewhere in
# this file) -- deliberately narrow to this pipeline's known output shape,
# not a general-purpose markdown page splitter.
_PAGE_MARKER_RE = re.compile(r"^##\s*Page\s+(\d+)\s*$", re.MULTILINE)


def _split_into_page_chunks(text: str) -> list:
    """
    Splits OCR'd multi-page markdown into per-page chunks on this
    pipeline's own "## Page N" markers, so a document too large for one
    model call can be extracted page-by-page instead -- every character
    still gets sent to the model, split across calls rather than
    truncated (see _run_json_mode's fallback-path branch and
    _merge_chunked_json_objects, which exist specifically so "the
    document didn't fit in one call" never means "part of it was
    dropped").

    Returns a list of (page_label, page_text) tuples in document order.
    Any text BEFORE the first page marker (the YAML frontmatter --
    attachment_id, source, engines, mean_confidence, etc.) is shared
    context relevant to every page, so it's prepended to EACH chunk's
    page_text rather than becoming its own separate chunk.

    Returns [] (signaling "don't chunk, use whole-document handling")
    when fewer than 2 page markers are found -- a single-page or
    marker-less document doesn't need this.
    """
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if len(matches) < 2:
        return []

    preamble = text[:matches[0].start()].strip()
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_label = f"Page {m.group(1)}"
        page_body = text[start:end]
        page_text = f"{preamble}\n\n{page_body}" if preamble else page_body
        chunks.append((page_label, page_text))
    return chunks


def _merge_chunked_json_objects(objects: list) -> dict:
    """
    Merges per-page JSON extraction results (see _split_into_page_chunks
    and its caller in _run_json_mode) into ONE combined object.
    Lossless by construction -- every chunk's own data is either kept
    as-is (scalars) or concatenated (lists); nothing from any chunk is
    ever dropped to make the merge "fit."

    - Scalar fields: the first chunk with a non-null/non-empty value for
      a given key wins. If a LATER chunk has a genuinely different
      non-null value for the same key (not just a repeat), that's
      recorded under "_chunk_merge_conflicts" rather than silently
      discarded -- e.g. two pages each stating their own distinct
      reference number under the same field name would be a real,
      worth-flagging disagreement, not noise to drop.
    - List fields (line_items, handwritten_notes,
      handwritten_annotations, source_backfill, needs_human_review,
      etc.): concatenated across every chunk, in chunk order, WITHOUT
      de-duplication -- a value appearing on two different pages is
      real information (e.g. two genuinely separate line items that
      happen to have identical text), and removing it would itself be
      an unrequested, silent truncation.
    - Dict fields present in more than one chunk (e.g. two chunks each
      producing their own "delivery_order" sub-object because the
      per-page prompt told each page to nest its own distinct document
      under its own key -- see _run_json_mode) are shallow-merged with
      the same first-chunk-wins-on-scalars rule, recursively.
    """
    def _merge_two(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                out[k] = _merge_two(out[k], v) if k in out else v
            return out
        return a  # scalar: first (a) wins; caller logs the conflict

    merged = {}
    conflicts = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            if key not in merged:
                merged[key] = value
                continue
            existing = merged[key]
            if isinstance(existing, list) or isinstance(value, list):
                merged[key] = (existing if isinstance(existing, list) else [existing]) + \
                              (value if isinstance(value, list) else [value])
            elif isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = _merge_two(existing, value)
            elif existing in (None, "", []) and value not in (None, "", []):
                merged[key] = value
            elif value not in (None, "", []) and existing != value:
                conflicts.append({"field": key, "kept": existing, "also_seen": value})
    if conflicts:
        merged["_chunk_merge_conflicts"] = conflicts
    return merged


# Matches a field name that is exactly "code" (any case) -- the generic
# fallback name the model reaches for instead of the real compound header
# (e.g. "bin_code") when it can't reconstruct the full header phrase.
_GENERIC_CODE_FIELD_RE = re.compile(r"^code$", re.IGNORECASE)

# Matches a real, specific "..._code" field name (e.g. "bin_code",
# "zone_code", "variant_code").
_SPECIFIC_CODE_FIELD_RE = re.compile(r"^[a-z][a-z0-9]*_code$", re.IGNORECASE)


def _repair_orphaned_code_fields(parsed):
    """
    Fixes a bare `"code"` field holding a value that really belongs to a
    specific null `"..._code"` sibling field (e.g.
    `{"zone_code": "N", "bin_code": null, "code": "AB12-99Z"}` where
    "AB12-99Z" is really the bin_code value). Only repairs the
    UNAMBIGUOUS case: exactly one null "..._code" sibling. Zero or
    multiple candidates are left untouched -- guessing among several
    would trade one silent error for another.
    """
    def _repair_dict(d):
        code_key = None
        for key in d:
            if _GENERIC_CODE_FIELD_RE.match(key):
                code_key = key
                break
        if code_key is None or d[code_key] in (None, ""):
            return
        null_specific_keys = [
            k for k in d
            if k != code_key
            and _SPECIFIC_CODE_FIELD_RE.match(k)
            and d[k] in (None, "")
        ]
        if len(null_specific_keys) == 1:
            target_key = null_specific_keys[0]
            d[target_key] = d[code_key]
            del d[code_key]

    def _walk(node):
        if isinstance(node, dict):
            _repair_dict(node)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return parsed


def _repair_bare_code_holding_freetext(parsed):
    """
    Fixes a different variant of the bare-"code"-field failure: the
    value under "code" is a multi-word free-text phrase (e.g. "ACME
    PORTABLE SPACE HEATER") that actually belongs under "description".
    Uses VALUE SHAPE (contains a space) rather than sibling null-
    frequency, since this failure can occur uniformly across every row
    at once, giving sibling-based checks no differential signal to work
    with.

    - If "description" exists and is null, the value is moved there.
    - If "description" already holds the EXACT same text (pure
      duplication), the redundant "code" key is simply deleted.
    - If "description" holds some OTHER value, both are left untouched
      -- which of the two is actually right is genuinely ambiguous, and
      guessing would trade one error for another.
    """
    def _repair_dict(d):
        code_key = None
        for key in d:
            if _GENERIC_CODE_FIELD_RE.match(key):
                code_key = key
                break
        if code_key is None:
            return
        value = d[code_key]
        if not isinstance(value, str) or " " not in value.strip():
            return
        existing_description = d.get("description")
        if existing_description in (None, ""):
            if "description" in d:
                d["description"] = value
                del d[code_key]
            return
        if value.strip() == existing_description.strip():
            del d[code_key]

    def _walk(node):
        if isinstance(node, dict):
            _repair_dict(node)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return parsed


def _repair_hallucinated_short_code_field(row_dict, row_text):
    """
    Fixes a THIRD variant of the bare-"code"-field failure, distinct
    from _repair_bare_code_holding_freetext (which only handles a
    multi-WORD value): a bare "code" field holding a short, single-
    token value (e.g. "707") that the model has invented a phantom
    field for -- confirmed on production data to be the model reacting
    to a genuinely blank wrapped column (e.g. "Variant Code", which
    prints across two separate physical lines and is easy to overlook
    or misjudge) by manufacturing SOME value for it rather than
    correctly outputting null, even after being given the header
    context needed to recognize the column and told explicitly not to
    guess -- prompt-level guidance alone did not reliably stop this,
    so this checks the claim against the row's own raw source text
    instead of trusting the model's judgment about it.

    Only usable in the per-row extraction path, where each row's own
    isolated source text is available to check the claim against
    directly -- this is what makes the fix a verification rather than
    another guess:

    - If the "code" value does NOT appear anywhere in this row's own
      raw text at all, it has zero grounding in what this row actually
      says -- confirmed hallucination, not a misplaced real value. The
      field is deleted outright rather than moved anywhere, since
      there's nothing to move it TO.
    - If the value DOES appear in the row's own raw text, and prepending
      it to the existing "description" value forms a phrase whose words
      ALL appear, in the same order, somewhere in the row's own raw
      text -- not necessarily contiguously, since a description's own
      wrapped continuation word can land on a later physical OCR line
      with unrelated column values interleaved in between -- this is a
      genuine split -- the value was chopped off the front of the real
      description, not invented -- and is merged back: description
      becomes "{value} {description}", and the stray "code" key is
      removed.
    - Any other case (value present in the text but doesn't cleanly
      recombine with description) is left untouched -- ambiguous enough
      that guessing would trade one error for another.
    """
    if not isinstance(row_dict, dict) or not isinstance(row_text, str):
        return row_dict
    code_key = None
    for key in row_dict:
        if _GENERIC_CODE_FIELD_RE.match(key):
            code_key = key
            break
    if code_key is None:
        return row_dict
    value = row_dict[code_key]
    if not isinstance(value, str) or not value.strip() or " " in value.strip():
        return row_dict

    normalized_text = re.sub(_OCR_WS_PLACEHOLDER + r"+", " ", row_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    value_clean = value.strip()

    if value_clean not in normalized_text:
        del row_dict[code_key]
        return row_dict

    def _words_appear_in_order(words, haystack):
        # Checks that every word appears in `haystack`, IN ORDER, but
        # NOT necessarily contiguously -- a plain substring check is too
        # strict here: a description's own wrapped continuation word can
        # land on a later physical OCR line, with unrelated column
        # values (zone/bin/quantity/unit) interleaved in between due to
        # the page's raster reading order, even though the words
        # genuinely belong to one description (confirmed on real data:
        # "707 INSTANT WATER" ... other columns' values ... "HEATER",
        # where the per-row model call had already correctly reunited
        # the wrap into "INSTANT WATER HEATER" -- a strict substring
        # check against the still-wrapped raw text would reject a
        # genuine, correct merge here, exactly the false negative
        # observed before this fix).
        pos = 0
        for w in words:
            idx = haystack.find(w, pos)
            if idx == -1:
                return False
            pos = idx + len(w)
        return True

    description = row_dict.get("description")
    if isinstance(description, str) and description.strip():
        candidate = f"{value_clean} {description.strip()}"
        if _words_appear_in_order(candidate.split(), normalized_text):
            row_dict["description"] = candidate
            del row_dict[code_key]
            return row_dict

    return row_dict


def _repair_zone_bin_code_confusion(parsed, timeout_secs: float = 20):
    """
    Fixes zone_code/bin_code confusion on any line item carrying both
    fields: a bin code and its own zone code are never identical in a
    real warehouse system (a bin is always a more specific location
    WITHIN a zone), so an equal pair -- or a null bin_code alongside a
    populated zone_code -- is a reliable tell, regardless of what a
    stray field holding the real value happens to be named that run.

    Classification of WHICH stray field actually holds the real
    bin_code value is delegated to a model call rather than a hardcoded
    list of "safe" field names to exclude from consideration -- the old
    approach had to enumerate every field name this specific warehouse
    schema happens to use, which silently breaks the moment a document
    uses a field name this codebase has never seen before. Asking "does
    this VALUE look like a bin code" instead of maintaining a field-NAME
    exclusion list generalizes to any schema.

    Structural candidates: any line item where `bin_code` looks wrong
    (missing, or equal to `zone_code`) gives something to compare
    against. When the row's own `zone_code` is ALSO empty, a cheap
    length-shape prefilter (learned from sibling rows' typical
    zone_code length, same idea as the original Rule 2) narrows which
    of the item's other fields are even worth asking the model about --
    this is a candidate-narrowing step, not a correctness decision; the
    model still makes the final call on every candidate it's shown.
    Every OTHER populated field on a qualifying item becomes a
    candidate; distinct (field_name, value) pairs are deduplicated
    across the whole document before the single classification call.

    Only acts when EXACTLY ONE field on a given item is classified as
    the bin code -- ambiguous (zero or multiple) matches are left
    untouched, same conservatism as before. Fails safe: any problem
    with the call or its response leaves `parsed` completely untouched.
    """
    triples = []  # (item, field_key, field_value)

    def _collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    items = [it for it in value if isinstance(it, dict)]
                    zone_lens = [
                        len(it["zone_code"]) for it in items
                        if isinstance(it.get("zone_code"), str) and it["zone_code"]
                    ]
                    typical_zone_len = (
                        max(set(zone_lens), key=zone_lens.count) if zone_lens else None
                    )
                    for item in items:
                        if "bin_code" not in item:
                            continue  # this document type doesn't use bin_code at all
                        zone_val = item.get("zone_code")
                        bin_val = item.get("bin_code")
                        bin_looks_wrong = bin_val in (None, "") or (
                            zone_val not in (None, "") and bin_val == zone_val
                        )
                        if not bin_looks_wrong:
                            continue
                        for k, v in item.items():
                            if k in ("zone_code", "bin_code"):
                                continue
                            if not (isinstance(v, str) and v.strip()):
                                continue
                            if zone_val in (None, "") and typical_zone_len is not None:
                                if len(v.strip()) <= typical_zone_len + 2:
                                    continue
                            triples.append((item, k, v.strip()))
                else:
                    _collect(value)
        elif isinstance(node, list):
            for it in node:
                _collect(it)

    _collect(parsed)
    if not triples:
        return parsed

    unique_pairs = []
    seen = set()
    for _, k, v in triples:
        pair = (k, v)
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    lines = [f'{i}. field_name: "{k}", value: "{v}"' for i, (k, v) in enumerate(unique_pairs, 1)]
    prompt = (
        "Each numbered entry below is a field currently extracted from one "
        "line item of a warehouse/inventory document, under the given "
        "field_name. For EACH entry, decide: does this value actually look "
        "like a warehouse BIN CODE -- a specific storage location WITHIN a "
        "zone, more granular than a zone code (e.g. \"A-12-03\" is bin-"
        "code-shaped; \"A\" alone is more zone-shaped)? Or does it "
        "genuinely belong under its current field_name (e.g. it's really "
        "an item number, a description, a quantity, a date)?\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object mapping each entry's number "
        "(as a string) to either \"bin_code\" or \"not_bin_code\". No "
        "prose, no markdown fences, just the JSON object."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.15, 8))
        classification, _ = _try_parse_json(raw)
    except Exception:
        return parsed  # fail safe -- no-op rather than guess

    if not isinstance(classification, dict):
        return parsed

    bin_code_pairs = set()
    for i, pair in enumerate(unique_pairs, 1):
        verdict = classification.get(str(i), classification.get(i))
        if isinstance(verdict, str) and verdict.strip().lower() == "bin_code":
            bin_code_pairs.add(pair)

    if not bin_code_pairs:
        return parsed

    item_matches = {}
    for item, k, v in triples:
        if (k, v) in bin_code_pairs:
            entry = item_matches.setdefault(id(item), (item, []))
            entry[1].append((k, v))

    for item, matches in item_matches.values():
        if len(matches) == 1:
            k, v = matches[0]
            item["bin_code"] = v
            del item[k]

    return parsed


# Collects the distinct candidate uom values (from items where lcn is
# empty) that need a classification decision, without walking the tree
# twice -- shared helper for _repair_uom_lcn_confusion below.
def _collect_uom_lcn_candidates(node, out: set):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "line_items" and isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    if "uom" not in item or "lcn" not in item:
                        continue
                    if item.get("lcn") not in (None, ""):
                        continue
                    uom_val = item.get("uom")
                    if isinstance(uom_val, str) and uom_val.strip():
                        out.add(uom_val.strip())
            else:
                _collect_uom_lcn_candidates(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_uom_lcn_candidates(item, out)


def _repair_uom_lcn_confusion(parsed, timeout_secs: float = 20):
    """
    Fixes uom/lcn confusion on line items: confirmed in production, a
    location/channel code ("WEB", "03SKS", "09CORP" -- all the SAME kind
    of value, same column position, on sibling rows of one table) gets
    filed under `uom` on some rows and `lcn` on others, purely because
    the model is inconsistent about which of the two adjacent header
    columns it lands a given cell in -- not because the value itself
    differs in kind. "WEB" is not a unit of measure (nothing is measured
    "in webs"); it's the same kind of code as "03SKS"/"09CORP" and
    belongs in the same field they do.

    Classification is delegated to a dedicated, targeted model call
    rather than a hardcoded whitelist of "real" units -- deliberately:
    a fixed list of unit abbreviations is a closed set that will always
    miss some legitimate unit this pipeline hasn't seen yet (a different
    vendor's documents, a different language, an uncommon abbreviation),
    silently misclassifying it as a code and wrongly moving it. "Is X a
    genuine unit of measure" is exactly the kind of open-ended judgment
    call a model is well suited to and a fixed list isn't -- the model's
    general knowledge already covers units this specific codebase was
    never told about.

    Only the DISTINCT candidate values (uom non-empty, lcn empty) are
    sent for classification, deduplicated once across the whole document
    -- typically 1-3 values even on a table with many rows -- to keep
    this a small, fast, single-purpose call rather than a per-row one.

    Fails safe: any problem with the call or its response (network
    error, timeout, non-JSON reply, wrong shape) leaves `parsed`
    completely untouched -- this is a repair, not a required step, and a
    failed classification should never risk guessing wrong or crashing
    the whole extraction.

    Rule (deliberately conservative, fires only on the unambiguous
    case): on any line item where `lcn` is null/empty and `uom` holds a
    non-empty string the model classifies as NOT a genuine unit, move
    that value from `uom` to `lcn`. Never fires when `lcn` is already
    populated -- this repairs a missing/misplaced value, it never
    overwrites one already present.
    """
    candidates = set()
    _collect_uom_lcn_candidates(parsed, candidates)
    if not candidates:
        return parsed

    prompt = (
        "For each of the following values, extracted from the 'Unit of "
        "Measure' (UOM) column of an invoice or purchase-order line-item "
        "table, decide whether it is a GENUINE unit of measure -- how a "
        "quantity is counted or measured, e.g. pieces, sets, kilograms, "
        "boxes, cartons, litres, dozens -- or whether it is actually some "
        "OTHER kind of code that has been misplaced into the UOM column, "
        "most commonly a location, warehouse, store, or sales-channel code "
        "(e.g. a branch code, or a channel indicator like an online/web "
        "sales flag).\n\n"
        "Values to classify:\n"
        + "\n".join(f"- {v}" for v in sorted(candidates))
        + "\n\nRespond with ONLY a JSON object mapping each value (exactly "
        "as given above) to either \"unit\" or \"not_unit\". No prose, no "
        "markdown fences, just the JSON object."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.15, 8))
        classification, _ = _try_parse_json(raw)
    except Exception:
        return parsed  # fail safe -- no-op rather than guess

    if not isinstance(classification, dict):
        return parsed

    not_units = {
        k.strip() for k, v in classification.items()
        if isinstance(v, str) and v.strip().lower() == "not_unit"
    }
    if not not_units:
        return parsed

    def _repair_item(d):
        if "uom" not in d or "lcn" not in d:
            return
        if d.get("lcn") not in (None, ""):
            return
        uom_val = d.get("uom")
        if isinstance(uom_val, str) and uom_val.strip() in not_units:
            d["lcn"] = uom_val
            d["uom"] = None

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            _repair_item(item)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return parsed


# Boundary marker used only internally by _normalize_with_boundary_marker
# below -- deliberately a control character (never appears in real OCR
# text or model output), so it can't collide with any actual value while
# still surviving the "collapse remaining whitespace" pass that follows.
_GAP_BOUNDARY_MARKER = "\x00"


def _normalize_with_boundary_marker(text: str) -> str:
    """
    Same "run of 2+ OCR placeholders = column/field boundary, single
    placeholder = normal word-spacing" signal used throughout this file
    (table headers, _ensure_source_labels_captured), applied here to
    produce ONE normalized string where a real column boundary is a
    distinct, searchable marker and everything else reads as plain,
    single-spaced text -- lets a caller check "were these two values
    adjacent in the same phrase" with a plain substring test rather than
    re-deriving gap widths itself.
    """
    text = _BIG_GAP_RE.sub(_GAP_BOUNDARY_MARKER, text)
    text = text.replace(_OCR_WS_PLACEHOLDER, " ")
    return re.sub(r"[ \t]+", " ", text)


def _find_description_model_merge_candidates(parsed, source_text):
    """
    Returns a list of (line_item_dict, description, model_value) for
    every line item where `model` is non-empty AND, per the source text,
    `description` and `model` sit right next to each other separated
    only by normal single-word spacing -- i.e. NOT across a real column
    boundary -- making `model` structurally look like it's actually just
    the tail end of the same description phrase rather than its own
    field. This only identifies STRUCTURAL candidates (a fast, cheap,
    deterministic text check); _repair_description_model_split below
    still gets the model's judgment on each one before merging anything,
    since adjacency alone doesn't prove a short code isn't genuinely a
    real, distinct model number that just happens to be printed right
    after the description.
    """
    normalized = _normalize_with_boundary_marker(source_text)
    candidates = []

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        desc = item.get("description")
                        model_val = item.get("model")
                        if not (isinstance(desc, str) and desc.strip()):
                            continue
                        if not (isinstance(model_val, str) and model_val.strip()):
                            continue
                        joined = f"{desc.strip()} {model_val.strip()}"
                        if joined in normalized:
                            candidates.append((item, desc.strip(), model_val.strip()))
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return candidates


def _repair_description_model_split(parsed, source_text, timeout_secs: float = 20):
    """
    Fixes description/model splitting on line items: confirmed in
    production, a single continuous description phrase in the source
    (e.g. "Sell-Out/Sell-Thru Supports", joined by ordinary single-space
    OCR gaps with no real column boundary between the two words) gets
    split across two fields on some rows -- `description: "Sell-Out/
    Sell-Thru"`, `model: "Supports"` -- while a sibling row with the
    exact same phrase in its `description` correctly keeps it intact and
    leaves `model` null. The split isn't random: the second word is real
    text that legitimately reads as a plausible "model" value in
    isolation, which is exactly why a fixed rule can't safely resolve
    it -- some short trailing words genuinely ARE separate model/product
    codes.

    Classification is delegated to a dedicated, targeted model call, the
    same design as _repair_uom_lcn_confusion above: candidates are found
    structurally first (see _find_description_model_merge_candidates --
    cheap, deterministic, and keeps the LLM call scoped to only genuinely
    ambiguous pairs instead of every line item), then the MODEL decides,
    per pair, whether the candidate is really a continuation of the
    description or a genuinely distinct value -- world knowledge about
    what looks like a real product code versus ordinary descriptive text
    isn't something a fixed rule in this codebase should be trusted to
    encode.

    Distinct (description, model) pairs are deduplicated before the call
    so a split repeated across many sibling rows (as in production) costs
    one classification, not one per row.

    Fails safe: any problem with the call or its response leaves
    `parsed` completely untouched, same as _repair_uom_lcn_confusion --
    this is a repair, not a required step.

    Only merges pairs the model calls "same_phrase"; every other pair
    (including anything the response doesn't mention) is left exactly as
    the extraction produced it.
    """
    candidates = _find_description_model_merge_candidates(parsed, source_text)
    if not candidates:
        return parsed

    unique_pairs = []
    seen = set()
    for _, desc, model_val in candidates:
        key = (desc, model_val)
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)

    lines = [
        f'{i}. description: "{desc}" | candidate_model: "{model_val}"'
        for i, (desc, model_val) in enumerate(unique_pairs, 1)
    ]

    prompt = (
        "Each numbered pair below was extracted from one line item of an "
        "invoice or purchase-order line-item table: a 'description' value, "
        "and a 'candidate_model' value that immediately follows it in the "
        "source text with only normal word-spacing between them (no real "
        "column gap). For EACH pair, decide: is candidate_model actually "
        "just a continuation of the SAME free-text description -- together "
        "they read as one natural descriptive phrase, and there is no real "
        "separate model/product code here -- or is it a genuinely distinct "
        "product code, model number, or SKU that happens to be printed "
        "right after the description?\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object mapping each pair's number "
        "(as a string, e.g. \"1\") to either \"same_phrase\" or "
        "\"distinct_model\". No prose, no markdown fences, just the JSON "
        "object."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.15, 8))
        classification, _ = _try_parse_json(raw)
    except Exception:
        return parsed  # fail safe -- no-op rather than guess

    if not isinstance(classification, dict):
        return parsed

    same_phrase_pairs = set()
    for i, pair in enumerate(unique_pairs, 1):
        verdict = classification.get(str(i), classification.get(i))
        if isinstance(verdict, str) and verdict.strip().lower() == "same_phrase":
            same_phrase_pairs.add(pair)

    if not same_phrase_pairs:
        return parsed

    for item, desc, model_val in candidates:
        if (desc, model_val) in same_phrase_pairs:
            item["description"] = f"{desc} {model_val}"
            item["model"] = None

    return parsed


def _looks_like_row_sequence(items, key: str = "item_no") -> bool:
    """
    Purely a numeric-SHAPE check, not a content judgment: does `key`'s
    value on every item in `items`, read in row order, form exactly the
    sequence 1, 2, 3, ...len(items)? A real product/item code in these
    documents is never a bare integer that happens to match its own row
    position -- this shape is specific to a row-sequence/"No." column,
    so it's a safe, deterministic signal that `key` on this list is
    holding the wrong column's value.
    """
    values = []
    for it in items:
        v = it.get(key)
        if v is None:
            return False
        s = str(v).strip()
        if not s.isdigit():
            return False
        values.append(int(s))
    return bool(values) and values == list(range(1, len(values) + 1))


def _repair_row_number_mistaken_for_item_code(parsed, source_text, timeout_secs: float = 20):
    """
    Fixes a distinct extraction failure from the repairs above: on a
    table whose header has SEPARATE "No." (row sequence) and "Item No."
    (real product/item code) columns, the model sometimes extracts the
    SEQUENCE number (1, 2, 3...) into the item_no field and drops the
    real product code entirely -- confirmed in production on a
    delivery-order table where item_no came out as "1"/"2"/"3" while
    the document's real item codes ("EWH-09-33C-7-11W", ...) never
    appeared anywhere in the output.

    Structural candidate detection: a `line_items` list only qualifies
    if EVERY item's item_no is a bare integer forming exactly the
    sequence 1, 2, 3... (see _looks_like_row_sequence) -- never fires
    on a list whose item_no values already look like real codes.

    For each qualifying item, the row's own raw source line is located
    by anchoring on its KNOWN sequence number as the first token of its
    physical line (sturdier than re-matching the model's already-
    merged `description`, which can fail across a wrap point no single
    raw line reproduces verbatim). That line AND the one immediately
    after it are given to the model AS TWO SEPARATE PIECES OF TEXT,
    explicitly labeled -- not pre-joined into one flat string -- because
    the real item code can itself wrap across that same line break (a
    prefix ending the first line, the rest starting the second), the
    same way a description can. Deliberately left for the MODEL to
    notice and reconstruct rather than a hardcoded rule in this code
    (e.g. "a trailing dash means treat the next line's first token as a
    continuation") -- that kind of formatting convention varies by
    document and OCR pass, while a model reading two adjacent lines of
    a table row can recognize a split code the way a person would.

    Grounding is verified generically against BOTH lines rather than
    requiring one contiguous match: the returned code is accepted if
    it's found whole in either line, OR if some split of it has its
    first part appearing anywhere in line 1 and its second part
    appearing anywhere in line 2 -- deliberately not constrained to the
    exact tail/head of each line, since a wrapped code's prefix can sit
    ahead of other row content (the rest of the description, a
    quantity) that follows it before the line actually ends. This
    covers a code entirely on one line and a code that wraps across
    the two, without encoding any assumption about WHERE on the line
    the wrap happens.

    Nothing is discarded: on a match, the ORIGINAL sequence number is
    preserved under a new `no` field (matching what that column is
    actually called on the source document -- "No." vs "Item No.")
    rather than being silently dropped, and `item_no` is overwritten
    with the real code.

    Fails safe: any problem with the call or its response leaves
    `parsed` completely untouched; any candidate the model can't find a
    grounded code for is left exactly as it was.
    """
    normalized_lines = []
    for line in source_text.splitlines():
        n = line.replace(_OCR_WS_PLACEHOLDER, " ")
        n = re.sub(r"\s+", " ", n).strip()
        normalized_lines.append(n)

    def _find_row_lines(row_no):
        target = str(row_no)
        for idx, n in enumerate(normalized_lines):
            if n == target or n.startswith(target + " "):
                line2 = normalized_lines[idx + 1] if idx + 1 < len(normalized_lines) else ""
                return n, line2
        return None, None

    def _is_grounded(code, line1, line2):
        code_c = re.sub(r"\s+", "", code.lower())
        l1_c = re.sub(r"\s+", "", line1.lower())
        l2_c = re.sub(r"\s+", "", line2.lower())
        if code_c in l1_c or code_c in l2_c:
            return True
        # A wrapped code's prefix isn't necessarily at the very END of
        # line 1 -- other row content (the rest of the description, a
        # quantity) can follow it before the line ends, since columns
        # interleave across the wrap rather than cleanly stopping at
        # the code. So this checks whether SOME split of the code has
        # its first part appearing anywhere in line 1 and its second
        # part appearing anywhere in line 2, not specifically at the
        # line's tail/head -- still a genuine grounding check (both
        # fragments must be real substrings of the row's own two
        # lines), just not over-constrained on position. A minimum
        # fragment length avoids a trivial single-character match.
        for split in range(2, len(code_c) - 1):
            if code_c[:split] in l1_c and code_c[split:] in l2_c:
                return True
        return False

    candidates = []  # (item, old_no, description, line1, line2)

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    items = [it for it in value if isinstance(it, dict)]
                    if items and all("item_no" in it for it in items) and _looks_like_row_sequence(items):
                        for it in items:
                            old_no = str(it["item_no"]).strip()
                            line1, line2 = _find_row_lines(old_no)
                            if line1:
                                candidates.append((it, old_no, it.get("description"), line1, line2))
                else:
                    _walk(value)
        elif isinstance(node, list):
            for it in node:
                _walk(it)

    _walk(parsed)
    if not candidates:
        return parsed

    lines = [
        f'{i}. row_sequence_number: "{no}", description: "{desc}", '
        f'line_1: "{l1}", line_2: "{l2}"'
        for i, (_, no, desc, l1, l2) in enumerate(candidates, 1)
    ]
    prompt = (
        "Each numbered entry below is one row from a document table, given "
        "as its two raw source text lines (line_1 is the row's own line; "
        "line_2 is whatever comes immediately after it, which may just be "
        "the NEXT row or may be a wrapped continuation of THIS row). The "
        "table has separate columns for a row SEQUENCE number and a real "
        "ITEM/PRODUCT CODE, but only the sequence number was extracted so "
        "far. For EACH entry, find the real item/product code: a distinct "
        "token (often a mix of letters, digits, and dashes, e.g. "
        "\"EWH-09-33C-7-11W\") that is NOT the row_sequence_number and NOT "
        "part of the description. IMPORTANT: the code itself may be split "
        "across line_1 and line_2 -- e.g. line_1 ends with \"EWH-R02-33C-\" "
        "and line_2 begins with \"COGM\", meaning the real code is "
        "\"EWH-R02-33C-COGM\" with the two fragments joined directly "
        "together (no space). Reconstruct the full code in that case. If "
        "there's genuinely no such code anywhere in line_1 or line_2 "
        "(only the sequence number and description), return null.\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object mapping each entry's number "
        "(as a string) to either the code as a string, or null. No prose, "
        "no markdown fences, just the JSON object."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.2, 10))
        classification, _ = _try_parse_json(raw)
    except Exception:
        return parsed  # fail safe -- no-op rather than guess

    if not isinstance(classification, dict):
        return parsed

    for i, (item, old_no, desc, line1, line2) in enumerate(candidates, 1):
        code = classification.get(str(i), classification.get(i))
        if not (isinstance(code, str) and code.strip()):
            continue
        code = code.strip()

        if not _is_grounded(code, line1, line2):
            continue
        if re.sub(r"\s+", "", code.lower()) == old_no.lower():
            continue

        if "no" not in item:
            item["no"] = old_no
        item["item_no"] = code

    return parsed


def _find_role_field_groups(node):
    """
    Finds top-level `{role}_name` fields on `node` (e.g. "customer_name",
    "supplier_name") and their sibling `{role}_*` fields (address, phone,
    fax, no, registration_number, ...). The role vocabulary here is
    whatever prefix the EXTRACTION ITSELF already chose -- this reads
    the schema's own field names, it doesn't invent or assume a fixed
    set of role words (customer/supplier/vendor/buyer/... all work
    identically, whatever the model happened to name them).
    """
    groups = {}
    if not isinstance(node, dict):
        return groups
    for key, value in node.items():
        if key.endswith("_name") and isinstance(value, str) and value.strip():
            prefix = key[: -len("_name")]
            if prefix:
                groups[prefix] = {"name": value.strip(), "fields": {}}
    if len(groups) < 2:
        return {}
    for key, value in node.items():
        for prefix in groups:
            if key != prefix + "_name" and key.startswith(prefix + "_"):
                groups[prefix]["fields"][key[len(prefix):]] = value
    return groups


_HEADER_LINE_COUNT = 8


def _find_document_body_start(normalized_lines) -> int:
    """
    Returns the index of the first line AFTER the YAML frontmatter block
    (delimited by --- ... ---) if present, else 0. Used only so "near
    the top of the document" is measured relative to the document's own
    body -- pipeline metadata lines (attachment_id, source, engines...)
    shouldn't count as part of a document's letterhead/header area.
    """
    if normalized_lines and normalized_lines[0].strip() == "---":
        for i in range(1, len(normalized_lines)):
            if normalized_lines[i].strip() == "---":
                return i + 1
    return 0


def _find_name_context(source_text, name, window: int = 2):
    """
    Locates the source line containing `name` (matched on its first few
    words, gap-collapsed) and returns it together with `window` lines
    on either side, joined -- e.g. a role label like "Supplier Name"
    printed on the line just above a company's name, which the model
    can then read directly rather than needing this code to know what
    a role label looks like.

    Also reports whether the match falls within the first
    _HEADER_LINE_COUNT lines of the document BODY -- a purely
    POSITIONAL fact, not a content judgment. This exists because a
    party's role isn't always recoverable from an explicit label at
    all: confirmed in production, one party had an explicit "Customer
    No." label nearby, but the OTHER party had no label whatsoever --
    just their own contact details sitting as the page's letterhead at
    the very top of the document, which is how a business document's
    own issuer conventionally appears. Surfacing that positional fact
    lets the model use it as a second signal when an explicit label
    isn't there for one side, without this code itself asserting "the
    company at the top is always the vendor" as a hardcoded rule -- the
    model still makes that call, same as everything else here.

    Returns (context, is_near_top), or (None, False) if not found.
    """
    words = name.strip().split()[:3]
    if not words:
        return None, False
    needle = " ".join(words).lower()
    normalized_lines = []
    for line in source_text.splitlines():
        n = line.replace(_OCR_WS_PLACEHOLDER, " ")
        n = re.sub(r"\s+", " ", n).strip()
        normalized_lines.append(n)
    body_start = _find_document_body_start(normalized_lines)
    for idx, n in enumerate(normalized_lines):
        if needle in n.lower():
            start = max(0, idx - window)
            end = min(len(normalized_lines), idx + window + 1)
            context = " | ".join(l for l in normalized_lines[start:end] if l)
            is_near_top = 0 <= (idx - body_start) < _HEADER_LINE_COUNT
            return context, is_near_top
    return None, False


def _repair_reversed_party_roles(parsed, source_text, timeout_secs: float = 20):
    """
    Fixes a distinct extraction failure from the field-content repairs
    above: on a two-party document, the extraction can assign the wrong
    ROLE prefix to each party's field group entirely -- confirmed in
    production on a purchase order where the source document itself
    explicitly labels one company "Supplier Name", but that company's
    details ended up under `customer_*` fields, while the OTHER party
    (the company that actually issued the PO) ended up mislabeled as
    `supplier_*`. This is a structural/semantic mismatch, not a missing
    or misspelled value -- every individual field is internally
    consistent, the whole GROUP is just filed under the wrong role.

    Structural candidate detection (_find_role_field_groups): finds
    `{role}_name` field groups already present in the extraction, using
    whatever role prefixes the extraction itself chose (no fixed
    vocabulary of role words is assumed or required). Only handles the
    common two-party case (exactly two role groups on one dict) --
    documents with more than two named parties are left untouched
    rather than guessed at.

    For each party, the raw source text around their name AND whether
    their name sits near the very top of the document (see
    _find_name_context) are handed to the model together with the two
    role prefixes actually in use. The model is asked which prefix the
    SOURCE DOCUMENT ITSELF assigns to each party, using BOTH signals --
    an explicit role word near their name (e.g. "Supplier Name", "Bill
    To") when present, and document position when it isn't (confirmed
    in production: a document had an explicit label for the customer
    but none at all for the vendor, who only appeared as the page's own
    letterhead -- text-only search alone answered "unclear" for that
    side and a genuine reversal went unfixed).

    Only acts on a clean, confident, SYMMETRIC swap: both parties' model-
    determined roles must be found, and they must be EXACTLY reversed
    from how the extraction currently has them. An "unclear" answer for
    either party, or a determination that doesn't cleanly swap, leaves
    everything untouched -- this only fires when the model is confident
    the assignment is backwards, not merely uncertain.

    Recurses into nested dicts, since a multi-document source (delivery
    order + goods receipt note, etc.) can have its own independent role
    groups at each level. Fails safe: any problem with the call or its
    response for a given node leaves that node's fields untouched.
    """
    def _repair_node(node):
        groups = _find_role_field_groups(node)
        if len(groups) != 2:
            return
        (prefix_a, group_a), (prefix_b, group_b) = list(groups.items())
        if group_a["name"].strip().lower() == group_b["name"].strip().lower():
            return

        ctx_a, top_a = _find_name_context(source_text, group_a["name"])
        ctx_b, top_b = _find_name_context(source_text, group_b["name"])
        if not ctx_a or not ctx_b:
            return

        position_note_a = (
            "This party's name appears near the very TOP of the document "
            "(within its first few lines) -- often, though not always, "
            "where a business document prints its own issuer as a "
            "letterhead."
            if top_a else
            "This party's name does not appear near the top of the "
            "document."
        )
        position_note_b = (
            "This party's name appears near the very TOP of the document "
            "(within its first few lines) -- often, though not always, "
            "where a business document prints its own issuer as a "
            "letterhead."
            if top_b else
            "This party's name does not appear near the top of the "
            "document."
        )

        prompt = (
            f"An extraction pipeline used the role labels \"{prefix_a}\" "
            f"and \"{prefix_b}\" for two parties named on a business "
            f"document. For EACH party below, determine which of the two "
            f"role labels (\"{prefix_a}\" or \"{prefix_b}\") the DOCUMENT "
            f"ITSELF actually assigns to them. Use TWO kinds of evidence, "
            f"in this priority order: (1) any explicit role word printed "
            f"near their name in the raw text (e.g. \"Supplier Name\", "
            f"\"Customer\", \"Vendor\", \"Bill To\", \"Deliver To\") -- "
            f"trust this over everything else when present; (2) if no "
            f"explicit role word is present for a party, you may use "
            f"where on the document their name appears (a company at the "
            f"very top of a business document is often, but not always, "
            f"that document's own issuer) as a weaker secondary signal. "
            f"If neither gives you real confidence for a party, answer "
            f"\"unclear\" for them rather than guessing.\n\n"
            f"Party 1: name = \"{group_a['name']}\", raw context = "
            f"\"{ctx_a}\". {position_note_a}\n"
            f"Party 2: name = \"{group_b['name']}\", raw context = "
            f"\"{ctx_b}\". {position_note_b}\n\n"
            f"Respond with ONLY a JSON object: "
            f'{{"party_1_role": "...", "party_2_role": "..."}}, where each '
            f"value is exactly \"{prefix_a}\", \"{prefix_b}\", or "
            f"\"unclear\". No prose, no markdown fences, just the JSON "
            f"object."
        )

        try:
            raw = _call_ollama(prompt, max(timeout_secs * 0.2, 10))
            result, _ = _try_parse_json(raw)
        except Exception:
            return  # fail safe -- no-op rather than guess

        if not isinstance(result, dict):
            return
        role_1 = result.get("party_1_role")
        role_2 = result.get("party_2_role")
        if not (isinstance(role_1, str) and isinstance(role_2, str)):
            return
        role_1, role_2 = role_1.strip(), role_2.strip()

        if role_1 == prefix_b and role_2 == prefix_a:
            for prefix in (prefix_a, prefix_b):
                for key in list(node.keys()):
                    if key == prefix + "_name" or key.startswith(prefix + "_"):
                        del node[key]
            node[prefix_b + "_name"] = group_a["name"]
            for suffix, val in group_a["fields"].items():
                node[prefix_b + suffix] = val
            node[prefix_a + "_name"] = group_b["name"]
            for suffix, val in group_b["fields"].items():
                node[prefix_a + suffix] = val

    def _walk(node):
        if isinstance(node, dict):
            _repair_node(node)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return parsed


def _find_line_item_key_variant_candidates(parsed):
    """
    Returns a list of `line_items` lists, each paired with the set of
    field-key pairs that look like they might be the SAME conceptual
    field spelled two different ways (e.g. "disc_%" vs "disc_pct").

    Structural signal, not a content judgment: a key pair is a candidate
    only if the two keys are MUTUALLY EXCLUSIVE across every item in
    that list -- no single item ever has both keys populated at once.
    That's the same shape a genuine spelling variant produces (a field
    the model names one way on some rows and another way on other rows
    of the SAME table), and it's cheap and safe to compute without any
    domain knowledge about what field names mean.
    """
    results = []

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    items = [it for it in value if isinstance(it, dict)]
                    if len(items) < 2:
                        continue
                    all_keys = set()
                    for it in items:
                        all_keys.update(it.keys())
                    candidate_pairs = set()
                    keys_list = sorted(all_keys)
                    for i in range(len(keys_list)):
                        for j in range(i + 1, len(keys_list)):
                            k1, k2 = keys_list[i], keys_list[j]
                            both_populated = any(
                                it.get(k1) not in (None, "") and it.get(k2) not in (None, "")
                                for it in items
                            )
                            if both_populated:
                                continue
                            # At least one item must use k1 and at least
                            # one (different) item must use k2, or this
                            # isn't actually a cross-row spelling split
                            # -- just two fields nobody happens to use
                            # together yet.
                            uses_k1 = any(it.get(k1) not in (None, "") for it in items)
                            uses_k2 = any(it.get(k2) not in (None, "") for it in items)
                            if uses_k1 and uses_k2:
                                candidate_pairs.add((k1, k2))
                    if candidate_pairs:
                        results.append((items, candidate_pairs))
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return results


# Expected spellings for the three numbers a line item's own arithmetic
# is reconciled against. Kept intentionally generic (not tied to any
# one doc_type) since this pipeline's extraction produces different
# spellings for the same concept across documents (this file's own
# quantity/amount fields alone: "quantity"/"qty", "unit_price"/
# "unitCost", "amt_sgd"/"amount"/"lineAmount" have all been observed --
# see the SIN26-02534/PO-TB10095304/GRN-B10096605/PPINV-B10102384
# examples this repair was written against). Matched via the same
# _match_field_alias exact-then-substring mechanism used for canonical
# fields elsewhere in this file.
_LINE_ITEM_QTY_ALIASES = ["quantity", "qty", "qty_ordered", "order_qty", "quantityordered"]
_LINE_ITEM_UNIT_PRICE_ALIASES = ["unit_price", "unitprice", "unit_cost", "unitcost", "price"]
_LINE_ITEM_AMOUNT_ALIASES = [
    "amt_sgd", "amount_sgd", "amount", "line_amount", "lineamount",
    "linetotal", "line_total", "total_amount", "ext_amount", "extamount",
]


def _to_line_item_number(value) -> "float | None":
    """Best-effort numeric coercion for a line-item field that might be
    a real number, a numeric string, or a currency-formatted string
    (commas, a leading currency symbol). Returns None for anything that
    doesn't reduce to a plain number -- never raises."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value)
        if not cleaned or cleaned in ("-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _repair_quantity_amount_reconciliation(parsed, qty_tolerance: float = 0.02):
    """
    Corrects a line item's quantity when it doesn't reconcile with that
    same row's OWN unit_price and amount -- quantity * unit_price
    should equal the line amount, by definition, on every real invoice/
    PO/GRN line. Confirmed production case this was written for: the
    same source document, re-extracted multiple times, sometimes
    correctly read quantity as 2 and sometimes as 9 -- a stray digit
    from elsewhere in the noisy OCR text (a "9" embedded in that row's
    own item code, "EWH-09-33C-7-11W", positioned much closer in the
    raw text stream than the real "2 pc" quantity value, which sat far
    to the right past a long run of placeholder/whitespace characters).
    unit_price (206.81) and amount (413.62) were extracted correctly on
    every run regardless -- only quantity varied -- so trusting the
    arithmetic relationship between the three numbers catches and fixes
    the wrong one deterministically, with no dependence on which
    specific value the model happened to misread that run.

    Deliberately NOT a model call: unlike _repair_zone_bin_code_confusion
    or _repair_uom_lcn_confusion (which need a model to judge whether an
    ambiguous stray FIELD NAME represents a given concept), there's no
    naming ambiguity to resolve here -- quantity, unit_price, and amount
    are already correctly identified fields (via the same alias-matching
    used for canonical fields elsewhere in this file). The only question
    is whether their VALUES agree, which is arithmetic, not judgment --
    a model call would add latency and a small chance of a wrong verdict
    for something plain division already answers exactly.

    Conservative on both ends:
    - Requires ALL THREE of quantity, unit_price, and amount to already
      be present as recognized fields on the item (via
      _LINE_ITEM_QTY_ALIASES / _LINE_ITEM_UNIT_PRICE_ALIASES /
      _LINE_ITEM_AMOUNT_ALIASES) -- never invents a new field name that
      wasn't already present, and skips entirely (no correction, no
      flag) if any of the three is missing or non-numeric, since there's
      nothing to reconcile against.
    - Only corrects quantity when amount / unit_price lands within
      `qty_tolerance` of a whole number. If it doesn't, that means
      unit_price or amount is more likely the field that's actually
      wrong (not quantity) -- guessing a fractional "correction" in
      that case would just substitute one wrong number for another, so
      this leaves the row untouched rather than guessing.
    - Only writes a correction when the CURRENT quantity doesn't already
      match the implied one within the same tolerance -- an
      already-consistent row is left completely alone.

    Every correction is also appended to top-level "needs_human_review"
    (see _flag_sibling_inconsistent_nulls for the same convention) with
    the original and corrected values and the arithmetic that justified
    the change, so an auto-corrected quantity is never silently
    invisible to a reviewer even though this function is confident
    enough to apply it automatically rather than only flagging it.

    Fails safe: any item that doesn't cleanly qualify is left untouched;
    this never raises.
    """
    flags = []

    def _repair_items(items):
        if not items or not all(isinstance(i, dict) for i in items):
            return
        for idx, item in enumerate(items):
            qty_key = _match_field_alias(item, _LINE_ITEM_QTY_ALIASES)
            price_key = _match_field_alias(item, _LINE_ITEM_UNIT_PRICE_ALIASES)
            amount_key = _match_field_alias(item, _LINE_ITEM_AMOUNT_ALIASES)
            if not (qty_key and price_key and amount_key):
                continue  # need all three already-identified fields to reconcile

            price = _to_line_item_number(item.get(price_key))
            amount = _to_line_item_number(item.get(amount_key))
            if price is None or amount is None or price == 0:
                continue

            implied_qty = amount / price
            rounded_qty = round(implied_qty)
            if rounded_qty <= 0 or abs(implied_qty - rounded_qty) > qty_tolerance:
                # Doesn't land near a whole number -- unit_price or
                # amount is more likely the actually-wrong field here,
                # not quantity. Don't guess; leave the row untouched.
                continue

            current_qty = _to_line_item_number(item.get(qty_key))
            if current_qty is not None and abs(current_qty - rounded_qty) <= qty_tolerance:
                continue  # already consistent, nothing to correct

            original = item.get(qty_key)
            item[qty_key] = rounded_qty
            flags.append({
                "item_index": idx,
                "field": qty_key,
                "original_value": original,
                "corrected_value": rounded_qty,
                "reason": (
                    f"{qty_key} corrected from {original!r} to {rounded_qty} -- "
                    f"reconciled from {amount_key} ({amount}) / {price_key} "
                    f"({price}) = {implied_qty:.4f}, which the original value "
                    f"did not match"
                ),
            })

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    _repair_items(value)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)

    if flags and isinstance(parsed, dict):
        parsed.setdefault("needs_human_review", []).extend(flags)

    return parsed


def _repair_line_item_field_name_variants(parsed, timeout_secs: float = 20):
    """
    Fixes field-name spelling drift within one `line_items` array:
    confirmed in production, the SAME conceptual field ("Disc %" from
    the table header) got extracted as `disc_%` on two sibling rows and
    `disc_pct` on a third -- same value shape, same column, just a
    different key name depending on the row. This isn't a missing
    value (the third row's discount genuinely is populated, "0.00",
    same as its siblings), but _flag_sibling_inconsistent_nulls still
    reports it as a null-looking gap, because it only compares by exact
    key name.

    Candidates are found structurally first (see
    _find_line_item_key_variant_candidates -- mutually exclusive key
    pairs within one line_items list, cheap and domain-agnostic), then
    the MODEL decides whether each candidate pair is really the same
    field under two spellings, or two genuinely different fields that
    simply haven't co-occurred yet. Same two-stage design as
    _repair_uom_lcn_confusion / _repair_description_model_split above,
    same reasoning for why: "are 'disc_%' and 'disc_pct' the same
    field" is a judgment call, not something a fixed rule should
    encode.

    When the model confirms a pair is the same field, every item in
    that list gets canonicalized onto ONE of the two keys -- whichever
    one more items already use (ties broken alphabetically, an
    arbitrary but deterministic rule with no domain meaning) -- and the
    other key is removed from every item it appeared on, after copying
    its value across first. Fails safe: any problem with the call or
    its response leaves `parsed` completely untouched.
    """
    groups = _find_line_item_key_variant_candidates(parsed)
    if not groups:
        return parsed

    # Dedup candidate pairs across all line_items lists in the document
    # (usually just one list, but this covers multi-table documents too)
    # so an identical pair repeated in several tables costs one
    # classification, not one per table.
    all_pairs = []
    seen = set()
    for _, pairs in groups:
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                all_pairs.append(pair)

    lines = [f'{i}. "{k1}" vs "{k2}"' for i, (k1, k2) in enumerate(all_pairs, 1)]
    prompt = (
        "Each numbered pair below lists two different JSON field-key names "
        "that both appeared somewhere across the SAME table's rows, on "
        "different rows (never both on the same row). For EACH pair, "
        "decide: are these two keys almost certainly the SAME real-world "
        "field, just spelled/formatted differently by an inconsistent "
        "extraction process (e.g. \"disc_%\" and \"disc_pct\" both meaning "
        "'discount percentage')? Or are they genuinely two DIFFERENT "
        "fields that simply haven't both had a value yet?\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object mapping each pair's number "
        "(as a string) to either \"same_field\" or \"different_fields\". "
        "No prose, no markdown fences, just the JSON object."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.15, 8))
        classification, _ = _try_parse_json(raw)
    except Exception:
        return parsed  # fail safe -- no-op rather than guess

    if not isinstance(classification, dict):
        return parsed

    same_field_pairs = set()
    for i, pair in enumerate(all_pairs, 1):
        verdict = classification.get(str(i), classification.get(i))
        if isinstance(verdict, str) and verdict.strip().lower() == "same_field":
            same_field_pairs.add(pair)

    if not same_field_pairs:
        return parsed

    for items, pairs in groups:
        for pair in pairs:
            if pair not in same_field_pairs:
                continue
            k1, k2 = pair
            k1_count = sum(1 for it in items if it.get(k1) not in (None, ""))
            k2_count = sum(1 for it in items if it.get(k2) not in (None, ""))
            if k1_count > k2_count:
                canonical, other = k1, k2
            elif k2_count > k1_count:
                canonical, other = k2, k1
            else:
                canonical, other = sorted((k1, k2))

            for it in items:
                if it.get(other) not in (None, "") and it.get(canonical) in (None, ""):
                    it[canonical] = it[other]
                if other in it:
                    del it[other]

    return parsed


def _flag_sibling_inconsistent_nulls(parsed, min_populated_ratio: float = 0.6):
    """
    Surfaces (does not attempt to fix) any line item whose field is null
    while that same field is populated on most sibling items in the same
    `line_items` array -- a self-referential signal computed fresh from
    each response's own data, so it generalizes across document types.
    Adds a top-level "needs_human_review" list of
    {"item_index", "field", "reason"} entries rather than guessing at a
    value.

    Two-tier threshold: `min_populated_ratio` (majority populated) is
    the strong-confidence case, flagged with the original wording. But
    a MINORITY populated -- even just one sibling out of many -- is
    still flagged, at lower confidence: confirmed production miss where
    only 1 of 3 line items had `qty` extracted despite the source OCR
    text showing the identical "<qty> <uom>" pattern on all three rows.
    Requiring a majority before flagging meant that miss was invisible
    to review even though the raw text made it obvious. The only case
    that stays silent is zero siblings populated -- that's ordinary
    "field doesn't apply to this document," not evidence of a miss.
    """
    flags = []

    def _scan_items_list(items):
        if not items or not all(isinstance(i, dict) for i in items):
            return
        all_keys = set()
        for item in items:
            all_keys.update(item.keys())
        for key in all_keys:
            populated = [i for i in items if i.get(key) not in (None, "")]
            if not populated:
                continue
            ratio = len(populated) / len(items)
            if ratio >= min_populated_ratio:
                reason = (
                    "null here but populated on {}/{} sibling rows -- likely "
                    "upstream OCR data loss rather than a genuinely blank "
                    "cell; verify against the source before trusting this value"
                ).format(len(populated), len(items))
            else:
                # Minority-populated: lower confidence than the majority
                # case above, but still worth a human glance rather than
                # silently trusting the null -- see docstring.
                reason = (
                    "null here but populated on {}/{} sibling rows (minority) -- "
                    "worth checking against the source; a shared row pattern "
                    "(e.g. the same '<qty> <uom>' shape) appearing on only "
                    "some rows often means the extraction missed it on the "
                    "others rather than it being genuinely absent"
                ).format(len(populated), len(items))
            for idx, item in enumerate(items):
                if item.get(key) in (None, ""):
                    flags.append({"item_index": idx, "field": key, "reason": reason})

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    _scan_items_list(value)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    if flags and isinstance(parsed, dict):
        parsed["needs_human_review"] = flags
    return parsed


def _verify_constant_fields_against_source(parsed, text, anchor_field="item_no"):
    """
    For any field with a DOMINANT value across the items in a
    `line_items` list (repeated on more than half the rows that have a
    non-null value for it, seen more than once -- e.g. a source/order
    number or date that legitimately repeats on every line item), and
    with at most 2 distinct non-null values overall (more than that
    means normal per-item variation, e.g. a "description" field, not a
    document-wide constant -- and the whole check backs off rather than
    guessing which values are "wrong"), cross-checks each item against
    its own row's slice of the raw OCR `text`, located via `anchor_field`
    (default "item_no").

    - NULL field, dominant value found in the row's own window: filled
      in directly (not a guess -- reading a value demonstrably present
      in that row's own source text).
    - Non-null field matching the dominant value, but NOT found in the
      row's own window: left untouched but flagged as possibly inferred
      from sibling pattern rather than actually read.
    - Non-null field DIFFERING from the dominant value: left untouched
      (may be a genuine exception) but flagged for review.

    Tolerant of a minority of rows disagreeing with the dominant value
    (rather than requiring total unanimity) -- a single corrupted or
    genuinely-different row must never veto recovery for every sibling.
    """
    def _windows_for(items):
        positions = []
        cursor = 0
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            anchor_val = item.get(anchor_field)
            if not isinstance(anchor_val, str) or not anchor_val:
                continue
            pos = text.find(anchor_val, cursor)
            if pos == -1:
                continue
            positions.append((idx, pos, len(anchor_val)))
            cursor = pos + len(anchor_val)
        if len(positions) < 2:
            return {}
        windows = {}
        for i, (idx, pos, alen) in enumerate(positions):
            start = 0 if i == 0 else positions[i - 1][1] + positions[i - 1][2]
            end = len(text) if i == len(positions) - 1 else positions[i + 1][1]
            windows[idx] = text[start:end]
        return windows

    def _process_items(items):
        flags = []
        windows = _windows_for(items)
        if not windows:
            return flags
        all_keys = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        for key in all_keys:
            if key == anchor_field:
                continue
            non_null_vals = [
                item[key] for item in items
                if isinstance(item, dict) and item.get(key) not in (None, "")
            ]
            if not non_null_vals or not all(isinstance(v, str) and v for v in non_null_vals):
                continue
            value_counts = {}
            for v in non_null_vals:
                value_counts[v] = value_counts.get(v, 0) + 1
            if len(value_counts) > 2:
                continue
            dominant_val, dominant_count = max(value_counts.items(), key=lambda kv: kv[1])
            if dominant_count < 2 or dominant_count <= len(non_null_vals) / 2:
                continue
            constant_val = dominant_val
            for idx, item in enumerate(items):
                if not isinstance(item, dict) or idx not in windows:
                    continue
                window = windows[idx]
                current = item.get(key)
                if current in (None, ""):
                    if constant_val in window:
                        item[key] = constant_val
                elif current == constant_val and constant_val not in window:
                    flags.append({
                        "item_index": idx,
                        "field": key,
                        "reason": (
                            f"value matches the dominant value used across "
                            f"sibling rows, but does not appear in this "
                            f"row's own source text -- may have been "
                            f"inferred from the sibling pattern rather "
                            f"than actually read; verify against source"
                        ),
                    })
                elif current not in (None, "") and current != constant_val:
                    flags.append({
                        "item_index": idx,
                        "field": key,
                        "reason": (
                            f"value differs from the dominant value used "
                            f"by {dominant_count}/{len(non_null_vals)} "
                            f"sibling rows -- may be a genuine exception "
                            f"or may reflect source corruption; verify "
                            f"against source before trusting it"
                        ),
                    })
        return flags

    all_flags = []

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    all_flags.extend(_process_items(value))
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    if all_flags and isinstance(parsed, dict):
        existing = parsed.get("needs_human_review", [])
        if not isinstance(existing, list):
            existing = []
        parsed["needs_human_review"] = existing + all_flags
    return parsed


def _looks_like_numbered_list(text: str, min_lines: int = 2, max_intro_words: int = 40) -> bool:
    """
    Ground-truth check for whether text is a CLEAN numbered list, not just
    a list that technically exists somewhere inside the text.

    Merely counting numbered lines isn't enough: a model asked for both a
    "4-6 sentence summary" AND a "numbered list" can satisfy the letter of
    both instructions by writing the full prose paragraph and THEN bolting
    a numbered list on afterward -- passing a naive "contains >=2 numbered
    lines" check while actually being redundant, doubled-up content (every
    item named twice: once in prose, once in the list). That is exactly
    the failure this stricter version catches.

    Requires:
      - at least `min_lines` separate numbered lines, AND
      - any text BEFORE the first numbered line is short (<= max_intro_words
        words) -- a brief one-line lead-in is fine, a full recap paragraph
        is not, AND
      - any text AFTER the last numbered line is similarly short -- no
        trailing prose recap of what the list already said.
    """
    lines = text.splitlines()
    match_indices = [i for i, line in enumerate(lines) if _NUMBERED_LIST_LINE_RE.match(line)]
    if len(match_indices) < min_lines:
        return False

    intro_text = "\n".join(lines[:match_indices[0]]).strip()
    outro_text = "\n".join(lines[match_indices[-1] + 1:]).strip()
    return len(intro_text.split()) <= max_intro_words and len(outro_text.split()) <= max_intro_words


_ECHO_CHECK_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for", "and",
    "or", "that", "this", "then", "every", "identify", "extract", "compare", "respond", "only",
    "name", "names", "list", "listing", "single", "specific", "distinct", "place", "places",
    "city", "cities", "country", "countries", "price", "prices", "mentioned", "text", "found",
    "destination", "attraction", "item", "items", "entity", "which", "what", "just", "nothing",
    "else", "with", "from", "also", "state", "stating", "explicitly", "provide", "summarize",
    "summarise", "full", "assessment", "quality", "reviews", "review", "how", "good", "nice",
    "stands", "stand", "out", "tied", "lowest", "highest", "top", "best", "rated", "rating",
    "who", "makes", "its", "their", "there", "these", "those", "one", "real", "actual", "given",
    "request", "original", "company", "person", "figure", "ticker", "symbol", "where", "lived",
    "found",
}


def _extraction_echoes_focus(summary: str, focus: str) -> bool:
    """
    Heuristic check for a different extraction failure than length: the
    model was asked to extract one KIND of entity (e.g. a place) but
    returned a proper noun that was already named IN the question itself
    (e.g. a person's own name), rather than genuinely new information
    pulled from the source text. Extraction exists to surface something
    the requester doesn't already know -- if a real content word from the
    returned entity also appears as a real content word in the focus text
    (after stripping our own instruction/boilerplate vocabulary), that's a
    strong, cheap signal it latched onto the wrong proper noun (e.g. the
    most-repeated name in a short biographical passage, instead of the
    actual place mentioned only once in passing).
    Word-level overlap rather than a whole-string substring check, since a
    multi-word wrong answer (e.g. "John Galt") may not appear verbatim in
    the focus even though its key content word ("Galt") does.
    """
    if not summary.strip() or not focus:
        return False
    summary_words = {
        w.lower() for w in re.findall(r"[A-Za-z]+", summary)
        if len(w) >= 3 and w.lower() not in _ECHO_CHECK_STOPWORDS
    }
    if not summary_words:
        return False
    focus_words = {
        w.lower() for w in re.findall(r"[A-Za-z]+", focus)
        if len(w) >= 3 and w.lower() not in _ECHO_CHECK_STOPWORDS
    }
    return bool(summary_words & focus_words)


_PLACE_TYPE_HINT_RE = re.compile(
    r"\b(cit(y|ies)|places?|locations?|towns?|regions?|countr(y|ies))\b",
    re.IGNORECASE,
)


def _extraction_looks_like_person_not_place(summary: str, focus: str, source_text: str) -> bool:
    """
    Heuristic for a THIRD extraction failure mode, distinct from both
    checks above: the focus explicitly asks for a PLACE-type entity (city,
    place, location, town, region, country), but the returned answer is
    actually being used as a PERSON's name in the source text -- e.g. "X
    is born ... as the oldest child of X" (a person mentioned twice) with
    the actual requested place ("Irvine, Scotland") appearing only once,
    in passing. _extraction_echoes_focus doesn't catch this: it only
    flags an answer that echoes words already present in the FOCUS text
    itself, but a focus phrased generically ("the subject") never names
    the person at all, so there's nothing to echo against -- the wrong
    answer is wrong relative to the SOURCE TEXT's usage, not the focus's
    wording. Cheap, mechanical proxy: does the returned name appear
    immediately before/after a person-indicating phrase ("X is born",
    "child of X", "son of X") in the source text? Can't positively verify
    an answer IS a place, but reliably flags this specific, previously
    observed failure shape.
    """
    if not summary.strip() or not focus or not source_text:
        return False
    if not _PLACE_TYPE_HINT_RE.search(focus):
        return False  # focus isn't asking for a place-type entity at all

    name = summary.strip().rstrip(".").lower()
    if not name:
        return False

    text_lower = source_text.lower()
    person_patterns = [
        f"{name} is born", f"{name} was born",
        f"child of {name}", f"son of {name}",
        f"daughter of {name}", f"children of {name}",
    ]
    return any(p in text_lower for p in person_patterns)


def _extraction_output_too_long(text: str, max_words: int = 6) -> bool:
    """
    Heuristic check for whether an extraction-mode response actually honored
    "respond with ONLY that entity, nothing else" -- catches the case where
    the model ignored extraction_override and returned a full sentence or
    paragraph instead of a bare name. Only ever called when
    needs_extraction_override was true for this request, so a positive here
    means the model's own output didn't match the mode it was asked for --
    not a judgment about the text's content.
    Deliberately permissive: multi-word entities ("Company A", "City B")
    are common and must NOT trip this, so the threshold is on the
    generous side and sentence count is checked as a second, independent
    signal (a bloated answer is almost always more than one sentence, even
    when short entity names alone would clear the word-count bar).
    """
    stripped = text.strip().rstrip(".")
    if not stripped:
        return False
    word_count = len(stripped.split())
    multiple_sentences = len(_SENTENCE_SPLIT_RE.split(stripped)) > 1
    return word_count > max_words or multiple_sentences


def _call_ollama(prompt: str, timeout_secs: float, guided_json: dict = None,
                  temperature: float = None) -> str:
    """
    Hoisted to module level (was previously a closure defined inside
    run()) so both the normal prose path and the JSON-output path below
    can share one implementation instead of duplicating the request
    plumbing.

    `guided_json`, when given, is passed straight through to vLLM as a
    top-level "guided_json" field -- vLLM's own constrained-decoding
    extension to the OpenAI-compatible endpoint (backed by outlines/
    xgrammar), NOT a prompt instruction. It restricts which tokens the
    model is even ALLOWED to sample at each step, so a field typed as a
    number in the schema literally cannot come back as a string with
    text stuck to it -- a stronger guarantee than asking nicely in the
    prompt, which the model can (and on production data, sometimes
    does) ignore. Schemas passed here are intentionally partial --
    typing only the specific field(s) known to be unreliable, with
    `additionalProperties: true` and no `required` list -- since the
    full column set varies per document and isn't known ahead of time;
    only the shape of the field(s) actually being fixed is enforced.

    `temperature`, when given, overrides the module default for this
    one call. Left as None (falls back to the existing 0.3 default) for
    every existing caller so this is purely additive; call sites doing
    constrained, structured extraction (as opposed to open-ended prose)
    can pass 0.0 to remove sampling variance on top of the schema
    constraint, since the two address different sources of
    inconsistency (what the model is ALLOWED to say vs. how much it
    varies run to run even within what's allowed).

    Raises RuntimeError with vLLM's OWN error response body included in
    the message on any non-2xx response -- plain r.raise_for_status()
    only produces a generic "400 Client Error: Bad Request for url:
    ..." with no indication of WHY, which meant every prior 400 (context
    length exceeded vs. a malformed request vs. anything else) had to be
    reverse-engineered from request size/timing after the fact instead
    of just being read off the response. vLLM's OpenAI-compatible error
    responses are JSON with a "message"/"error" field describing the
    actual problem (e.g. "This model's maximum context length is X
    tokens..."); fall back to the raw response text if the body isn't
    JSON-shaped, and to str(e) if even reading the body fails, so this
    is strictly more informative than before, never less.
    """
    # Process-wide gate: whatever pool this call came from (a per-document
    # ThreadPoolExecutor sized by MAX_CONCURRENT_MODEL_CALLS, or a totally
    # separate document being processed concurrently by this same waitress
    # process), only VLLM_CONCURRENCY_LIMIT of these are ever actually in
    # flight against vLLM at once. Everything past that queues here rather
    # than piling onto vLLM and risking a read timeout downstream.
    #
    # Bounded, NOT indefinite: acquire() has its own deadline
    # (VLLM_QUEUE_WAIT_SECS, independent of this call's own timeout_secs --
    # see that constant's comment for why). An unbounded .acquire() would
    # block silently for however long the backend stays saturated, and that
    # wait time counts against nothing in the timeout math above it --
    # including _timed_run()'s own overall per-job deadline
    # (payload.timeout_secs), which is enforced from OUTSIDE this thread and
    # can't be caught by any try/except in here. A job killed that way
    # returns no TaskResult at all ("stopped responding"), instead of the
    # graceful partial-failure result (_chunk_extraction_incomplete_pages
    # etc.) this file is designed to produce. Failing fast here with a
    # normal exception keeps saturation inside the same catchable, reported
    # failure path as any other model-call error.
    if not _vllm_semaphore.acquire(timeout=VLLM_QUEUE_WAIT_SECS):
        raise RuntimeError(
            f"vLLM request slot unavailable after waiting {VLLM_QUEUE_WAIT_SECS:.0f}s "
            f"-- VLLM_CONCURRENCY_LIMIT={VLLM_CONCURRENCY_LIMIT} is saturated by "
            f"other concurrent documents/calls right now."
        )
    try:
        body = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else 0.3,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if guided_json is not None:
            body["guided_json"] = guided_json
        r = _req.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=body,
            timeout=int(timeout_secs),
        )
    finally:
        _vllm_semaphore.release()
    if not r.ok:
        try:
            body = r.json()
            detail = (
                body.get("error", {}).get("message") if isinstance(body.get("error"), dict)
                else body.get("error") or body.get("message") or json.dumps(body)
            )
        except Exception:
            detail = (r.text or "")[:2000]
        raise RuntimeError(
            f"{r.status_code} {r.reason} from {VLLM_BASE_URL}/v1/chat/completions "
            f"-- vLLM said: {detail}"
        )
    return r.json()["choices"][0]["message"]["content"].strip()


# Strips a ```json ... ``` or bare ``` ... ``` fence the model sometimes
# wraps its JSON output in despite being told not to -- tried before
# json.loads() rather than instead of the "no fences" instruction, since
# instruction-following on formatting asks isn't guaranteed (same lesson
# as every other "HARD GUARANTEE" / verify-don't-trust pattern already in
# this file for list mode and extraction mode).
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_json_fence(text: str) -> str:
    m = _JSON_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def _try_parse_json(text: str):
    """Returns (parsed_or_None, cleaned_text_that_was_attempted)."""
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned), cleaned
    except (json.JSONDecodeError, ValueError):
        return None, cleaned


def _looks_like_meta_refusal(parsed, timeout_secs: float = 20) -> bool:
    """
    Ground-truth check for TWO distinct JSON-mode refusal shapes observed
    in production, both syntactically valid JSON so the parse-failure
    check alone can't catch either:
      1. The model notices leftover prose-formatting text in `focus` (a
         stray sentence-count instruction predating JSON mode) and treats
         it as contradicting the JSON-only rule, returning an object that
         comments on the contradiction instead of extracting data.
      2. The model notices the source document isn't one of the specific
         types a rule higher up described (e.g. "if the source is an
         invoice, receipt, quote, PO, or bill...") and refuses the ENTIRE
         extraction as a result, even though the document plainly
         contains other extractable content (item descriptions,
         quantities, reference numbers, parties, dates) -- just not
         pricing. This is the more damaging of the two: it discards
         everything rather than adapting the schema, whereas the first
         shape at least tends to preserve real fields alongside the
         complaint.
    Both surface as a single string-valued top-level field named error/
    note/message containing meta-language about the request or the
    document's type/suitability rather than actual document content, so
    one check covers both.

    The STRUCTURAL gate (single-key dict, key named error/note/message)
    stays a fixed rule -- that's the JSON shape itself, not a content
    judgment, and is what keeps this call rare (it's the near-total
    majority of successful extractions that never reach the classification
    step at all). Whether the TEXT in that field is actually refusal
    language, though, is delegated to a model call rather than a
    hardcoded phrase list: the old list ("contradictory", "cannot be
    extracted", "cannot be determined"...) only catches refusals phrased
    in one of those specific ways, in English -- a model has no such
    blind spot.

    Fails safe toward NOT retrying: if the classification call itself
    fails, this returns False (not a refusal) rather than True, so a
    call failure here never triggers an unnecessary retry -- consistent
    with every other repair in this file defaulting to "leave it alone"
    when its own verification step can't complete.
    """
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return False
    key = next(iter(parsed))
    if key.lower() not in ("error", "note", "message"):
        return False
    value = parsed[key]
    if not isinstance(value, str) or not value.strip():
        return False

    prompt = (
        f"The text below is the entire content of a JSON object returned "
        f"by a document-extraction task, under a single key named "
        f"\"{key}\". Decide: is this text a REFUSAL or META-COMMENTARY "
        f"about the request itself (e.g. complaining about contradictory "
        f"instructions, declining because the source document isn't a "
        f"specific expected type, saying required fields can't be "
        f"extracted) -- or is it genuine content that was actually found "
        f"ON the document (e.g. a real printed error code, a real note "
        f"field, a real short message field)?\n\n"
        f"Text: \"{value.strip()}\"\n\n"
        f"Respond with ONLY the single word \"refusal\" or "
        f"\"real_content\". No prose, no markdown fences, no punctuation, "
        f"just that one word."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.1, 6))
    except Exception:
        return False  # fail safe -- don't trigger a retry we can't confirm is needed

    return raw.strip().strip('"').strip(".").lower().startswith("refusal")


# Cheap lexical signal that a `line_items` entry's "description" is really
# legal/administrative boilerplate (a retention-of-title clause, interest
# terms, a footer disclaimer) rather than an actual product/service row --
# seen in production when a table header had zero real data rows beneath
# it and the model grabbed the nearest sentence-shaped text near the
# totals instead. Genuine product/service descriptions essentially never
# contain this kind of legal/administrative phrasing, so this is a
# reliable (if not exhaustive) filter, not a content judgment about
# whether an item is a "real" product.
def _find_boilerplate_line_item_candidates(parsed):
    """
    Structural candidates for _strip_boilerplate_line_items below: any
    line item whose ONLY populated field is `description`, everything
    else null/empty/zero. A real product/service row essentially always
    has SOME other field populated (a quantity, a code, a price) --
    description-only is already a strong shape signal on its own,
    independent of what the description text actually says, which is
    what makes it safe to narrow candidates this way before asking the
    model anything.
    """
    candidates = []

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        desc = item.get("description")
                        if not isinstance(desc, str) or not desc.strip():
                            continue
                        other_fields = {k: v for k, v in item.items() if k != "description"}
                        if all(v in (None, "", 0) for v in other_fields.values()):
                            candidates.append((item, desc.strip()))
                else:
                    _walk(value)
        elif isinstance(node, list):
            for it in node:
                _walk(it)

    _walk(parsed)
    return candidates


def _strip_boilerplate_line_items(parsed, timeout_secs: float = 20):
    """
    Removes any `line_items` entry that's actually legal/administrative
    boilerplate (a retention-of-title clause, interest terms, a footer
    disclaimer) rather than a real product/service row -- seen in
    production when a table header had zero real data rows beneath it
    and the model grabbed the nearest sentence-shaped text near the
    totals instead.

    Structural candidates come from _find_boilerplate_line_item_candidates
    (description-only items -- a shape signal, not a content judgment).
    Whether the description TEXT itself is genuinely boilerplate is then
    a model call rather than a hardcoded English phrase list -- the old
    list ("remains the property", "E&OE"...) only catches boilerplate
    phrased in one of those exact ways; a model generalizes to any
    phrasing or language. Distinct description texts are deduplicated
    across the whole document before the single classification call, so
    a repeated boilerplate line (e.g. the same footer text reappearing)
    costs one classification, not one per occurrence.

    A genuine line item that happens to mention payment terms in its own
    description text, but ALSO has a real quantity/price/code populated,
    is never even a candidate -- the null-fields co-occurrence from the
    structural filter is what distinguishes "this was never a real row"
    from "this is a real row with unusual wording," same as before.

    Recurses into nested dicts/lists rather than only checking a
    top-level `line_items` key, since a multi-document source (see the
    MULTI-PAGE / MULTI-DOCUMENT SOURCE CHECK prompt rule) is expected to
    nest each document's own line_items under its own key. Fails safe:
    any problem with the call or its response leaves every item exactly
    as it was -- nothing is ever removed without the model's affirmative
    classification.
    """
    candidates = _find_boilerplate_line_item_candidates(parsed)
    if not candidates:
        return parsed

    unique_descs = []
    seen = set()
    for _, desc in candidates:
        if desc not in seen:
            seen.add(desc)
            unique_descs.append(desc)

    lines = [f'{i}. "{d}"' for i, d in enumerate(unique_descs, 1)]
    prompt = (
        "Each numbered text below is the ONLY populated field on what was "
        "extracted as a line item from an invoice/PO-style document table "
        "(every other field on it is empty). For EACH one, decide: is this "
        "genuinely a product or service line item -- something that was "
        "bought, sold, delivered, or billed? Or is it actually legal or "
        "administrative boilerplate that doesn't belong as a line item at "
        "all -- e.g. a retention-of-title clause, interest/payment terms, "
        "a signature disclaimer, or a footer note like \"E&OE\"?\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object mapping each entry's number "
        "(as a string) to either \"real_item\" or \"boilerplate\". No "
        "prose, no markdown fences, just the JSON object."
    )

    try:
        raw = _call_ollama(prompt, max(timeout_secs * 0.15, 8))
        classification, _ = _try_parse_json(raw)
    except Exception:
        return parsed  # fail safe -- no-op rather than guess

    if not isinstance(classification, dict):
        return parsed

    boilerplate_descs = set()
    for i, desc in enumerate(unique_descs, 1):
        verdict = classification.get(str(i), classification.get(i))
        if isinstance(verdict, str) and verdict.strip().lower() == "boilerplate":
            boilerplate_descs.add(desc)

    if not boilerplate_descs:
        return parsed

    def _clean_items_list(items):
        def _is_bogus(item):
            if not isinstance(item, dict):
                return False
            if item.get("description") not in boilerplate_descs:
                return False
            other_fields = {k: v for k, v in item.items() if k != "description"}
            return all(v in (None, "", 0) for v in other_fields.values())
        return [item for item in items if not _is_bogus(item)]

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "line_items" and isinstance(value, list):
                    node[key] = _clean_items_list(value)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return parsed


# Matches one line of the OCR pipeline's own fixed-format handwriting
# section, e.g.:
#   - Page 1: handwriting 6: "GOODS RECEIVET& VISUAL!' IN2P::"E"
#   - Page 2: handwriting 4 (mark): "[unreadable]"
# The trailing `"(.*)"\s*$` is intentionally greedy: some OCR'd entries
# contain a stray embedded quote character in the annotation text itself
# (see the example above) -- greedy matching to the LAST quote on the
# line, rather than the first, is what keeps that entry intact instead
# of truncating at the embedded quote.
_HANDWRITING_LINE_RE = re.compile(
    r'^(?:-\s*)?Page\s+(\d+)\s*:\s*handwriting\s+\d+\s*(?:\([^)]*\))?\s*:\s*"(.*)"\s*$',
    re.IGNORECASE,
)


def _extract_handwriting_by_page(text: str) -> dict:
    """
    Deterministically parses the OCR pipeline's "## Detected handwritten/
    colored annotations" section (see _HANDWRITING_LINE_RE) into
    {page_number: [annotation_text, ...]}, in source order. This is a
    mechanical parse of a FIXED format the OCR tool always emits when
    present -- not a guess about arbitrary text -- so it's used as a
    ground truth to backstop the model's own (unreliable) inclusion of
    this section in the extracted JSON. Returns {} if the section, or
    no individual annotation lines, are present.
    """
    by_page = {}
    for line in text.splitlines():
        m = _HANDWRITING_LINE_RE.match(line.strip())
        if not m:
            continue
        page = int(m.group(1))
        by_page.setdefault(page, []).append(m.group(2))
    return by_page


# Same "large gap = field/column boundary" signal already used elsewhere
# in this file for table headers (see the character-position-alignment
# docstring above) -- reused here for document-metadata label:value pairs
# rather than table columns. A line like
# "Tax⌷Invoice⌷No.⌷⌷⌷⌷PSINV-B10596837" has single placeholder gaps WITHIN
# the label phrase and WITHIN the value, but 4 placeholders BETWEEN them
# -- splitting on runs of 2+ placeholders turns that into exactly two
# segments, ["Tax Invoice No.", "PSINV-B10596837"], with the multi-word
# label and value each correctly reassembled with real spaces.
_BIG_GAP_RE = re.compile(_OCR_WS_PLACEHOLDER + "{2,}")


def _split_line_into_gap_segments(line: str) -> list:
    segments = _BIG_GAP_RE.split(line)
    out = []
    for seg in segments:
        seg = seg.replace(_OCR_WS_PLACEHOLDER, " ")
        seg = re.sub(r"\s+", " ", seg).strip()
        if seg:
            out.append(seg)
    return out


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "field"


def _collect_string_values(node, out: list):
    """Recursively collects every string AND number leaf value in a
    JSON-like structure (dict/list nesting), used to build the 'has this
    value already been captured ANYWHERE in the output, under ANY field
    name' haystack below -- deliberately not keyed on field names, since
    the whole problem this backstops is the model using a different
    field name (or a different nesting) than expected, not necessarily a
    missing VALUE. Numbers are included (stringified) because the same
    monetary/reference value this function is checking for can just as
    easily have been captured as a real JSON number (e.g. 373.49) rather
    than the string this function finds in the source text ("373.49") --
    without this, an already-numeric field would look uncaptured and get
    wrongly re-added as a duplicate string field."""
    if isinstance(node, dict):
        for v in node.values():
            _collect_string_values(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_string_values(v, out)
    elif isinstance(node, str):
        out.append(node)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.append(str(node))


def _find_label_value_candidates(source_text: str, already_captured) -> list:
    """
    Deterministically enumerates candidate text spans that MIGHT encode a
    field label followed by its value -- purely from the OCR pipeline's
    own physical whitespace-gap encoding (splitting each line on runs of
    2+ placeholder characters, the same mechanical signal already used
    elsewhere in this file for table-header column boundaries). This is
    NOT a content/domain judgment -- it doesn't know what a "label" or
    "value" looks like, doesn't check word shape, digit counts, or
    language-specific suffix words. It only reads the document's own
    encoded spacing metadata, the same way splitting a CSV on commas
    isn't "hardcoding" anything about the data inside it. ALL semantic
    judgment -- is this really a label:value pair, and if so where does
    one end and the other begin -- is left entirely to the model in
    _ensure_source_labels_captured below.

    Three candidate shapes are produced, matching the OCR gap patterns
    confirmed in production:
      (a) each individual gap-segment, covering a label and its value
          glued by ONLY single gaps throughout (e.g. "UEN No.:
          199304709Z", where nothing in the source marks where the
          label ends and the value begins);
      (b) each ADJACENT PAIR of gap-segments joined with a space,
          covering a label and value that live in two separate,
          big-gap-delimited segments (e.g. "Tax Invoice No." followed
          by "PSINV-B10596837" as its own segment);
      (c) each pair of gap-segments with exactly ONE segment SKIPPED
          between them, covering a label and value separated by a
          short stray/misread segment sitting between them (confirmed
          in production: a "Total Excl. GST" label and its "165.21"
          value ended up as three separate gap-segments -- label,
          stray "1", value -- because of a single misread character
          in between, so the (b) adjacent-pair shape alone never
          produced a candidate containing both label and value
          together). This is still a purely structural/positional
          shape, not a content judgment about what the skipped
          segment means -- the model decides, same as any other
          candidate, whether skipping it was the right read.

    `already_captured(text)` is called to drop any candidate whose full
    text is already present somewhere in the current extraction, so only
    genuinely-uncaptured spans get sent to the model at all.
    """
    candidates = []
    seen = set()
    for line in source_text.splitlines():
        segments = _split_line_into_gap_segments(line)
        for seg in segments:
            if 3 <= len(seg) <= 160 and seg not in seen and not already_captured(seg):
                seen.add(seg)
                candidates.append(seg)
        for i in range(len(segments) - 1):
            combo = f"{segments[i]} {segments[i + 1]}"
            if 3 <= len(combo) <= 160 and combo not in seen and not already_captured(combo):
                seen.add(combo)
                candidates.append(combo)
        for i in range(len(segments) - 2):
            combo = f"{segments[i]} {segments[i + 2]}"
            if 3 <= len(combo) <= 160 and combo not in seen and not already_captured(combo):
                seen.add(combo)
                candidates.append(combo)
    return candidates


# Per-call chunk size, and an overall cap on how many chunks (i.e. how
# many model calls) one extraction will make -- payload-size and cost
# bounds only, not content filters. Confirmed necessary in production:
# a single-page invoice with a lot of blank/boilerplate fields (empty
# "Pick Order No.", "Team No.", "Delivery Time"...) produced 98
# candidates, and a single hard cutoff at 60 silently dropped the
# invoice's own GST/total lines because they happened to sit near the
# bottom of the page -- position in the document is not a signal of
# importance, so truncating by position is never safe. Chunking instead
# of truncating means every candidate gets a classification chance
# regardless of where in the document it falls; the chunk-count cap
# just bounds worst-case cost on an unusually candidate-heavy document.
_LABEL_VALUE_CHUNK_SIZE = 50
# Raised from 4 to 8 after adding the skip-one-segment candidate shape
# above roughly doubled typical candidate counts (confirmed in
# production: 231 candidates on one page after that change, versus
# ~100-125 before it) -- without raising this, the same position-based
# truncation loss this cap exists to prevent would recur simply because
# there are now more candidates to get truncated.
_MAX_LABEL_VALUE_CHUNKS = 8


def _classify_label_value_chunk(candidates: list, timeout_secs: float) -> dict:
    """
    One model call classifying up to _LABEL_VALUE_CHUNK_SIZE candidate
    spans. Returns {index (1-based, as given) -> {"label":..., "value":
    ...}} for every span the model resolved to a real field; entries the
    model returned null for, or that fail the grounding check, are
    simply absent from the result. Returns {} (not an exception) on any
    call/parse failure, so a bad chunk never breaks the others.
    """
    lines = [f'{i}. "{c}"' for i, c in enumerate(candidates, 1)]
    prompt = (
        "Each numbered text span below was pulled from a scanned business "
        "document (OCR output) using only whitespace-gap heuristics -- "
        "NONE of them have been checked yet for whether they actually "
        "represent a meaningful labeled field. For EACH span, decide: does "
        "it contain a field label immediately followed by that field's "
        "value (e.g. \"Tax Invoice No. PSINV-B10596837\" -> label "
        "\"Tax Invoice No.\", value \"PSINV-B10596837\"; \"UEN No.: "
        "199304709Z\" -> label \"UEN No.\", value \"199304709Z\"; "
        "\"Payment Terms CASH\" -> label \"Payment Terms\", value "
        "\"CASH\")? If so, extract the label and the value separately, "
        "exactly as they appear in the span (don't paraphrase or "
        "reformat) -- EXCEPT that a bracketed inline marker like "
        "\"[handwriting 3]\", \"[illegible]\", or \"[unreadable]\" is the "
        "OCR pipeline's own annotation, not real document content; if one "
        "appears in the middle of an otherwise-sensible label or value, "
        "treat it as noise and extract the label/value with it removed "
        "rather than concluding the span isn't a real field. If the span "
        "is NOT a real labeled field -- e.g. it's "
        "a company name, an address fragment, page boilerplate, a stray "
        "OCR artifact, or otherwise doesn't cleanly split into a label "
        "and its value -- return null for it.\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON object mapping each span's number "
        "(as a string) to either {\"label\": \"...\", \"value\": \"...\"} "
        "or null. No prose, no markdown fences, just the JSON object."
    )

    try:
        raw = _call_ollama(prompt, timeout_secs)
        classification, _ = _try_parse_json(raw)
    except Exception:
        return {}

    if not isinstance(classification, dict):
        return {}

    result = {}
    for i in range(1, len(candidates) + 1):
        entry = classification.get(str(i), classification.get(i))
        if isinstance(entry, dict):
            result[i] = entry
    return result


def _ensure_source_labels_captured(parsed: dict, source_text: str, timeout_secs: float = 20) -> dict:
    """
    Completeness safety net, same philosophy as _ensure_handwriting_
    captured above, for the more general failure it doesn't cover: on
    some runs the model drops entire document-metadata fields it isn't
    told about by name (confirmed in production -- an invoice's own
    "Tax Invoice No.", "Order Date", "Payment Terms", "Salesperson", and
    all three GST/total lines missing on one run of the SAME source
    document that had them present on another run). Prompt wording can
    raise the odds the model tries; only a re-check of the source text
    can GUARANTEE nothing the document actually states goes missing.

    Candidate text spans are found by _find_label_value_candidates
    above, using only the source's own OCR whitespace-gap encoding --
    that part is mechanical, not a content judgment. Whether each
    candidate genuinely IS a label:value pair, and where the label ends
    and the value begins, is then decided by the model (see
    _classify_label_value_chunk) rather than a fixed rule -- earlier
    versions of this function used a regex "does this look like a
    label" shape test plus a small hardcoded list of label-ending words
    (":", "no.", "date", "terms"...) to avoid mis-pairing, and that
    fixed list still missed real cases (confirmed in production: "UEN
    No.: 199304709Z", where label and value are glued by ordinary
    single-word spacing throughout with no gap-boundary between them at
    all, so there was no separate "value segment" for the old pairing
    logic to even find). Candidates are sent to the model in CHUNKS
    (see _LABEL_VALUE_CHUNK_SIZE / _MAX_LABEL_VALUE_CHUNKS) rather than
    one hard-truncated batch, specifically so a candidate's classification
    chance never depends on where in the document it happens to fall --
    confirmed in production that a single-cutoff design silently dropped
    an invoice's own GST/total lines purely because of their position on
    the page.

    Every value the model returns is validated against the candidate
    span it came from (whitespace-insensitive substring check) before
    being trusted -- if the model's answer isn't actually grounded in
    the source text it was given, it's discarded rather than trusted,
    consistent with the "verify, don't just trust" pattern already used
    throughout this file (JSON-fence stripping, meta-refusal retry,
    etc.).

    Fails safe per chunk: no candidates found -> no call made at all; a
    failed or unparseable chunk response contributes nothing (other
    chunks are unaffected). This is a repair, not a required step.

    Mutates and returns `parsed`. No-op if `parsed` isn't a dict.
    """
    if not isinstance(parsed, dict):
        return parsed

    existing_values = []
    _collect_string_values(parsed, existing_values)
    haystack = "\n".join(existing_values).lower()
    # Secondary, whitespace-collapsed comparison list -- kept as
    # per-value entries (never joined into one string) specifically to
    # avoid a cross-value boundary bleed: joining first and THEN
    # stripping whitespace would erase the newline separators between
    # entries, letting the tail of one value and the head of the next
    # accidentally concatenate into a false match. This catches
    # formatting-only mismatches the plain `haystack` substring check
    # above misses -- confirmed in production: the model wrote
    # "TRACY TING HENG MUI / 11184" (space before the slash) while the
    # OCR source has "TRACY TING HENG MUI /11184" (no space), which are
    # the same value but don't substring-match without this.
    existing_values_ws_collapsed = [re.sub(r"\s+", "", v.lower()) for v in existing_values]

    def _already_captured(text: str) -> bool:
        if text.lower() in haystack:
            return True
        collapsed = re.sub(r"\s+", "", text.lower())
        return any(collapsed in v for v in existing_values_ws_collapsed)

    candidates = _find_label_value_candidates(source_text, _already_captured)
    if not candidates:
        return parsed

    max_candidates = _LABEL_VALUE_CHUNK_SIZE * _MAX_LABEL_VALUE_CHUNKS
    candidates = candidates[:max_candidates]

    existing_keys = set(parsed.keys())
    backfilled = []
    chunk_timeout = max(timeout_secs * 0.25, 10)

    chunks = [
        candidates[i:i + _LABEL_VALUE_CHUNK_SIZE]
        for i in range(0, len(candidates), _LABEL_VALUE_CHUNK_SIZE)
    ]

    # Classification calls are independent of each other -- each chunk
    # only classifies its own candidates -- so they're dispatched
    # CONCURRENTLY rather than one chunk waiting on the last (this is
    # what made an earlier candidate-heavy document cost up to 8
    # sequential model calls just for this one repair). Results are
    # collected indexed by chunk position, THEN applied in that same
    # original order below, one at a time -- the APPLICATION step (the
    # dedup/grounding checks against `_already_captured`/`existing_keys`
    # /`haystack`) still has to run sequentially, since a value
    # backfilled from chunk 0 needs to correctly suppress a duplicate
    # found in chunk 1's results; only the network waiting is
    # parallelized, not the bookkeeping that depends on ordering.
    pool_size = min(len(chunks), MAX_CONCURRENT_MODEL_CALLS)
    classifications = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max(pool_size, 1)) as executor:
        chunk_futures = {
            executor.submit(_classify_label_value_chunk, chunk, chunk_timeout): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(chunk_futures):
            classifications[chunk_futures[future]] = future.result()

    for chunk, classification in zip(chunks, classifications):
        for local_i, candidate in enumerate(chunk, 1):
            entry = classification.get(local_i)
            if not isinstance(entry, dict):
                continue
            label, value = entry.get("label"), entry.get("value")
            if not (isinstance(label, str) and label.strip()):
                continue
            if not (isinstance(value, str) and value.strip()):
                continue
            label, value = label.strip(), value.strip()

            # Grounding check: the model's extracted value must actually
            # appear in the source span it was given -- guards against a
            # hallucinated or reformatted value that isn't really what
            # the document says.
            candidate_collapsed = re.sub(r"\s+", "", candidate.lower())
            value_collapsed = re.sub(r"\s+", "", value.lower())
            if value_collapsed not in candidate_collapsed:
                continue

            if _already_captured(value):
                continue  # may already have been captured earlier this pass

            key = _slugify_label(label)
            unique_key = key
            suffix = 2
            while unique_key in existing_keys:
                unique_key = f"{key}_{suffix}"
                suffix += 1
            existing_keys.add(unique_key)

            parsed[unique_key] = value
            backfilled.append({"field": unique_key, "label": label, "value": value})
            haystack += "\n" + value.lower()
            existing_values_ws_collapsed.append(value_collapsed)

    if backfilled:
        parsed["source_backfill"] = backfilled

    return parsed


def _extract_review_metadata(parsed: dict) -> dict:
    """
    Pops the dev-facing diagnostic fields -- "needs_human_review" (see
    _flag_sibling_inconsistent_nulls / _verify_constant_fields_against_
    source) and "source_backfill" (see _ensure_source_labels_captured)
    -- OFF of a finished extraction result and returns them as their
    own dict, so callers can surface them in a separate top-level
    output key instead of leaving them inside `structured`.

    These two fields describe the EXTRACTION PROCESS itself (which
    line-item cells look suspiciously null next to populated siblings,
    which top-level fields got backfilled from a stray source label
    the model missed) -- they are not real content transcribed from the
    source document. Leaving them inside `structured` means any
    consumer that renders `structured` directly as "the document" (a
    frontend, an AP Store record, a downstream export) shows this
    internal bookkeeping to an end user right alongside genuine
    invoice/PO/GRN/DO fields, which is confusing and not itself part of
    the document.

    Mutates `parsed` in place (pops the keys, if present) AND returns
    them, since every caller wants both effects together. Returns {} 
    (and leaves `parsed` untouched) if `parsed` isn't a dict or neither
    key is present -- always safe to call unconditionally.
    """
    review_metadata = {}
    if isinstance(parsed, dict):
        if "needs_human_review" in parsed:
            review_metadata["needs_human_review"] = parsed.pop("needs_human_review")
        if "source_backfill" in parsed:
            review_metadata["source_backfill"] = parsed.pop("source_backfill")
    return review_metadata


def _ensure_handwriting_captured(parsed, text):
    """
    Deterministic safety net, same philosophy as
    _strip_boilerplate_line_items above: the model inconsistently
    carries the OCR pipeline's handwriting section into the extracted
    JSON -- sometimes every annotation ends up correctly nested under
    its own document (e.g. `delivery_order.handwritten_notes` for page
    1's annotations, `goods_receipt_note.handwritten_notes` for page
    2's, matching the MULTI-PAGE/MULTI-DOCUMENT prompt rule's own
    per-page grouping), sometimes the whole section is silently dropped.
    Nothing in the prompt can be tuned to GUARANTEE this the way a
    mechanical check can, so this checks the source text directly and
    backstops whatever the model did:

      1. Parse the source's handwriting section by page (see
         _extract_handwriting_by_page). No-op if there is none.
      2. Collect every dict inside `parsed` that looks like its own
         per-document object (has a "document_type" field), in the
         order they appear. If there's more than one distinct page with
         handwriting AND that count matches the number of such
         document objects, assume page N's annotations belong to the
         Nth object (same ordering the multi-document prompt rule
         already produces) and deterministically OVERWRITE that
         object's `handwritten_notes` with the complete, source-
         verified list. This corrects a partial list exactly the same
         way it fills in a completely missing one.
      3. Otherwise (a single document, or a page/document-object count
         mismatch too ambiguous to map safely) OVERWRITE a single
         top-level `handwritten_notes` field on `parsed` with every
         annotation from every page, in page order -- guaranteed
         present even when the nesting can't be done safely.

    Mutates and returns `parsed`. No-op if `parsed` isn't a dict.
    """
    if not isinstance(parsed, dict):
        return parsed

    by_page = _extract_handwriting_by_page(text)
    if not by_page:
        return parsed

    pages_in_order = sorted(by_page)

    def _find_document_objects(node):
        found = []
        if isinstance(node, dict):
            if "document_type" in node:
                found.append(node)
            for v in node.values():
                found.extend(_find_document_objects(v))
        elif isinstance(node, list):
            for item in node:
                found.extend(_find_document_objects(item))
        return found

    doc_objects = _find_document_objects(parsed)

    if len(pages_in_order) > 1 and len(doc_objects) == len(pages_in_order):
        for page, doc_obj in zip(pages_in_order, doc_objects):
            doc_obj["handwritten_notes"] = list(by_page[page])
    else:
        parsed["handwritten_notes"] = [
            ann for page in pages_in_order for ann in by_page[page]
        ]

    return parsed


class SummarizeAgent(Agent):
    """
    Accepts inputs['text'] and returns a 4-6 sentence summary
    via Ollama running on this box.
    """
    name = "byoa-summarize-agent"
    version = "2.1.0"
    description = ("Accepts plain text and returns a 4-6 sentence summary with good detail "
                   "using Ollama. Stateless, GPU-accelerated.")
    capabilities = ["text-summarize"]

    dag = DAG([
        Step("validate", depends_on=[], description="Validate inputs.text"),
        Step("summarize", depends_on=["validate"], description="Call Ollama"),
    ])

    def _run_json_mode_per_row(self, payload: TaskPayload, text: str,
                                header_context: str, row_texts: list,
                                row_start: int, row_end: int) -> TaskResult:
        """
        Row-isolated extraction path, used whenever `_find_table_row_blocks`
        can confidently identify per-row text blocks (see that function's
        docstring for the detection method). Each row is extracted with
        its OWN model call, seeing ONLY that row's text plus shared
        column-header context -- never any other row's text. This makes
        "exactly one JSON object per row, with that row's own fields"
        true BY CONSTRUCTION rather than by prompt request.

        This exists because prompt-only row-boundary instructions were
        tried first (marking rows inline in one big prompt and telling
        the model the exact expected count) and failed on production
        data: the model still merged/split/duplicated rows and bled
        fields between them despite the boundaries being spelled out
        explicitly. Telling a model something is not the same as
        guaranteeing it -- so the guarantee is moved into the CALL
        STRUCTURE instead of the prompt text. A row-count mismatch is no
        longer possible for successfully-parsed rows: `line_items` always
        has exactly `len(row_texts)` entries, because each entry comes
        from its own independent call, one per row block, no more and no
        fewer.

        Trade-off, stated plainly: this issues len(row_texts) + 2 model
        calls (header checklist, per-row calls, one document-metadata
        call) instead of 1-2 -- meaningfully more latency and cost per
        document IF run sequentially. Every one of those per-row calls
        is fully independent (row N's extraction never depends on row
        M's result), and the document-metadata call only depends on
        `doc_level_text`, not on any row's output either -- so all of
        them are dispatched CONCURRENTLY below via a thread pool rather
        than in a loop, and the model-call cost above is against wall-
        clock time, not against how much is spent waiting serially. This
        still buys the same structural elimination of cross-row field
        bleed the sequential version did; parallelizing changes nothing
        about correctness, only how much of the latency cost is actually
        felt.
        """
        header_checklist_block = self._build_header_checklist_block(payload, text)

        # Fix codes (e.g. item_no) that OCR split across two physical
        # lines before any row goes to the model -- see
        # _reconstruct_split_row_values. Row boundaries are already
        # final at this point, so rewriting text WITHIN one row's own
        # span here can't affect how rows were split.
        row_texts = [_reconstruct_split_row_values(rt) for rt in row_texts]

        per_row_timeout = max(payload.timeout_secs * 0.6 / max(len(row_texts), 1), 8)
        line_items = []
        any_row_repaired = False

        # Inference-time constraint (not a prompt instruction) for the two
        # fields production data showed the model getting inconsistent on:
        # quantity coming back as a string with a unit word glued onto it
        # ("6 pc" instead of 6), or the unit word going missing/left in
        # description instead of its own field. Deliberately partial --
        # only these two properties are typed, `additionalProperties` is
        # left true, and nothing is `required` -- because the real column
        # set differs per document and isn't known here; this schema
        # doesn't attempt to describe the whole row, only to make it
        # IMPOSSIBLE for these two fields specifically to come back in the
        # shape that caused the bug, regardless of how well the model
        # followed the surrounding prompt text.
        row_json_schema = {
            "type": "object",
            "properties": {
                "quantity": {"type": ["number", "null"]},
                "quantity_unit": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        }

        row_prompt_rules = (
            "STRICT OUTPUT RULES:\n"
            "- Output ONLY a single valid JSON object representing this ONE "
            "row -- not an array, not a list of rows, just one flat object. "
            "No preamble, no explanation, no markdown code fences, nothing "
            "before the opening '{' or after the closing '}'.\n"
            "- You are shown ONLY this one row's own text. There is no "
            "other row's data anywhere in your input right now -- do not "
            "reference, invent, or borrow a value from any other row.\n"
            "- Use null for any field that is genuinely blank, illegible, "
            "or not stated in THIS row's text -- never invent a plausible-"
            "looking value just because a value is expected there.\n"
            "- Keep field names in lower_snake_case, one flat field per "
            "column -- never merge two adjacent columns into one hybrid "
            "field, even when one of them is short (e.g. a single "
            "character next to a longer code): each must be its own "
            "field, using the FULL header phrase as the field name (e.g. "
            "\"bin_code\", never a bare \"code\").\n"
            "- If a header checklist is provided below, include a field "
            "for every listed header that applies to this table, using "
            "null for any that are blank in this row's own text -- never "
            "omit a listed header's field entirely.\n"
            "- Never manufacture a column's value by splitting text out of "
            "a neighboring column (most commonly the description text) "
            "just because a header checklist expects a value there.\n"
            "- QUANTITY WITH AN ATTACHED UNIT: a value under a "
            "Quantity-type column is sometimes printed as a bare number "
            "(e.g. \"20\"), and sometimes as a number immediately "
            "followed by a short unit-of-measure word directly next to "
            "it (e.g. \"6 pc\", \"10 sets\", \"5kg\") -- do NOT assume "
            "either shape; read what THIS row actually shows. There is "
            "no fixed list of what the unit word can be, so recognize it "
            "by PATTERN, not by matching it against known words: a short "
            "run of letters (no digits, no currency symbols) sitting "
            "immediately beside a number, with no other column's content "
            "between them, is that number's unit -- whatever the letters "
            "actually say. When you see that pattern: put the bare "
            "number in the quantity field (as a number, not a string), "
            "and put the unit word in its own \"quantity_unit\" field -- "
            "never leave the unit word sitting inside the description, "
            "and never drop it entirely. When the row's quantity value "
            "is a bare number with nothing attached, output that number "
            "for quantity and null for quantity_unit -- do not invent a "
            "unit that isn't actually printed next to it. If a number-"
            "plus-word pair like this appears somewhere in the row's "
            "text but NOT in the position the quantity column occupies "
            "for this row (e.g. it's clearly part of a wrapped "
            "description line instead), it is still description text, "
            "not the quantity -- only apply this rule to the value that "
            "is actually this row's own quantity.\n"
            "- OCR TYPO CORRECTION (narrow): fix only an obvious one-"
            "character OCR misread of a common/standard word you already "
            "know with certainty (e.g. \"Sihgapore\" -> \"Singapore\"). Do "
            "NOT extend this to item codes, model numbers, or reference "
            "numbers -- transcribe those exactly as they appear even if "
            "they look unusual.\n\n"
        )

        def _extract_one_row(row_text: str) -> dict:
            """
            Runs entirely independently for one row -- no shared mutable
            state touched here, so this is safe to call concurrently
            from multiple threads. Returns {"row_dict": ..., "repaired":
            bool} so the caller can fold `any_row_repaired` together
            after all rows complete, without any thread needing to
            touch a shared variable directly.
            """
            row_prompt = (
                "Extract the single line-item row shown below into one "
                "flat JSON object, one field per column. Here is the "
                "table's own column header text, for context on what "
                "each value means (do NOT extract this header text "
                "itself as data). IMPORTANT: OCR sometimes prints a "
                "column header across TWO separate physical lines -- "
                "e.g. one short line containing just one word (like "
                "\"Variant\" or \"Unit of\" on its own), whose other half "
                "appears on a DIFFERENT line, reuniting into one real "
                "header phrase (e.g. \"Variant Code\", \"Unit of "
                "Measure\"). If you see a short isolated line like that "
                "below, treat it as one whole header column -- it is "
                "NOT evidence that some number or word in the row's own "
                "data belongs to a new field; a column like this is "
                "very often genuinely blank for this row, and the "
                "correct output is null for it, not a value copied or "
                "guessed from elsewhere in the row's own text:\n"
                + header_context + "\n\n"
                + header_checklist_block
                + row_prompt_rules
                + "ROW TEXT TO EXTRACT:\n" + row_text
            )
            row_dict = None
            raw = ""
            repaired = False
            # Bounded retry loop, not unlimited -- see MAX_TRANSIENT_MODEL_
            # RETRIES' comment for why unbounded retrying on a saturated
            # backend is worse, not safer: it just delays hitting the
            # document's own overall deadline instead of avoiding it. This
            # many attempts rides out realistic contention spikes; a row
            # that still can't get a slot after all of them is reported as
            # failed rather than silently blocking further.
            for attempt in range(MAX_TRANSIENT_MODEL_RETRIES):
                try:
                    raw = _call_ollama(
                        row_prompt, per_row_timeout,
                        guided_json=row_json_schema, temperature=0.0,
                    )
                    row_dict, _ = _try_parse_json(raw)
                    break
                except Exception:
                    row_dict = None
                    continue

            if not isinstance(row_dict, dict) and raw:
                try:
                    repaired_raw = _call_ollama(
                        "The text below was supposed to be a single valid "
                        "JSON object but isn't. Fix ONLY the syntax/format "
                        "-- return exactly one flat JSON object (not an "
                        "array), nothing else:\n\n" + raw,
                        max(per_row_timeout * 0.5, 8),
                    )
                    repaired_dict, _ = _try_parse_json(repaired_raw)
                    if isinstance(repaired_dict, dict):
                        row_dict = repaired_dict
                        repaired = True
                except Exception:
                    pass

            if not isinstance(row_dict, dict):
                row_dict = {"_extraction_failed": True}
            else:
                row_dict = _repair_hallucinated_short_code_field(row_dict, row_text)

            return {"row_dict": row_dict, "repaired": repaired}

        # Document-level metadata: everything OUTSIDE the table's own
        # span, extracted with a single separate call, explicitly told
        # not to include line_items (those are added separately below).
        # Built here (before dispatch) so its call can be submitted to
        # the SAME thread pool as the row calls below, rather than
        # waiting for every row to finish first -- it depends only on
        # `doc_level_text`, never on any row's result.
        doc_level_text = text[:row_start] + text[row_end:]
        doc_prompt = (
            "Read the document metadata below (this is everything in the "
            "source EXCEPT the line-item table rows themselves, which "
            "have already been extracted separately) and respond with a "
            "single valid JSON object capturing document-level details "
            "only: things like document type/title, reference numbers, "
            "dates, assigned user/location codes, vendor/customer names, "
            "and any handwritten annotation notes. Do NOT include a "
            "`line_items` field -- that is added separately afterward. "
            "Use null for any field genuinely blank or not stated. "
            "Output ONLY the JSON object, no preamble, no markdown "
            "fences:\n\n" + doc_level_text
        )

        def _extract_doc_metadata() -> "dict | None":
            try:
                doc_raw = _call_ollama(doc_prompt, max(payload.timeout_secs * 0.25, 10))
                parsed, _ = _try_parse_json(doc_raw)
                return parsed
            except Exception:
                return None

        # Dispatch every row's extraction AND the document-metadata
        # extraction concurrently -- all are independent of each
        # other's results (see docstring above). Row order in the
        # final `line_items` list is preserved by indexing futures back
        # to their original position after they complete, not by the
        # order threads happen to finish in.
        line_items = [None] * len(row_texts)
        any_row_repaired = False
        pool_size = min(len(row_texts) + 1, MAX_CONCURRENT_MODEL_CALLS)

        with ThreadPoolExecutor(max_workers=max(pool_size, 1)) as executor:
            row_futures = {
                executor.submit(_extract_one_row, row_text): idx
                for idx, row_text in enumerate(row_texts)
            }
            doc_future = executor.submit(_extract_doc_metadata)

            for future in as_completed(row_futures):
                idx = row_futures[future]
                result = future.result()
                line_items[idx] = result["row_dict"]
                if result["repaired"]:
                    any_row_repaired = True

            doc_parsed = doc_future.result()

        if not isinstance(doc_parsed, dict):
            doc_parsed = {}
        doc_parsed.pop("line_items", None)

        doc_parsed["line_items"] = line_items
        doc_parsed = _repair_bare_code_holding_freetext(doc_parsed)
        doc_parsed = _repair_orphaned_code_fields(doc_parsed)
        doc_parsed = _repair_zone_bin_code_confusion(doc_parsed, payload.timeout_secs)
        doc_parsed = _repair_uom_lcn_confusion(doc_parsed, payload.timeout_secs)
        doc_parsed = _repair_description_model_split(doc_parsed, text, payload.timeout_secs)
        doc_parsed = _repair_row_number_mistaken_for_item_code(doc_parsed, text, payload.timeout_secs)
        doc_parsed = _repair_reversed_party_roles(doc_parsed, text, payload.timeout_secs)
        doc_parsed = _repair_line_item_field_name_variants(doc_parsed, payload.timeout_secs)
        doc_parsed = _repair_quantity_amount_reconciliation(doc_parsed)
        doc_parsed = _verify_constant_fields_against_source(doc_parsed, text)
        doc_parsed = _flag_sibling_inconsistent_nulls(doc_parsed)
        doc_parsed = _ensure_handwriting_captured(doc_parsed, text)
        # Deliberately scans the FULL `text`, not `doc_level_text`, for
        # this specific call -- confirmed in production that
        # _find_table_row_blocks's row-region boundary can overshoot and
        # swallow genuinely document-level content (this document's own
        # GST/total summary lines) into what it treats as "table area",
        # which then gets excluded from doc_level_text entirely. That's
        # a different bug, in row-boundary detection, not in candidate
        # classification -- scanning the full text sidesteps it here
        # without touching that detection logic (lower risk than trying
        # to fix the boundary heuristic itself, which other documents
        # may depend on behaving exactly as it does now). Scanning the
        # row region too can't cause harm: any real line-item cell value
        # is already captured under `line_items` and gets filtered out
        # by the "already captured" check same as anywhere else.
        doc_parsed = _ensure_source_labels_captured(doc_parsed, text, payload.timeout_secs)

        # Pulled OUT of `structured` here -- these are dev-facing
        # diagnostics about the extraction itself (null-vs-siblings
        # flags, backfilled stray labels), not real document content.
        # See _extract_review_metadata. Popped BEFORE AP submission too,
        # so the AP Document Store record only ever contains genuine
        # document fields, never this internal bookkeeping.
        review_metadata = _extract_review_metadata(doc_parsed)

        # Guarantee job_id (and any other requested frontmatter keys)
        # end up in doc_parsed BEFORE submission, regardless of whether
        # the probabilistic label:value backfill above happened to
        # classify the frontmatter's "job_id: ..." line as a real field
        # this run. Never overwrites a value already present.
        for key, value in _extract_frontmatter_fields(text, keys=("job_id",)).items():
            doc_parsed.setdefault(key, value)

        doc_label = doc_parsed.get("document_type")

        ap_submission = _finalize_ap_submission(doc_parsed, text, payload.timeout_secs)

        # Rewrites doc_parsed["document_type"] in place if AP submission
        # resolved a different type than the model's own guess (header
        # title / pricing tie-break) -- see _reconcile_document_type_label.
        doc_label = _reconcile_document_type_label(doc_parsed, ap_submission)
        outputs_summary = (
            f"JSON extraction ({doc_label}) -- see 'structured' field."
            if doc_label else
            "JSON extraction complete -- see 'structured' field for the full object."
        )

        return TaskResult(
            task_id=payload.task_id,
            status=TaskStatus.SUCCESS,
            outputs={
                "summary": outputs_summary,
                "structured": doc_parsed,
                "review_metadata": review_metadata,
                "output_format": "json",
                "json_parse_failed": False,
                "json_repaired": any_row_repaired,
                "json_meta_refusal_retried": False,
                "input_length": len(text),
                "model": VLLM_MODEL,
                "extraction_mode": "per_row",
                "row_count": len(row_texts),
                "doc_id": ap_submission.get("doc_id"),
                "api_doc_type": ap_submission.get("api_doc_type"),
                "ap_submission": ap_submission,
                # TEMPORARY debug fields -- echoes back the actual text this
                # agent received (capped to avoid bloating huge-doc
                # responses), so the exact input can be inspected directly
                # from the task result instead of reverse-engineering it
                # from upstream pipeline state. debug_input_text_length is
                # the length of the (possibly-capped) string in
                # debug_input_text specifically -- compare it against
                # whatever length a pasted/copied version of that string
                # comes out to, to catch silent copy-paste truncation
                # before debugging against a corrupted copy (confirmed
                # failure mode: manually transcribing a large escaped-
                # Unicode JSON string through chat lost ~1200 characters
                # with no visible sign anything was wrong). Remove once
                # the row-count detection issue on production MD input is
                # resolved.
                "debug_input_text": text[:8000],
                "debug_input_text_length": len(text[:8000]),
            },
        )

    def _build_header_checklist_block(self, payload: TaskPayload, text: str) -> str:
        """
        Shared header-identification pre-pass, factored out so both the
        per-row path and the whole-document fallback path can use the
        identical logic instead of duplicating it. Best-effort: returns
        "" if the call fails or returns nothing useful, in which case
        extraction proceeds without the checklist rather than failing
        the whole task over it.
        """
        try:
            header_raw = _call_ollama(
                "List every column header label that appears above an item/"
                "line-item table in the source text below, exactly as "
                "printed (fix only obvious single-character OCR garbling of "
                "a header word). Include EVERY column, including short or "
                "single-letter-looking ones. Each header label must be a "
                "complete, sensible 1-4 word phrase on its own -- NEVER a "
                "single stray letter or an obviously truncated fragment as "
                "its own entry. OCR sometimes runs multiple header labels "
                "together with no clear boundary, or splits one header "
                "phrase across two separate lines that get picked up out "
                "of order -- reconstruct the full, sensible phrase in "
                "either case using the shape of the data rows below as a "
                "guide. Output ONLY the header list, columns comma-"
                "separated, nothing else -- no JSON, no commentary:\n\n" + text,
                max(payload.timeout_secs * 0.15, 10),
            )
            header_raw = header_raw.strip()
            if header_raw and len(header_raw) < 2000:
                return (
                    "The following column headers were identified verbatim "
                    "in the source table. Include a field for EVERY header "
                    "listed here that applies -- using null for a row "
                    "whose cell under that header is genuinely blank -- "
                    "even for headers that look minor, short, or redundant "
                    "with a neighboring one. Never combine two of these "
                    "listed headers into one field:\n"
                    f"{header_raw}\n\n"
                )
        except Exception:
            pass
        return ""

    def _run_json_mode(self, payload: TaskPayload, text: str, focus: str, focus_rules: str = "") -> TaskResult:
        """
        JSON-output path. Two model calls at most: one draft, and -- only
        if the draft doesn't parse as valid JSON -- one repair call whose
        ONLY job is fixing the JSON syntax of the existing content (not
        re-deriving it from the source), mirroring the same
        verify-don't-trust pattern the reformat/condense steps in the
        prose path already use. If the repair call also fails to produce
        parseable JSON, the raw model output is still returned (under
        `summary`, with `json_parse_failed=True`) rather than erroring out
        -- a caller further down the pipeline can decide whether a
        malformed-but-present answer is still useful, the same "don't lose
        the result entirely" philosophy every fallback in the prose path
        already follows.
        """
        text = _strip_isolated_noise_char_lines(text)
        text = _unwrap_cancelled_annotation(text)
        text = _recover_header_bar_labels(text)
        row_texts, header_context, row_start, row_end = _find_table_row_blocks(
            _body_for_row_detection(text)
        )
        if len(row_texts) >= 2:
            return self._run_json_mode_per_row(
                payload, text, header_context, row_texts, row_start, row_end
            )

        focus_block = (
            f"The user's original request was: \"{focus}\". Some of this text may "
            f"contain leftover prose-formatting instructions from before JSON mode "
            f"was selected (e.g. a sentence count, a 'summarize briefly' phrasing, "
            f"or a length target) -- those do NOT apply here and must be ignored "
            f"entirely; a sentence-count instruction never overrides the JSON-only "
            f"output rule above, there is no real contradiction to resolve or "
            f"comment on. Only follow the parts of this request that describe WHICH "
            f"fields, entities, or values to include and how to handle ones that are "
            f"missing -- apply those as guidance for what the JSON object should "
            f"contain.\n\n"
        ) if focus else ""

        # focus_rules carries the controller's always-on safety-net rules
        # (OCR row-alignment, blank-field-not-inferred, financial-document
        # checks, etc) -- kept as a SEPARATE field from `focus` itself (see
        # `run()` below) so those rules never leak into the prose path's
        # cue-detection regexes, but JSON mode still wants their content,
        # so it's appended here same as it always was when it lived inside
        # `focus`.
        focus_rules_block = f"{focus_rules}\n\n" if focus_rules else ""

        # HEADER PRE-PASS: asking the model to both identify every table
        # column AND correctly extract/format every row in one single call
        # has repeatedly resulted in silently dropped or merged columns
        # (confirmed on two unrelated documents with different table
        # layouts -- a "Variant Code" column and a "Jancode" column, both
        # genuinely present in the source, both missing from the output)
        # despite explicit prose rules telling it not to. Splitting header
        # identification into its own small, focused call and handing the
        # literal result back as a checklist removes the need for the
        # model to remember to self-audit its own column coverage while
        # also juggling row-value extraction and JSON syntax -- it's just
        # told which fields must exist. Best-effort: if this call fails or
        # returns nothing useful, extraction proceeds without the
        # checklist rather than failing the whole task over it.
        header_checklist_block = ""
        try:
            header_raw = _call_ollama(
                "List every column header label that appears above an item/"
                "line-item table in the source text below, exactly as "
                "printed (fix only obvious single-character OCR garbling of "
                "a header word, e.g. a stray symbol in place of a letter). "
                "Include EVERY column, including short or single-letter-"
                "looking ones -- do not skip a column because it looks "
                "minor. If there is more than one table (e.g. a delivery "
                "order table and a separate goods-receipt table), label "
                "each table's header list separately using that table's own "
                "heading.\n"
                "OCR often runs multiple header labels together on one "
                "line with no clear boundary (e.g. \"Zone Code Bin Code "
                "Quantity Code\" printed as one fused string, or \"Item No "
                "Code\" where \"Code\" actually belongs to a DIFFERENT "
                "nearby header, not to \"Item No\"). Each header label must "
                "be a complete, sensible 1-4 word phrase on its own (like "
                "\"Zone Code\", \"Bin Code\", \"Item No\") -- NEVER a single "
                "stray letter or an obviously truncated fragment as its own "
                "entry. To resolve a fused run of header words correctly, "
                "look at the actual DATA rows below the header: the number "
                "and shape of distinct values in each row (e.g. one short "
                "single-character value immediately followed by one longer "
                "alphanumeric code) tells you how many separate columns "
                "are really there and where each header phrase's boundary "
                "falls, even though the header text itself ran together.\n"
                "A SPECIFIC common cause of this fusion: a header that "
                "visually wraps onto two lines on the page (any two-word "
                "header can be split this way -- e.g. a generic \"Foo\" / "
                "\"Bar\" pair printed on separate lines) gets its two "
                "pieces picked up at DIFFERENT points in the OCR reading "
                "order than the single-line headers sitting beside it, "
                "because OCR reads roughly line-by-line across the page "
                "rather than column-by-column. The result is an isolated "
                "one-word line (the first half of the wrapped header, on "
                "its own with no second word) followed shortly after by "
                "an unrelated run of single-line headers that has an "
                "extra orphan word tacked onto its end (that orphan word "
                "is the missing second half of the wrapped header, not "
                "part of the header it now sits next to). When you spot "
                "an isolated short header line with no obvious second "
                "word, check nearby lines (including a run of otherwise-"
                "unrelated headers) for an orphan trailing word that "
                "would complete it, and reunite them into one header "
                "phrase based on what the two fragments actually spell "
                "out together -- rather than treating the isolated word "
                "as its own column or leaving the orphan word attached "
                "to the wrong neighbor. Do NOT assume any specific header "
                "pair by name -- apply this purely as a structural/"
                "positional pattern to whatever fragments actually "
                "appear in THIS document.\n"
                "Output ONLY the header lists, one table per line, "
                "columns comma-separated, nothing else -- no JSON, no "
                "commentary:\n\n" + text,
                max(payload.timeout_secs * 0.15, 10),
            )
            header_raw = header_raw.strip()
            if header_raw and len(header_raw) < 2000:
                header_checklist_block = (
                    "The following column headers were identified verbatim "
                    "in the source table(s). Every line_items object for "
                    "the corresponding table MUST include a field for EVERY "
                    "header listed for that table -- using null for a row "
                    "whose cell under that header is genuinely blank -- "
                    "even for headers that look minor, short, or "
                    "redundant with a neighboring one. Never combine two "
                    "of these listed headers into one field:\n"
                    f"{header_raw}\n\n"
                )
        except Exception:
            pass

        def _build_draft_prompt(content_text: str, page_note: str = "") -> str:
            return (
                "Read the source text below and respond with a single valid JSON "
                "object capturing its key details -- nothing else. "
                "STRICT OUTPUT RULES:\n"
                "- Output ONLY the JSON object. No preamble, no explanation, no "
                "markdown code fences, nothing before the opening '{' or after "
                "the closing '}'.\n"
                "- Use null for any field that is genuinely blank, illegible, or "
                "not stated in the source -- never invent a plausible-looking "
                "value to fill a field just because a value is expected there. "
                "This applies to numeric fields (prices, totals, taxes) exactly "
                "as much as text fields.\n"
            "- Keep field names in lower_snake_case and use arrays of "
            "objects for repeated structures (e.g. a `line_items` array), "
            "not a flat field per repeated item.\n"
            "- When pairing a labeled row's fields together (e.g. an item "
            "code with its own description and price), use only the "
            "pairing the source text actually presents for THAT row -- do "
            "not assume adjacent lines belong together just because OCR "
            "reflow placed them near each other.\n"
            "- A `line_items` array must contain ONLY genuine product/service "
            "rows that actually have their own item code, description, "
            "quantity, or price data in the source -- never populate it with "
            "legal boilerplate, payment-terms text, ownership/retention-of-"
            "title clauses, footer notes, or any other sentence that merely "
            "happens to sit near the totals in the source layout. If the "
            "source shows a line-item table HEADER (e.g. column names like "
            "'Item No.', 'Description', 'Quantity', 'Unit Price') but no "
            "actual data rows underneath it, that means the rows were not "
            "captured or genuinely don't exist here -- `line_items` must be "
            "an empty array in that case, not a placeholder entry built from "
            "unrelated nearby text.\n"
            "- NEVER refuse the whole extraction, and never return a single "
            "{\"error\": \"...\"} object in place of real data, just because "
            "the source document is a different TYPE than some other "
            "instruction above happened to describe (e.g. a delivery order "
            "or goods receipt note instead of a priced invoice). Extract "
            "whatever fields the document actually contains -- item codes, "
            "descriptions, quantities, reference/PO/GRN/DO numbers, dates, "
            "vendor/customer/deliver-to parties, purchaser/recipient names -- "
            "and use null ONLY for the specific fields that document type "
            "genuinely has no data for (e.g. unit_price and amount on a "
            "delivery order that lists quantities but no prices at all). A "
            "document missing pricing information is still full of "
            "extractable content; it is never a reason to extract nothing.\n"
            "- MULTI-PAGE / MULTI-DOCUMENT SOURCE CHECK: the source text below "
            "may contain MORE THAN ONE distinct document stitched together "
            "(e.g. a page-1 delivery order followed by a page-2 goods receipt "
            "note, or an invoice followed by a delivery note) -- look at "
            "'## Page' markers and document-type headings (e.g. 'DELIVERY "
            "ORDER', 'GOODS RECEIPT NOTE', 'TAX INVOICE') to identify each "
            "one. Do NOT stop extracting after the first item table or "
            "reference number you find -- check EVERY page/section for its "
            "OWN item table, its OWN reference numbers, and its OWN party "
            "names, even if an earlier page already had similar-looking "
            "data. If two pages list the same underlying goods but use "
            "DIFFERENT item/model codes for them (e.g. a supplier's own "
            "model code on one page vs. the buyer's internal SKU/item number "
            "on another), these are not duplicates to merge or discard -- "
            "represent each page's own numbering as its own line_items "
            "list (e.g. group fields under a key per document, such as "
            "`delivery_order` and `goods_receipt_note`, each with its own "
            "line_items array and its own reference number/date/party "
            "fields, rather than flattening everything into one top-level "
            "line_items array that only reflects the first table found). "
            "Also make sure every distinct reference number is captured "
            "even if it's labeled differently per page (e.g. a 'Pay-to "
            "Vendor No.' on one page is a real, distinct field from a "
            "'Customer Account' number on another -- capture both, don't "
            "assume they're the same value or that one supersedes the "
            "other), and capture every named person mentioned in a role "
            "(e.g. a 'Purchaser', 'Sales Executive', 'Entered By') as its "
            "own field.\n"
            "- DO NOT DROP A TABLE COLUMN just because its values look "
            "terse, sparse, repetitive, or hard to interpret (e.g. a "
            "column of single letters, a mostly-blank column, or a column "
            "whose header got garbled by OCR reflow into a fragment sitting "
            "next to other header words). If the source's own header row "
            "names or implies MORE columns than you can confidently label, "
            "still capture every one of them as its own field per row using "
            "your best reading of that column's header text, rather than "
            "silently keeping only the columns that were easiest to parse. "
            "Use null only for the specific rows where that particular "
            "column's own value is genuinely absent -- never drop the "
            "column itself from the schema, even if EVERY row happens to "
            "be blank for it (a column that is blank for all rows should "
            "appear as its own field set to null in every line item, not "
            "be omitted from the schema entirely).\n"
            "- FIELD NAMES MUST BE COMPLETE, SENSIBLE WORDS, never a single "
            "stray letter or an obviously truncated fragment (e.g. a field "
            "literally named \"n\" or \"code\" on its own is wrong; the "
            "underlying header was truncated or fused with a neighbor by "
            "OCR, and the fix is to reconstruct the FULL header phrase, not "
            "to name the field after the leftover fragment). If a header "
            "checklist is provided below, use each listed header's full "
            "phrase to derive the field name.\n"
            "- DO NOT MERGE ADJACENT COLUMNS INTO ONE HYBRID FIELD, even "
            "when their values sit right next to each other on the page and "
            "one of them is short. Count the header row's own column names "
            "literally: if the source names N columns as separate headers, "
            "the output must have N separate fields for that row, one per "
            "header, never fewer -- do not fold a short single-character "
            "column's value together with its longer alphanumeric neighbor "
            "into one combined field or one combined name, and do not "
            "invent a hybrid field name that matches none of the source's "
            "actual headers. Concretely, using made-up placeholder values "
            "purely to illustrate the PATTERN (do not expect these exact "
            "letters/codes in the real source): if a row's data reads "
            "\"...  X   AB12-99Z  10  PCS\" under headers \"Zone Code | Bin "
            "Code | Quantity | Unit\", that is `\"zone_code\": \"X\", "
            "\"bin_code\": \"AB12-99Z\"` as two separate, correctly-named "
            "fields -- NOT a single field that discards one of the real "
            "values, and NOT a vague/truncated field name like a bare "
            "\"code\" holding a value that belongs in a field named after "
            "ITS OWN header. If a column header exists but that row's cell "
            "under it is blank in the source, keep the field with a null "
            "value rather than omitting it or letting a neighboring "
            "column's value slide into it.\n"
            "- NEVER MANUFACTURE A COLUMN'S VALUE BY SPLITTING TEXT FROM A "
            "DIFFERENT COLUMN. If a header checklist lists a column and "
            "that column's cell is genuinely blank for a given row, set "
            "that field to null for that row. Do NOT take words from a "
            "neighboring column's own text (most commonly the description/"
            "item-name column, which often wraps onto a second line) and "
            "reassign them to the blank column just because the checklist "
            "expects a value there. When in doubt about whether text "
            "belongs to the blank column or to its neighbor, leave the "
            "neighbor's text intact and whole, and use null for the blank "
            "column.\n"
            "- A REPEATED VALUE ACROSS CONSECUTIVE ROWS IS NORMAL, NOT A "
            "SIGN OF DUPLICATION. Line-item tables routinely repeat the "
            "same reference value (e.g. an order/source number, a date) "
            "identically on every row, because that value genuinely "
            "applies to every line item. If a row's own text plainly "
            "shows a value for a column -- even a value identical to the "
            "one you already used for the row above it -- output it for "
            "THIS row too. Never null out a field just because you've "
            "already seen that exact value on a previous row.\n"
            "- OCR TYPO CORRECTION (narrow, only for common/standard words): "
            "OCR sometimes garbles a well-known word into something that "
            "isn't a real word but is obviously ONE character away from a "
            "standard term whose correct spelling you already know with "
            "certainty -- e.g. a country name like \"Sihgapore\" for "
            "\"Singapore\", or a standard company-type suffix garbled from "
            "context, like \"(FL)\" where the surrounding text and company "
            "type make it clearly meant to read \"(PTE)\". For cases like "
            "this ONLY, output the corrected standard spelling rather than "
            "reproducing the scanning glitch verbatim. This is fixing a "
            "recognizable OCR misread of a word that IS actually present on "
            "the document, not inventing new information -- it does NOT "
            "extend to proper nouns, brand/product names, item or model "
            "codes, reference numbers, or any other field where more than "
            "one correction could plausibly fit; for those, if the source "
            "text is genuinely garbled or ambiguous, transcribe it exactly "
            "as it appears rather than guessing at a cleaner version.\n\n"
            + page_note
            + focus_block + focus_rules_block + header_checklist_block
            + content_text
            )

        def _run_single_pass(content_text: str, budget: float, page_note: str = ""):
            """
            One draft-extraction call plus its syntax-repair retry.
            Returns (parsed_or_None, cleaned_text_or_None, repaired: bool).
            Raises on total call failure (network/HTTP error, including
            the context-length 400 this whole chunking path exists to
            avoid) -- CALLERS decide whether that's fatal (the
            single-document path below treats it as fatal, unchanged
            from before) or a per-page skip (the chunked path treats a
            single page's failure as just that page failing, not the
            whole document).
            """
            prompt = _build_draft_prompt(content_text, page_note)
            raw = _call_ollama(prompt, budget)
            parsed_pass, cleaned_pass = _try_parse_json(raw)
            repaired_pass = False
            if parsed_pass is None:
                try:
                    repaired_raw = _call_ollama(
                        "The text below was supposed to be a single valid JSON "
                        "object but has a syntax error (e.g. a trailing comma, "
                        "an unescaped quote, missing braces). Fix ONLY the JSON "
                        "syntax -- do not add, remove, invent, or change any "
                        "field name or value that is already present and "
                        "syntactically fine. Return ONLY the corrected JSON "
                        "object, nothing else:\n\n" + cleaned_pass,
                        max(budget * 0.4, 10),
                    )
                    repaired_parsed_pass, repaired_cleaned_pass = _try_parse_json(repaired_raw)
                    if repaired_parsed_pass is not None:
                        parsed_pass, cleaned_pass = repaired_parsed_pass, repaired_cleaned_pass
                        repaired_pass = True
                except Exception:
                    pass
            return parsed_pass, cleaned_pass, repaired_pass

        # Trimmed from the previous 0.55/0.9 split to leave room for the
        # header pre-pass call above, which now also draws from the same
        # overall timeout_secs budget.
        draft_budget = payload.timeout_secs * (0.4 if focus else 0.65)

        # CHUNKED-BY-PAGE FALLBACK: reached only when _find_table_row_blocks
        # found no per-row structure to isolate (otherwise _run_json_mode_
        # per_row already handled this above) AND the source has 2+ "##
        # Page" markers -- i.e. exactly the situation that used to send an
        # entire multi-page, OCR-placeholder-dense document as ONE prompt
        # and get a hard 400 (context-length) failure from vLLM with zero
        # output, confirmed on a real production document (a Delivery Note
        # + Goods Receipt Note combo whose price column OCR'd as
        # unrecognized-artifact markers instead of numbers, which is
        # ALSO why the per-row amount-anchor detector above found nothing
        # to isolate on). Chunking, not truncating: every page's full text
        # still gets sent to the model, just as its own smaller call
        # instead of all pages crammed into one; _merge_chunked_json_
        # objects then combines every page's result losslessly. See that
        # function and _split_into_page_chunks for the merge/split rules.
        page_chunks = _split_into_page_chunks(text)
        if page_chunks:
            per_page_budget = max(draft_budget / len(page_chunks), 15)
            page_parsed_objects = []
            any_page_repaired = False
            chunk_failures = []
            for page_label, page_text in page_chunks:
                page_note = (
                    f"\n\nNOTE: this is {page_label} of a {len(page_chunks)}-page "
                    f"source, being extracted separately from its other pages "
                    f"(each page is sent to you on its own, then every page's "
                    f"result is merged afterward) -- extract ONLY what THIS "
                    f"page's own text below actually contains; do not invent "
                    f"or assume fields that would only appear on a different "
                    f"page. Identify what KIND of document THIS page is (e.g. "
                    f"'Delivery Note', 'Goods Receipt Note', 'Tax Invoice', "
                    f"'Purchase Order') from its own heading, even if that "
                    f"heading sits alongside fields that also make it read "
                    f"like another type (e.g. a page titled 'Delivery Note' "
                    f"that also has its own 'Invoice No.' field is still its "
                    f"own document -- classify it by whichever labels/numbers "
                    f"it actually carries, don't leave it unclassified just "
                    f"because it isn't a single pure type). ALWAYS nest ALL of "
                    f"this page's extracted fields under ONE top-level key "
                    f"named after that document type in lower_snake_case (e.g. "
                    f"\"delivery_order\", \"goods_receipt_note\", \"invoice\") "
                    f"-- this applies EQUALLY to every page, including the "
                    f"first/primary one; never leave one page's fields flat at "
                    f"the top level while another page's are nested, since "
                    f"every page is its own peer document, not a main document "
                    f"plus an attachment. That nested object MUST itself "
                    f"include a \"document_type\" field naming what this page "
                    f"is (e.g. \"document_type\": \"goods_receipt_note\"), the "
                    f"same as any single-page document would.\n\n"
                )
                page_parsed = None
                page_error = None
                for attempt in range(MAX_TRANSIENT_MODEL_RETRIES):
                    try:
                        parsed_page, _cleaned_page, repaired_page = _run_single_pass(
                            page_text, per_page_budget, page_note
                        )
                        page_parsed = parsed_page
                        page_error = None
                        break
                    except Exception as e:
                        page_error = e
                        continue
                if page_error is not None:
                    chunk_failures.append({"page": page_label, "error": str(page_error)})
                    continue
                parsed_page = page_parsed
                if isinstance(parsed_page, dict):
                    page_parsed_objects.append(parsed_page)
                    any_page_repaired = any_page_repaired or repaired_page
                else:
                    chunk_failures.append({"page": page_label, "error": "model output for this page was not valid JSON"})

            if not page_parsed_objects:
                return TaskResult(
                    task_id=payload.task_id,
                    status=TaskStatus.FAILURE,
                    error_msg=(
                        f"Chunked JSON extraction failed on every page "
                        f"({len(page_chunks)} pages attempted): {chunk_failures}"
                    ),
                )

            parsed = _merge_chunked_json_objects(page_parsed_objects)
            if chunk_failures:
                # Surfaced, never silently dropped -- "no truncation" means
                # a genuinely-failed page is REPORTED, not pretended away.
                parsed["_chunk_extraction_incomplete_pages"] = chunk_failures
            cleaned = json.dumps(parsed, indent=2, ensure_ascii=False)
            json_repaired = any_page_repaired
        else:
            try:
                parsed, cleaned, json_repaired = _run_single_pass(text, draft_budget)
            except Exception as e:
                return TaskResult(task_id=payload.task_id,
                                  status=TaskStatus.FAILURE,
                                  error_msg=f"Summarization failed: {e}")

        json_parse_failed = parsed is None
        if not json_parse_failed:
            parsed = _strip_boilerplate_line_items(parsed, payload.timeout_secs)
            parsed = _repair_bare_code_holding_freetext(parsed)
            parsed = _repair_orphaned_code_fields(parsed)
            parsed = _repair_zone_bin_code_confusion(parsed, payload.timeout_secs)
            parsed = _repair_uom_lcn_confusion(parsed, payload.timeout_secs)
            parsed = _repair_description_model_split(parsed, text, payload.timeout_secs)
            parsed = _repair_row_number_mistaken_for_item_code(parsed, text, payload.timeout_secs)
            parsed = _repair_reversed_party_roles(parsed, text, payload.timeout_secs)
            parsed = _repair_line_item_field_name_variants(parsed, payload.timeout_secs)
            parsed = _repair_quantity_amount_reconciliation(parsed)
            parsed = _verify_constant_fields_against_source(parsed, text)
            parsed = _flag_sibling_inconsistent_nulls(parsed)
            parsed = _ensure_handwriting_captured(parsed, text)
            parsed = _ensure_source_labels_captured(parsed, text, payload.timeout_secs)
            cleaned = json.dumps(parsed, indent=2, ensure_ascii=False)

        # Content-level guard, separate from the syntax-repair retry above:
        # the JSON parsed fine but is a meta-commentary refusal object
        # rather than actual extracted data (see _looks_like_meta_refusal).
        # One forced-extraction retry with the contradiction explicitly
        # pre-resolved, reusing the source text directly rather than
        # feeding the refusal object back in (nothing useful to repair
        # there, unlike the syntax-fix retry above).
        json_meta_refusal_retried = False
        if not json_parse_failed and _looks_like_meta_refusal(parsed, payload.timeout_secs):
            _forced_extraction_instruction = (
                "Your previous attempt at this task returned a JSON object "
                "refusing to extract data -- either because it saw a "
                "perceived contradiction between instructions, or because "
                "it decided the source document isn't the specific type "
                "(e.g. \"invoice\") some instruction described. Neither is "
                "a valid reason to return nothing: a document missing "
                "pricing information (e.g. a delivery order or goods "
                "receipt note) still has real extractable content -- item "
                "codes, descriptions, quantities, reference/PO/GRN/DO "
                "numbers, dates, and party names. Output ONLY a JSON "
                "object of the source document's ACTUAL details using "
                "whatever fields fit what this specific document "
                "contains (vendor, document number, dates, line_items, "
                "subtotal, tax, total, etc.), using null ONLY for the "
                "individual fields that are genuinely blank or not "
                "stated -- never null or omit an entire field category "
                "just because the document type differs from a typical "
                "invoice. Do not include any 'error' or 'note' field "
                "about the request itself. No prose, no markdown "
                "fences, just the JSON object of extracted data:\n\n"
            )
            try:
                if page_chunks:
                    # Same reasoning as the draft path above: never
                    # re-embed the whole (possibly oversized) document in
                    # one retry call just because the FIRST attempt's
                    # merged result looked like a refusal -- redo the
                    # forced-extraction retry per page instead, then
                    # merge again.
                    retry_page_objects = []
                    for page_label, page_text in page_chunks:
                        try:
                            retry_raw = _call_ollama(
                                _forced_extraction_instruction + page_text,
                                max(payload.timeout_secs * 0.35 / len(page_chunks), 15),
                            )
                            rp, _ = _try_parse_json(retry_raw)
                            if isinstance(rp, dict) and not _looks_like_meta_refusal(rp, payload.timeout_secs):
                                retry_page_objects.append(rp)
                        except Exception:
                            # Same one-retry treatment as the main page-chunk
                            # loop above and _extract_one_row -- a saturated
                            # backend shouldn't permanently drop this page's
                            # retry attempt on the first bad moment.
                            try:
                                retry_raw = _call_ollama(
                                    _forced_extraction_instruction + page_text,
                                    max(payload.timeout_secs * 0.35 / len(page_chunks), 15),
                                )
                                rp, _ = _try_parse_json(retry_raw)
                                if isinstance(rp, dict) and not _looks_like_meta_refusal(rp, payload.timeout_secs):
                                    retry_page_objects.append(rp)
                            except Exception:
                                continue
                    retry_parsed = _merge_chunked_json_objects(retry_page_objects) if retry_page_objects else None
                else:
                    retry_raw = _call_ollama(
                        _forced_extraction_instruction + text,
                        max(payload.timeout_secs * 0.35, 10),
                    )
                    retry_parsed, _ = _try_parse_json(retry_raw)
                    if retry_parsed is not None and _looks_like_meta_refusal(retry_parsed, payload.timeout_secs):
                        retry_parsed = None

                if retry_parsed is not None:
                    retry_parsed = _strip_boilerplate_line_items(retry_parsed, payload.timeout_secs)
                    retry_parsed = _repair_bare_code_holding_freetext(retry_parsed)
                    retry_parsed = _repair_orphaned_code_fields(retry_parsed)
                    retry_parsed = _repair_zone_bin_code_confusion(retry_parsed, payload.timeout_secs)
                    retry_parsed = _repair_uom_lcn_confusion(retry_parsed, payload.timeout_secs)
                    retry_parsed = _repair_description_model_split(retry_parsed, text, payload.timeout_secs)
                    retry_parsed = _repair_row_number_mistaken_for_item_code(retry_parsed, text, payload.timeout_secs)
                    retry_parsed = _repair_reversed_party_roles(retry_parsed, text, payload.timeout_secs)
                    retry_parsed = _repair_line_item_field_name_variants(retry_parsed, payload.timeout_secs)
                    retry_parsed = _repair_quantity_amount_reconciliation(retry_parsed)
                    retry_parsed = _verify_constant_fields_against_source(retry_parsed, text)
                    retry_parsed = _flag_sibling_inconsistent_nulls(retry_parsed)
                    retry_parsed = _ensure_handwriting_captured(retry_parsed, text)
                    retry_parsed = _ensure_source_labels_captured(retry_parsed, text, payload.timeout_secs)
                    retry_cleaned = json.dumps(retry_parsed, indent=2, ensure_ascii=False)
                    parsed, cleaned = retry_parsed, retry_cleaned
                    json_meta_refusal_retried = True
            except Exception:
                # Retry failed -- keep the refusal object rather than losing
                # the result entirely; json_parse_failed stays False since
                # it IS valid JSON, just not useful content, and the flag
                # below tells the caller this happened.
                pass

        # `summary` is deliberately NOT a duplicate of the JSON text once
        # parsing succeeds -- `structured` already IS that JSON, as a real
        # nested object rather than an escaped string, and having both
        # fields carry the same content (one readable, one escaped) is
        # pure duplication with no upside: any consumer that wants the
        # parsed data reads `structured` directly, and any consumer that
        # wants the JSON as text (final_answer, synthesis context) has the
        # controller re-serialize `structured` once itself (see
        # _extract_raw_content_for_synthesis) rather than reading a second,
        # already-duplicated copy out of `summary`. Only when parsing
        # genuinely failed does `summary` fall back to holding the raw
        # text, since in that case it's the ONLY representation of what
        # the model actually returned -- `structured` is None then, and
        # there's nothing to duplicate.
        ap_submission = {"status": "skipped", "reason": "json_parse_failed"}
        review_metadata = {}
        if json_parse_failed:
            outputs_summary = cleaned
        else:
            # Pulled OUT of `structured` here -- same reasoning as the
            # per-row path (see _extract_review_metadata): dev-facing
            # extraction diagnostics, not real document content. Popped
            # BEFORE AP submission so the AP Document Store record only
            # ever contains genuine document fields.
            review_metadata = _extract_review_metadata(parsed)
            ap_submission = _finalize_ap_submission(parsed, text, payload.timeout_secs)

            # Same reconciliation as the per-row path above -- keeps
            # `structured.document_type`/`summary` in sync with whatever
            # type AP submission actually resolved to, rather than
            # leaving them stuck on the model's original (possibly
            # overridden) guess.
            doc_label = _reconcile_document_type_label(parsed, ap_submission) if isinstance(parsed, dict) else None
            outputs_summary = (
                f"JSON extraction ({doc_label}) -- see 'structured' field."
                if doc_label else
                "JSON extraction complete -- see 'structured' field for the full object."
            )

        return TaskResult(
            task_id=payload.task_id,
            status=TaskStatus.SUCCESS,
            outputs={
                "summary": outputs_summary,
                "structured": parsed,
                "review_metadata": review_metadata,
                "output_format": "json",
                "json_parse_failed": json_parse_failed,
                "json_repaired": json_repaired,
                "json_meta_refusal_retried": json_meta_refusal_retried,
                "input_length": len(text),
                "model": VLLM_MODEL,
                "doc_id": ap_submission.get("doc_id"),
                "api_doc_type": ap_submission.get("api_doc_type"),
                "ap_submission": ap_submission,
                # TEMPORARY debug fields -- see matching comment in
                # _run_json_mode_per_row above.
                "debug_input_text": text[:8000],
                "debug_input_text_length": len(text[:8000]),
            },
        )

    def run(self, payload: TaskPayload) -> TaskResult:
        text = str(payload.inputs.get("text", "")).strip()
        focus = str(payload.inputs.get("focus", "")).strip()
        # focus_rules is the controller's always-on safety-net text
        # (financial-document check, OCR row-alignment, date check, etc).
        # It used to arrive concatenated INTO `focus` itself -- which meant
        # every summarize call, prose or not, had this boilerplate run
        # through _LIST_CUES_RE / _EXTRACTION_CUES_RE below, and since the
        # boilerplate happens to contain the word "list" ("...logistics
        # documents that list items and quantities..."), EVERY call was
        # being wrongly flagged as list-mode regardless of what the task
        # actually asked for. The controller now sends it as its own field
        # so `focus` stays exactly what the task/goal author actually
        # wrote, and cue-detection below only ever sees that. Falls back to
        # empty string for any caller/version that doesn't send it yet.
        focus_rules = str(payload.inputs.get("focus_rules", "")).strip()

        if not text:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg="inputs.text is required")
        if len(text) < 50:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"inputs.text too short (min 50 chars, got {len(text)})")

        output_format = str(payload.inputs.get("output_format", "")).strip().lower()
        if output_format == "json":
            # Deliberately a SEPARATE path rather than bolting a JSON mode
            # onto the prose path below. The prose path's list_instruction /
            # extraction_override / check_instruction machinery is all
            # built around producing and verifying SENTENCES -- a numbered
            # list of sentences, a bare-entity sentence fragment, a trailing
            # "CHECK: complete" sentence. None of that maps cleanly onto
            # "produce one valid JSON object with no surrounding prose",
            # and trying to make it do double duty risks the same kind of
            # instruction conflict already documented above (see the note
            # on check_instruction vs extraction_override). JSON mode still
            # benefits from `focus` -- including whatever OCR-alignment,
            # blank-field, and financial-document rules the controller's
            # _build_enriched_focus appended to it -- it just doesn't run
            # any of the prose-specific post-processing steps below.
            return self._run_json_mode(payload, text, focus, focus_rules)

        focus_instruction = (
            f"The user's original request was: \"{focus}\". "
            f"Prioritize information in the text that is directly relevant to this request. "
            f"If the text is only tangentially related to the request (e.g. it's a generic "
            f"landing page rather than substantive content), say so explicitly in the summary "
            f"rather than padding it with unrelated details.\n\n"
        ) if focus else ""

        # Computed early (rather than alongside extraction_override further
        # below) so check_instruction can also see it. Attaching BOTH
        # check_instruction ("end your response with a CHECK: line") and
        # extraction_override ("respond with ONLY the entity, no extra
        # sentences, nothing else") to the same call gives the model two
        # directly contradictory instructions -- append a trailing line vs.
        # output nothing but a bare name -- which in production caused the
        # model to resolve the conflict by padding a bare-entity answer into
        # a full paragraph instead of reliably doing either. See
        # extraction_override below for the disqualifier logic itself. Also
        # disqualified by _LIST_CUES_RE: a focus asking for a specific COUNT
        # or LIST of items (e.g. "extract exactly 5 distinct principles,
        # listing them") contains an extraction cue word ("extract") but is
        # unambiguously a multi-item list request, not a single-entity
        # lookup -- without this check it would wrongly collapse to
        # bare-entity mode and return just one name pulled from the text
        # (e.g. a person's name mentioned in passing) instead of the
        # requested list, with no downstream check to catch the loss since
        # extraction mode skips check_instruction entirely.
        # The bare "and" in _EXTRACTION_DISQUALIFIERS_RE is meant to catch
        # genuine multi-part asks ("explain X and evaluate Y"), but it also
        # matches "and" sitting harmlessly inside a parenthetical describing
        # the shape of a SINGLE answer -- e.g. "identify the date (day and
        # month)" is one date, not two things, yet the disqualifier fired
        # and silently downgraded the request out of extraction mode into a
        # full prose+check-instruction answer instead of the bare date the
        # focus explicitly asked for. Stripping parentheticals before this
        # check only (the real prompt below still uses the untouched
        # `focus`) removes that false positive without weakening the "and"
        # disqualifier for genuine multi-clause requests written outside
        # parentheses.
        _focus_for_disqualifier_check = re.sub(r"\([^)]*\)", "", focus)
        needs_extraction_override = (
            bool(focus)
            and bool(_EXTRACTION_CUES_RE.search(focus))
            and not bool(_EXTRACTION_DISQUALIFIERS_RE.search(_focus_for_disqualifier_check))
            and not bool(_LIST_CUES_RE.search(focus))
        )

        check_instruction = (
            "- Finally, end your entire response with one line, on its own, in exactly "
            "this format: \"CHECK: complete\" if the summary above already names the "
            "specific entity, figure, or list/ranking of items the request asked for, using "
            "details that are present in the text below -- or \"CHECK: incomplete\" if the "
            "text below contains a specific entity, name, figure, OR a list/ranking of named "
            "items (e.g. a \"top 10\" table, a ranked list, a set of named options) that is "
            "directly relevant to the request and that the summary above failed to name. A "
            "summary that only gives aggregate statistics (a score, a price range, a count) "
            "about a list WITHOUT naming any of the list's actual items is incomplete, even if "
            "those statistics are accurate. IMPORTANT: if the request asks WHO or WHAT performs "
            "some role in relation to another named thing (e.g. \"who makes it\", \"who "
            "manufactures it\", \"which company owns it\", \"who makes X\"), the summary is "
            "INCOMPLETE unless it states that relationship as its own explicit fact (e.g. \"It "
            "is made by Company A\" / \"Company A manufactures it\") -- the company's name "
            "merely appearing as part of the product's own name (e.g. \"Company A Product X\") "
            "does NOT "
            "count as answering who makes it, since that requires the reader to infer the "
            "connection themselves rather than being told it directly. This must be the "
            "literal last line, exactly as specified, so it can be parsed programmatically.\n\n"
        ) if (focus and not needs_extraction_override) else ""

        # Only attached when focus actually looks like a list/ranking ask --
        # otherwise this ~180-word exception block is pure dead weight the
        # model has to read for no behavioral benefit. See PERFORMANCE NOTE.
        # An explicit prose/sentence-count instruction in the same focus
        # wins over an incidental list-cue word (see
        # _PROSE_FORMAT_OVERRIDE_RE above) -- the requester already told us
        # what shape they want, and a bare cue-word match shouldn't second
        # -guess that.
        needs_list_instruction = (
            bool(focus)
            and bool(_LIST_CUES_RE.search(focus))
            and not bool(_PROSE_FORMAT_OVERRIDE_RE.search(focus))
        )
        list_instruction = (
            "IMPORTANT: if the user's original request above asks for a LIST, RANKING, or "
            "ENUMERATION of items (e.g. \"top 10 X\", \"list the reasons\", \"name the "
            "competitors\", \"what are the options\"), and the text below contains such a "
            "list or ranking (a table, a numbered list, a set of named entries), your summary "
            "MUST name the actual items from it -- as many as you can fit while staying "
            "reasonably concise. Do NOT satisfy the '4-6 sentences' guidance by describing "
            "aggregate statistics about the list (e.g. score ranges, price ranges, counts) "
            "while omitting the items' actual names -- that fails to answer what was asked. "
            "If fitting all requested items breaks the 4-6 sentence guidance, prioritize "
            "naming the items over strict sentence-count adherence. FORMAT REQUIREMENT: "
            "you MUST present these items as a numbered list, one item per line, in the "
            "exact form \"1. Item name -- brief supporting detail\" (score/rank/price/etc, "
            "whatever detail the text supports) -- NOT as a comma-separated sentence or "
            "prose paragraph, even if that means the response reads more like a list than "
            "a 4-6 sentence narrative. A short 1-2 sentence intro line before the numbered "
            "list is fine, but the items themselves must each be their own numbered line.\n"
            "If the text contains MULTIPLE differently-dated or conflicting lists/rankings "
            "on the same topic (e.g. a \"top 10\" from several months apart with different "
            "items), use the single list whose date or framing best matches what the request "
            "is asking for (e.g. the most recent one, or one explicitly labeled for the "
            "requested timeframe) -- name items from THAT one list, rather than blending "
            "items or statistics from different lists together into one inconsistent answer.\n\n"
        ) if needs_list_instruction else ""

        # needs_extraction_override was already computed above (before
        # check_instruction, so that instruction could also account for it).
        # Only attached when focus looks like a bare-entity extraction ask
        # AND doesn't also contain a disqualifying cue (evaluate/describe/
        # "and"/etc. -- the same signals the instruction itself already
        # tells the model to watch for). See PERFORMANCE NOTE.
        extraction_override = (
            "IMPORTANT EXCEPTION -- read carefully, this applies in ONE narrow case only: "
            "if the user's original request above is asking you to IDENTIFY, NAME, or EXTRACT "
            "one specific entity (e.g. a destination city, a person, a company, a ticker "
            "symbol, a single figure) and NOTHING ELSE -- no evaluation, no description, no "
            "further question attached -- then ignore the '4-6 sentences' instruction and "
            "respond with ONLY that entity, in as few words as possible (e.g. \"Company "
            "A\"), no extra sentences, nothing else.\n"
            "CRITICAL TYPE CHECK: the request names the KIND of entity it wants (a city, a "
            "place, a person, a company, a figure, etc.) -- before answering, confirm your "
            "answer is actually THAT kind of thing, not merely the most prominent or most "
            "repeated proper noun in the text. A short biographical passage often names a "
            "PERSON many times while the actual requested PLACE (or vice versa) appears only "
            "once, in passing -- e.g. if asked for the city or place a person lived in, do "
            "NOT return that person's own name just because it appears more often; find the "
            "actual place name (a city, town, region, or country) mentioned in the text, even "
            "if it's mentioned only briefly. The same applies to any other type mismatch: "
            "asked for a company, don't return a product name; asked for a person, don't "
            "return an organization they belong to.\n"
            "This exception does NOT apply if the request also asks you to describe, "
            "evaluate, assess, explain, or answer anything ABOUT that entity -- e.g. \"...and "
            "whether it's a good place to visit\", \"...and is it worth it\", \"...and how is "
            "it performing\", \"...and what do reviews say\". In every one of those cases the "
            "request has TWO parts (name something, THEN say something about it), and you "
            "MUST write the normal 4-6 sentence summary covering BOTH parts using the source "
            "text -- do not collapse it down to just the entity's name. When in doubt (the "
            "request has more than one clause, or contains words like \"and\", \"whether\", "
            "\"how\", \"why\", \"should\"), treat it as a normal summary, not an extraction. "
            "This exception also does NOT apply if the request asks for a list/ranking of "
            "several items (see the list instruction above) -- naming multiple items is not "
            "the same as extracting one entity.\n"
            "DISAMBIGUATION: \"as few words as possible\" does NOT mean stripping out "
            "distinguishing context the source text itself provides. Many entity names are "
            "shared by multiple different real-world things -- e.g. \"Irvine\" names both a "
            "town in Scotland and a city in California; \"Cambridge\" names places in both "
            "England and Massachusetts; \"Springfield\" names dozens of unrelated US towns; "
            "common company/person names can likewise collide. If the source text specifies "
            "a distinguishing qualifier for the entity (a country, state/region, or similar "
            "disambiguator), you MUST include it as part of your answer -- e.g. \"Irvine, "
            "Scotland\" rather than bare \"Irvine\" -- since a bare, common name will very "
            "likely be misread as a different, unrelated entity by whatever downstream step "
            "consumes your answer next (e.g. it could trigger a web search about the wrong "
            "place entirely). Only include qualifiers the source text itself actually states "
            "-- never invent or infer one from outside knowledge. This still overrides the "
            "'as few words as possible' framing above whenever the source text supports a "
            "qualifier; it does not license adding unrelated detail beyond that.\n\n"
        ) if needs_extraction_override else ""

        # Unconditional -- always included when present, never gated behind
        # any of the cue regexes above. This is the controller's safety-net
        # text (financial-doc check, OCR alignment, date check, etc); it
        # still needs to reach the model's prompt exactly as before, it
        # just must never be examined BY _LIST_CUES_RE / _EXTRACTION_CUES_RE
        # / _EXTRACTION_DISQUALIFIERS_RE, which all run on `focus` alone,
        # above this point.
        focus_rules_block = f"{focus_rules}\n\n" if focus_rules else ""

        # Draft call. When a focus is present, the same call also asks the
        # model to self-check its own draft against the request and append a
        # CHECK marker -- no separate validation call needed for this step.
        draft_budget = payload.timeout_secs * (0.45 if focus else 0.9)
        try:
            draft = _call_ollama(
                "Summarize the following text in 4-6 sentences with enough supporting "
                "detail to feel like a complete, informative answer -- not just the "
                "bare minimum needed to technically cover the request. "
                "Focus on key points only. No preamble. "
                "Write directly about the subject matter using what the text actually "
                "supports -- never write ABOUT the text itself (e.g. never say \"the "
                "text does not identify...\", \"the provided text fails to state...\", "
                "\"the source does not clearly rank...\"). If the text only partially "
                "supports the request, state the best-supported answer plainly and "
                "confidently using the strongest available evidence, rather than "
                "describing the gap or critiquing the text's completeness. "
                "If the text contains specific numbers, statistics, percentages, "
                "or monetary figures, you MUST include at least one of them in the summary — "
                "do not drop all figures in favour of qualitative description.\n\n"
                + focus_instruction + list_instruction + extraction_override + check_instruction + focus_rules_block + text,
                draft_budget,
            )
        except Exception as e:
            return TaskResult(task_id=payload.task_id,
                              status=TaskStatus.FAILURE,
                              error_msg=f"Summarization failed: {e}")

        summary, incomplete = _parse_check_marker(draft) if focus else (draft, False)
        gap_fill_triggered = False
        extraction_condensed = False

        # Guard for extraction-mode calls: the model was told to respond with
        # ONLY the entity, but instruction-following isn't guaranteed, and a
        # bloated response here is worse than a normal-summary bloat -- this
        # text is typically about to be spliced verbatim into a downstream
        # task's query or input (per the controller's dependent-entity
        # chaining), so an unexpectedly long "entity" can drown out whatever
        # else that downstream task actually needed to ask about. Cheap fix:
        # one short, fast follow-up call that only has to condense already-
        # correct content down to the bare name, not re-derive anything from
        # the source text.
        if needs_extraction_override and _extraction_output_too_long(summary):
            try:
                condensed = _call_ollama(
                    "The text below was supposed to be a bare answer -- just one "
                    "specific entity's name (e.g. a place, person, company, or "
                    "figure) and NOTHING else -- but it contains extra sentences "
                    "or padding instead. Extract and return ONLY that single "
                    "entity, in as few words as possible, with no extra "
                    "commentary, no punctuation beyond what the name itself "
                    "requires, and nothing else. EXCEPTION: if the entity name is "
                    "one that could plausibly refer to multiple different "
                    "real-world things (e.g. a place name that exists in more "
                    "than one country/region), and the text below states a "
                    "distinguishing qualifier for it, keep that qualifier in "
                    "your answer (e.g. \"Irvine, Scotland\", not bare "
                    "\"Irvine\") -- do not strip it out just to minimize word "
                    "count:\n\n" + summary,
                    max(payload.timeout_secs * 0.2, 10),
                )
                if condensed.strip():
                    summary = condensed.strip()
                    extraction_condensed = True
            except Exception:
                # Condensing failed -- keep the verbose-but-correct draft
                # rather than losing the result entirely.
                pass

        # Separate guard, different failure mode than the length check
        # above: the response was short (so it passed length-wise) but is
        # just an echo of a proper noun already named in the FOCUS itself --
        # e.g. focus asks "identify the city where Galt lived" and the
        # answer comes back as "Galt" (the person, not the place). A short
        # answer that repeats the question is not a correct extraction, so
        # this needs a distinct re-derivation call pointed back at the
        # source text rather than the condensing call above (which only
        # trims already-correct content, it doesn't fix wrong content).
        extraction_echo_retried = False
        if needs_extraction_override and _extraction_echoes_focus(summary, focus):
            try:
                reextracted = _call_ollama(
                    f"The request was: \"{focus}\"\n\n"
                    f"Your previous answer was: \"{summary}\"\n\n"
                    "That answer just repeats a term already given IN the request itself -- "
                    "it cannot be correct, since the whole point of this task is to find "
                    "something the requester does not already know. Look again at the "
                    "source text below and find the ACTUAL entity being asked for -- it is "
                    "a DIFFERENT proper noun than the one already named in the request (e.g. "
                    "if asked for the place a named person lived, find the actual city/town/"
                    "region/country mentioned in the text, not the person's own name again). "
                    "If that entity's name could plausibly refer to multiple different "
                    "real-world things (e.g. a place name shared by more than one country/"
                    "region) and the text states a distinguishing qualifier for it, include "
                    "that qualifier in your answer (e.g. \"Irvine, Scotland\", not bare "
                    "\"Irvine\"). Respond with ONLY that different entity (plus qualifier if "
                    "applicable), nothing else:\n\n" + text,
                    max(payload.timeout_secs * 0.25, 10),
                )
                candidate = reextracted.strip()
                if candidate and candidate.lower() not in focus.lower():
                    summary = candidate
                    extraction_echo_retried = True
            except Exception:
                # Re-derivation failed -- keep the (likely wrong) echoed
                # answer rather than losing the result entirely; better to
                # surface something the caller can sanity-check than error out.
                pass

        # Third, independent extraction-failure check: the focus asked for
        # a PLACE but the source text shows the returned name being used
        # as a PERSON (see _extraction_looks_like_person_not_place). Runs
        # regardless of whether the echo-retry above fired, since it
        # catches a different failure signature than that check does.
        extraction_type_corrected = False
        if needs_extraction_override and _extraction_looks_like_person_not_place(summary, focus, text):
            try:
                retyped = _call_ollama(
                    "The request asked for a PLACE (a city, town, region, or country) -- "
                    f"not a person. Your previous answer was: \"{summary}\"\n\n"
                    f"But the source text below shows \"{summary}\" being used as a "
                    "PERSON's name (e.g. it appears right before/after phrasing like "
                    "\"is born\" or \"child of\"), not a place. Look again at the source "
                    "text and find the ACTUAL place name (a city, town, region, or "
                    "country) mentioned -- it may appear only once, in passing, even "
                    "if a person's name appears more often nearby. If that place's name "
                    "could plausibly refer to multiple different real-world things and "
                    "the text states a distinguishing qualifier (a country or region), "
                    "keep that qualifier in your answer (e.g. \"Irvine, Scotland\", not "
                    "bare \"Irvine\"). Respond with ONLY that place name, nothing "
                    "else:\n\n" + text,
                    max(payload.timeout_secs * 0.25, 10),
                )
                candidate = retyped.strip()
                if candidate and candidate.lower() != summary.strip().lower():
                    summary = candidate
                    extraction_type_corrected = True
            except Exception:
                # Re-derivation failed -- keep the (likely wrong) answer
                # rather than losing the result entirely.
                pass

        # Only spend a second LLM call when the draft flagged itself as
        # missing something -- and even then, reuse the text already in
        # hand rather than doing another search+crawl round trip.
        if incomplete:
            try:
                revised = _call_ollama(
                    f"The user's original request was: \"{focus}\".\n\n"
                    f"Here is a draft summary of a source text:\n{summary}\n\n"
                    f"Here is the full source text the draft was drawn from:\n{text}\n\n"
                    "The draft is missing a specific entity, name, figure, OR a list/ranking "
                    "of named items that the source text actually contains and that would "
                    "more directly answer the request. Rewrite the answer so it includes the "
                    "missing specific detail(s), using only what the source text actually "
                    "contains. If the request asked for a list/ranking of several items (e.g. "
                    "\"top 10 X\"), name as many of the actual items as you can fit -- a "
                    "numbered list of names is acceptable and preferred over prose here, even "
                    "if it runs longer than 4-6 sentences; do not just restate aggregate "
                    "statistics without naming items. Otherwise, only keep the revised answer "
                    "to just one entity's name if the original request was asking SOLELY to "
                    "identify/name/extract it with nothing else attached -- if the request "
                    "also asks you to describe, evaluate, or answer anything ABOUT that entity "
                    "(e.g. \"...and whether it's good\", \"...and how is it performing\"), "
                    "write the normal 4-6 sentences covering every part of the request "
                    "instead. If the request asks WHO or WHAT performs some role in relation to "
                    "another named thing (e.g. \"who makes it\"), that role MUST be covered by a "
                    "direct statement of the relationship (e.g. \"It is made by Company A\") -- "
                    "simply having the company's name already appear inside the product's own "
                    "name is not a fix, since the reader still has to infer the connection "
                    "themselves. This direct statement is ONE clause to ADD inside the normal "
                    "4-6 sentence summary alongside everything else the request asked about --"
                    "it is NOT a replacement for that summary. Do not collapse the whole answer "
                    "down to just that one sentence: keep the full 4-6 sentences of substantive "
                    "detail from the source text, with the direct statement woven in as one of "
                    "them, not standing alone as the entire response. Write directly about "
                    "the subject matter using what the source text actually supports -- "
                    "never write ABOUT the text itself (e.g. never say \"the text does not "
                    "identify...\", \"the source fails to state...\", \"the text does not "
                    "clearly rank...\"). If the source material is genuinely mixed or "
                    "inconsistent, state the best-supported answer plainly and confidently "
                    "using the strongest available evidence, rather than describing the "
                    "inconsistency or critiquing the text's completeness. No "
                    "preamble, and do not include any "
                    "marker or meta-commentary -- just the revised answer.",
                    payload.timeout_secs * 0.45,
                )
                summary = revised.strip()
                gap_fill_triggered = True
            except Exception:
                # Second call failing shouldn't lose the result -- fall back
                # to the honest-if-incomplete draft rather than erroring out.
                pass

        # HARD GUARANTEE for list-mode requests: the prompt-level
        # FORMAT REQUIREMENT in list_instruction is a strong ask, but ask
        # is all it is -- instruction-following isn't certain, and this
        # gap-fill/extraction pipeline already has multiple examples above
        # of the model not doing what it was told. So when this request
        # was flagged as list-type, actually VERIFY the structural shape
        # of the current summary with a regex, independent of anything the
        # model claims about its own output, and if it doesn't already
        # look like a numbered list, force one more call whose ONLY job is
        # reformatting -- not re-deriving content -- then verify that
        # result too before accepting it. If even the reformat call fails
        # the structural check, fall back to a pure-code line-splitter
        # rather than ever silently accepting an unformatted list.
        list_reformatted = False
        if needs_list_instruction and not _looks_like_numbered_list(summary):
            try:
                reformatted = _call_ollama(
                    "Reformat the text below so every distinct item it names becomes "
                    "its own numbered line, in the exact form \"1. Item name -- brief "
                    "supporting detail\" (one item per line, using whatever detail -- "
                    "score, rank, price, etc -- the text already provides for that "
                    "item). This is a PURE FORMATTING pass: do NOT add, remove, "
                    "invent, reorder, or drop any item, fact, or number already present "
                    "in the text below. A short 1-sentence intro line before the "
                    "numbered lines is fine, but the FINAL RESULT MUST BE ONLY that "
                    "optional short intro plus the numbered lines -- if the text below "
                    "already contains both a prose paragraph AND a numbered list "
                    "covering the same items, DELETE the redundant prose entirely and "
                    "keep only the numbered list (plus, at most, one short intro "
                    "sentence). Do not leave any narrative recap before or after the "
                    "numbered lines that restates items the list already names. "
                    "Return ONLY the reformatted result, no extra commentary:\n\n" + summary,
                    max(payload.timeout_secs * 0.2, 10),
                )
                if reformatted.strip() and _looks_like_numbered_list(reformatted):
                    summary = reformatted.strip()
                    list_reformatted = True
            except Exception:
                # Reformat call itself failed -- fall through to the
                # code-level fallback below rather than giving up.
                pass

            if not _looks_like_numbered_list(summary):
                # Last-resort, model-free fallback: split on the most
                # common separators a comma/semicolon-joined list uses and
                # number the pieces by hand. This can't fix genuinely
                # malformed input, but guarantees SOME numbered structure
                # rather than ever returning unnumbered prose once this
                # request has been flagged as list-type.
                pieces = [p.strip(" .;") for p in re.split(r",\s+(?:and\s+)?|;\s+", summary) if p.strip(" .;")]
                if len(pieces) >= 2:
                    summary = "\n".join(f"{i+1}. {p}" for i, p in enumerate(pieces))
                    list_reformatted = True

        return TaskResult(
            task_id=payload.task_id,
            status=TaskStatus.SUCCESS,
            outputs={
                "summary": summary,
                "input_length": len(text),
                "model": VLLM_MODEL,
                "gap_fill_triggered": gap_fill_triggered,
                "extraction_condensed": extraction_condensed,
                "extraction_echo_retried": extraction_echo_retried,
                "extraction_type_corrected": extraction_type_corrected,
                "list_reformatted": list_reformatted,
            },
        )


agent = SummarizeAgent()

agent.generate_manifest(
    endpoint=AGENT_ENDPOINT,
    input_schema={
        "text": {"type": "string", "required": True},
        "focus": {"type": "string", "required": False},
        # "json" switches the agent into the JSON-output path (see
        # _run_json_mode): a single valid JSON object with no surrounding
        # prose, rather than the default 4-6 sentence summary. Any other
        # value, or omitting this field, keeps the existing prose behavior
        # unchanged -- older callers that don't know about this field are
        # unaffected.
        "output_format": {"type": "string", "required": False},
        # Always-on safety-net rules from the controller (financial-doc
        # check, OCR alignment, date check, etc), kept separate from
        # `focus` so the list/extraction cue-detection heuristics above
        # only ever see the task's own actual ask -- see the note at the
        # top of run() for why this used to be folded into `focus` itself
        # and what that broke.
        "focus_rules": {"type": "string", "required": False},
    },
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "input_length": {"type": "integer"},
            "model": {"type": "string"},
            "gap_fill_triggered": {"type": "boolean"},
            "extraction_condensed": {"type": "boolean"},
            "extraction_echo_retried": {"type": "boolean"},
            "extraction_type_corrected": {"type": "boolean"},
            "list_reformatted": {"type": "boolean"},
            # Only populated when output_format="json" was requested.
            # `structured` is the parsed JSON object (null if parsing
            # failed even after the repair retry -- callers should check
            # `json_parse_failed` rather than assuming `structured` is
            # always present). `summary` still carries the raw JSON text
            # in this mode, so existing callers that only read `summary`
            # keep working unchanged, just receiving JSON text instead of
            # prose.
            "structured": {"type": ["object", "null"]},
            "output_format": {"type": "string"},
            "json_parse_failed": {"type": "boolean"},
            "json_repaired": {"type": "boolean"},
            "json_meta_refusal_retried": {"type": "boolean"},
            # AP Document Store integration (only populated when
            # output_format="json" produced a parseable object). doc_id
            # and api_doc_type are None when submission was skipped --
            # callers should check ap_submission.status rather than
            # assuming doc_id is always present.
            "doc_id": {"type": ["string", "null"]},
            "api_doc_type": {"type": ["string", "null"]},
            "ap_submission": {"type": ["object", "null"]},
        },
    },
)

# extra_protected_paths tells the HMAC middleware to also guard /run_async
# from the moment the app is built, rather than the route being added to
# the Flask app afterwards where the middleware never sees it.
app = build_flask_app(agent, BYOA_SECRET, extra_protected_paths={"/run_async"})

# ── Async job-queue routes ───────────────────────────────────────────────────
# Added because summarize is the one agent slow enough (multi-tens-of-seconds
# on long crawled text) to occasionally exceed Cloudflare Tunnel's ~100s edge
# timeout when called synchronously via /run. Every other agent (webcrawl,
# websearch, fx) stays on the fast synchronous /run path -- unchanged.
#
# /run_async returns immediately with a job_id and runs agent._timed_run()
# in a background thread. The controller then polls /status/<job_id> every
# few seconds; each individual HTTP request (submit + each poll) completes
# in well under a second, so none of them can ever hit the tunnel's edge
# timeout, no matter how long the underlying summarization actually takes.
#
# This now comes from byoa_adk (v1.1.0+) instead of being hand-rolled here.
# The previous inline version registered /run_async directly on `app`
# *after* HMACMiddleware had already been attached to only intercept /run --
# so /run_async was reachable with no HMAC check at all. attach_async_routes()
# is meant to be paired with extra_protected_paths={"/run_async"} above so
# the middleware actually covers it.
attach_async_routes(app, agent)


if __name__ == "__main__":
    verify_vllm_ready()
    print("=" * 60)
    print(f"Summarize Agent -- {agent.name} v{agent.version}")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/agent.md")
    print(f"  POST {AGENT_ENDPOINT}/run")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)
