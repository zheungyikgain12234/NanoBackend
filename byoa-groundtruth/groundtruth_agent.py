"""
byoa-groundtruth -- fetches the ground-truth JSON for a document from the
DocStore API, keyed by DOCUMENT NUMBER.

Sits between byoa-md-json and byoa-val-json in the fixed route:
    byoa-md-json > byoa-groundtruth > byoa-val-json

Inputs : {"actual": <obj|JSON str>   # byoa-md-json's `structured` output
          "document_no": str}        # either/both; explicit document_no wins
Outputs: {"found": bool, "document_no": str, "ground_truth": {...},
          "source_file": str, "tried": [...], "method": str}

Finding the document number (field names differ per document, so none is fixed):
  1. `document_no` itself, when given as an input or present as a key.
  2. Variants of it by key name (document_number, doc_no, invoice_no, ...).
  3. Model inference: when 1-2 yield nothing, or every candidate they yield is
     unknown to the DocStore, the extracted JSON's scalar fields are shown to
     the model, which picks WHICH FIELD is the document number. The model only
     returns a field path; the value is read from the JSON, so it can never
     invent a number, and the DocStore lookup is the final check.

Load-friendliness (no limits, no timeouts, nothing blocks):
  * Served async (/run_async + /status/<job_id>), so no HTTP request to this
    agent ever stays open for the lookup's duration.
  * The DocStore call has a CONNECT timeout only; the read timeout is None, so
    a slow or very large response is never cut off. Response bodies are
    streamed to a buffer; there is no size cap on either side.
  * One GET per candidate document number, in priority order, stopping at the
    first hit (normally exactly 1 call). Connection errors / 429 / 502-504
    are retried with capped exponential backoff, honouring Retry-After; a 202
    ("still preparing") is polled the same way. 404 is final for that number.
"""
import json
import os
import re
import sys
import time
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent, attach_async_routes
from byoa_adk.llm_client import call_llm
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5800"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://groundtruth.zackloh.dev")
DOCSTORE_BASE_URL = os.environ.get("DOCSTORE_BASE_URL", "http://localhost:5900").rstrip("/")
DOCSTORE_TOKEN = os.environ.get("DOCSTORE_TOKEN", "")  # optional bearer for the real API

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")
LLM_TIMEOUT = None            # no client-side timeout; the agent is served async
CONNECT_TIMEOUT = float(os.environ.get("DOCSTORE_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = None            # deliberately unlimited
MAX_CANDIDATES = int(os.environ.get("DOCSTORE_MAX_CANDIDATES", "6"))
BACKOFF_START, BACKOFF_CAP = 1.0, 30.0

# Key names (normalised) tried first, best first. `documentno` is the canonical one.
_ID_PRIORITY = ["documentno", "documentnumber", "docno", "docnumber", "docid",
                "invoiceno", "invoicenumber", "invno", "deliveryno", "deliverynumber",
                "grnno", "ppinvno"]
# Variant keys: look like "<doc-ish> ... <no/num/id>" and are not some other kind of number.
_VARIANT_HAS = re.compile(r"(doc|invoice|inv|delivery|grn|voucher|receipt|note)")
_VARIANT_ENDS = re.compile(r"(no|num|number|id|ref)$")
_VARIANT_NOT = re.compile(r"(external|po|purchase|order|customer|account|reg|gst|tel|phone|"
                          r"label|heading|type|date|page|bank|swift|bill|company|sales)")

_session = requests.Session()


def _nk(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _load(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _body(actual):
    if isinstance(actual, dict) and isinstance(actual.get("structured"), dict):
        return actual["structured"]
    return actual


def _scalar_leaves(node, path="$", out=None):
    """(path, value) for scalar fields in nested objects. Lists (line items) are skipped:
    a document number is a header field, never a table cell."""
    out = [] if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}"
            if isinstance(v, dict):
                _scalar_leaves(v, p, out)
            elif isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip():
                out.append((p, str(v).strip()))
    return out


def keyed_candidates(actual) -> list:
    """Tiers 1-2: document number by field NAME -- `document_no` first, then variants."""
    leaves = _scalar_leaves(_body(actual))
    ranked = []
    for path, value in leaves:
        nk = _nk(path.rsplit(".", 1)[-1])
        if nk in _ID_PRIORITY:
            ranked.append((_ID_PRIORITY.index(nk), value))
        elif _VARIANT_HAS.search(nk) and _VARIANT_ENDS.search(nk) and not _VARIANT_NOT.search(nk):
            ranked.append((len(_ID_PRIORITY), value))
    out = []
    for _, v in sorted(ranked, key=lambda t: t[0]):
        if v not in out:
            out.append(v)
    return out[:MAX_CANDIDATES]


_LLM_SCHEMA = {"type": "object",
               "properties": {"paths": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
               "required": ["paths"]}
_LLM_PROMPT = """Below are the fields of a business document (an invoice, delivery note, goods receipt note, etc.) extracted from OCR text. Field names vary from document to document.

Which field holds THE DOCUMENT'S OWN NUMBER -- the number that identifies this document itself (e.g. "Invoice No.", "Document No.", "Delivery Note No.", "GRN No.")?
NOT the purchase-order / customer-order / sales-order number, not a registration, GST, account, phone or bank number, and not a date.

Answer as JSON: {{"paths": [...]}} with up to 3 field paths, most likely first, copied EXACTLY from the list. Use an empty list if none of the fields is a document number.

Fields:
{fields}
"""


def llm_candidates(actual, exclude=()) -> list:
    """Tier 3: ask the model WHICH FIELD is the document number. It returns paths only;
    values are read from the JSON, so nothing can be invented."""
    leaves = _scalar_leaves(_body(actual))
    by_path = dict(leaves)
    if not by_path:
        return []
    fields = "\n".join(f"{p} = {v[:100]}" for p, v in leaves)
    resp = call_llm(
        VLLM_BASE_URL, "/v1/chat/completions",
        json={"model": VLLM_MODEL,
              "messages": [{"role": "user", "content": _LLM_PROMPT.format(fields=fields)}],
              "temperature": 0,
              "guided_json": _LLM_SCHEMA,
              "chat_template_kwargs": {"enable_thinking": False}},
        timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    parsed = json.loads(text)
    # the model sometimes answers with a bare array instead of {"paths": [...]}
    paths = parsed.get("paths", []) if isinstance(parsed, dict) else parsed
    paths = [x for x in paths if isinstance(x, str)] if isinstance(paths, list) else []
    out = []
    for p in paths:
        v = by_path.get(p)
        if v and v not in out and v not in exclude:
            out.append(v)
    return out


def fetch_ground_truth(doc_no: str) -> dict | None:
    """GET /documents/<doc_no>. None on 404. Retries transient failures forever
    (bounded only by the caller's job lifetime); never applies a read timeout."""
    url = f"{DOCSTORE_BASE_URL}/documents/{quote(doc_no, safe='')}"
    headers = {"Accept": "application/json"}
    if DOCSTORE_TOKEN:
        headers["Authorization"] = f"Bearer {DOCSTORE_TOKEN}"
    delay = BACKOFF_START
    while True:
        try:
            r = _session.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except (requests.ConnectionError, requests.exceptions.ConnectTimeout):
            time.sleep(delay)
            delay = min(delay * 2, BACKOFF_CAP)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (202, 429, 502, 503, 504):
            try:
                wait = float(r.headers.get("Retry-After", delay))
            except ValueError:
                wait = delay
            time.sleep(min(max(wait, 0.5), BACKOFF_CAP))
            delay = min(delay * 2, BACKOFF_CAP)
            continue
        r.raise_for_status()


class GroundTruthAgent(Agent):
    name = "byoa-groundtruth"
    version = "1.0.0"
    description = ("Fetches the ground-truth JSON for a document from the DocStore API by "
                   "document number. Pass the ENTIRE extracted JSON as `actual` (e.g. "
                   "$task_1.structured); the agent finds the document number itself, "
                   "whatever the field is called.")
    capabilities = ["data-fetch"]
    llm_base_url = VLLM_BASE_URL

    input_schema = {
        "actual": {"type": "object", "required": False,
                   "description": "The whole extracted JSON (pass it entirely, not one field). "
                                  "The document number is located inside it."},
        "document_no": {"type": "string", "required": False,
                        "description": "Only if the number is already known exactly."},
    }
    output_schema = {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "document_no": {"type": "string"},
            "ground_truth": {"type": "object"},
            "source_file": {"type": "string"},
            "tried": {"type": "array"},
            "method": {"type": "string"},
        },
    }

    def run(self, payload: TaskPayload) -> TaskResult:
        try:
            explicit = str(payload.inputs.get("document_no") or "").strip()
            actual = _load(payload.inputs["actual"]) if payload.inputs.get("actual") is not None else None
            if not explicit and actual is None:
                return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE,
                                  error_msg="No input: pass `actual` (the extracted JSON) or `document_no`.")
            tried, llm_error = [], ""

            def attempt(cands, method):
                for doc_no in cands:
                    if doc_no in tried:
                        continue
                    tried.append(doc_no)
                    doc = fetch_ground_truth(doc_no)
                    if doc is not None:
                        return TaskResult(task_id=payload.task_id, status=TaskStatus.SUCCESS, outputs={
                            "found": True,
                            "document_no": doc.get("document_no", doc_no),
                            "ground_truth": doc["ground_truth"],
                            "source_file": doc.get("source_file", ""),
                            "tried": list(tried),
                            "method": method,
                        })
                return None

            # 1-2: explicit number, then `document_no` / its variants by field name
            hit = attempt(([explicit] if explicit else []) + (keyed_candidates(actual) if actual else []),
                          "explicit" if explicit else "field_name")
            # 3: model inference over the extracted JSON's fields
            if hit is None and actual is not None:
                try:
                    hit = attempt(llm_candidates(actual, exclude=tried), "llm_inference")
                except Exception as exc:
                    llm_error = f" (model inference failed: {exc})"
            if hit is not None:
                return hit
            return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE,
                              error_msg=f"No ground truth in DocStore for document number(s) {tried}"
                                        f"{llm_error}" if tried else
                                        f"Could not find a document number in the input{llm_error}")
        except Exception as exc:
            return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE,
                              error_msg=f"Ground-truth lookup failed: {exc}")


agent = GroundTruthAgent()
agent.generate_manifest(endpoint=AGENT_ENDPOINT)
app = build_flask_app(agent, BYOA_SECRET, extra_protected_paths={"/run_async"})
attach_async_routes(app, agent)


if __name__ == "__main__":
    if "--lookup" in sys.argv:       # python groundtruth_agent.py --lookup DOC_NO
        print(json.dumps(fetch_ground_truth(sys.argv[sys.argv.index("--lookup") + 1]), indent=2))
        sys.exit(0)
    print(f"byoa-groundtruth v{agent.version} on :{PORT}  docstore={DOCSTORE_BASE_URL}")
    serve(app, host="0.0.0.0", port=PORT, threads=8)
