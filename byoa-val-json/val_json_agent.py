"""
BYOA Val-JSON Agent -- compares a JSON produced by byoa-summarize against a
ground-truth JSON and reports whether the VALUES are accurate.

The comparison is deliberately NOT strict about structure:
  - Field names may differ ("invoice_no" vs "invoiceNumber" is fine).
  - Nesting / ordering may differ (array order is irrelevant).
  - Extra fields in the actual JSON are reported but never penalised.
Only the ground-truth VALUES must be found, accurately, in the actual JSON.

Matching strategy (cheap and deterministic first, LLM only for the rest):
  1. Flatten both documents to scalar leaves (path -> value).
  2. Deterministic pass: pair each ground-truth leaf with an unused actual
     leaf holding an equivalent value after normalisation (numbers with
     currency/thousand separators, dates, case/whitespace/punctuation).
     When several actual leaves hold the same value, the one whose field
     name is most similar wins (so two "0" values don't get crossed).
  3. LLM pass (platform vLLM, via the sanctioned call_llm path) for the
     ground-truth leaves still unmatched: pairs semantically equivalent
     values ("Singapore Dollar" ~ "SGD") and flags genuinely different ones.
     If the LLM is unavailable the agent still returns the deterministic
     result, with llm_used=false.
  4. Whatever is left is "missing" (no counterpart in actual) or a
     "mismatch" (a counterpart field exists but the value is wrong).

Inputs : {"actual": <obj|JSON str>, "ground_truth": <obj|JSON str>,
          "numeric_tolerance": 0.01 (optional, absolute)}
Outputs: {"match": bool, "score": float, ...}  (see output_schema)
"""
import difflib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv
from waitress import serve

from byoa_adk import Agent, attach_async_routes
from byoa_adk.llm_client import call_llm
from byoa_adk.schema import TaskPayload, TaskResult, TaskStatus
from byoa_adk.server import build_flask_app

load_dotenv()

PORT = int(os.environ.get("PORT", "5600"))
BYOA_SECRET = os.environ.get("BYOA_SECRET", "byoa-dev-secret-change-in-prod")
os.environ["BYOA_SECRET"] = BYOA_SECRET
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://val-json.zackloh.dev")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")

# Cap on leaves sent to the LLM in one call -- keeps the prompt bounded for
# very large documents; anything beyond it is simply reported as unmatched.
MAX_LLM_LEAVES = int(os.environ.get("MAX_LLM_LEAVES", "150"))
# No client-side timeout on the vLLM call: big documents legitimately take a
# while. The agent is served async (/run_async + /status/<job_id>), so no
# single HTTP request ever has to stay open for the duration.
LLM_TIMEOUT = None
# Each LLM alignment call covers at most this many ground-truth fields; calls
# for different records / chunks run in parallel (vLLM batches them natively).
LLM_CHUNK = int(os.environ.get("LLM_CHUNK", "25"))
LLM_WORKERS = int(os.environ.get("LLM_WORKERS", "4"))

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


# --------------------------------------------------------------------------
# Parsing / flattening / normalisation
# --------------------------------------------------------------------------

def _load(value, label: str):
    """Accept an already-parsed object/array or a JSON string (fences ok)."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = re.sub(r'^```(?:json)?\s*|\s*```$', '', value.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(s)
        except Exception as exc:
            raise ValueError(f"'{label}' is not valid JSON: {exc}")
    raise ValueError(f"'{label}' must be a JSON object, array or JSON string")


def _flatten(node, path="$"):
    """Yield (path, scalar) for every scalar leaf. Nulls/empty containers skipped."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _flatten(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _flatten(v, f"{path}[{i}]")
    elif node is not None and node != "":
        yield path, node


def _key_of(path: str) -> str:
    """Last field-name segment of a path, lowercased, alphanumerics only."""
    last = re.sub(r'\[\d+\]', '', path.rsplit(".", 1)[-1])
    return re.sub(r'[^a-z0-9]', '', last.lower())


def _to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        neg = s.startswith("(") and s.endswith(")")
        s = s.strip("()").strip()
        # currency code/symbol only as a prefix/suffix next to the number, so
        # ids like 'INV-9' or 'GX21L58557' are never mistaken for numbers
        s = re.sub(r'(?i)^(?:S\$|RM|[A-Z]{3}(?=[\s\d]))', '', s)
        s = re.sub(r'(?i)(?<=[\d\s])[A-Z]{3}$', '', s)
        s = re.sub(r'[$€£¥\s]', '', s)
        if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', s):
            s = s.replace(",", "")
        elif re.fullmatch(r'-?\d+(\.\d+)?', s):
            pass
        else:
            return None
        n = float(s)
        return -n if neg else n
    return None


def _to_date(v):
    if not isinstance(v, str):
        return None
    s = v.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _norm_str(v) -> str:
    s = re.sub(r'\s+', ' ', str(v)).strip().casefold()
    return s.strip(" .,;:-_'\"")


def _equivalent(a, b, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        if isinstance(a, bool) and isinstance(b, bool):
            return a == b
        other = b if isinstance(a, bool) else a
        flag = a if isinstance(a, bool) else b
        words = {"true": True, "yes": True, "y": True, "t": True, "1": True,
                 "false": False, "no": False, "n": False, "f": False, "0": False}
        if isinstance(other, str):
            return words.get(other.strip().lower()) is flag
        return isinstance(other, (int, float)) and other in (0, 1) and bool(other) is flag
    na, nb = _to_number(a), _to_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= tol
    da, db = _to_date(a), _to_date(b)
    if da and db:
        return da == db
    ua, ub = _num_unit(a), _num_unit(b)
    if ua and ub:
        return ua[1] == ub[1] and abs(ua[0] - ub[0]) <= tol
    pa, pb = _phone_digits(a), _phone_digits(b)
    if pa and pb:
        return pa == pb
    return _norm_str(a) == _norm_str(b)


# --------------------------------------------------------------------------
# LLM semantic pass
# --------------------------------------------------------------------------

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gt": {"type": "integer"},
                    "actual": {"type": "integer"},
                    "same_value": {"type": "boolean"},
                },
                "required": ["gt", "actual", "same_value"],
            },
        }
    },
    "required": ["pairs"],
}

_LLM_PROMPT = """You are aligning the fields of an ACTUAL extracted JSON to a \
GROUND TRUTH JSON. The two use different field names, nesting and ordering for \
the same underlying information. Decide which ACTUAL field corresponds to each \
GROUND TRUTH field using BOTH the field name/path AND the value as evidence.

Each line is `id: path = value`. Paths show structure: fields sharing an array \
element (e.g. `$.items[0].qty` and `$.items[0].desc`) belong to the same record, \
so align whole records together -- use the description/name fields to decide \
which actual line item is which ground-truth line item, then align their \
qty/amount fields within that pair.

{hint}GROUND TRUTH:
{gt}

ACTUAL:
{actual}

For every ground-truth field that has a corresponding ACTUAL field, output \
{{"gt": <id>, "actual": <id>, "same_value": <bool>}}.
 - Align by meaning of the field, not by value alone. Two fields merely holding \
the same number are NOT a pair unless the names/context say they are the same \
kind of information.
 - same_value=true if the VALUES are equivalent in meaning (formatting, \
abbreviations, currency names vs codes, paraphrases of the same fact).
 - same_value=false if it is the same field but the value is factually \
different (wrong number, wrong name, wrong date).
 - Omit a ground-truth field if the ACTUAL JSON has no field for that \
information at all.
Use each ACTUAL id at most once. Respond with ONLY a JSON object: \
{{"pairs": [...]}}"""


def _short(v, n=80) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    fenced = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
    for cand in (content, fenced):
        try:
            return json.loads(cand)
        except Exception:
            pass
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"unparseable model response: {content[:200]!r}")


_SAME_RECORD_HINT = ("NOTE: these two groups of fields come from two records that are ALREADY "
    "KNOWN to describe the same item (e.g. the same line item). Every ground-truth field "
    "that has any counterpart in the ACTUAL group should be paired -- match by field "
    "meaning (qty~quantity, price~unit_cost, ...). Only omit a field if the ACTUAL "
    "group truly has nothing for it.\n\n")


def _as_idx(x):
    """Model ids sometimes come back as "3" despite the integer schema."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, (str, float)) and re.fullmatch(r'\s*\d+(\.0+)?\s*', str(x)):
        return int(float(x))
    return None


def _llm_pair(gt_left, act_left, same_record=False):
    """Return list of (gt_idx, act_idx, same_value, reason) over the given lists."""
    gt_txt = "\n".join(f"{i}: {p} = {_short(v)}" for i, (p, v) in enumerate(gt_left))
    act_txt = "\n".join(f"{i}: {p} = {_short(v)}" for i, (p, v) in enumerate(act_left))
    resp = call_llm(
        VLLM_BASE_URL,
        "/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": _LLM_PROMPT.format(gt=gt_txt, actual=act_txt,
                                                                    hint=_SAME_RECORD_HINT if same_record else "")}],
            "temperature": 0,
            "guided_json": _LLM_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    data = _parse_llm_json(resp.json()["choices"][0]["message"]["content"])
    if os.environ.get("VAL_JSON_DEBUG"):
        print(f"[LLM job] {len(gt_left)} gt x {len(act_left)} actual fields"
              f"{' (same record)' if same_record else ''}", file=sys.stderr)
        for pr in data.get("pairs", []):
            g, a = _as_idx(pr.get("gt")), _as_idx(pr.get("actual"))
            if g is not None and a is not None and g < len(gt_left) and a < len(act_left):
                print(f"   {gt_left[g][0]} = {_short(gt_left[g][1], 30)!r}  <->  "
                      f"{act_left[a][0]} = {_short(act_left[a][1], 30)!r}  "
                      f"same_value={pr.get('same_value')}", file=sys.stderr)
    out, used_act, used_gt = [], set(), set()
    for p in data.get("pairs", []):
        g, a = _as_idx(p.get("gt")), _as_idx(p.get("actual"))
        if (isinstance(g, int) and isinstance(a, int)
                and 0 <= g < len(gt_left) and 0 <= a < len(act_left)
                and g not in used_gt and a not in used_act):
            used_gt.add(g)
            used_act.add(a)
            out.append((g, a, bool(p.get("same_value")), ""))
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

_NUM_UNIT = re.compile(r'^\s*(-?\d[\d,]*(?:\.\d+)?|-?\.\d+)\s*(%|[A-Za-z]{1,5})\s*$')


def _num_unit(v):
    """'19.800 KG' -> (19.8, 'kg'); '9.00%' -> (9.0, '%'); else None."""
    if isinstance(v, str):
        m = _NUM_UNIT.match(v)
        if m:
            return float(m.group(1).replace(",", "")), m.group(2).lower()
    return None


def _phone_digits(v):
    """Digits of a phone-like string ('(65) 6298 0888', '+65-62980888'), else None."""
    if isinstance(v, str) and re.fullmatch(r'\+?[\d\s().-]{7,}', v.strip()):
        d = re.sub(r'\D', '', v)
        return d if len(d) >= 7 else None
    return None


def _contained(gv, av) -> bool:
    """GT string appears as whole word(s) inside the actual string."""
    if not (isinstance(gv, str) and isinstance(av, str)):
        return False
    g, a = _norm_str(gv), _norm_str(av)
    return len(g) >= 2 and re.search(r'(?<!\w)' + re.escape(g) + r'(?!\w)', a) is not None


def _canon(v):
    if isinstance(v, bool):
        return ("b", v)
    n = _to_number(v)
    if n is not None:
        return ("n", round(n, 2))
    d = _to_date(v)
    if d:
        return ("d", d)
    nu = _num_unit(v)
    if nu:
        return ("nu", round(nu[0], 2), nu[1])
    ph = _phone_digits(v)
    if ph:
        return ("p", ph)
    return ("s", _norm_str(v))


def _distinctive(v) -> bool:
    nu = _num_unit(v)
    n = nu[0] if nu else _to_number(v)
    if n is not None:
        return abs(n) >= 100 or n != int(n)
    return isinstance(v, str) and len(_norm_str(v)) >= 5


def _record_of(path: str):
    """'$.line_items[2].qty' -> '$.line_items[2]'; None for non-record leaves."""
    m = re.match(r'^(.*?\[\d+\])', path)
    return m.group(1) if m else None


def compare(actual, ground_truth, tol: float = 0.01, use_llm: bool = True) -> dict:
    gt_leaves = list(_flatten(ground_truth))
    act_leaves = list(_flatten(actual))
    act_used = [False] * len(act_leaves)
    gt_done = [False] * len(gt_leaves)
    gt_rec = [_record_of(p) for p, _ in gt_leaves]
    act_rec = [_record_of(p) for p, _ in act_leaves]

    matches, mismatches = [], []
    pair_of = {}  # gt leaf index -> actual leaf index

    def _record(gi, ai, same_llm, reason, method):
        """Pair a GT leaf with an actual leaf; CODE decides if values agree."""
        gp, gv = gt_leaves[gi]
        ap, av = act_leaves[ai]
        gt_done[gi] = act_used[ai] = True
        pair_of[gi] = ai
        if _equivalent(gv, av, tol):
            matches.append({"ground_truth_path": gp, "actual_path": ap,
                            "value": gv, "method": method})
        elif (same_llm and not re.search(r'\d', str(gv)) and not re.search(r'\d', str(av))
              and not isinstance(gv, bool)):
            # Semantic equivalence (e.g. "Singapore Dollar" ~ "SGD") -- only
            # trusted for free text. Anything containing a digit (numbers,
            # dates, phone/id numbers) is decided by deterministic code,
            # never by the model.
            matches.append({"ground_truth_path": gp, "actual_path": ap,
                            "value": gv, "method": "llm"})
        elif _contained(gv, av):
            # e.g. GT currency "SGD" inside actual "SGD 21.00"
            matches.append({"ground_truth_path": gp, "actual_path": ap,
                            "value": gv, "method": "contained"})
        else:
            mismatches.append({"ground_truth_path": gp, "ground_truth_value": gv,
                               "actual_path": ap, "actual_value": av,
                               "reason": reason or "value differs"})

    # Step 1 -- anchors: a distinctive value present the same number of times
    # on both sides is the same information (barcodes, ids, names, totals).
    # Paired in code, with field-name similarity choosing between duplicates.
    by_val_gt, by_val_act = {}, {}
    for i, (p, v) in enumerate(gt_leaves):
        if _distinctive(v):
            by_val_gt.setdefault(_canon(v), []).append(i)
    for i, (p, v) in enumerate(act_leaves):
        if _distinctive(v):
            by_val_act.setdefault(_canon(v), []).append(i)
    votes = {}
    def _sim(g, a):
        return difflib.SequenceMatcher(None, _key_of(gt_leaves[g][0]),
                                       _key_of(act_leaves[a][0])).ratio()

    def _vote(g, a):
        if gt_rec[g] and act_rec[a]:
            votes[(gt_rec[g], act_rec[a])] = votes.get((gt_rec[g], act_rec[a]), 0) + 1

    for c, gis in by_val_gt.items():
        ais = by_val_act.get(c, [])
        if not ais:
            continue
        for _, g, a in sorted(((_sim(g, a), g, a) for g in gis for a in ais), reverse=True):
            if gt_done[g] or act_used[a]:
                continue
            _record(g, a, False, "", "value-anchor")
            _vote(g, a)
        # GT prints the same value more times than the actual has copies
        # (e.g. gst_amount + gst_amount_2): same information, reuse the
        # best-named actual field for the extra copies.
        for g in gis:
            if not gt_done[g]:
                a = max(ais, key=lambda a: _sim(g, a))
                _record(g, a, False, "", "value-anchor")
                _vote(g, a)

    # Step 2 -- pair array records (line items) by anchor votes.
    rec_pair, used_act_rec = {}, set()
    for (gr, ar), _n in sorted(votes.items(), key=lambda kv: -kv[1]):
        if gr not in rec_pair and ar not in used_act_rec:
            rec_pair[gr] = ar
            used_act_rec.add(ar)

    # Step 2b -- inside a record pair already known to be the same item, a
    # value that occurs exactly once on each side is the same field
    # (e.g. unit_price 21.0 <-> price_per_unit "SGD 21.00").
    for gr, ar in rec_pair.items():
        g_by, a_by = {}, {}
        for i in range(len(gt_leaves)):
            if gt_rec[i] == gr and not gt_done[i] and not isinstance(gt_leaves[i][1], bool):
                g_by.setdefault(_canon(gt_leaves[i][1]), []).append(i)
        for i in range(len(act_leaves)):
            if act_rec[i] == ar and not act_used[i] and not isinstance(act_leaves[i][1], bool):
                a_by.setdefault(_canon(act_leaves[i][1]), []).append(i)
        for c, gis in g_by.items():
            if len(gis) == 1 and len(a_by.get(c, [])) == 1:
                _record(gis[0], a_by[c][0], False, "", "record-anchor")

    # Step 3 -- LLM aligns what's left, in SMALL local jobs run in parallel:
    # one per paired record, one for unpaired records, one for top-level fields.
    jobs = []  # (gt_indices, act_indices)
    def _left(side_done, recs, pred):
        return [i for i in range(len(recs)) if not side_done[i] and pred(recs[i])]
    for gr, ar in rec_pair.items():
        jobs.append((_left(gt_done, gt_rec, lambda r, gr=gr: r == gr),
                     _left(act_used, act_rec, lambda r, ar=ar: r == ar), True))
    unp_gt = _left(gt_done, gt_rec, lambda r: r is not None and r not in rec_pair)
    unp_act = _left(act_used, act_rec, lambda r: r is not None and r not in used_act_rec)
    jobs.append((unp_gt, unp_act, False))
    jobs.append((_left(gt_done, gt_rec, lambda r: r is None),
                 _left(act_used, act_rec, lambda r: r is None), False))
    # chunk the GT side so no single prompt/response gets large
    sized = []
    for g_idx, a_idx, same_rec in jobs:
        if not g_idx or not a_idx:
            continue
        for k in range(0, len(g_idx), LLM_CHUNK):
            sized.append((g_idx[k:k + LLM_CHUNK], a_idx[:MAX_LLM_LEAVES], same_rec))

    llm_used, errors = False, []
    if use_llm and sized:
        def _run(job):
            g_idx, a_idx, same_rec = job
            return g_idx, a_idx, _llm_pair([gt_leaves[i] for i in g_idx],
                                           [act_leaves[i] for i in a_idx], same_rec)
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futs = [pool.submit(_run, j) for j in sized]
            results = []
            for f in futs:
                try:
                    results.append(f.result())
                    llm_used = True
                except Exception as exc:
                    errors.append(str(exc))
        for g_idx, a_idx, pairs in results:
            for g, a, same, reason in pairs:
                gi, ai = g_idx[g], a_idx[a]
                if not gt_done[gi] and not act_used[ai]:
                    _record(gi, ai, same, reason, "llm-aligned")
    llm_error = "; ".join(sorted(set(errors)))

    # Step 4 (fallback) -- deterministic value match for anything still
    # unpaired: everything if the LLM failed, else only distinctive values so
    # trivial ones (0, 1, "USD") can't pair with unrelated fields by accident.
    for gi, (gp, gv) in enumerate(gt_leaves):
        if gt_done[gi] or (llm_used and not errors and not _distinctive(gv)):
            continue
        best, best_sim = None, -1.0
        gk = _key_of(gp)
        for ai, (ap, av) in enumerate(act_leaves):
            if act_used[ai] or not _equivalent(gv, av, tol):
                continue
            sim = difflib.SequenceMatcher(None, gk, _key_of(ap)).ratio()
            if sim > best_sim:
                best, best_sim = ai, sim
        if best is not None:
            act_used[best] = gt_done[gi] = True
            matches.append({"ground_truth_path": gp, "actual_path": act_leaves[best][0],
                            "value": gv, "method": "value-fallback"})

    # Step 5 -- last-chance, still deterministic:
    #  a) a duplicate-suffixed GT field (freight_charge_2) whose twin
    #     (freight_charge) holds the same value inherits the twin's pairing;
    #  b) a GT string (currency "SGD") that appears as whole word(s) inside a
    #     field of the paired actual record ("SGD 21.00") counts as present.
    for gi, (gp, gv) in enumerate(gt_leaves):
        if gt_done[gi]:
            continue
        base = re.sub(r'\d+$', '', _key_of(gp))
        src = next((j for j in list(pair_of) if j != gi and gt_rec[j] == gt_rec[gi]
                    and re.sub(r'\d+$', '', _key_of(gt_leaves[j][0])) == base
                    and _canon(gt_leaves[j][1]) == _canon(gv)), None)
        if src is not None:
            _record(gi, pair_of[src], False, "", "duplicate-field")
            continue
        ar = rec_pair.get(gt_rec[gi]) if gt_rec[gi] else None
        if ar and isinstance(gv, str):
            for ai, (ap, av) in enumerate(act_leaves):
                if act_rec[ai] == ar and _contained(gv, av):
                    _record(gi, ai, False, "", "contained")
                    break

    missing = [{"ground_truth_path": gp, "ground_truth_value": gv}
               for (gp, gv), d in zip(gt_leaves, gt_done) if not d]
    extra = [ap for (ap, _), u in zip(act_leaves, act_used) if not u]

    total = len(gt_leaves)
    score = round(len(matches) / total, 4) if total else 1.0
    return {
        "match": not mismatches and not missing,
        "score": score,
        "total_ground_truth_fields": total,
        "matched_count": len(matches),
        "matches": matches,
        "mismatches": mismatches,
        "missing": missing,
        "extra_fields": extra,
        "llm_used": llm_used,
        "llm_error": llm_error or "",
    }


# --------------------------------------------------------------------------
# Short final answer
# --------------------------------------------------------------------------
# The controller prints an agent's `summary` string verbatim as the goal's
# final answer when an output carries one (and no `structured`). So the full
# comparison detail stays in the other output keys, and `summary` holds only
# the short verdict a person wants to read.

# Ground-truth fields that decide whether the extraction is trustworthy:
# every line item, subtotals/totals (incl. tax amount), and names of people/businesses.
_LINE_ITEM_RE = re.compile(r"line_?items?|\bitems?\[|\blines?\[|\bproducts?\[", re.I)
_TOTAL_KEY_RE = re.compile(r"(sub_?total|grand_?total|total|amount_?due|amount_?payable|"
                           r"net_?amount|gst_?amount|tax_?amount|vat_?amount)", re.I)
_NAME_KEY_RE = re.compile(r"(name|^company$|^supplier$|^customer$|^vendor$|^buyer$|^seller$|"
                          r"^bill_?to$|^ship_?to$|^sold_?to$|^deliver(y)?_?to$)", re.I)
_NOT_NAME_RE = re.compile(r"(file_?name|user_?name|host_?name|label|heading)", re.I)


def _is_important(gt_path: str) -> bool:
    key = re.sub(r"\[\d+\]", "", gt_path).rsplit(".", 1)[-1]
    if _LINE_ITEM_RE.search(gt_path):
        return True
    if _TOTAL_KEY_RE.search(key) and not re.search(r"label|heading", key, re.I):
        return True
    return bool(_NAME_KEY_RE.search(key)) and not _NOT_NAME_RE.search(key)


def final_summary(result: dict) -> dict:
    """The short verdict: match, score, missing, mismatches, all_important_fields_match."""
    paths = [m["ground_truth_path"] for m in result["matches"]] \
        + [m["ground_truth_path"] for m in result["mismatches"]] \
        + [m["ground_truth_path"] for m in result["missing"]]
    important = {p for p in paths if _is_important(p)}
    failed = {m["ground_truth_path"] for m in result["mismatches"]} \
        | {m["ground_truth_path"] for m in result["missing"]}
    return {
        "match": result["match"],
        "score": result["score"],
        # True only if every line item, subtotal/total and name/business-name field matched.
        # None when the ground truth has none of those fields (nothing to judge).
        "all_important_fields_match": (not (important & failed)) if important else None,
        "missing": [{"field": m["ground_truth_path"], "expected": m["ground_truth_value"]}
                    for m in result["missing"]],
        "mismatches": [{"field": m["ground_truth_path"], "expected": m["ground_truth_value"],
                        "actual": m["actual_value"]} for m in result["mismatches"]],
    }


class ValJsonAgent(Agent):
    """
    Validates an actual JSON (e.g. byoa-summarize output) against a ground-truth
    JSON, comparing values only -- field names and structure may differ.
    """
    name         = "byoa-val-json"
    version      = "1.0.0"
    description  = ("Compares an actual JSON against a ground-truth JSON by value, "
                    "tolerating different field names, nesting and ordering.")
    capabilities = ["data-matching"]

    input_schema = {
        "actual":            {"type": "object", "required": True},
        "ground_truth":      {"type": "object", "required": True},
        "numeric_tolerance": {"type": "number", "required": False},
    }
    output_schema = {
        "type": "object",
        "properties": {
            "match":                     {"type": "boolean"},
            "score":                     {"type": "number"},
            "total_ground_truth_fields": {"type": "integer"},
            "matched_count":             {"type": "integer"},
            "matches":                   {"type": "array"},
            "mismatches":                {"type": "array"},
            "missing":                   {"type": "array"},
            "extra_fields":              {"type": "array"},
            "llm_used":                  {"type": "boolean"},
            "llm_error":                 {"type": "string"},
            "all_important_fields_match": {"type": "boolean", "nullable": True},
            "summary":                   {"type": "string"},
        },
    }

    llm_base_url = VLLM_BASE_URL

    def run(self, payload: TaskPayload) -> TaskResult:
        try:
            actual = _load(payload.inputs["actual"], "actual")
            truth = _load(payload.inputs["ground_truth"], "ground_truth")
            tol = float(payload.inputs.get("numeric_tolerance", 0.01))
            outputs = compare(actual, truth, tol)
            short = final_summary(outputs)
            outputs["all_important_fields_match"] = short["all_important_fields_match"]
            outputs["summary"] = json.dumps(short, indent=2, ensure_ascii=False)
        except Exception as exc:
            return TaskResult(task_id=payload.task_id, status=TaskStatus.FAILURE,
                              error_msg=f"Validation failed: {exc}")
        return TaskResult(task_id=payload.task_id, status=TaskStatus.SUCCESS, outputs=outputs)


agent = ValJsonAgent()
agent.generate_manifest(endpoint=AGENT_ENDPOINT)
# /run_async returns a job_id immediately and runs the comparison on a
# background thread; callers poll GET /status/<job_id>. /run_async must be
# listed as protected so the HMAC middleware covers it. Sync /run still works.
app = build_flask_app(agent, BYOA_SECRET, extra_protected_paths={"/run_async"})
attach_async_routes(app, agent)


def _selftest():
    """Run with: python val_json_agent.py --selftest"""
    truth = {"invoice_no": "INV-001", "total": 1234.5, "currency": "SGD",
             "date": "2025-03-04",
             "items": [{"desc": "Widget", "qty": 2}, {"desc": "Gadget", "qty": 5}]}
    actual = {"invoiceNumber": "inv-001", "amount": "S$1,234.50",
              "issued": "04 Mar 2025", "currency_name": "Singapore Dollar",
              "lines": [{"qty": 5, "name": "gadget"}, {"qty": 2, "name": "WIDGET"}],
              "notes": "extra field"}

    r = compare(actual, truth, use_llm=False)
    assert r["matched_count"] == 7 and r["missing"][0]["ground_truth_value"] == "SGD", r
    assert r["extra_fields"] == ["$.currency_name", "$.notes"], r
    print(f"PASS deterministic: score={r['score']} missing={len(r['missing'])}")

    wrong = dict(actual, amount="999")
    r = compare(wrong, truth, use_llm=False)
    assert not r["match"] and r["matched_count"] == 6, r
    print("PASS wrong value detected")

    r = compare(actual, truth)  # needs vLLM; SGD ~ Singapore Dollar
    print(f"LLM pass: llm_used={r['llm_used']} match={r['match']} score={r['score']}"
          + (f" error={r['llm_error']}" if r["llm_error"] else ""))
    print("\nSelf-test done.")


def _compare_files(argv):
    """python val_json_agent.py --compare actual.json ground_truth.json [--no-llm]"""
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 2:
        sys.exit("usage: val_json_agent.py --compare ACTUAL.json GROUND_TRUTH.json [--no-llm]")
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            docs.append(_load(f.read(), p))
    result = compare(docs[0], docs[1], use_llm="--no-llm" not in argv)
    if result["llm_error"]:
        print(f"WARNING: LLM alignment failed ({result['llm_error']}); result below is "
              "the value-only fallback and will over-report missing fields.",
              file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if "--compare" in sys.argv:
        _compare_files([a for a in sys.argv[1:] if a != "--compare"])
        sys.exit(0)
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)

    print("=" * 60)
    print(f"Val-JSON Agent -- {agent.name} v{agent.version}")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT}")
    print(f"  GET  {AGENT_ENDPOINT}/health")
    print(f"  GET  {AGENT_ENDPOINT}/agent.md")
    print(f"  POST {AGENT_ENDPOINT}/run")
    print(f"  POST {AGENT_ENDPOINT}/run_async   (then poll GET /status/<job_id>)")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)
