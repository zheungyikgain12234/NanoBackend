"""
Fill in the TODO sections of a generated agent.py from a text prompt, using the
local model (qwen via vLLM).

SCOPE -- "TODO only": the model never sees a way to change anything except
  * the body of the `try:` block in run()   (plain agents), id  "run_body"
  * the body of each `step_<name>` function (DAG agents),   id  "step_<name>"
  * extra top-level `import` lines (plain import / from-import only)
Everything else in the file (class attributes, schemas, server setup) is copied
through byte-for-byte: we ask the model for the region bodies only and splice
them into the original text ourselves.

SECURITY: nothing here executes generated code. It is parsed (ast) and checked
(agent_check.check_agent_code), then handed back for a person to review.
Region code that imports dangerous modules or calls eval/exec/os.system etc. is
rejected and the model is asked to try again.
"""
import ast
import os
import re
import textwrap
import threading
import time
import uuid

import requests

from agent_check import check_agent_code, reference

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:11500").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-fp8")
MAX_ATTEMPTS = int(os.environ.get("GEN_MAX_ATTEMPTS", "3"))
# How many generations may use the model at once (the extraction agents share it).
GEN_CONCURRENCY = int(os.environ.get("GEN_CONCURRENCY", "2"))
MAX_PROMPT_CHARS = 20_000
JOB_TTL_SECS = 3600

_FORBIDDEN_MODULES = {"subprocess", "ctypes", "pickle", "marshal", "shelve", "pty", "socketserver"}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "system", "popen", "Popen"}
_IMPORT_LINE = re.compile(r"^(import [A-Za-z_][\w.]*( as \w+)?|from [A-Za-z_][\w.]* import [\w, ]+( as \w+)?)$")


class GenerateError(ValueError):
    """Bad request (not the model's fault): e.g. no fill-in sections in the code."""


# --------------------------------------------------------------------------
# Finding the regions the model may write
# --------------------------------------------------------------------------

def find_regions(src: str) -> list:
    """[{id, start, end, indent}] with 1-based inclusive line numbers of each body."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise GenerateError(f"agent.py has a syntax error (line {e.lineno}): {e.msg}. Fix it first.")
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                and any((isinstance(b, ast.Name) and b.id == "Agent")
                        or (isinstance(b, ast.Attribute) and b.attr == "Agent") for b in n.bases)), None)
    run = next((n for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run is None:
        raise GenerateError("No Agent class with a run() method was found.")
    regions = []
    for node in ast.walk(run):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("step_") and node.body:
            regions.append({"id": node.name, "start": node.lineno + 1,
                            "end": node.body[-1].end_lineno, "indent": node.body[0].col_offset})
    if not regions:   # plain agent: the try-block is the TODO. (In a DAG agent it just runs the steps.)
        for node in run.body:
            if isinstance(node, ast.Try) and node.body:
                regions.append({"id": "run_body", "start": node.lineno + 1,
                                "end": node.body[-1].end_lineno, "indent": node.body[0].col_offset})
                break
    if not regions:
        raise GenerateError("No fill-in sections found. Expected the generated try-block in run() "
                            "or step_<name> functions. Generate the project from the wizard first.")
    return sorted(regions, key=lambda r: r["start"])


def annotate(src: str, regions: list) -> str:
    """The file as shown to the model, with each writable section fenced by marker lines."""
    lines = src.split("\n")
    for r in sorted(regions, key=lambda r: r["start"], reverse=True):
        pad = " " * r["indent"]
        lines.insert(r["end"], f"{pad}# <<< END REGION {r['id']} >>>")
        lines.insert(r["start"] - 1, f"{pad}# <<< REGION {r['id']} : you write this body >>>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Model I/O
# --------------------------------------------------------------------------

def _system_prompt() -> str:
    ref = reference()
    return f"""You complete the TODO sections of a BYOA agent written with the ADK (version {ref['adk_version']}).

You are given the agent's agent.py. Some parts are fenced by lines
`# <<< REGION <id> : you write this body >>>` ... `# <<< END REGION <id> >>>`.
You may write ONLY those bodies. Nothing else in the file can change.

Rules for every region body:
- Write plain Python statements (no `def`/class wrapper, no fences). Indent from column 0; it will be re-indented.
- Inputs are already read into local variables at the top of run() (for plain agents).
- In a step, `ctx` is a dict. The task's inputs are `ctx["payload"].inputs["name"]` (payload is an OBJECT: use `.inputs`, never `["inputs"]`).
  Values returned by earlier steps are plain keys in ctx (e.g. ctx["cleaned_text"]).
- region `run_body`: must assign a dict named `outputs` whose keys match output_schema exactly (right types).
  An exception raised here becomes a FAILURE result, so do not catch-and-hide errors.
- region `step_<name>`: must end with `return {{...}}` (a dict of new context values). Read earlier steps' results from ctx.
- Call a model only through `ask_llm(prompt)` and only if that function is defined in the file. If it is not defined, do not call any model.
- Use the standard library, or `requests` for HTTP. Never hardcode secrets; read config from os.environ.
- Never use eval, exec, subprocess, os.system, pickle or ctypes.
- Keep it simple and correct. Prefer clear code over clever code.

Reply with NOTHING except these sections (no explanations, no code fences):
### IMPORTS
<extra top-level import lines, one per line; leave empty if none>
### REGION <id>
<the body for that region>
(one ### REGION section per region, using the exact ids from the file)"""


def _user_prompt(prompt: str, annotated: str, regions: list) -> str:
    ids = ", ".join(r["id"] for r in regions)
    return (f"What the agent should do:\n{prompt.strip()}\n\n"
            f"Regions to write: {ids}\n\nagent.py:\n{annotated}\n")


def _call_llm(messages: list) -> str:
    """One chat call to the local vLLM. No client timeout: long generations are fine
    because the caller is a background job that the UI polls."""
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={"model": VLLM_MODEL, "messages": messages, "temperature": 0.2,
              "chat_template_kwargs": {"enable_thinking": False}},
        timeout=None,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_reply(text: str, region_ids: list):
    """-> (imports: list[str], bodies: dict[id, str]) from the `### ...` sectioned reply."""
    text = re.sub(r"^```[a-zA-Z]*\s*$", "", text, flags=re.MULTILINE)
    # the model sometimes echoes the fence markers it was shown: drop them
    text = re.sub(r"^\s*(###\s+END\b.*|#\s*<<<.*>>>\s*|#{3,})\s*$", "", text, flags=re.MULTILINE)
    sections, cur, buf = {}, None, []

    def flush():
        # a repeated header (the model closing a region with the header again) must not
        # wipe out the code it just wrote
        if cur is not None and ("\n".join(buf).strip() or cur not in sections):
            sections[cur] = "\n".join(buf)

    for line in text.split("\n"):
        m = re.match(r"^###\s+(IMPORTS|REGION\s+(\S+))\s*$", line.strip())
        if m:
            flush()
            cur, buf = ("IMPORTS" if m.group(1) == "IMPORTS" else m.group(2)), []
        elif cur is not None:
            buf.append(line)
    flush()
    imports = [l.strip() for l in sections.pop("IMPORTS", "").split("\n") if l.strip()]
    missing = [i for i in region_ids if i not in sections or not sections[i].strip()]
    if missing:
        raise ValueError(f"Your reply had no code for region(s): {', '.join(missing)}. "
                         f"Use the exact '### REGION <id>' headers.")
    # the model often copies the file's indentation: bodies are re-indented by splice()
    return imports, {i: textwrap.dedent(sections[i]).strip("\n") for i in region_ids}


# --------------------------------------------------------------------------
# Validation and splicing
# --------------------------------------------------------------------------

def _validate_imports(imports: list) -> list:
    out = []
    for line in imports:
        if not _IMPORT_LINE.match(line):
            raise ValueError(f"Import line not allowed: {line!r}. Use plain `import x` or `from x import y`.")
        root = re.split(r"[ .]", line.split()[1])[0]
        if root in _FORBIDDEN_MODULES:
            raise ValueError(f"Module not allowed: {root}.")
        out.append(line)
    return out


def _validate_body(rid: str, body: str):
    try:
        tree = ast.parse(body)
    except SyntaxError as e:
        raise ValueError(f"Region {rid} has a syntax error on its line {e.lineno}: {e.msg}.")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise ValueError(f"Region {rid}: put imports in the ### IMPORTS section, not in a region.")
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in _FORBIDDEN_CALLS:
                raise ValueError(f"Region {rid} calls {name}(), which is not allowed.")
    if rid == "run_body":
        if not any(isinstance(t, ast.Name) and t.id == "outputs"
                   for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets):
            raise ValueError("Region run_body must assign a dict named `outputs`.")
    else:
        if not any(isinstance(n, ast.Return) for n in ast.walk(tree)):
            raise ValueError(f"Region {rid} must end with `return {{...}}` (a dict).")


def splice(src: str, regions: list, bodies: dict, imports: list) -> str:
    """Put each body into its region; add new import lines. Nothing else is touched."""
    lines = src.split("\n")
    for r in sorted(regions, key=lambda r: r["start"], reverse=True):
        pad = " " * r["indent"]
        new = [(pad + l) if l.strip() else "" for l in bodies[r["id"]].split("\n")]
        lines[r["start"] - 1:r["end"]] = new
    wanted = [i for i in imports if i not in lines]
    if wanted:
        tree = ast.parse(src)
        last = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
        lines[last:last] = wanted
    return "\n".join(lines)


def generate(prompt: str, code: str, llm=None, on_attempt=None) -> dict:
    """Run the fill-in loop. Returns {code, check, attempts, added_imports, regions}.
    `llm(messages) -> str` can be swapped for tests."""
    llm = llm or _call_llm
    regions = find_regions(code)
    ids = [r["id"] for r in regions]
    messages = [{"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(prompt, annotate(code, regions), regions)}]
    best, last_error = None, ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if on_attempt:
            on_attempt(attempt)
        reply = llm(messages)
        messages.append({"role": "assistant", "content": reply})
        problems = []
        try:
            imports, bodies = parse_reply(reply, ids)
            imports = _validate_imports(imports)
            for rid, body in bodies.items():
                _validate_body(rid, body)
            new_code = splice(code, regions, bodies, imports)
            check = check_agent_code(new_code)
            errors = [i for i in check["issues"] if i["severity"] == "error"]
            best = {"code": new_code, "check": check, "attempts": attempt,
                    "added_imports": [i for i in imports if i not in code.split("\n")], "regions": ids}
            if not errors:
                return best
            problems = [f"line {i['line']}: {i['message']}" for i in errors]
        except ValueError as e:
            last_error = str(e)
            problems = [last_error]
        messages.append({"role": "user", "content":
                         "That attempt has problems:\n- " + "\n- ".join(problems) +
                         "\nFix them and reply again with the full set of sections."})
    if best is None:
        raise GenerateError(f"The model did not produce usable code after {MAX_ATTEMPTS} attempts. Last problem: {last_error}")
    return best  # has check errors; the UI shows them


# --------------------------------------------------------------------------
# Background jobs (the UI polls; no request ever waits on the model)
# --------------------------------------------------------------------------

_jobs: dict = {}
_jobs_lock = threading.Lock()
_slots = threading.BoundedSemaphore(GEN_CONCURRENCY)


def _prune():
    cutoff = time.time() - JOB_TTL_SECS
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if v["status"] in ("done", "error") and v["updated"] < cutoff]:
            del _jobs[jid]


def start_job(owner: str, prompt: str, code: str) -> str:
    """Validate quickly (raises GenerateError), then run in the background. Returns job_id."""
    if not prompt.strip():
        raise GenerateError("Describe what the agent should do.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise GenerateError(f"The description is too long (max {MAX_PROMPT_CHARS} characters).")
    find_regions(code)  # fail fast on code that has nothing to fill in
    _prune()
    jid = uuid.uuid4().hex
    job = {"owner": owner, "status": "queued", "attempt": 0, "created": time.time(), "updated": time.time()}
    with _jobs_lock:
        _jobs[jid] = job

    def work():
        with _slots:
            job.update(status="running", updated=time.time())
            try:
                result = generate(prompt, code, on_attempt=lambda n: job.update(attempt=n, updated=time.time()))
                job.update(status="done", result=result, updated=time.time())
            except Exception as e:  # report, never crash the thread silently
                job.update(status="error", error=str(e) or e.__class__.__name__, updated=time.time())

    threading.Thread(target=work, daemon=True).start()
    return jid


def get_job(owner: str, job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job["owner"] != owner:
        return None
    out = {k: v for k, v in job.items() if k in ("status", "attempt", "result", "error")}
    out["max_attempts"] = MAX_ATTEMPTS
    return out
