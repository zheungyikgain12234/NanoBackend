"""
Static check of agent source code written in the admin app's editor.

SECURITY: this module only PARSES the code (ast.parse) and inspects the tree.
It never imports, compiles to bytecode for execution, or runs what it is given.
"""
import ast
import re

from byoa_adk import __version__ as ADK_VERSION
from byoa_adk.schema import ALLOWED_CAPABILITIES

# Mirrors ASYNC_CAPABILITIES in byoa-controller/platform_controller.py: the
# controller only uses submit-then-poll (/run_async) for these capabilities.
ASYNC_CAPABILITIES = ("text-summarize", "image-generate", "data-matching")

MAX_SOURCE_BYTES = 200 * 1024
_NAME_RE = re.compile(r"^[a-z0-9\-]+$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def reference() -> dict:
    return {
        "adk_version": ADK_VERSION,
        "capabilities": list(ALLOWED_CAPABILITIES),
        "async_capabilities": list(ASYNC_CAPABILITIES),
    }


def _issue(sev, line, msg, code):
    return {"severity": sev, "line": line or 1, "message": msg, "code": code}


def _lit(node):
    """Literal value of a node, or a sentinel when it is not a plain literal."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return _NOT_LITERAL


_NOT_LITERAL = object()


def _call_name(node):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _class_attrs(cls: ast.ClassDef) -> dict:
    out = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = stmt
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
            out[stmt.target.id] = stmt
    return out


def _is_agent_subclass(cls: ast.ClassDef) -> bool:
    for b in cls.bases:
        if (isinstance(b, ast.Name) and b.id == "Agent") or (isinstance(b, ast.Attribute) and b.attr == "Agent"):
            return True
    return False


def _check_dag(call: ast.Call, issues):
    """Validate a literal DAG([Step(...), ...]) declaration."""
    if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
        return None
    steps, names = {}, []
    for el in call.args[0].elts:
        if not (isinstance(el, ast.Call) and _call_name(el) == "Step"):
            continue
        name = _lit(el.args[0]) if el.args else None
        for kw in el.keywords:
            if kw.arg == "name":
                name = _lit(kw.value)
        deps = []
        for kw in el.keywords:
            if kw.arg == "depends_on":
                v = _lit(kw.value)
                deps = v if isinstance(v, list) else []
        if len(el.args) > 1:
            v = _lit(el.args[1])
            deps = v if isinstance(v, list) else deps
        if not isinstance(name, str):
            issues.append(_issue("warning", el.lineno, "Step name is not a plain string, so the DAG cannot be checked.", "dag-dynamic"))
            return None
        if name in steps:
            issues.append(_issue("error", el.lineno, f"Duplicate DAG step name {name!r}.", "dag-duplicate"))
        steps[name] = (deps, el.lineno)
        names.append(name)
    for n, (deps, ln) in steps.items():
        for d in deps:
            if d not in steps:
                issues.append(_issue("error", ln, f"Step {n!r} depends on unknown step {d!r}.", "dag-unknown-dep"))
    # cycle check (Kahn)
    indeg = {n: sum(1 for d in deps if d in steps) for n, (deps, _) in steps.items()}
    queue, seen = [n for n, d in indeg.items() if d == 0], 0
    while queue:
        cur = queue.pop()
        seen += 1
        for n, (deps, _) in steps.items():
            if cur in deps:
                indeg[n] -= 1
                if indeg[n] == 0:
                    queue.append(n)
    if steps and seen != len(steps):
        issues.append(_issue("error", call.lineno, "The DAG contains a cycle. Check the depends_on lists.", "dag-cycle"))
    return names


def check_agent_code(src: str) -> dict:
    issues = []
    if len(src.encode("utf-8", "ignore")) > MAX_SOURCE_BYTES:
        return {"ok": False, "issues": [_issue("error", 1, "Source is larger than 200 KB.", "too-large")]}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"ok": False, "issues": [_issue("error", e.lineno, f"Syntax error: {e.msg}", "syntax")]}
    except (RecursionError, MemoryError, ValueError) as e:
        return {"ok": False, "issues": [_issue("error", 1, f"Could not parse the file ({e.__class__.__name__}).", "unparseable")]}

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and _is_agent_subclass(n)]
    if not classes:
        issues.append(_issue("error", 1, "No class that inherits from Agent was found. Write: class MyAgent(Agent):", "no-agent-class"))
    elif len(classes) > 1:
        issues.append(_issue("warning", classes[1].lineno, "More than one Agent class found; the platform serves one agent per process.", "multi-agent"))

    declared_caps, has_llm_url, dag_names, uses_dag_run = [], False, None, False
    for cls in classes[:1]:
        attrs = _class_attrs(cls)
        for req in ("name", "version", "description", "capabilities"):
            if req not in attrs:
                issues.append(_issue("error", cls.lineno, f"Agent class is missing required attribute {req!r}.", f"missing-{req}"))
        def val(k):
            return _lit(attrs[k].value) if k in attrs else _NOT_LITERAL
        n, v, d, c = val("name"), val("version"), val("description"), val("capabilities")
        if "name" in attrs and (not isinstance(n, str) or not _NAME_RE.match(n)):
            issues.append(_issue("error", attrs["name"].lineno, "name must be lowercase letters, digits and dashes only (e.g. 'my-agent').", "bad-name"))
        if "version" in attrs and (not isinstance(v, str) or not _VERSION_RE.match(v)):
            issues.append(_issue("error", attrs["version"].lineno, "version must look like 1.0.0.", "bad-version"))
        if "description" in attrs and (not isinstance(d, str) or len(d) < 10):
            issues.append(_issue("error", attrs["description"].lineno, "description must be at least 10 characters.", "bad-description"))
        if "capabilities" in attrs:
            if not isinstance(c, list) or not c or not all(isinstance(x, str) for x in c):
                issues.append(_issue("error", attrs["capabilities"].lineno, "capabilities must be a non-empty list of strings.", "bad-capabilities"))
            else:
                declared_caps = c
                for x in c:
                    if x not in ALLOWED_CAPABILITIES:
                        issues.append(_issue("error", attrs["capabilities"].lineno,
                                             f"Unknown capability {x!r}. Allowed: {', '.join(ALLOWED_CAPABILITIES)}", "unknown-capability"))
        manifest_kwargs = {kw.arg for x in ast.walk(tree) if isinstance(x, ast.Call)
                           and _call_name(x) == "generate_manifest" for kw in x.keywords}
        for sch in ("input_schema", "output_schema"):
            # The ADK also accepts schemas passed to generate_manifest(...).
            if sch not in attrs and sch not in manifest_kwargs:
                issues.append(_issue("warning", cls.lineno, f"No {sch} declared. Declaring it makes the ADK validate every request/response for you.", f"no-{sch}"))
        has_llm_url = "llm_base_url" in attrs
        run = next((s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "run"), None)
        if run is None:
            issues.append(_issue("error", cls.lineno, "The class has no run(self, payload) method.", "no-run"))
        else:
            if [a.arg for a in run.args.args][:2] != ["self", "payload"]:
                issues.append(_issue("warning", run.lineno, "run should be declared as run(self, payload).", "run-signature"))
            if not any(isinstance(x, ast.Call) and _call_name(x) == "TaskResult" for x in ast.walk(run)):
                issues.append(_issue("warning", run.lineno, "run() never builds a TaskResult. It must return one.", "no-taskresult"))
        if "dag" in attrs and isinstance(attrs["dag"].value, ast.Call) and _call_name(attrs["dag"].value) == "DAG":
            dag_names = _check_dag(attrs["dag"].value, issues)

    calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call)]
    call_names = {_call_name(x) for x in calls}
    uses_dag_run = any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "run"
                       and isinstance(x.func.value, ast.Attribute) and x.func.value.attr == "dag" for x in calls)
    if dag_names is not None and not uses_dag_run:
        issues.append(_issue("warning", 1, "A DAG is declared but self.dag.run(...) is never called, so it only documents the steps and does not execute them.", "dag-unused"))

    # --- serving / async -------------------------------------------------
    bfa = [x for x in calls if _call_name(x) == "build_flask_app"]
    if classes and not bfa and "serve" not in call_names:
        issues.append(_issue("warning", 1, "build_flask_app(agent, secret) is never called, so nothing serves /health and /run.", "no-server"))
    if "attach_async_routes" in call_names:
        protected = False
        for x in bfa:
            for kw in x.keywords:
                if kw.arg == "extra_protected_paths":
                    protected = "/run_async" in ast.dump(kw.value)
        if not protected:
            issues.append(_issue("error", (bfa[0].lineno if bfa else 1),
                                 'attach_async_routes needs build_flask_app(..., extra_protected_paths={"/run_async"}). '
                                 "Without it /run_async is not covered by the signature check.", "async-unprotected"))
        if declared_caps and not (set(declared_caps) & set(ASYNC_CAPABILITIES)):
            issues.append(_issue("warning", 1,
                                 "The controller only uses /run_async for: " + ", ".join(ASYNC_CAPABILITIES) +
                                 ". With your capabilities it will call /run (sync) unless the controller is updated.", "async-not-used"))

    # --- secrets / llm ---------------------------------------------------
    for x in calls:
        if isinstance(x.func, ast.Attribute) and x.func.attr == "get" and len(x.args) >= 2:
            k, dflt = _lit(x.args[0]), _lit(x.args[1])
            if k == "BYOA_SECRET" and isinstance(dflt, str) and dflt:
                issues.append(_issue("warning", x.lineno, "BYOA_SECRET has a hard-coded fallback. Remove it so a missing secret fails loudly instead of using a guessable one.", "default-secret"))
    if "call_llm" in call_names and not has_llm_url:
        issues.append(_issue("warning", 1, "call_llm is used but the class has no llm_base_url. Declare it so Agent.md tells the platform which model endpoint you use.", "llm-undeclared"))
    for x in calls:
        if isinstance(x.func, ast.Attribute) and x.func.attr == "post" and isinstance(x.func.value, ast.Name) and x.func.value.id in ("requests", "_req"):
            u = x.args[0] if x.args else None
            if u is not None and "chat/completions" in ast.dump(u):
                issues.append(_issue("warning", x.lineno, "Call your model through byoa_adk.llm_client.call_llm, not requests.post, so only approved endpoints are used.", "raw-llm-call"))

    ok = not any(i["severity"] == "error" for i in issues)
    issues.sort(key=lambda i: (i["line"], i["severity"] != "error"))
    return {"ok": ok, "issues": issues}
