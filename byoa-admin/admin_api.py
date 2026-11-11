"""
BYOA Admin API -- backs the Flutter admin app.

  POST /auth/login                 -> short-lived session token
  GET  /credentials                -> onboarded agents (never secrets)
  POST /onboard                    -> create an agent secret, shown ONCE
  POST /agents/<agent_id>/revoke   -> invalidate an agent's secret
  POST /agents/<agent_id>/unregister -> take it out of service (reversible)
  DELETE /agents/<agent_id>        -> remove registry rows + credential (irreversible)
  GET  /agents                     -> registered agents (from the registration server)
  GET  /code/reference             -> ADK facts for the editor (capabilities, version)
  POST /code/check                 -> static check of agent source (parse only, never run)
  POST /code/generate              -> start a job: the model fills the TODO sections of
                                      agent.py from a text prompt -> {job_id} (202)
  GET  /code/generate/<job_id>     -> poll that job: queued | running | done | error
  POST /verify-secret              -> check the RUNNING agent accepts its stored secret
  POST /register                   -> register an agent; this API signs the
                                      identity token with the agent's stored
                                      secret, so no secret is ever pasted into
                                      a browser

Binds to 127.0.0.1 by default. Put it behind HTTPS (e.g. a Cloudflare tunnel)
before exposing it; do not bind it to a public interface directly.

Env: PLATFORM_DB_URL, ADMIN_JWT_SECRET (required); PORT (4150),
     ADMIN_HOST (127.0.0.1), REGISTRATION_URL (http://localhost:4100),
     ADMIN_CORS_ORIGINS (comma list, no wildcard), ADMIN_AUDIT_LOG,
     VLLM_BASE_URL (http://localhost:11500), VLLM_MODEL (qwen3.6-35b-fp8),
     GEN_CONCURRENCY (2), GEN_MAX_ATTEMPTS (3).
"""
import json
import os
import re
import threading
import time
import uuid
from functools import wraps

import jwt
import requests
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from waitress import serve

load_dotenv()

from agent_credentials import (  # noqa: E402
    get_agent_secret, init_credentials_table, list_credentials,
    onboard_agent, revoke_agent,
)
from admin_users import init_admin_table, verify_login  # noqa: E402
from registry import delete_agent, unregister_agent  # noqa: E402
from agent_check import check_agent_code, reference  # noqa: E402
import code_generate  # noqa: E402
from byoa_adk.manifest import parse_agent_md  # noqa: E402
from byoa_adk.schema import ALLOWED_CAPABILITIES  # noqa: E402
from byoa_adk.security import issue_agent_token, sign_request  # noqa: E402

ADMIN_JWT_SECRET = os.environ["ADMIN_JWT_SECRET"]
if len(ADMIN_JWT_SECRET) < 32:
    raise SystemExit("ADMIN_JWT_SECRET must be at least 32 characters.")
PORT = int(os.environ.get("PORT", "4150"))
HOST = os.environ.get("ADMIN_HOST", "127.0.0.1")
REGISTRATION_URL = os.environ.get("REGISTRATION_URL", "http://localhost:4100").rstrip("/")
AUDIT_LOG = os.environ.get("ADMIN_AUDIT_LOG", "admin_audit.log")
CORS_ORIGINS = [o.strip() for o in os.environ.get("ADMIN_CORS_ORIGINS", "").split(",") if o.strip()]

SESSION_TTL_SECS = 3600
SESSION_ISS = "byoa-admin-api"
AGENT_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,62}")
MAX_FAILS, FAIL_WINDOW_SECS = 5, 900

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
if CORS_ORIGINS:
    CORS(app, origins=CORS_ORIGINS, allow_headers=["Authorization", "Content-Type"])

_fails: dict = {}
_fails_lock = threading.Lock()
_audit_lock = threading.Lock()


def audit(event: str, **fields) -> None:
    entry = {"ts": time.time(), "event": event, "remote_addr": request.remote_addr, **fields}
    with _audit_lock, open(AUDIT_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _locked_out(key) -> bool:
    now = time.time()
    with _fails_lock:
        attempts = [t for t in _fails.get(key, []) if now - t < FAIL_WINDOW_SECS]
        _fails[key] = attempts
        return len(attempts) >= MAX_FAILS


def _record_fail(key) -> None:
    with _fails_lock:
        _fails.setdefault(key, []).append(time.time())


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing session token."}), 401
        try:
            claims = jwt.decode(
                auth[7:], ADMIN_JWT_SECRET, algorithms=["HS256"], issuer=SESSION_ISS,
                options={"require": ["exp", "sub", "iss"]},
            )
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid or expired session. Please sign in again."}), 401
        g.admin = claims["sub"]
        return fn(*args, **kwargs)
    return wrapper


@app.after_request
def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/auth/login")
def login():
    body = request.get_json(silent=True) or {}
    username, password = str(body.get("username", "")), str(body.get("password", ""))
    key = (request.remote_addr, username.lower())
    if _locked_out(key):
        audit("login.locked", username=username)
        return jsonify({"error": "Too many failed attempts. Try again later."}), 429
    if not username or not password or not verify_login(username, password):
        _record_fail(key)
        audit("login.failed", username=username)
        return jsonify({"error": "Invalid username or password."}), 401
    now = int(time.time())
    token = jwt.encode(
        {"iss": SESSION_ISS, "sub": username, "iat": now, "exp": now + SESSION_TTL_SECS},
        ADMIN_JWT_SECRET, algorithm="HS256",
    )
    audit("login.success", username=username)
    return jsonify({"token": token, "expires_in": SESSION_TTL_SECS})


@app.get("/credentials")
@require_admin
def credentials():
    return jsonify(list_credentials())


@app.post("/onboard")
@require_admin
def onboard():
    body = request.get_json(silent=True) or {}
    agent_id = str(body.get("agent_id", "")).strip()
    owner = str(body.get("owner", "")).strip()[:100]
    if not AGENT_ID_RE.fullmatch(agent_id):
        return jsonify({"error": "agent_id must be kebab-case: lowercase letters, digits and dashes (2-63 chars)."}), 422
    try:
        secret = onboard_agent(agent_id, owner_label=owner)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    audit("agent.onboarded", admin=g.admin, agent_id=agent_id, owner=owner)
    return jsonify({
        "agent_id": agent_id,
        "secret": secret,
        "note": "Shown once. Set it as BYOA_SECRET in the agent's own .env. It cannot be retrieved again.",
    }), 201


@app.post("/agents/<agent_id>/revoke")
@require_admin
def revoke(agent_id):
    if not AGENT_ID_RE.fullmatch(agent_id):
        return jsonify({"error": "Invalid agent_id."}), 422
    if not revoke_agent(agent_id):
        return jsonify({"error": "No active credential for that agent."}), 404
    audit("agent.revoked", admin=g.admin, agent_id=agent_id)
    return jsonify({"agent_id": agent_id, "revoked": True})


@app.post("/agents/<agent_id>/unregister")
@require_admin
def unregister(agent_id):
    if not AGENT_ID_RE.fullmatch(agent_id):
        return jsonify({"error": "Invalid agent_id."}), 422
    changed = unregister_agent(agent_id)
    if changed == 0:
        return jsonify({"error": "No registered entry to unregister for that agent."}), 404
    audit("agent.unregistered", admin=g.admin, agent_id=agent_id, rows=changed)
    return jsonify({"agent_id": agent_id, "unregistered": True, "rows": changed})


@app.delete("/agents/<agent_id>")
@require_admin
def delete(agent_id):
    """Irreversible. The caller must echo the agent id in confirm_agent_id."""
    if not AGENT_ID_RE.fullmatch(agent_id):
        return jsonify({"error": "Invalid agent_id."}), 422
    body = request.get_json(silent=True) or {}
    if body.get("confirm_agent_id") != agent_id:
        return jsonify({"error": "Confirmation does not match the agent id."}), 400
    result = delete_agent(agent_id)
    if result["registry_rows"] == 0 and result["credential_rows"] == 0:
        return jsonify({"error": "No such agent."}), 404
    audit("agent.deleted", admin=g.admin, agent_id=agent_id, **result)
    return jsonify({"agent_id": agent_id, "deleted": True, **result})


@app.get("/agents")
@require_admin
def agents():
    try:
        resp = requests.get(f"{REGISTRATION_URL}/agents", timeout=10)
        return jsonify(resp.json()), resp.status_code
    except (requests.RequestException, ValueError) as exc:
        return jsonify({"error": f"Registration server unavailable: {exc}"}), 502


@app.get("/code/reference")
@require_admin
def code_reference():
    """ADK facts the editor needs (allowed capabilities, version), so the UI never drifts from the ADK."""
    return jsonify(reference())


@app.post("/code/check")
@require_admin
def code_check():
    """Static check of editor source. Parses only; the code is never executed."""
    body = request.get_json(silent=True) or {}
    code = body.get("code")
    if not isinstance(code, str):
        return jsonify({"error": "code (a string) is required."}), 400
    return jsonify(check_agent_code(code))


@app.post("/code/generate")
@require_admin
def code_generate_start():
    """Start filling agent.py's TODO sections from a prompt. Returns at once; poll the job."""
    body = request.get_json(silent=True) or {}
    prompt, code = body.get("prompt"), body.get("code")
    if not isinstance(prompt, str) or not isinstance(code, str):
        return jsonify({"error": "prompt and code (strings) are required."}), 400
    try:
        job_id = code_generate.start_job(g.admin, prompt, code)
    except code_generate.GenerateError as exc:
        return jsonify({"error": str(exc)}), 422
    audit("code_generate_start", admin=g.admin, job_id=job_id, prompt_chars=len(prompt))
    return jsonify({"job_id": job_id}), 202


@app.get("/code/generate/<job_id>")
@require_admin
def code_generate_status(job_id):
    job = code_generate.get_job(g.admin, job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired job."}), 404
    return jsonify(job)


@app.post("/verify-secret")
@require_admin
def verify_secret():
    """
    Checks that the running agent was given the secret we have on file, with
    no side effects: send a correctly signed /run call for a capability the
    agent does NOT declare. The agent verifies the signature first, so:
      401 -> signature rejected: the agent has a different secret
      403 / 422 -> signature accepted, then the dummy capability was refused
    No agent code runs either way.
    """
    body = request.get_json(silent=True) or {}
    agent_md = body.get("agent_md")
    if not isinstance(agent_md, str) or not agent_md.strip():
        return jsonify({"error": "agent_md (the text of Agent.md) is required."}), 400
    try:
        manifest = parse_agent_md(agent_md)
    except Exception as exc:
        return jsonify({"error": "Agent.md is not valid.", "details": str(exc)}), 422
    secret = get_agent_secret(manifest.agent_id)
    if secret is None:
        return jsonify({"error": f"{manifest.agent_id!r} is not onboarded (or was revoked). Onboard it first."}), 409
    spare = [c for c in ALLOWED_CAPABILITIES if c not in manifest.capabilities]
    if not spare:
        return jsonify({"status": "unexpected", "message": "This agent declares every known capability, so it cannot be probed safely."})

    task_id, goal_id, capability = f"verify-{uuid.uuid4().hex[:12]}", "secret-check", spare[0]
    sig, ts = sign_request(secret, task_id, goal_id, capability)
    url = manifest.endpoint_url.rstrip("/") + manifest.invoke_path
    try:
        resp = requests.post(url, json={
            "task_id": task_id, "goal_id": goal_id, "capability": capability,
            "inputs": {}, "timestamp": ts, "hmac_sig": sig,
        }, timeout=15, allow_redirects=False)
    except requests.RequestException as exc:
        audit("agent.verify_unreachable", admin=g.admin, agent_id=manifest.agent_id)
        return jsonify({"status": "unreachable", "message": f"Could not reach {url}. Is the agent running? ({exc.__class__.__name__})"})

    if resp.status_code == 401:
        status, msg = "mismatch", "The agent rejected the signature. It is not using the secret shown at onboarding: set BYOA_SECRET in its .env and restart it."
    elif resp.status_code in (403, 422):
        status, msg = "ok", "The agent accepted the signature, so it has the right secret."
    else:
        status, msg = "unexpected", f"Unexpected answer from the agent (HTTP {resp.status_code})."
    audit("agent.verify_secret", admin=g.admin, agent_id=manifest.agent_id, result=status)
    return jsonify({"status": status, "message": msg, "http_status": resp.status_code})


@app.post("/register")
@require_admin
def register():
    body = request.get_json(silent=True) or {}
    agent_md = body.get("agent_md")
    if not isinstance(agent_md, str) or not agent_md.strip():
        return jsonify({"error": "agent_md (the text of Agent.md) is required."}), 400
    try:
        manifest = parse_agent_md(agent_md)
    except Exception as exc:
        return jsonify({"error": "Agent.md is not valid.", "details": str(exc)}), 422

    secret = get_agent_secret(manifest.agent_id)
    if secret is None:
        return jsonify({"error": f"{manifest.agent_id!r} is not onboarded (or was revoked). Onboard it first."}), 409

    token = issue_agent_token(secret, manifest.agent_id)
    try:
        resp = requests.post(
            f"{REGISTRATION_URL}/register",
            files={"agent_md": ("Agent.md", agent_md, "text/markdown")},
            headers={"Authorization": f"Bearer {token}"},
            timeout=45,
        )
    except requests.RequestException as exc:
        audit("agent.register_failed", admin=g.admin, agent_id=manifest.agent_id, detail=str(exc))
        return jsonify({"error": f"Registration server unavailable: {exc}"}), 502
    try:
        payload = resp.json()
    except ValueError:
        payload = {"error": f"Unexpected response from registration server (HTTP {resp.status_code})."}
    audit("agent.registered" if resp.ok else "agent.register_failed",
          admin=g.admin, agent_id=manifest.agent_id, status=resp.status_code)
    return jsonify(payload), resp.status_code


if __name__ == "__main__":
    init_credentials_table()
    init_admin_table()
    print(f"BYOA admin API on http://{HOST}:{PORT} -> registration {REGISTRATION_URL}", flush=True)
    serve(app, host=HOST, port=PORT, threads=4)
