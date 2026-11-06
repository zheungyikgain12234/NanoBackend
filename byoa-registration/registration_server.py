"""
BYOA Agent Registration Server -- standalone GX10 service.

Same logic as byoa-gx10-agent-registration-server.ipynb, unchanged:
Agent.md parsing/validation via byoa_adk's parse_agent_md/AgentManifest,
upsert into registered_agents, post-registration health check, /agents
and /agents/<id> listing endpoints.

What's different from the Kaggle version:
  - No pyngrok. Public exposure is handled by the systemd-managed
    `cloudflared` unit on this box, proxying
    https://register.zackloh.dev -> http://localhost:PORT.
  - No `while True: sleep(60)` keepalive cell -- systemd keeps this alive.
  - Config comes from a .env file instead of hardcoded notebook cell values.
  - Table init (registered_agents DDL) still runs at startup, same as the
    notebook -- idempotent, so safe on every restart.
"""
import json
import os

import jwt as _jwt
import psycopg2
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pydantic import ValidationError
from waitress import serve

from byoa_adk.manifest import parse_agent_md
from byoa_adk.schema import AgentManifest
from byoa_adk.security import verify_agent_token

from agent_credentials import init_credentials_table, get_agent_secret

load_dotenv()

PLATFORM_DB_URL = os.environ["PLATFORM_DB_URL"]
PORT = int(os.environ.get("PORT", "5100"))
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "https://register.zackloh.dev")

HEALTH_CHECK_TIMEOUT = 15


def get_conn():
    return psycopg2.connect(PLATFORM_DB_URL)


def init_registry_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS registered_agents (
            id                  SERIAL PRIMARY KEY,
            agent_id            TEXT        NOT NULL,
            version             TEXT        NOT NULL,
            display_name        TEXT        NOT NULL,
            endpoint_url        TEXT        NOT NULL,
            health_check_path   TEXT        NOT NULL DEFAULT '/health',
            invoke_path         TEXT        NOT NULL,
            invoke_method       TEXT        NOT NULL DEFAULT 'POST',
            capabilities        JSONB       NOT NULL DEFAULT '[]',
            input_schema        JSONB       NOT NULL DEFAULT '{}',
            output_schema       JSONB       NOT NULL DEFAULT '{}',
            runtime_meta        JSONB       NOT NULL DEFAULT '{}',
            manifest            JSONB       NOT NULL DEFAULT '{}',
            status              TEXT        NOT NULL DEFAULT 'PENDING',
            last_health_check   TIMESTAMPTZ,
            health_check_status TEXT,
            registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(agent_id, version)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_registered_agents_capabilities
        ON registered_agents USING gin(capabilities)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_registered_agents_status
        ON registered_agents(status)
    """)
    conn.commit()
    conn.close()
    print("OK Registry tables ready")


app = Flask(__name__)


def parse_and_validate_manifest(md_content: str):
    try:
        manifest = parse_agent_md(md_content)
        return manifest, []
    except ValueError as e:
        return None, [str(e)]
    except ValidationError as e:
        errors = []
        for err in e.errors():
            field = " -> ".join(str(x) for x in err["loc"])
            errors.append(f'{field}: {err["msg"]}')
        return None, errors


LIVE_MANIFEST_TIMEOUT = 10

# Fields worth cross-checking. Deliberately excludes endpoint_url itself
# (that's the address we're fetching FROM) and adk_version (informational,
# not a safety-relevant field to hard-reject on).
_COMPARE_FIELDS = ("version", "description", "capabilities", "input_schema", "output_schema")


def fetch_live_manifest(endpoint_url: str):
    """
    Fetch and parse the agent's OWN live /agent.md -- generated fresh by
    its actual running code every time it's asked, via the exact same
    generate_manifest() call every byoa_adk agent already exposes.

    This is the closest available proxy to "check the submission against
    the real implementation" that doesn't require ever receiving anyone's
    source code: the live endpoint IS the implementation, observed rather
    than read.

    Returns (manifest, error, error_kind). error is None on success.
    error_kind distinguishes two very different situations:
      "unreachable" -- couldn't even connect (agent may be mid-deploy;
                        network issue). Not the agent's fault, necessarily.
      "invalid"     -- DID connect and get a response, but what it says
                        about itself fails validation (e.g. an unapproved
                        llm_base_url, or a malformed manifest). This is a
                        real, actionable policy violation, not a transient
                        availability problem, and deserves a clearer error.
    """
    url = endpoint_url.rstrip("/") + "/agent.md"
    try:
        resp = requests.get(url, timeout=LIVE_MANIFEST_TIMEOUT)
    except Exception as e:
        return None, str(e), "unreachable"

    try:
        resp.raise_for_status()
        manifest = parse_agent_md(resp.text)
        return manifest, None, None
    except Exception as e:
        # We successfully talked to the endpoint -- it responded -- but
        # what it said about itself doesn't pass validation.
        return None, str(e), "invalid"


def compare_manifests(submitted: AgentManifest, live: AgentManifest) -> list:
    """
    Compare the submitted Agent.md against what the agent's own live
    endpoint currently reports about itself. Returns a list of
    human-readable mismatch descriptions (empty list = consistent).
    """
    mismatches = []
    for field_name in _COMPARE_FIELDS:
        submitted_val = getattr(submitted, field_name)
        live_val = getattr(live, field_name)
        if submitted_val != live_val:
            mismatches.append(
                f'{field_name}: submitted={submitted_val!r} but live endpoint reports {live_val!r}'
            )
    return mismatches


def run_health_check(manifest: dict) -> dict:
    url = manifest["endpoint_url"].rstrip("/") + manifest["health_check_path"]
    try:
        resp = requests.get(
            url,
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}
        success = resp.status_code == 200 and body.get("status") in ("ok", "healthy")
        return {"success": success, "status_code": resp.status_code,
                "response_body": body, "error": None, "url": url}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "status_code": None, "response_body": None,
                "error": f"ConnectionError: {e}", "url": url}
    except requests.exceptions.Timeout:
        return {"success": False, "status_code": None, "response_body": None,
                "error": f"Timeout after {HEALTH_CHECK_TIMEOUT}s", "url": url}
    except Exception as e:
        return {"success": False, "status_code": None, "response_body": None,
                "error": str(e), "url": url}


def upsert_registration(manifest) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO registered_agents (
            agent_id, version, display_name,
            endpoint_url, health_check_path, invoke_path, invoke_method,
            capabilities, input_schema, output_schema, runtime_meta, manifest,
            status, registered_at, updated_at
        ) VALUES (
            %(agent_id)s, %(version)s, %(display_name)s,
            %(endpoint_url)s, %(health_check_path)s, %(invoke_path)s, %(invoke_method)s,
            %(capabilities)s, %(input_schema)s, %(output_schema)s, %(runtime_meta)s, %(manifest)s,
            'PENDING', NOW(), NOW()
        )
        ON CONFLICT (agent_id, version) DO UPDATE SET
            endpoint_url        = EXCLUDED.endpoint_url,
            health_check_path   = EXCLUDED.health_check_path,
            capabilities        = EXCLUDED.capabilities,
            input_schema        = EXCLUDED.input_schema,
            output_schema       = EXCLUDED.output_schema,
            runtime_meta        = EXCLUDED.runtime_meta,
            manifest            = EXCLUDED.manifest,
            status              = 'PENDING',
            updated_at          = NOW()
        RETURNING id
    """, {
        "agent_id": manifest.agent_id,
        "version": manifest.version,
        "display_name": manifest.display_name,
        "endpoint_url": manifest.endpoint_url,
        "health_check_path": manifest.health_check_path,
        "invoke_path": manifest.invoke_path,
        "invoke_method": manifest.invoke_method,
        "capabilities": json.dumps(manifest.capabilities),
        "input_schema": json.dumps(manifest.input_schema),
        "output_schema": json.dumps(manifest.output_schema),
        "runtime_meta": json.dumps({"invoke_content_type": manifest.invoke_content_type, "adk_version": manifest.adk_version}),
        "manifest": json.dumps(manifest.model_dump()),
    })
    row_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return row_id


def update_status(agent_id: str, version: str, status: str, health_detail: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE registered_agents
        SET status              = %s,
            health_check_status = %s,
            last_health_check   = NOW(),
            updated_at          = NOW()
        WHERE agent_id = %s AND version = %s
    """, (status, health_detail, agent_id, version))
    conn.commit()
    conn.close()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/register", methods=["POST"])
def register():
    # --- Identity check, multi-tenant version. ---
    # We can't verify a token's signature without first knowing WHICH
    # agent's secret to check it against -- but the token itself is what
    # claims the identity. So: peek at the UNVERIFIED "sub" claim just to
    # know which secret to fetch, then verify for real with that secret.
    # A forged/tampered token will still fail the real verification below,
    # even if the peek succeeded -- the peek is only a lookup key, never
    # treated as proof of anything on its own.
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing identity token. Include 'Authorization: Bearer <token>'."}), 401
    token = auth_header[len("Bearer "):]

    try:
        unverified = _jwt.decode(token, options={"verify_signature": False})
        claimed_agent_id = unverified.get("sub")
    except Exception as e:
        return jsonify({"error": f"Malformed token: {e}"}), 401
    if not claimed_agent_id:
        return jsonify({"error": "Token has no 'sub' claim -- cannot determine claimed identity."}), 401

    agent_secret = get_agent_secret(claimed_agent_id)
    if agent_secret is None:
        return jsonify({"error": f"Unknown or revoked agent_id: {claimed_agent_id!r}. Contact the platform owner to be onboarded."}), 401

    try:
        decoded_token = verify_agent_token(agent_secret, token)
    except Exception as e:
        return jsonify({"error": f"Invalid or expired identity token: {e}"}), 401

    if "agent_md" not in request.files:
        return jsonify({"error": "Missing file field: agent_md"}), 400
    file = request.files["agent_md"]
    if file.filename == "":
        return jsonify({"error": "agent_md file is empty"}), 400
    try:
        md_content = file.read().decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Failed to read Agent.md: {str(e)}"}), 400

    # The submitted file is used for EXACTLY ONE thing: telling us where
    # to look. Its content is never trusted as the record of truth --
    # only endpoint_url is read from it at all. This is a deliberate
    # simplification over "submit + cross-check": rather than trusting
    # the submission and only catching a disagreement with the live
    # server, we simply never let the submission's claims about
    # capabilities/schemas/description reach the database in the first
    # place. Only what the agent's OWN live /agent.md reports, fetched
    # fresh right now, is ever written down.
    submitted_manifest, errors = parse_and_validate_manifest(md_content)
    if errors:
        return jsonify({"error": "Manifest validation failed", "details": errors}), 422

    live_manifest, fetch_error, error_kind = fetch_live_manifest(submitted_manifest.endpoint_url)
    if live_manifest is None:
        if error_kind == "invalid":
            # The endpoint is genuinely live and responded -- but what it
            # reports about itself violates platform policy (e.g. an
            # unapproved llm_base_url). This is not "try again later" --
            # it's a real, actionable rejection.
            return jsonify({
                "error": "The agent's live /agent.md was fetched successfully but failed validation.",
                "endpoint_url": submitted_manifest.endpoint_url,
                "details": fetch_error,
            }), 422
        # Genuinely unreachable -- no soft-tolerant path either way, since
        # there is nothing trustworthy to register from, but this IS a
        # different, more forgiving situation than an active policy
        # violation (e.g. the agent might simply be mid-deploy).
        return jsonify({
            "error": "Could not reach the agent's live /agent.md -- registration requires a reachable, live endpoint.",
            "endpoint_url": submitted_manifest.endpoint_url,
            "fetch_error": fetch_error,
        }), 503

    # Identity is checked against the LIVE manifest now -- the one
    # actually-trusted source -- not against whatever the submission claimed.
    if decoded_token["sub"] != live_manifest.agent_id:
        return jsonify({
            "error": "Identity mismatch: token is for a different agent than the one the live endpoint reports.",
            "token_identity": decoded_token["sub"],
            "live_agent_id": live_manifest.agent_id,
        }), 403

    manifest = live_manifest  # everything from here on uses ONLY the live, verified manifest

    try:
        row_id = upsert_registration(manifest)
    except Exception as e:
        return jsonify({"error": f"DB error: {str(e)}"}), 500
    health_result = run_health_check(manifest.model_dump())
    if health_result["success"]:
        status = "REGISTERED_ACTIVE"
        health_detail = f'OK {health_result["status_code"]} -- {json.dumps(health_result["response_body"])}'
    else:
        status = "HEALTH_CHECK_FAILED"
        err_detail = health_result["error"] or f'HTTP {health_result["status_code"]}'
        health_detail = f"FAILED: {err_detail}"
    try:
        update_status(manifest.agent_id, manifest.version, status, health_detail)
    except Exception as e:
        return jsonify({"error": f"DB status update error: {str(e)}"}), 500
    http_code = 200 if health_result["success"] else 502
    return jsonify({
        "status": status,
        "agent_id": manifest.agent_id,
        "version": manifest.version,
        "display_name": manifest.display_name,
        "endpoint_url": manifest.endpoint_url,
        "capabilities": manifest.capabilities,
        "row_id": row_id,
        "health_check": {
            "url": health_result["url"],
            "status_code": health_result["status_code"],
            "response": health_result["response_body"],
            "error": health_result["error"],
        },
        "source_of_truth": "live /agent.md (submitted file was used only to locate the endpoint)",
    }), http_code


@app.route("/agents", methods=["GET"])
def list_agents():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT agent_id, version, display_name, endpoint_url,
               capabilities, status, last_health_check, registered_at
        FROM registered_agents
        ORDER BY registered_at DESC
    """)
    cols = [d[0] for d in cur.description]
    rows = []
    for row in cur.fetchall():
        r = dict(zip(cols, row))
        r["last_health_check"] = r["last_health_check"].isoformat() if r["last_health_check"] else None
        r["registered_at"] = r["registered_at"].isoformat() if r["registered_at"] else None
        rows.append(r)
    conn.close()
    return jsonify(rows), 200


@app.route("/agents/<agent_id>", methods=["GET"])
def get_agent(agent_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT agent_id, version, display_name, endpoint_url,
               capabilities, input_schema, output_schema, runtime_meta,
               status, health_check_status, last_health_check, registered_at, updated_at
        FROM registered_agents
        WHERE agent_id = %s
        ORDER BY registered_at DESC
    """, (agent_id,))
    cols = [d[0] for d in cur.description]
    rows = []
    for row in cur.fetchall():
        r = dict(zip(cols, row))
        for ts in ("last_health_check", "registered_at", "updated_at"):
            r[ts] = r[ts].isoformat() if r[ts] else None
        rows.append(r)
    conn.close()
    if not rows:
        return jsonify({"error": f"Agent not found: {agent_id}"}), 404
    return jsonify(rows), 200


if __name__ == "__main__":
    init_registry_tables()
    init_credentials_table()
    print("=" * 60)
    print("BYOA Agent Registration Server is live")
    print("=" * 60)
    print(f"  Listening on : http://0.0.0.0:{PORT}")
    print(f"  Public via   : {AGENT_ENDPOINT} (cloudflared, managed by systemd)")
    print(f"  POST {AGENT_ENDPOINT}/register")
    print(f"  GET  {AGENT_ENDPOINT}/agents")
    print(f"  GET  {AGENT_ENDPOINT}/agents/<agent_id>")
    print("=" * 60)
    serve(app, host="0.0.0.0", port=PORT, threads=4)
