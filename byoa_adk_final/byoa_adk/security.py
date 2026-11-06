"""
byoa_adk/security.py
~~~~~~~~~~~~~~~~~~~~
HMAC-SHA256 request signing/verification and JWT identity tokens.
This module is the security boundary of the ADK.
"""
from __future__ import annotations
import hashlib
import hmac
import time
import logging
from typing import Optional

import jwt                      # PyJWT
from pydantic import BaseModel

logger = logging.getLogger("byoa_adk.security")

# How many seconds either side of now we accept a signed request.
HMAC_WINDOW_SECS = 300  # 5 min window for dev/Postman testing

# JWT lifetime for agent->platform identity tokens.
JWT_TTL_SECS = 300   # 5 minutes


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------

def _canonical_message(task_id: str, goal_id: str,
                        capability: str, timestamp: int) -> bytes:
    """
    Build the byte string that is signed/verified.
    Changing this function is a breaking change — bump adk_version.
    """
    return f"{task_id}:{goal_id}:{capability}:{timestamp}".encode()


def sign_request(
    secret: str,
    task_id: str,
    goal_id: str,
    capability: str,
    timestamp: Optional[int] = None,
) -> tuple[str, int]:
    """
    Called by the **platform** before dispatching a task call.
    Returns (hex_digest, timestamp_used).
    """
    ts  = timestamp or int(time.time())
    msg = _canonical_message(task_id, goal_id, capability, ts)
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return sig, ts


def verify_request(
    secret: str,
    task_id: str,
    goal_id: str,
    capability: str,
    timestamp: int,
    provided_sig: str,
) -> None:
    """
    Called by the **ADK middleware** on every inbound POST /run.
    Raises ValueError with a safe (non-leaking) message on failure.

    Checks
    ------
    1. Timestamp is within HMAC_WINDOW_SECS of now (replay protection).
    2. HMAC digest matches (constant-time comparison — no timing oracle).
    """
    now = int(time.time())
    age = abs(now - timestamp)
    if age > HMAC_WINDOW_SECS:
        logger.warning("HMAC timestamp out of window: age=%ds", age)
        raise ValueError("Request timestamp outside acceptable window.")

    expected_sig, _ = sign_request(secret, task_id, goal_id, capability, timestamp)
    if not hmac.compare_digest(expected_sig, provided_sig):
        logger.warning("HMAC verification failed for task_id=%s", task_id)
        raise ValueError("HMAC signature mismatch.")

    logger.debug("HMAC OK for task_id=%s", task_id)


# ---------------------------------------------------------------------------
# JWT helpers (agent -> platform identity)
# ---------------------------------------------------------------------------

def issue_agent_token(secret: str, agent_name: str) -> str:
    """
    Issue a short-lived JWT the agent presents when calling the platform.
    Signed with the same shared secret (HS256).
    """
    now = int(time.time())
    payload = {
        "iss": "byoa-adk",
        "sub": agent_name,
        "iat": now,
        "exp": now + JWT_TTL_SECS,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_agent_token(secret: str, token: str) -> dict:
    """
    Called by the **platform** to verify an inbound agent JWT.
    Returns the decoded payload dict, or raises jwt.InvalidTokenError.
    """
    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        options={"require": ["iss", "sub", "exp"]},
    )


# ---------------------------------------------------------------------------
# Flask middleware (applied in server.py)
# ---------------------------------------------------------------------------

class HMACMiddleware:
    """
    WSGI middleware that verifies HMAC on every POST request to a
    protected path before the Flask view function is called.

    Attach via:  app.wsgi_app = HMACMiddleware(app.wsgi_app, secret, agent)

    protected_paths defaults to {"/run"} for backward compatibility, but
    any route that triggers agent work -- e.g. /run_async -- MUST be
    included here. A route added to the Flask app after build_flask_app()
    without also adding it to protected_paths runs with NO HMAC check at
    all, silently. That gap is exactly how /run_async shipped unauthenticated
    when it was bolted onto individual agents' apps by hand instead of
    through this middleware.
    """
    def __init__(self, wsgi_app, secret: str, agent_ref, protected_paths=None):
        self._app    = wsgi_app
        self._secret = secret
        self._agent  = agent_ref
        self._protected_paths = frozenset(protected_paths) if protected_paths else frozenset({"/run"})

    def __call__(self, environ, start_response):
        if environ.get("PATH_INFO") in self._protected_paths and environ.get("REQUEST_METHOD") == "POST":
            import json as _json
            from io import BytesIO
            from .audit import record as _audit_record, EVENT_HMAC_REJECTED

            content_length = int(environ.get("CONTENT_LENGTH") or 0)
            body_bytes = environ["wsgi.input"].read(content_length) if content_length > 0 else b""
            environ["wsgi.input"] = BytesIO(body_bytes)  # rewind for Flask

            body: dict = {}
            try:
                body = _json.loads(body_bytes) if body_bytes else {}
                verify_request(
                    self._secret,
                    task_id     = body["task_id"],
                    goal_id     = body["goal_id"],
                    capability  = body["capability"],
                    timestamp   = body.get("timestamp", 0),
                    provided_sig = body["hmac_sig"],
                )
            except (ValueError, KeyError, _json.JSONDecodeError) as exc:
                _audit_record(
                    EVENT_HMAC_REJECTED,
                    agent_name  = self._agent.name if self._agent is not None else "unknown",
                    outcome     = "rejected",
                    task_id     = body.get("task_id"),
                    goal_id     = body.get("goal_id"),
                    capability  = body.get("capability"),
                    detail      = str(exc),
                    remote_addr = environ.get("REMOTE_ADDR"),
                )
                if self._agent is not None:
                    from .audit import check_and_flag_suspicious
                    check_and_flag_suspicious(self._agent.name, remote_addr=environ.get("REMOTE_ADDR"))
                status  = "401 Unauthorized"
                headers = [("Content-Type", "application/json")]
                start_response(status, headers)
                return [_json.dumps({"error": str(exc)}).encode()]
        return self._app(environ, start_response)
