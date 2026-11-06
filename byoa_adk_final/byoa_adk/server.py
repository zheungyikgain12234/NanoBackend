"""
byoa_adk/server.py
~~~~~~~~~~~~~~~~~~
Flask scaffold — pre-wired routes + HMAC middleware + structured logging.
"""
from __future__ import annotations
import logging
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from . import __version__
from .schema               import AgentStatus, HealthResponse, TaskPayload, TaskStatus
from .security             import HMACMiddleware
from .request_validator    import validate_task_payload
from .response_validator   import validate_task_result, validate_task_input
from .audit                import (
    record as audit_record, include_payload as audit_include_payload,
    EVENT_INPUT_REJECTED, EVENT_CAPABILITY_REJECTED,
    EVENT_OUTPUT_REJECTED, EVENT_RUN_SUCCESS, EVENT_RUN_FAILURE,
)

if TYPE_CHECKING:
    from .base import Agent

logger = logging.getLogger("byoa_adk.server")


def build_flask_app(agent: "Agent", secret: str, extra_protected_paths=None) -> Flask:
    """
    Build and return a configured Flask application for the given agent.
    HMAC middleware is attached to the WSGI layer, not as a decorator,
    so it fires before ANY view function is called.

    extra_protected_paths: pass {"/run_async"} (or similar) here if you plan
    to call attach_async_routes() on this app afterwards, so HMAC covers it
    from the start rather than being bolted on unauthenticated later.
    """
    app = Flask(agent.name)
    _configure_logging()
    protected_paths = {"/run"} | set(extra_protected_paths or [])

    # -------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------

    @app.get("/health")
    def health():
        resp = HealthResponse(
            name         = agent.name,
            version      = agent.version,
            status       = AgentStatus.HEALTHY,
            capabilities = agent.capabilities,
            adk_version  = __version__,
        )
        return jsonify(resp.model_dump()), 200

    @app.get("/agent.md")
    def serve_manifest():
        """Serve Agent.md so the platform can fetch and re-validate it."""
        content = agent.generate_manifest()
        return content, 200, {"Content-Type": "text/markdown"}

    @app.post("/run")
    @validate_task_payload
    def run_task():
        """
        Main task execution endpoint.
        Request Contract Validator runs first (via decorator).
        HMAC is verified by HMACMiddleware at WSGI layer before this view.
        """
        from flask import g
        payload_dict = g.task

        try:
            payload = TaskPayload.model_validate(payload_dict)
        except Exception as exc:
            logger.warning("Invalid TaskPayload: %s", exc)
            audit_record(
                EVENT_INPUT_REJECTED, agent_name=agent.name, outcome="rejected",
                task_id=payload_dict.get('task_id'), goal_id=payload_dict.get('goal_id'),
                capability=payload_dict.get('capability'), detail=str(exc),
                remote_addr=request.remote_addr,
            )
            return jsonify({"error": f"Invalid payload: {exc}"}), 422

        try:
            agent._validate_capability(payload.capability)
        except ValueError as exc:
            audit_record(
                EVENT_CAPABILITY_REJECTED, agent_name=agent.name, outcome="rejected",
                task_id=payload.task_id, goal_id=payload.goal_id,
                capability=payload.capability, detail=str(exc),
                remote_addr=request.remote_addr,
            )
            return jsonify({"error": str(exc)}), 403

        # Request Contract Validator: check REAL inputs against the agent's
        # own declared input_schema, before run() ever executes.
        input_schema = agent.input_schema if hasattr(agent, 'input_schema') else {}
        input_rv = validate_task_input(payload.inputs, input_schema)
        if not input_rv.valid:
            audit_record(
                EVENT_INPUT_REJECTED, agent_name=agent.name, outcome="rejected",
                task_id=payload.task_id, goal_id=payload.goal_id,
                capability=payload.capability, detail="; ".join(input_rv.errors),
                remote_addr=request.remote_addr,
            )
            return jsonify({
                "error":   "Request contract violation",
                "details": input_rv.errors,
                "task_id": payload.task_id,
            }), 422

        result = agent._timed_run(payload)

        # Response Contract Validator
        output_schema = agent.output_schema if hasattr(agent, 'output_schema') else {}
        rv = validate_task_result(result, output_schema)
        if not rv.valid:
            logger.warning("Response validation failed: %s", rv.errors)
            audit_record(
                EVENT_OUTPUT_REJECTED, agent_name=agent.name, outcome="rejected",
                task_id=result.task_id, goal_id=payload.goal_id,
                capability=payload.capability, detail="; ".join(rv.errors),
                duration_ms=result.duration_ms, remote_addr=request.remote_addr,
            )
            return jsonify({
                "error":   "Response contract violation",
                "details": rv.errors,
                "task_id": result.task_id,
            }), 502

        logger.info(
            "task_id=%s capability=%s status=%s duration_ms=%s warnings=%s",
            result.task_id, payload.capability,
            result.status, result.duration_ms, rv.warnings,
        )
        audit_record(
            EVENT_RUN_SUCCESS if result.status == TaskStatus.SUCCESS else EVENT_RUN_FAILURE,
            agent_name  = agent.name,
            outcome     = "allowed" if result.status == TaskStatus.SUCCESS else "task_failed",
            task_id     = result.task_id,
            goal_id     = payload.goal_id,
            capability  = payload.capability,
            detail      = result.error_msg,
            duration_ms = result.duration_ms,
            remote_addr = request.remote_addr,
            extra       = (
                {"inputs": payload.inputs, "outputs": result.outputs}
                if audit_include_payload() else None
            ),
        )

        http_status = 200 if result.status == TaskStatus.SUCCESS else 207
        return jsonify(result.model_dump()), http_status

    # -------------------------------------------------------------------
    # HMAC middleware (WSGI layer)
    # -------------------------------------------------------------------
    app.wsgi_app = HMACMiddleware(app.wsgi_app, secret, agent, protected_paths=protected_paths)

    return app


def attach_async_routes(app: Flask, agent: "Agent", job_ttl_secs: int = 3600) -> None:
    """
    Adds POST /run_async and GET /status/<job_id> to an app built by
    build_flask_app(agent, secret, extra_protected_paths={"/run_async"}).

    Use this for any agent slow enough to risk exceeding an edge/proxy
    timeout (e.g. Cloudflare Tunnel's ~100s) when called synchronously.
    /run_async returns a job_id immediately and runs agent._timed_run() on
    a background thread; the caller polls /status/<job_id>, and each of
    those requests completes in well under a second regardless of how long
    the underlying work takes.

    This is the same job-queue pattern that was previously hand-written
    inline in individual agent files (e.g. summarize_agent.py). Duplicating
    it per-agent is how /run_async ended up outside the HMAC boundary --
    each hand-rolled copy has to remember to also update the middleware,
    and at least one didn't. Routing every agent's async needs through
    this single function means the fix only has to happen once, here.
    """
    _jobs: dict = {}
    _jobs_lock = threading.Lock()

    def _prune_stale_jobs():
        cutoff = time.time() - job_ttl_secs
        with _jobs_lock:
            stale = [jid for jid, j in _jobs.items()
                     if j.get("status") in ("done", "error") and j.get("completed_at", cutoff) < cutoff]
            for jid in stale:
                del _jobs[jid]

    @app.post("/run_async")
    @validate_task_payload
    def run_async():
        from flask import g
        payload = TaskPayload.model_validate(g.task)
        try:
            agent._validate_capability(payload.capability)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 403

        input_schema = agent.input_schema if hasattr(agent, 'input_schema') else {}
        input_rv = validate_task_input(payload.inputs, input_schema)
        if not input_rv.valid:
            return jsonify({
                "error":   "Request contract violation",
                "details": input_rv.errors,
                "task_id": payload.task_id,
            }), 422

        _prune_stale_jobs()

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"status": "running", "result": None, "error": None, "completed_at": None}

        def _worker():
            result = agent._timed_run(payload)
            with _jobs_lock:
                _jobs[job_id] = {
                    "status": "done" if result.status == TaskStatus.SUCCESS else "error",
                    "result": result.model_dump(),
                    "error": result.error_msg,
                    "completed_at": time.time(),
                }

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({"job_id": job_id}), 202

    @app.get("/status/<job_id>")
    def job_status(job_id):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "unknown or expired job_id"}), 404
        return jsonify(job), 200


def _configure_logging() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )
