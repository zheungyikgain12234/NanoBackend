"""
byoa_adk/request_validator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Flask decorator that validates the incoming TaskPayload from the platform
before the agent's handler runs.

Used automatically by server.py — agent developers never call this directly.

On failure: returns 422 { valid: false, errors: [...] } — agent never executes.
On success: sets flask.g.task and flask.g.inputs for the handler.
"""
from __future__ import annotations
import re
import json as _json
from functools import wraps
from flask import request, jsonify, g
from .schema import ALLOWED_CAPABILITIES

TASK_ID_PAT = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _extract(req) -> dict:
    """Extract TaskPayload fields from JSON body or multipart form-data."""
    ct = req.content_type or ''
    if 'multipart' in ct or 'form' in ct:
        raw = req.form.get('inputs', '{}')
        try: inputs = _json.loads(raw)
        except: inputs = {}
        return {
            'task_id':      req.form.get('task_id', ''),
            'goal_id':      req.form.get('goal_id', ''),
            'capability':   req.form.get('capability', ''),
            'inputs':       inputs,
            'timeout_secs': req.form.get('timeout_secs', '60'),
            'hmac_sig':     req.form.get('hmac_sig', ''),
            'timestamp':    req.form.get('timestamp', '0'),
        }
    body = req.get_json(silent=True, force=True) or {}
    return {
        'task_id':      body.get('task_id', ''),
        'goal_id':      body.get('goal_id', ''),
        'capability':   body.get('capability', ''),
        'inputs':       body.get('inputs', {}),
        'timeout_secs': body.get('timeout_secs', 60),
        'hmac_sig':     body.get('hmac_sig', ''),
        'timestamp':    body.get('timestamp', 0),
    }


def _validate(p: dict) -> list[str]:
    """Validate all TaskPayload fields. Returns list of errors (empty = valid)."""
    errors = []

    # task_id
    tid = str(p.get('task_id', '')).strip()
    if not tid:
        errors.append('task_id is required')
    elif not TASK_ID_PAT.match(tid):
        errors.append(f'task_id must be alphanumeric/hyphens/underscores. Got: "{tid}"')

    # goal_id
    if not str(p.get('goal_id', '')).strip():
        errors.append('goal_id is required')

    # capability
    cap = str(p.get('capability', '')).strip()
    if not cap:
        errors.append('capability is required')
    elif cap not in ALLOWED_CAPABILITIES:
        errors.append(f'Unknown capability: "{cap}". Allowed: {ALLOWED_CAPABILITIES}')

    # inputs
    inputs = p.get('inputs')
    if not isinstance(inputs, dict):
        errors.append(f'inputs must be a JSON object. Got: {type(inputs).__name__}')

    # timeout_secs
    try:
        t = int(p.get('timeout_secs', 60))
        if t <= 0:
            errors.append(f'timeout_secs must be positive. Got: {t}')
    except (TypeError, ValueError):
        errors.append('timeout_secs must be an integer')

    return errors


def validate_task_payload(fn):
    """
    Decorator for Flask route handlers (e.g. POST /run).

    Validates the incoming TaskPayload before the handler runs.
    On pass → sets flask.g.task and flask.g.inputs, calls handler.
    On fail → returns 422 { valid: false, errors: [...] }, handler never called.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        payload = _extract(request)
        errors  = _validate(payload)
        if errors:
            from flask import current_app
            from .audit import record as _audit_record, EVENT_INPUT_REJECTED
            _audit_record(
                EVENT_INPUT_REJECTED,
                agent_name  = current_app.name,   # build_flask_app() sets Flask(agent.name)
                outcome     = "rejected",
                task_id     = payload.get('task_id') or None,
                goal_id     = payload.get('goal_id') or None,
                capability  = payload.get('capability') or None,
                detail      = "; ".join(errors),
                remote_addr = request.remote_addr,
            )
            return jsonify({
                'valid':   False,
                'errors':  errors,
                'task_id': payload.get('task_id') or None,
            }), 422
        try:
            payload['timeout_secs'] = int(payload['timeout_secs'])
            payload['timestamp']    = int(payload['timestamp'])
        except (TypeError, ValueError):
            pass
        g.task   = payload
        g.inputs = payload['inputs']
        return fn(*args, **kwargs)
    return wrapper
