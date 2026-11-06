"""
byoa_adk/audit.py
~~~~~~~~~~~~~~~~~~
Structured audit logging for agent activity, with two backends:

  - Postgres (preferred): set AUDIT_DB_URL, e.g.
      postgresql://user:pass@host:5432/byoa_audit
    All agents writing to the SAME database means audit history is
    queryable across the whole fleet with one SQL query, not N log files.

  - JSONL file (fallback): used automatically if AUDIT_DB_URL is unset,
    OR if a Postgres write fails for any reason (so a transient DB outage
    never means a lost audit entry -- it just lands in the file instead).

Every agent gets this automatically via build_flask_app() -- it is NOT
something an individual agent has to set up or call itself.

Deliberately does NOT log request/response payload contents by default --
only routing metadata (task_id, goal_id, capability, outcome, timing,
error detail) -- since payloads may contain sensitive business data.
Set AUDIT_LOG_INCLUDE_PAYLOAD=1 to opt into full payload logging for
local debugging; leave it unset in production.

Env vars
--------
AUDIT_DB_URL               : Postgres connection string. If unset, file-only mode.
AUDIT_LOG_PATH             : JSONL fallback/only-mode path. Default: ./audit.log
AUDIT_LOG_INCLUDE_PAYLOAD  : "1" to include raw inputs/outputs. Default: unset (off)
"""
from __future__ import annotations
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("byoa_adk.audit")

_file_lock = threading.Lock()
_pool_lock = threading.Lock()
_pool = None             # lazily created psycopg2 connection pool, or None
_pg_unavailable = False  # set True after the FIRST failed attempt, so we
                         # don't retry a broken connection string on every request

EVENT_HMAC_REJECTED        = "hmac.rejected"
EVENT_INPUT_REJECTED       = "input.rejected"
EVENT_CAPABILITY_REJECTED  = "capability.rejected"
EVENT_OUTPUT_REJECTED      = "output.rejected"
EVENT_RUN_SUCCESS          = "run.success"
EVENT_RUN_FAILURE          = "run.failure"
EVENT_SUSPICIOUS_ACTIVITY  = "suspicious.activity"

_REJECTION_EVENTS = (EVENT_HMAC_REJECTED, EVENT_INPUT_REJECTED, EVENT_CAPABILITY_REJECTED)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    event        TEXT NOT NULL,
    agent        TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    task_id      TEXT,
    goal_id      TEXT,
    capability   TEXT,
    detail       TEXT,
    duration_ms  INTEGER,
    remote_addr  TEXT,
    extra        JSONB
);
CREATE INDEX IF NOT EXISTS idx_audit_log_agent  ON audit_log (agent);
CREATE INDEX IF NOT EXISTS idx_audit_log_event  ON audit_log (event);
CREATE INDEX IF NOT EXISTS idx_audit_log_ts     ON audit_log (ts);
CREATE INDEX IF NOT EXISTS idx_audit_log_taskid ON audit_log (task_id);
"""

_INSERT_SQL = """
INSERT INTO audit_log
    (ts, event, agent, outcome, task_id, goal_id, capability, detail, duration_ms, remote_addr, extra)
VALUES
    (to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _log_path() -> Path:
    return Path(os.environ.get("AUDIT_LOG_PATH", "./audit.log"))


def include_payload() -> bool:
    return os.environ.get("AUDIT_LOG_INCLUDE_PAYLOAD", "0") == "1"


def _get_pool():
    """
    Lazily create a small connection pool on first use. Cached for the
    life of the process. If creation fails once, we stop retrying for
    this process (so a bad AUDIT_DB_URL doesn't add connection-timeout
    latency to every single request) -- fixed by restarting the service
    after correcting the connection string.
    """
    global _pool, _pg_unavailable
    if _pool is not None or _pg_unavailable:
        return _pool

    db_url = os.environ.get("AUDIT_DB_URL")
    if not db_url:
        return None

    with _pool_lock:
        if _pool is not None or _pg_unavailable:
            return _pool
        try:
            import psycopg2
            import psycopg2.pool
            pool = psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=db_url, connect_timeout=3)
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                conn.commit()
            finally:
                pool.putconn(conn)
            _pool = pool
            logger.info("Audit logging: connected to Postgres, table ready.")
        except Exception as exc:
            logger.warning(
                "Audit logging: could not connect to AUDIT_DB_URL (%s). "
                "Falling back to file logging at %s for this process.",
                exc, _log_path(),
            )
            _pg_unavailable = True
            _pool = None
    return _pool


def _write_file(entry: dict) -> None:
    line = json.dumps(entry, default=str)
    try:
        with _file_lock:
            with open(_log_path(), "a") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.warning("Audit logging: failed to write fallback file entry: %s", exc)


def _write_postgres(entry: dict) -> bool:
    """Returns True on success, False if it should fall back to file."""
    pool = _get_pool()
    if pool is None:
        return False
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(_INSERT_SQL, (
                entry["ts"], entry["event"], entry["agent"], entry["outcome"],
                entry["task_id"], entry["goal_id"], entry["capability"], entry["detail"],
                entry["duration_ms"], entry["remote_addr"],
                json.dumps(entry["extra"]) if entry.get("extra") else None,
            ))
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("Audit logging: Postgres write failed (%s); falling back to file for this entry.", exc)
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn is not None:
            pool.putconn(conn)


def check_and_flag_suspicious(
    agent_name: str,
    window_minutes: int = 5,
    threshold: int = 5,
    remote_addr: str | None = None,
) -> bool:
    """
    Look at this agent's own recent rejection history and flag a pattern,
    not a single event. A burst of rejections in a short window is the
    signature of an attack IN PROGRESS -- someone guessing signatures,
    replaying stale requests, or probing capabilities -- and is worth
    catching before any of it succeeds.

    Only meaningful with the Postgres backend (it has to READ past
    entries, not just write new ones). No-ops quietly in file-only mode.

    Returns True if a suspicious.activity event was raised this call.

    Deliberately NOT a compromise detector: this looks at requests
    ARRIVING at /run, not at the agent's own process internals. A fully
    compromised agent process could bypass /run entirely, so this catches
    attackers still trying to get IN, not ones already inside.
    """
    pool = _get_pool()
    if pool is None:
        return False  # file-only mode: no history to query against

    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM audit_log
                WHERE agent = %s
                  AND event = ANY(%s)
                  AND ts > now() - (%s || ' minutes')::interval
                """,
                (agent_name, list(_REJECTION_EVENTS), str(window_minutes)),
            )
            recent_count = cur.fetchone()[0]
    except Exception as exc:
        logger.warning("Audit logging: suspicious-activity check failed: %s", exc)
        return False
    finally:
        if conn is not None:
            pool.putconn(conn)

    if recent_count >= threshold:
        record(
            EVENT_SUSPICIOUS_ACTIVITY,
            agent_name=agent_name,
            outcome="alert",
            detail=f"{recent_count} rejected requests in the last {window_minutes} minute(s) (threshold: {threshold})",
            remote_addr=remote_addr,
        )
        logger.warning(
            "SUSPICIOUS ACTIVITY: agent=%s had %d rejections in %d min (threshold %d)",
            agent_name, recent_count, window_minutes, threshold,
        )
        return True
    return False


def record(
    event: str,
    *,
    agent_name: str,
    outcome: str,
    task_id: Optional[str] = None,
    goal_id: Optional[str] = None,
    capability: Optional[str] = None,
    detail: Optional[str] = None,
    duration_ms: Optional[int] = None,
    remote_addr: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Record one structured audit entry.

    Tries Postgres first if AUDIT_DB_URL is configured; falls back to the
    JSONL file if it's not configured, or if the write fails for any
    reason. Never raises -- a failure to record an audit entry must never
    turn an otherwise-successful request into a failed one.
    """
    entry: dict[str, Any] = {
        "ts"          : time.time(),
        "event"       : event,
        "agent"       : agent_name,
        "outcome"     : outcome,
        "task_id"     : task_id,
        "goal_id"     : goal_id,
        "capability"  : capability,
        "detail"      : detail,
        "duration_ms" : duration_ms,
        "remote_addr" : remote_addr,
        "extra"       : extra,
    }

    if os.environ.get("AUDIT_DB_URL"):
        if _write_postgres(entry):
            return
        # fall through to file on failure

    _write_file(entry)
