"""
agent_credentials.py
~~~~~~~~~~~~~~~~~~~~~
Per-agent secret storage, replacing the single shared BYOA_SECRET.

Every agent -- yours or a third party's -- gets its OWN secret, stored
in the `agent_credentials` table. This is what makes it safe to let
third parties register and run their own agents: a leaked or stolen
secret only compromises the ONE agent it belongs to, not the whole
platform.

Onboarding a new agent (yours or a third party's) is a deliberate,
platform-owner-controlled action -- see onboard_agent.py. This module
only handles LOOKING UP an already-onboarded agent's secret; it never
creates one on its own.
"""
import os
import secrets as _secrets

import psycopg2

PLATFORM_DB_URL = os.environ["PLATFORM_DB_URL"]


def get_conn():
    return psycopg2.connect(PLATFORM_DB_URL)


def init_credentials_table() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_credentials (
            agent_id     TEXT PRIMARY KEY,
            secret       TEXT        NOT NULL,
            owner_label  TEXT,
            revoked      BOOLEAN     NOT NULL DEFAULT FALSE,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            revoked_at   TIMESTAMPTZ
        )
    """)
    conn.commit()
    conn.close()


def generate_secret() -> str:
    """A fresh, random, sufficiently long secret for a newly onboarded agent."""
    return _secrets.token_urlsafe(32)


def onboard_agent(agent_id: str, owner_label: str = "") -> str:
    """
    Create a brand-new credential for agent_id and return its secret.
    Raises ValueError if agent_id is already onboarded (and not revoked).
    This is a deliberate, one-time action -- the secret is shown here
    and nowhere else; it is not retrievable again after this call.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT revoked FROM agent_credentials WHERE agent_id = %s",
        (agent_id,),
    )
    row = cur.fetchone()
    if row is not None and not row[0]:
        conn.close()
        raise ValueError(f"agent_id {agent_id!r} is already onboarded and active.")

    secret = generate_secret()
    cur.execute("""
        INSERT INTO agent_credentials (agent_id, secret, owner_label, revoked)
        VALUES (%s, %s, %s, FALSE)
        ON CONFLICT (agent_id) DO UPDATE SET
            secret = EXCLUDED.secret, owner_label = EXCLUDED.owner_label,
            revoked = FALSE, revoked_at = NULL, created_at = NOW()
    """, (agent_id, secret, owner_label))
    conn.commit()
    conn.close()
    return secret


def get_agent_secret(agent_id: str) -> str | None:
    """Returns the active secret for agent_id, or None if unknown/revoked."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT secret FROM agent_credentials WHERE agent_id = %s AND revoked = FALSE",
        (agent_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def revoke_agent(agent_id: str) -> bool:
    """Immediately invalidates an agent's secret. Returns True if a row was updated."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE agent_credentials SET revoked = TRUE, revoked_at = NOW() WHERE agent_id = %s AND revoked = FALSE",
        (agent_id,),
    )
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def list_credentials() -> list:
    """Onboarded agents' metadata for the admin UI. Never returns secrets."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT agent_id, owner_label, revoked, created_at, revoked_at "
        "FROM agent_credentials ORDER BY created_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "agent_id": r[0], "owner": r[1] or "", "revoked": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "revoked_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]
