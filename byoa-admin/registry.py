"""
Registry-side admin actions that the registration server doesn't expose.

  unregister_agent -- take an agent out of service but keep its history and
                      credentials. The controller only routes to agents whose
                      status is REGISTERED_ACTIVE, so this takes effect for
                      new lookups straight away. Reversible: register again.
  delete_agent     -- remove the agent entirely: every registry row AND its
                      credential, in one transaction. Not reversible; the agent
                      must be onboarded again with a new secret.
"""
from agent_credentials import get_conn

UNREGISTERED = "UNREGISTERED"


def unregister_agent(agent_id: str) -> int:
    """Marks every version of agent_id as UNREGISTERED. Returns rows changed."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE registered_agents SET status = %s, updated_at = NOW() "
            "WHERE agent_id = %s AND status <> %s",
            (UNREGISTERED, agent_id, UNREGISTERED),
        )
        changed = cur.rowcount
        conn.commit()
        return changed
    finally:
        conn.close()


def delete_agent(agent_id: str) -> dict:
    """Deletes registry rows and the credential together, or neither."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM registered_agents WHERE agent_id = %s", (agent_id,))
        registry_rows = cur.rowcount
        cur.execute("DELETE FROM agent_credentials WHERE agent_id = %s", (agent_id,))
        credential_rows = cur.rowcount
        conn.commit()
        return {"registry_rows": registry_rows, "credential_rows": credential_rows}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
