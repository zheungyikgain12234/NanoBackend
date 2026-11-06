"""
Admin user storage for the BYOA admin API.

Users are created ONLY by create_admin.py, run locally on the host -- there
is deliberately no HTTP endpoint that creates or changes admin accounts.
Passwords are stored as salted scrypt hashes (werkzeug), never in plain text.
"""
import psycopg2
from werkzeug.security import check_password_hash, generate_password_hash

from agent_credentials import get_conn

# Compared against when the username doesn't exist, so a wrong username and a
# wrong password take about the same time and give the same answer.
_DUMMY_HASH = generate_password_hash("not-a-real-password")


def init_admin_table() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            username      TEXT PRIMARY KEY,
            password_hash TEXT        NOT NULL,
            disabled      BOOLEAN     NOT NULL DEFAULT FALSE,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    conn.commit()
    conn.close()


def create_admin(username: str, password: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO admin_users (username, password_hash) VALUES (%s, %s)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise ValueError(f"Admin {username!r} already exists.")
    finally:
        conn.close()


def verify_login(username: str, password: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, disabled FROM admin_users WHERE username = %s", (username,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        check_password_hash(_DUMMY_HASH, password)
        return False
    ok = check_password_hash(row[0], password)
    return ok and not row[1]
