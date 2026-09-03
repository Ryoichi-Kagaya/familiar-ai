"""Partition companion-facing memory by user.

Existing rows cannot be attributed reliably because older versions did not
store an owner.  Keep them intact under ``__legacy__`` so they can be audited,
but do not inject them into any named user's conversation.
"""

from __future__ import annotations

import sqlite3


LEGACY_USER_ID = "__legacy__"


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


def _add_owner(conn: sqlite3.Connection, table: str) -> None:
    if not _has_column(conn, table, "user_id"):
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT '{LEGACY_USER_ID}'"
        )


def _rebuild_semantic_facts(conn: sqlite3.Connection) -> None:
    if _has_column(conn, "semantic_facts", "user_id"):
        return
    conn.execute(
        """
        CREATE TABLE semantic_facts_scoped (
            id TEXT PRIMARY KEY,
            fact_key TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            source_memory_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            tags TEXT NOT NULL DEFAULT '',
            last_seen_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT '__legacy__',
            UNIQUE(user_id, fact_key)
        )
        """
    )
    conn.execute(
        "INSERT INTO semantic_facts_scoped "
        "(id, fact_key, fact_text, source_memory_id, confidence, tags, last_seen_at, "
        "created_at, updated_at, user_id) "
        "SELECT id, fact_key, fact_text, source_memory_id, confidence, tags, last_seen_at, "
        "created_at, updated_at, ? FROM semantic_facts",
        (LEGACY_USER_ID,),
    )
    conn.execute("DROP TABLE semantic_facts")
    conn.execute("ALTER TABLE semantic_facts_scoped RENAME TO semantic_facts")
    conn.execute(
        "CREATE INDEX idx_semantic_facts_last_seen ON semantic_facts(user_id, last_seen_at)"
    )
    conn.execute("CREATE INDEX idx_semantic_facts_source ON semantic_facts(source_memory_id)")


def _rebuild_behavior_policies(conn: sqlite3.Connection) -> None:
    if _has_column(conn, "behavior_policies", "user_id"):
        return
    conn.execute(
        """
        CREATE TABLE behavior_policies_scoped (
            id TEXT PRIMARY KEY,
            policy_key TEXT NOT NULL,
            policy_text TEXT NOT NULL,
            trigger_context TEXT NOT NULL DEFAULT '',
            action_hint TEXT NOT NULL DEFAULT '',
            source_memory_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            last_seen_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT '__legacy__',
            UNIQUE(user_id, policy_key)
        )
        """
    )
    conn.execute(
        "INSERT INTO behavior_policies_scoped "
        "(id, policy_key, policy_text, trigger_context, action_hint, source_memory_id, "
        "confidence, last_seen_at, created_at, updated_at, user_id) "
        "SELECT id, policy_key, policy_text, trigger_context, action_hint, source_memory_id, "
        "confidence, last_seen_at, created_at, updated_at, ? FROM behavior_policies",
        (LEGACY_USER_ID,),
    )
    conn.execute("DROP TABLE behavior_policies")
    conn.execute("ALTER TABLE behavior_policies_scoped RENAME TO behavior_policies")
    conn.execute(
        "CREATE INDEX idx_behavior_policies_last_seen ON behavior_policies(user_id, last_seen_at)"
    )
    conn.execute("CREATE INDEX idx_behavior_policies_source ON behavior_policies(source_memory_id)")


def upgrade(conn: sqlite3.Connection) -> None:
    _add_owner(conn, "observations")
    _add_owner(conn, "unfinished_business")
    _add_owner(conn, "episodes")
    _add_owner(conn, "memory_revisions")
    _rebuild_semantic_facts(conn)
    _rebuild_behavior_policies(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_user_time ON observations(user_id, timestamp)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_unfinished_user_status "
        "ON unfinished_business(user_id, status, created_at DESC)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id, updated_at)")
